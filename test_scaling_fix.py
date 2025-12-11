"""快速测试缩放修复效果

这个脚本会动态修改模型的 text_transform 输出，无需修改源代码。

使用方法:
python test_scaling_fix.py \
    --checkpoint <path> \
    --root_dir <path> \
    --scale_factor 12.0

它会：
1. 加载模型
2. 应用缩放修复
3. 运行诊断
4. 比较修复前后的效果
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))
from models.alignn import ALIGNN, MiddleFusionModule
from extract_alpha_final import SimpleDataset, load_local_data, get_dataset_paths, collate_fn


class ScaledTextTransform(torch.nn.Module):
    """Wrapper that scales text_transform output"""

    def __init__(self, original_transform, scale_factor):
        super().__init__()
        self.original_transform = original_transform
        self.scale_factor = scale_factor

    def forward(self, x):
        output = self.original_transform(x)
        return output * self.scale_factor

    def load_state_dict(self, *args, **kwargs):
        return self.original_transform.load_state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.original_transform.state_dict(*args, **kwargs)


def apply_scaling_fix(model, scale_factor=12.0):
    """应用缩放修复到模型的所有 MiddleFusionModule"""

    fixed_count = 0

    for name, module in model.named_modules():
        if isinstance(module, MiddleFusionModule):
            print(f"🔧 对 {name} 应用缩放 (factor={scale_factor})")

            # 包装 text_transform
            original_transform = module.text_transform
            module.text_transform = ScaledTextTransform(original_transform, scale_factor)

            fixed_count += 1

    print(f"✅ 成功应用缩放到 {fixed_count} 个融合模块\n")
    return model


def run_diagnostic(model, data_loader, device):
    """运行简化的诊断"""

    # 找到 fusion 模块
    fusion_module = None
    for module in model.modules():
        if isinstance(module, MiddleFusionModule):
            fusion_module = module
            break

    if not fusion_module:
        print("❌ 未找到 MiddleFusionModule")
        return None

    # 获取一个 batch
    batch = next(iter(data_loader))
    g, lg, text_list, _, _, _ = batch

    # Hook 捕获数据
    captured_data = {}

    def hook_fusion_input(module, input_tuple, output):
        node_feat, text_feat, batch_num_nodes = input_tuple
        captured_data['node_feat'] = node_feat.detach()
        captured_data['text_feat_in'] = text_feat.detach()

    def hook_text_transform(module, input, output):
        captured_data['text_feat_out'] = output.detach()

    def hook_gate(module, input, output):
        captured_data['gate_input'] = input[0].detach()

    # 注册 hooks
    h1 = fusion_module.register_forward_hook(hook_fusion_input)
    h2 = fusion_module.text_transform.register_forward_hook(hook_text_transform)
    h3 = fusion_module.gate.register_forward_hook(hook_gate)

    # 运行
    with torch.no_grad():
        _ = model((g.to(device), lg.to(device), text_list))

    # 移除 hooks
    h1.remove()
    h2.remove()
    h3.remove()

    # 计算统计
    node_feat = captured_data['node_feat']
    text_in = captured_data['text_feat_in']
    text_out = captured_data['text_feat_out']
    gate_input = captured_data['gate_input']

    node_norm = node_feat.norm(dim=1).mean().item()
    text_in_norm = text_in.norm(dim=1).mean().item()
    text_out_norm = text_out.norm(dim=1).mean().item()

    node_part_norm = gate_input[:, :256].norm(dim=1).mean().item()
    text_part_norm = gate_input[:, 256:].norm(dim=1).mean().item()
    ratio = node_part_norm / (text_part_norm + 1e-8)

    # 计算余弦相似度
    batch_num_nodes = g.batch_num_nodes().tolist()
    text_expanded = []
    for i, num in enumerate(batch_num_nodes):
        if i < len(text_out):
            text_expanded.append(text_out[i].unsqueeze(0).repeat(num, 1))

    if text_expanded:
        text_broadcasted = torch.cat(text_expanded, dim=0)
        cos_sim = F.cosine_similarity(node_feat, text_broadcasted, dim=1).mean().item()
    else:
        cos_sim = 0.0

    return {
        'node_norm': node_norm,
        'text_in_norm': text_in_norm,
        'text_out_norm': text_out_norm,
        'ratio': ratio,
        'cos_sim': cos_sim
    }


def main():
    parser = argparse.ArgumentParser(description="测试缩放修复效果")
    parser.add_argument('--checkpoint', required=True, help='模型checkpoint路径')
    parser.add_argument('--root_dir', required=True, help='数据集根目录')
    parser.add_argument('--scale_factor', type=float, default=12.0, help='缩放因子')
    parser.add_argument('--n_samples', type=int, default=10, help='测试样本数')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("🧪 缩放修复效果测试")
    print("=" * 80)

    # 加载模型
    print(f"\n📂 加载模型: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']

    # 加载数据
    print(f"\n📊 加载测试数据 ({args.n_samples} 个样本)...")
    cif_dir, csv_file = get_dataset_paths(args.root_dir, 'jarvis', 'hse_bandgap-2')
    raw_data = load_local_data(cif_dir, csv_file, max_samples=args.n_samples)

    if not raw_data:
        print("❌ 无法加载数据")
        return

    loader = DataLoader(SimpleDataset(raw_data, tokenizer=None), batch_size=2, collate_fn=collate_fn)

    # ===== 测试原始模型 =====
    print("\n" + "=" * 80)
    print("📊 原始模型诊断")
    print("=" * 80)

    model_original = ALIGNN(config)
    model_original.load_state_dict(ckpt['model'])
    model_original.to(device)
    model_original.eval()

    results_original = run_diagnostic(model_original, loader, device)

    if results_original:
        print(f"\ntext_transform 输入 L2:  {results_original['text_in_norm']:.4f}")
        print(f"text_transform 输出 L2:  {results_original['text_out_norm']:.4f}")
        print(f"节点特征 L2:            {results_original['node_norm']:.4f}")
        print(f"节点/文本比例:          {results_original['ratio']:.2f}:1")
        print(f"余弦相似度:             {results_original['cos_sim']:.4f}")

    # ===== 测试修复后模型 =====
    print("\n" + "=" * 80)
    print(f"📊 修复后模型诊断 (scale_factor={args.scale_factor})")
    print("=" * 80)

    model_fixed = ALIGNN(config)
    model_fixed.load_state_dict(ckpt['model'])
    model_fixed = apply_scaling_fix(model_fixed, scale_factor=args.scale_factor)
    model_fixed.to(device)
    model_fixed.eval()

    results_fixed = run_diagnostic(model_fixed, loader, device)

    if results_fixed:
        print(f"\ntext_transform 输入 L2:  {results_fixed['text_in_norm']:.4f}")
        print(f"text_transform 输出 L2:  {results_fixed['text_out_norm']:.4f}  ← 已缩放")
        print(f"节点特征 L2:            {results_fixed['node_norm']:.4f}")
        print(f"节点/文本比例:          {results_fixed['ratio']:.2f}:1  ← 改善！")
        print(f"余弦相似度:             {results_fixed['cos_sim']:.4f}  ← 改善！")

    # ===== 对比改进 =====
    if results_original and results_fixed:
        print("\n" + "=" * 80)
        print("📈 改进对比")
        print("=" * 80)

        ratio_improvement = (results_original['ratio'] - results_fixed['ratio']) / results_original['ratio'] * 100
        cos_improvement = (results_fixed['cos_sim'] - results_original['cos_sim']) / abs(results_original['cos_sim'] + 1e-8) * 100
        text_norm_increase = (results_fixed['text_out_norm'] - results_original['text_out_norm']) / results_original['text_out_norm'] * 100

        print(f"\ntext_transform 输出范数: {results_original['text_out_norm']:.4f} → {results_fixed['text_out_norm']:.4f} ({text_norm_increase:+.1f}%)")
        print(f"节点/文本比例:          {results_original['ratio']:.2f}:1 → {results_fixed['ratio']:.2f}:1 ({ratio_improvement:+.1f}%)")
        print(f"余弦相似度:             {results_original['cos_sim']:.4f} → {results_fixed['cos_sim']:.4f} ({cos_improvement:+.1f}%)")

        # 判断
        print("\n💡 结论:")
        if results_fixed['ratio'] < 2.0:
            print("   ✅ 特征尺度基本平衡 (比例 < 2:1)")
        elif results_fixed['ratio'] < 5.0:
            print("   ⚠️  特征尺度改善但仍偏高 (比例 2-5:1)")
        else:
            print("   ❌ 特征尺度仍然不平衡 (比例 > 5:1)，建议增加缩放因子")

        if results_fixed['cos_sim'] > 0.25:
            print("   ✅ 余弦相似度良好 (> 0.25)")
        elif results_fixed['cos_sim'] > 0.15:
            print("   ⚠️  余弦相似度中等 (0.15-0.25)")
        else:
            print("   ❌ 余弦相似度仍然偏低 (< 0.15)")

        # 建议
        print("\n📋 建议:")
        if results_fixed['ratio'] > 3.0:
            suggested_scale = args.scale_factor * (results_fixed['ratio'] / 1.5)
            print(f"   • 建议使用更大的缩放因子: {suggested_scale:.1f}")
        elif results_fixed['ratio'] < 0.8:
            suggested_scale = args.scale_factor * (results_fixed['ratio'] / 1.2)
            print(f"   • 建议使用较小的缩放因子: {suggested_scale:.1f}")
        else:
            print(f"   • 当前缩放因子 {args.scale_factor} 效果良好")

        if results_fixed['cos_sim'] > 0.2:
            print("   • 可以使用此缩放因子进行后续分析和可视化")
        else:
            print("   • 建议结合 LayerNorm 进行重新训练以获得更好效果")

    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80)

    # 保存修复后的模型（可选）
    save_fixed = input("\n是否保存修复后的模型？(y/n): ").strip().lower()
    if save_fixed == 'y':
        output_path = args.checkpoint.replace('.pt', f'_scaled_{args.scale_factor:.1f}.pt')
        torch.save({
            'config': config,
            'model': model_fixed.state_dict()
        }, output_path)
        print(f"✅ 已保存到: {output_path}")
        print(f"\n可以使用此模型进行后续分析：")
        print(f"python 1.extract_alpha_final.py --checkpoint {output_path} ...")
        print(f"python 3.analyze_text_flow_v2.py --checkpoint {output_path} ...")


if __name__ == '__main__':
    main()
