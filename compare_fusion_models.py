"""比较原始模型和改进模型的融合效果

这个脚本会并排比较两个模型：
1. 原始模型（或手动缩放的模型）
2. 使用 LayerNorm + 可学习缩放重新训练的模型

使用方法:
python compare_fusion_models.py \
    --model1 best_test_model_scaled_12.0.pt \
    --model2 outputs/improved_fusion/best_model.pt \
    --root_dir <data_root> \
    --n_samples 50
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


def analyze_model(model, data_loader, device, model_name):
    """分析单个模型的融合效果"""

    print(f"\n{'=' * 80}")
    print(f"📊 分析模型: {model_name}")
    print(f"{'=' * 80}")

    # 找到 fusion 模块
    fusion_module = None
    for name, module in model.named_modules():
        if isinstance(module, MiddleFusionModule):
            fusion_module = module
            fusion_module_name = name
            break

    if not fusion_module:
        print("❌ 未找到 MiddleFusionModule")
        return None

    print(f"✅ 找到 MiddleFusionModule: {fusion_module_name}")

    # 检查是否使用了新功能
    has_gate_norm = hasattr(fusion_module, 'gate_norm') and fusion_module.use_gate_norm
    has_learnable_scale = hasattr(fusion_module, 'use_learnable_scale') and fusion_module.use_learnable_scale

    print(f"\n功能检测:")
    print(f"   Gate LayerNorm: {'✅ 启用' if has_gate_norm else '❌ 未启用'}")
    print(f"   可学习缩放: {'✅ 启用' if has_learnable_scale else '❌ 未启用'}")

    if has_learnable_scale:
        text_scale = fusion_module.text_scale.item()
        print(f"   text_scale 值: {text_scale:.4f}")

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

    def hook_gate_input(module, input, output):
        captured_data['gate_input'] = input[0].detach()

    def hook_gate_values(module, input, output):
        captured_data['gate_values'] = output.detach()

    # 注册 hooks
    h1 = fusion_module.register_forward_hook(hook_fusion_input)
    h2 = fusion_module.text_transform.register_forward_hook(hook_text_transform)
    h3 = fusion_module.gate.register_forward_hook(hook_gate_input)
    h4 = fusion_module.gate.register_forward_hook(hook_gate_values)

    # 运行
    with torch.no_grad():
        _ = model((g.to(device), lg.to(device), text_list))

    # 移除 hooks
    h1.remove()
    h2.remove()
    h3.remove()
    h4.remove()

    # 计算统计
    node_feat = captured_data['node_feat']
    text_in = captured_data['text_feat_in']
    text_out = captured_data['text_feat_out']
    gate_input = captured_data['gate_input']
    gate_values = captured_data['gate_values']

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

    # 计算 Alpha 统计
    alpha_values = gate_values.mean(dim=1).cpu().numpy()
    alpha_mean = alpha_values.mean()
    alpha_std = alpha_values.std()
    alpha_min = alpha_values.min()
    alpha_max = alpha_values.max()

    results = {
        'model_name': model_name,
        'has_gate_norm': has_gate_norm,
        'has_learnable_scale': has_learnable_scale,
        'text_scale': text_scale if has_learnable_scale else None,
        'node_norm': node_norm,
        'text_in_norm': text_in_norm,
        'text_out_norm': text_out_norm,
        'ratio': ratio,
        'cos_sim': cos_sim,
        'alpha_mean': alpha_mean,
        'alpha_std': alpha_std,
        'alpha_min': alpha_min,
        'alpha_max': alpha_max,
        'alpha_range': alpha_max - alpha_min
    }

    # 打印结果
    print(f"\n特征范数分析:")
    print(f"   text_transform 输入 L2:  {text_in_norm:8.4f}")
    print(f"   text_transform 输出 L2:  {text_out_norm:8.4f}")
    print(f"   节点特征 L2:            {node_norm:8.4f}")
    print(f"   节点/文本比例:          {ratio:8.2f}:1")

    print(f"\n融合效果:")
    print(f"   余弦相似度:             {cos_sim:8.4f}")

    print(f"\nAlpha 统计:")
    print(f"   均值:                   {alpha_mean:8.4f}")
    print(f"   标准差:                 {alpha_std:8.4f}")
    print(f"   范围:                   [{alpha_min:.4f}, {alpha_max:.4f}]")
    print(f"   跨度:                   {alpha_range:8.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="比较两个模型的融合效果")
    parser.add_argument('--model1', required=True, help='模型 1 路径（原始或手动缩放）')
    parser.add_argument('--model2', required=True, help='模型 2 路径（改进训练）')
    parser.add_argument('--root_dir', required=True, help='数据集根目录')
    parser.add_argument('--n_samples', type=int, default=50, help='测试样本数')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("🔬 模型融合效果对比")
    print("=" * 80)

    # 加载数据
    print(f"\n📊 加载测试数据 ({args.n_samples} 个样本)...")
    cif_dir, csv_file = get_dataset_paths(args.root_dir, 'jarvis', 'hse_bandgap-2')
    raw_data = load_local_data(cif_dir, csv_file, max_samples=args.n_samples)

    if not raw_data:
        print("❌ 无法加载数据")
        return

    loader = DataLoader(SimpleDataset(raw_data, tokenizer=None), batch_size=4, collate_fn=collate_fn)

    # 加载模型 1
    print(f"\n📂 加载模型 1: {args.model1}")
    ckpt1 = torch.load(args.model1, map_location=device, weights_only=False)
    config1 = ckpt1['config']
    model1 = ALIGNN(config1)
    model1.load_state_dict(ckpt1['model'])
    model1.to(device)
    model1.eval()

    # 加载模型 2
    print(f"📂 加载模型 2: {args.model2}")
    ckpt2 = torch.load(args.model2, map_location=device, weights_only=False)
    config2 = ckpt2['config']
    model2 = ALIGNN(config2)
    model2.load_state_dict(ckpt2['model'])
    model2.to(device)
    model2.eval()

    # 分析两个模型
    results1 = analyze_model(model1, loader, device, "Model 1 (Original/Manual Scaling)")
    results2 = analyze_model(model2, loader, device, "Model 2 (Improved Training)")

    if results1 is None or results2 is None:
        print("\n❌ 分析失败")
        return

    # 对比分析
    print(f"\n{'=' * 80}")
    print("📈 对比分析")
    print(f"{'=' * 80}")

    def calc_improvement(old_val, new_val):
        if abs(old_val) < 1e-8:
            return 0.0
        return (new_val - old_val) / abs(old_val) * 100

    print(f"\n{'指标':<30} {'Model 1':<15} {'Model 2':<15} {'变化':<15}")
    print("-" * 80)

    # 特征范数
    print(f"{'text_transform 输出 L2':<30} {results1['text_out_norm']:>14.4f} {results2['text_out_norm']:>14.4f} {calc_improvement(results1['text_out_norm'], results2['text_out_norm']):>13.1f}%")
    print(f"{'节点/文本比例':<30} {results1['ratio']:>13.2f}:1 {results2['ratio']:>13.2f}:1 {calc_improvement(results1['ratio'], results2['ratio']):>13.1f}%")

    # 融合效果
    print(f"{'余弦相似度':<30} {results1['cos_sim']:>14.4f} {results2['cos_sim']:>14.4f} {calc_improvement(results1['cos_sim'], results2['cos_sim']):>13.1f}%")

    # Alpha 统计
    print(f"{'Alpha 均值':<30} {results1['alpha_mean']:>14.4f} {results2['alpha_mean']:>14.4f} {calc_improvement(results1['alpha_mean'], results2['alpha_mean']):>13.1f}%")
    print(f"{'Alpha 标准差':<30} {results1['alpha_std']:>14.4f} {results2['alpha_std']:>14.4f} {calc_improvement(results1['alpha_std'], results2['alpha_std']):>13.1f}%")
    print(f"{'Alpha 范围':<30} {results1['alpha_range']:>14.4f} {results2['alpha_range']:>14.4f} {calc_improvement(results1['alpha_range'], results2['alpha_range']):>13.1f}%")

    # 评估改进
    print(f"\n{'=' * 80}")
    print("💡 改进评估")
    print(f"{'=' * 80}")

    improvements = []

    # 评估余弦相似度改进
    cos_improvement = calc_improvement(results1['cos_sim'], results2['cos_sim'])
    if cos_improvement > 10:
        improvements.append(f"✅ 余弦相似度显著提升 ({cos_improvement:+.1f}%)")
    elif cos_improvement > 0:
        improvements.append(f"⚠️  余弦相似度小幅提升 ({cos_improvement:+.1f}%)")
    else:
        improvements.append(f"❌ 余弦相似度下降 ({cos_improvement:+.1f}%)")

    # 评估 Alpha 多样性改进
    alpha_std_improvement = calc_improvement(results1['alpha_std'], results2['alpha_std'])
    if alpha_std_improvement > 20:
        improvements.append(f"✅ Alpha 多样性显著提升 ({alpha_std_improvement:+.1f}%)")
    elif alpha_std_improvement > 0:
        improvements.append(f"⚠️  Alpha 多样性小幅提升 ({alpha_std_improvement:+.1f}%)")
    else:
        improvements.append(f"❌ Alpha 多样性下降 ({alpha_std_improvement:+.1f}%)")

    # 评估特征平衡改进
    ratio_improvement = calc_improvement(results1['ratio'], results2['ratio'])
    if ratio_improvement < -20:  # 比例降低是好事
        improvements.append(f"✅ 特征尺度平衡显著改善 ({ratio_improvement:+.1f}%)")
    elif ratio_improvement < 0:
        improvements.append(f"⚠️  特征尺度平衡小幅改善 ({ratio_improvement:+.1f}%)")
    else:
        improvements.append(f"❌ 特征尺度平衡恶化 ({ratio_improvement:+.1f}%)")

    for item in improvements:
        print(f"\n{item}")

    # 功能对比
    print(f"\n{'=' * 80}")
    print("🔧 功能对比")
    print(f"{'=' * 80}")

    print(f"\nModel 1:")
    print(f"   Gate LayerNorm: {'✅ 启用' if results1['has_gate_norm'] else '❌ 未启用'}")
    print(f"   可学习缩放: {'✅ 启用' if results1['has_learnable_scale'] else '❌ 未启用'}")
    if results1['text_scale'] is not None:
        print(f"   text_scale: {results1['text_scale']:.4f}")

    print(f"\nModel 2:")
    print(f"   Gate LayerNorm: {'✅ 启用' if results2['has_gate_norm'] else '❌ 未启用'}")
    print(f"   可学习缩放: {'✅ 启用' if results2['has_learnable_scale'] else '❌ 未启用'}")
    if results2['text_scale'] is not None:
        print(f"   text_scale: {results2['text_scale']:.4f}")

        # 如果两个模型都有 text_scale，比较学习到的值
        if results1['text_scale'] is not None:
            scale_change = results2['text_scale'] - results1['text_scale']
            print(f"\ntext_scale 变化: {results1['text_scale']:.4f} → {results2['text_scale']:.4f} ({scale_change:+.4f})")

            if abs(scale_change) < 1.0:
                print("   ✅ text_scale 基本稳定，初始值已接近最优")
            elif scale_change > 0:
                print(f"   ⚠️  text_scale 增加了 {scale_change:.2f}，模型需要更强的文本特征")
            else:
                print(f"   ⚠️  text_scale 减少了 {abs(scale_change):.2f}，模型调整了文本权重")

    # 总结
    print(f"\n{'=' * 80}")
    print("✅ 对比完成")
    print(f"{'=' * 80}")

    print(f"\n建议:")
    if cos_improvement > 10 and alpha_std_improvement > 10:
        print("   ✅ Model 2 在融合效果和 Alpha 多样性上都有显著改进")
        print("   ✅ 建议使用 Model 2 进行后续分析和论文图表生成")
    elif cos_improvement > 0 and alpha_std_improvement > 0:
        print("   ⚠️  Model 2 有改进，但提升不显著")
        print("   ⚠️  可以尝试调整 initial_scale 或启用更多融合层重新训练")
    else:
        print("   ❌ Model 2 未能带来明显改进")
        print("   ❌ 建议检查训练配置或使用 Model 1（手动缩放）")


if __name__ == '__main__':
    main()
