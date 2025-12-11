"""深度诊断工具：检查中期融合的有效性

使用方法:
python diagnose_fusion_effectiveness.py --checkpoint <path> --root_dir <path>
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))
from models.alignn import ALIGNN, MiddleFusionModule

from extract_alpha_final import SimpleDataset, load_local_data, get_dataset_paths, collate_fn

def diagnose_fusion_effectiveness(checkpoint_path, root_dir):
    """诊断中期融合的有效性"""

    print("=" * 80)
    print("🔬 中期融合有效性深度诊断")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    print(f"\n📂 加载模型...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = ALIGNN(config)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    # 查找 fusion 模块
    fusion_module = None
    for name, module in model.named_modules():
        if isinstance(module, MiddleFusionModule):
            fusion_module = module
            print(f"✅ 找到融合模块: {name}")
            break

    if not fusion_module:
        print("❌ 未找到 MiddleFusionModule!")
        return

    # 加载数据
    print("\n📊 加载测试数据...")
    cif_dir, csv_file = get_dataset_paths(root_dir, 'jarvis', 'hse_bandgap-2')
    raw_data = load_local_data(cif_dir, csv_file, max_samples=10)
    loader = DataLoader(SimpleDataset(raw_data, tokenizer=None), batch_size=2, collate_fn=collate_fn)

    # ===== 诊断 1: 检查 text_transform 的输出分布 =====
    print("\n" + "=" * 80)
    print("📊 诊断 1: text_transform 输出分析")
    print("=" * 80)

    batch = next(iter(loader))
    g, lg, text_list, _, _, _ = batch

    # Hook text_transform 的输入和输出
    transform_data = {}

    def hook_transform(module, input, output):
        transform_data['input'] = input[0].detach()
        transform_data['output'] = output.detach()

    hook = fusion_module.text_transform.register_forward_hook(hook_transform)

    with torch.no_grad():
        _ = model((g.to(device), lg.to(device), text_list))

    hook.remove()

    text_in = transform_data['input']  # [Batch, 64]
    text_out = transform_data['output']  # [Batch, 256]

    print(f"\n📥 text_transform 输入:")
    print(f"   - 形状: {text_in.shape}")
    print(f"   - L2 范数 (均值): {text_in.norm(dim=1).mean():.4f}")
    print(f"   - 各维度均值: {text_in.mean(dim=0).mean():.4f}")
    print(f"   - 各维度标准差: {text_in.std(dim=0).mean():.4f}")

    print(f"\n📤 text_transform 输出:")
    print(f"   - 形状: {text_out.shape}")
    print(f"   - L2 范数 (均值): {text_out.norm(dim=1).mean():.4f}")
    print(f"   - 各维度均值: {text_out.mean(dim=0).mean():.4f}")
    print(f"   - 各维度标准差: {text_out.std(dim=0).mean():.4f}")

    # ===== 诊断 2: 检查 Gate 输入的特征分布 =====
    print("\n" + "=" * 80)
    print("📊 诊断 2: Gate 网络输入分析")
    print("=" * 80)

    gate_input_data = {}

    def hook_gate(module, input, output):
        gate_input_data['input'] = input[0].detach()
        gate_input_data['output'] = output.detach()

    hook = fusion_module.gate.register_forward_hook(hook_gate)

    with torch.no_grad():
        _ = model((g.to(device), lg.to(device), text_list))

    hook.remove()

    gate_in = gate_input_data['input']  # [Total_Atoms, 512] (256+256)
    gate_out = gate_input_data['output']  # [Total_Atoms, 256]

    print(f"\n📥 Gate 输入 (concat[node_feat, text_feat]):")
    print(f"   - 形状: {gate_in.shape}")
    print(f"   - 前256维 (节点特征) L2范数: {gate_in[:, :256].norm(dim=1).mean():.4f}")
    print(f"   - 后256维 (文本特征) L2范数: {gate_in[:, 256:].norm(dim=1).mean():.4f}")
    print(f"   - 比例: {gate_in[:, :256].norm(dim=1).mean() / gate_in[:, 256:].norm(dim=1).mean():.2f}:1")

    print(f"\n📤 Gate 输出 (Sigmoid 后的 alpha):")
    print(f"   - 形状: {gate_out.shape}")
    print(f"   - 均值: {gate_out.mean():.4f}")
    print(f"   - 标准差: {gate_out.std():.4f}")
    print(f"   - 最小值: {gate_out.min():.4f}")
    print(f"   - 最大值: {gate_out.max():.4f}")

    # 检查是否饱和
    saturated_low = (gate_out < 0.1).float().mean().item()
    saturated_high = (gate_out > 0.9).float().mean().item()
    mid_range = ((gate_out >= 0.3) & (gate_out <= 0.7)).float().mean().item()

    print(f"\n   饱和度分析:")
    print(f"   - 接近 0 (<0.1): {saturated_low*100:.1f}%")
    print(f"   - 接近 1 (>0.9): {saturated_high*100:.1f}%")
    print(f"   - 中间范围 (0.3-0.7): {mid_range*100:.1f}%")

    if mid_range > 0.8:
        print("\n   ⚠️  警告: 80%+ 的 gate 值在 0.3-0.7 范围，缺乏区分度！")

    # ===== 诊断 3: 融合前后的特征变化 =====
    print("\n" + "=" * 80)
    print("📊 诊断 3: 融合前后特征变化")
    print("=" * 80)

    fusion_io_data = {}

    def hook_fusion(module, input, output):
        node_feat, text_feat, batch_num_nodes = input
        fusion_io_data['node_feat_in'] = node_feat.detach()
        fusion_io_data['text_feat_in'] = text_feat.detach()
        fusion_io_data['node_feat_out'] = output.detach()

    hook = fusion_module.register_forward_hook(hook_fusion)

    with torch.no_grad():
        _ = model((g.to(device), lg.to(device), text_list))

    hook.remove()

    node_in = fusion_io_data['node_feat_in']  # [Total_Atoms, 256]
    node_out = fusion_io_data['node_feat_out']  # [Total_Atoms, 256]

    print(f"\n📥 融合前的节点特征:")
    print(f"   - L2 范数: {node_in.norm(dim=1).mean():.4f}")
    print(f"   - 各维度标准差: {node_in.std(dim=0).mean():.4f}")

    print(f"\n📤 融合后的节点特征:")
    print(f"   - L2 范数: {node_out.norm(dim=1).mean():.4f}")
    print(f"   - 各维度标准差: {node_out.std(dim=0).mean():.4f}")

    # 计算融合前后的变化
    diff = node_out - node_in
    diff_norm = diff.norm(dim=1).mean().item()
    relative_change = diff_norm / node_in.norm(dim=1).mean().item()

    print(f"\n📈 融合带来的变化:")
    print(f"   - 绝对变化 (L2范数): {diff_norm:.4f}")
    print(f"   - 相对变化: {relative_change*100:.2f}%")

    if relative_change < 0.05:
        print("\n   ⚠️  警告: 融合带来的变化 < 5%，融合效果微弱！")
    elif relative_change > 0.5:
        print("\n   ⚠️  警告: 融合带来的变化 > 50%，可能过度依赖文本！")
    else:
        print(f"\n   ✅ 融合变化适中")

    # ===== 诊断 4: 检查 gate 和实际融合的关系 =====
    print("\n" + "=" * 80)
    print("📊 诊断 4: Gate 值与融合效果的关系")
    print("=" * 80)

    # 重新计算以获取 gate_values
    with torch.no_grad():
        _ = model((g.to(device), lg.to(device), text_list))

    gate_vals = gate_out  # 之前保存的
    gate_mean = gate_vals.mean(dim=1)  # [Total_Atoms]

    # 计算每个原子的融合变化
    atom_diff = diff.norm(dim=1)  # [Total_Atoms]

    # 计算相关性
    correlation = torch.corrcoef(torch.stack([gate_mean, atom_diff]))[0, 1].item()

    print(f"\n   Gate 值（平均）与融合变化的相关性: {correlation:.4f}")

    if abs(correlation) < 0.1:
        print("   ⚠️  警告: 相关性极低！Gate 值与实际融合效果无关！")
    elif correlation > 0.5:
        print("   ✅ 正相关: Gate 值越大，融合变化越大（符合预期）")
    elif correlation < -0.5:
        print("   ⚠️  负相关: Gate 值与融合变化反向（异常！）")

    # ===== 总结和建议 =====
    print("\n" + "=" * 80)
    print("📋 诊断总结与建议")
    print("=" * 80)

    issues = []
    recommendations = []

    # 检查特征尺度
    scale_ratio = gate_in[:, :256].norm(dim=1).mean() / gate_in[:, 256:].norm(dim=1).mean()
    if scale_ratio > 2.0:
        issues.append("特征尺度不匹配（节点特征远大于文本特征）")
        recommendations.append("在 Gate 前添加 LayerNorm 归一化输入特征")

    # 检查 gate 多样性
    if gate_out.std() < 0.05:
        issues.append("Gate 值缺乏多样性（标准差 < 0.05）")
        recommendations.append("添加多样性正则化或调整 gate 网络结构")

    # 检查融合效果
    if relative_change < 0.05:
        issues.append("融合效果微弱（相对变化 < 5%）")
        recommendations.append("增大文本特征的权重或调整 gate 初始化")

    # 检查 gate 相关性
    if abs(correlation) < 0.1:
        issues.append("Gate 值与融合效果无关")
        recommendations.append("检查训练过程，可能需要重新训练")

    if issues:
        print("\n🚨 发现的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")

        print("\n💡 改进建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print("\n✅ 未发现明显问题，融合机制工作正常")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="诊断中期融合的有效性")
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--root_dir', required=True)
    args = parser.parse_args()

    diagnose_fusion_effectiveness(args.checkpoint, args.root_dir)
