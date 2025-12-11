"""诊断脚本：检查 alpha 提取和文本流分析的问题

使用方法:
python debug_alpha_extraction.py --checkpoint <path> --root_dir <path>
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

# 复用数据加载
from extract_alpha_final import SimpleDataset, load_local_data, get_dataset_paths, collate_fn

def diagnose_model(checkpoint_path, root_dir):
    """诊断模型和数据"""

    print("=" * 80)
    print("🔍 Alpha 提取诊断工具")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 设备: {device}")

    # 1. 加载模型
    print(f"\n📂 加载模型: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 检查 config
    if 'config' not in ckpt:
        print("❌ 错误: checkpoint 中没有 'config' 键")
        print(f"   可用的键: {list(ckpt.keys())}")
        return

    config = ckpt['config']
    print(f"✅ Config 加载成功")
    print(f"   - use_middle_fusion: {config.use_middle_fusion}")
    print(f"   - middle_fusion_layers: {config.middle_fusion_layers if config.use_middle_fusion else 'N/A'}")

    if not config.use_middle_fusion:
        print("\n❌ 致命错误: 模型未启用 middle_fusion!")
        print("   请使用启用了 middle_fusion 的模型checkpoint")
        return

    model = ALIGNN(config)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    print("✅ 模型加载成功")

    # 2. 查找 Fusion 模块
    print("\n🔎 检查 MiddleFusionModule...")
    fusion_modules = []
    for name, module in model.named_modules():
        if isinstance(module, MiddleFusionModule):
            fusion_modules.append((name, module))
            print(f"   ✅ 找到: {name}")
            print(f"      - node_dim: {module.node_dim}")
            print(f"      - text_dim: {module.text_dim}")
            print(f"      - hidden_dim: {module.hidden_dim}")

    if not fusion_modules:
        print("   ❌ 错误: 未找到 MiddleFusionModule!")
        return

    fusion_module = fusion_modules[0][1]  # 使用第一个

    # 3. 加载少量数据
    print("\n📊 加载测试数据...")
    cif_dir, csv_file = get_dataset_paths(root_dir, 'jarvis', 'hse_bandgap-2')
    raw_data = load_local_data(cif_dir, csv_file, max_samples=5)

    if not raw_data:
        print("❌ 错误: 无法加载数据")
        return

    print(f"✅ 加载了 {len(raw_data)} 个样本")

    loader = DataLoader(SimpleDataset(raw_data, tokenizer=None),
                       batch_size=2,
                       collate_fn=collate_fn)

    # 4. 前向传播并检查
    print("\n🚀 执行前向传播...")

    batch = next(iter(loader))
    g, lg, text_list, targets, jids, atom_types_list = batch

    print(f"   - Batch size: {len(text_list)}")
    print(f"   - 图节点数: {g.batch_num_nodes().tolist()}")
    print(f"   - 文本示例: '{text_list[0][:100]}...'")

    # === 关键: 添加 Hook 捕获中间值 ===
    captured_data = {}

    def capture_fusion_input(module, input_tuple, output):
        """捕获 fusion 模块的输入和输出"""
        node_feat, text_feat, batch_num_nodes = input_tuple
        captured_data['node_feat_shape'] = node_feat.shape
        captured_data['text_feat_shape'] = text_feat.shape
        captured_data['node_feat_norm'] = node_feat.norm(dim=1).mean().item()
        captured_data['text_feat_norm'] = text_feat.norm(dim=1).mean().item()
        captured_data['output_norm'] = output.norm(dim=1).mean().item()

    def capture_gate_values(module, input_tuple, output):
        """捕获 gate 的输出"""
        # gate 是一个 Sequential，我们Hook它的输出
        captured_data['gate_output'] = output.detach()

    # 注册 Hooks
    fusion_hook = fusion_module.register_forward_hook(capture_fusion_input)
    gate_hook = fusion_module.gate.register_forward_hook(capture_gate_values)

    with torch.no_grad():
        _ = model((g.to(device), lg.to(device), text_list))

    fusion_hook.remove()
    gate_hook.remove()

    # 5. 检查捕获的数据
    print("\n📈 Fusion 模块诊断:")
    print(f"   - 节点特征形状: {captured_data.get('node_feat_shape', 'N/A')}")
    print(f"   - 文本特征形状: {captured_data.get('text_feat_shape', 'N/A')}")
    print(f"   - 节点特征L2范数(均值): {captured_data.get('node_feat_norm', 0):.4f}")
    print(f"   - 文本特征L2范数(均值): {captured_data.get('text_feat_norm', 0):.4f}")
    print(f"   - 输出特征L2范数(均值): {captured_data.get('output_norm', 0):.4f}")

    # 6. 检查 stored_alphas
    print("\n🔍 检查 stored_alphas:")
    if hasattr(fusion_module, 'stored_alphas') and fusion_module.stored_alphas is not None:
        alphas = fusion_module.stored_alphas.numpy()
        print(f"   ✅ stored_alphas 形状: {alphas.shape}")
        print(f"   - 均值: {alphas.mean():.4f}")
        print(f"   - 标准差: {alphas.std():.4f}")
        print(f"   - 最小值: {alphas.min():.4f}")
        print(f"   - 最大值: {alphas.max():.4f}")
        print(f"   - 25%分位: {np.percentile(alphas, 25):.4f}")
        print(f"   - 50%分位: {np.percentile(alphas, 50):.4f}")
        print(f"   - 75%分位: {np.percentile(alphas, 75):.4f}")
    else:
        print("   ❌ stored_alphas 为空!")

    # 7. 检查 gate_values 的原始值
    if 'gate_output' in captured_data:
        gate_vals = captured_data['gate_output']
        print(f"\n🔍 检查原始 gate_values (Sigmoid 后):")
        print(f"   - 形状: {gate_vals.shape}")
        print(f"   - 均值: {gate_vals.mean().item():.4f}")
        print(f"   - 标准差: {gate_vals.std().item():.4f}")
        print(f"   - 最小值: {gate_vals.min().item():.4f}")
        print(f"   - 最大值: {gate_vals.max().item():.4f}")

        # 检查是否有梯度饱和
        if gate_vals.min().item() > 0.3 and gate_vals.max().item() < 0.5:
            print("   ⚠️  警告: Gate 值范围太窄 (0.3-0.5)，可能发生了饱和!")
            print("      建议检查:")
            print("      1. Gate 网络的权重初始化")
            print("      2. 输入特征的归一化")
            print("      3. 训练过程中的学习率")

    # 8. 分析文本特征和节点特征的余弦相似度
    print("\n🔍 计算文本-节点余弦相似度:")
    node_feat_raw = captured_data.get('node_feat_shape')
    text_feat_raw = captured_data.get('text_feat_shape')

    # 重新运行一次以获取实际tensor（之前只保存了shape）
    captured_tensors = {}

    def capture_tensors(module, input_tuple, output):
        node_feat, text_feat, batch_num_nodes = input_tuple
        captured_tensors['node_feat'] = node_feat.detach()
        captured_tensors['text_feat'] = text_feat.detach()
        captured_tensors['batch_num_nodes'] = batch_num_nodes

    hook = fusion_module.register_forward_hook(capture_tensors)
    with torch.no_grad():
        _ = model((g.to(device), lg.to(device), text_list))
    hook.remove()

    if 'node_feat' in captured_tensors and 'text_feat' in captured_tensors:
        node_feat = captured_tensors['node_feat']
        text_feat = captured_tensors['text_feat']
        batch_num_nodes = captured_tensors['batch_num_nodes']

        # 广播text特征到节点
        text_expanded = []
        for i, num in enumerate(batch_num_nodes):
            text_expanded.append(text_feat[i].unsqueeze(0).repeat(num, 1))
        text_broadcasted = torch.cat(text_expanded, dim=0)

        # 计算余弦相似度
        cos_sim = F.cosine_similarity(node_feat, text_broadcasted, dim=1)
        print(f"   - 余弦相似度均值: {cos_sim.mean().item():.4f}")
        print(f"   - 余弦相似度标准差: {cos_sim.std().item():.4f}")
        print(f"   - 余弦相似度范围: [{cos_sim.min().item():.4f}, {cos_sim.max().item():.4f}]")

        if abs(cos_sim.mean().item()) < 0.1:
            print("   ⚠️  警告: 余弦相似度极低! 可能原因:")
            print("      1. 文本和图特征在不同的向量空间")
            print("      2. 特征归一化问题")
            print("      3. 文本特征没有有效融入图编码")

    print("\n" + "=" * 80)
    print("✅ 诊断完成")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="诊断 alpha 提取问题")
    parser.add_argument('--checkpoint', required=True, help='模型checkpoint路径')
    parser.add_argument('--root_dir', required=True, help='数据集根目录')
    args = parser.parse_args()

    diagnose_model(args.checkpoint, args.root_dir)
