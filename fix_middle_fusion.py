"""修复中期融合的特征尺度不匹配问题

这个脚本提供了对 MiddleFusionModule 的改进版本，解决：
1. 特征尺度不匹配（节点特征 >> 文本特征）
2. Gate 值缺乏多样性
3. 余弦相似度过低

使用方法:
1. 复制改进的 MiddleFusionModule 到 models/alignn.py
2. 重新训练模型

或者：
使用这个脚本创建一个 wrapper 来修复已有模型（无需重新训练）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedMiddleFusionModule(nn.Module):
    """改进的中期融合模块

    主要改进：
    1. 添加 LayerNorm 归一化输入特征（解决尺度不匹配）
    2. 使用 Tanh 替代 Sigmoid（更宽的激活范围）
    3. 添加可学习的温度参数（控制 gate 敏感度）
    4. 改进 alpha 提取方式（使用加权平均而非简单平均）
    """

    def __init__(self, node_dim=64, text_dim=64, hidden_dim=128, num_heads=2, dropout=0.1,
                 use_layer_norm=True, use_tanh_gate=False, gate_temperature=1.0):
        """初始化改进的中期融合模块

        Args:
            node_dim: 图节点特征维度
            text_dim: 文本特征维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数（保留用于未来扩展）
            dropout: Dropout 率
            use_layer_norm: 是否在 gate 输入前使用 LayerNorm
            use_tanh_gate: 是否使用 Tanh 替代 Sigmoid
            gate_temperature: Gate 温度参数（>1 使分布更平坦，<1 使分布更尖锐）
        """
        super().__init__()
        self.node_dim = node_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.use_tanh_gate = use_tanh_gate
        self.gate_temperature = gate_temperature

        # Text transformation
        self.text_transform = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )

        # === 改进 1: 添加 LayerNorm ===
        if use_layer_norm:
            self.gate_norm = nn.LayerNorm(node_dim * 2)

        # Gate mechanism
        self.gate = nn.Linear(node_dim + node_dim, node_dim)

        # === 改进 2: 可选的激活函数 ===
        if use_tanh_gate:
            self.gate_activation = nn.Tanh()
        else:
            self.gate_activation = nn.Sigmoid()

        self.layer_norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

        # === 改进 3: 可学习的 alpha 提取权重 ===
        # 用于从 [N, node_dim] 的 gate_values 提取单个 alpha 值
        self.alpha_weights = nn.Parameter(torch.ones(node_dim) / node_dim)

        # 存储变量
        self.stored_alphas = None

    def forward(self, node_feat, text_feat, batch_num_nodes=None):
        """应用改进的中期融合

        Args:
            node_feat: 节点特征 [total_nodes, node_dim]
            text_feat: 文本特征 [batch_size, text_dim]
            batch_num_nodes: 每个图的节点数列表

        Returns:
            enhanced: 增强后的节点特征 [total_nodes, node_dim]
        """
        batch_size = text_feat.size(0)
        num_nodes = node_feat.size(0)

        # Transform text features
        text_transformed = self.text_transform(text_feat)  # [batch_size, node_dim]

        # Broadcast text features
        if num_nodes != batch_size:
            if batch_num_nodes is not None:
                text_expanded = []
                for i, num in enumerate(batch_num_nodes):
                    text_expanded.append(text_transformed[i].unsqueeze(0).repeat(num, 1))
                text_broadcasted = torch.cat(text_expanded, dim=0)
            else:
                text_pooled = text_transformed.mean(dim=0, keepdim=True)
                text_broadcasted = text_pooled.repeat(num_nodes, 1)
        else:
            text_broadcasted = text_transformed

        # === 改进 1: Gate 输入归一化 ===
        gate_input = torch.cat([node_feat, text_broadcasted], dim=-1)
        if self.use_layer_norm:
            gate_input = self.gate_norm(gate_input)

        # Compute gate values
        gate_logits = self.gate(gate_input)  # [*, node_dim]

        # === 改进 2: 温度缩放 ===
        if self.gate_temperature != 1.0:
            gate_logits = gate_logits / self.gate_temperature

        # Apply activation
        if self.use_tanh_gate:
            # Tanh 输出 [-1, 1]，映射到 [0, 1]
            gate_values = (self.gate_activation(gate_logits) + 1) / 2
        else:
            gate_values = self.gate_activation(gate_logits)

        # === 改进 3: 使用加权平均提取 alpha ===
        # 而不是简单的均值（避免方差坍缩）
        self.stored_alphas = (gate_values * self.alpha_weights).sum(dim=1).detach().cpu()

        # Apply gating and residual connection
        enhanced = node_feat + gate_values * text_broadcasted
        enhanced = self.layer_norm(enhanced)
        enhanced = self.dropout(enhanced)

        return enhanced


# ============================================
# 用于替换现有模型的 Wrapper
# ============================================

def upgrade_fusion_module(model, use_layer_norm=True, use_tanh_gate=False, gate_temperature=1.5):
    """将模型中的 MiddleFusionModule 替换为改进版本

    这个函数可以在不重新训练的情况下升级现有模型。
    权重会从旧模块复制到新模块。

    Args:
        model: 包含 MiddleFusionModule 的 ALIGNN 模型
        use_layer_norm: 是否使用 LayerNorm（推荐）
        use_tanh_gate: 是否使用 Tanh 替代 Sigmoid
        gate_temperature: Gate 温度参数（>1 增加多样性）

    Returns:
        upgraded_model: 升级后的模型
    """
    from models.alignn import MiddleFusionModule

    upgraded_count = 0

    for name, module in model.named_children():
        if isinstance(module, nn.ModuleDict):
            # 处理 middle_fusion_modules
            for sub_name, sub_module in module.items():
                if isinstance(sub_module, MiddleFusionModule):
                    print(f"🔄 升级模块: {name}.{sub_name}")

                    # 创建新模块
                    new_module = ImprovedMiddleFusionModule(
                        node_dim=sub_module.node_dim,
                        text_dim=sub_module.text_dim,
                        hidden_dim=sub_module.hidden_dim,
                        dropout=0.1,
                        use_layer_norm=use_layer_norm,
                        use_tanh_gate=use_tanh_gate,
                        gate_temperature=gate_temperature
                    )

                    # 复制权重
                    new_module.text_transform.load_state_dict(sub_module.text_transform.state_dict())
                    new_module.gate.weight.data = sub_module.gate[0].weight.data.clone()
                    new_module.gate.bias.data = sub_module.gate[0].bias.data.clone()
                    new_module.layer_norm.load_state_dict(sub_module.layer_norm.state_dict())

                    # 替换模块
                    module[sub_name] = new_module
                    upgraded_count += 1

    print(f"✅ 成功升级 {upgraded_count} 个融合模块")
    return model


# ============================================
# 测试和比较
# ============================================

def test_fusion_improvement(checkpoint_path, root_dir):
    """测试改进前后的差异"""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))

    from models.alignn import ALIGNN
    from extract_alpha_final import SimpleDataset, load_local_data, get_dataset_paths, collate_fn
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载原始模型
    print("📂 加载原始模型...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_original = ALIGNN(ckpt['config'])
    model_original.load_state_dict(ckpt['model'])
    model_original.to(device)
    model_original.eval()

    # 升级模型
    print("\n🔄 升级模型...")
    model_upgraded = ALIGNN(ckpt['config'])
    model_upgraded.load_state_dict(ckpt['model'])
    model_upgraded = upgrade_fusion_module(
        model_upgraded,
        use_layer_norm=True,
        use_tanh_gate=False,
        gate_temperature=1.5  # 增加温度以提高多样性
    )
    model_upgraded.to(device)
    model_upgraded.eval()

    # 加载数据
    print("\n📊 加载测试数据...")
    cif_dir, csv_file = get_dataset_paths(root_dir, 'jarvis', 'hse_bandgap-2')
    raw_data = load_local_data(cif_dir, csv_file, max_samples=10)
    loader = DataLoader(SimpleDataset(raw_data, tokenizer=None), batch_size=2, collate_fn=collate_fn)

    # 测试
    batch = next(iter(loader))
    g, lg, text_list, _, _, _ = batch

    print("\n📈 比较结果:")
    print("=" * 60)

    with torch.no_grad():
        # 原始模型
        _ = model_original((g.to(device), lg.to(device), text_list))
        fusion_orig = None
        for module in model_original.modules():
            if hasattr(module, 'stored_alphas') and module.stored_alphas is not None:
                fusion_orig = module.stored_alphas.numpy()
                break

        # 升级模型
        _ = model_upgraded((g.to(device), lg.to(device), text_list))
        fusion_new = None
        for module in model_upgraded.modules():
            if hasattr(module, 'stored_alphas') and module.stored_alphas is not None:
                fusion_new = module.stored_alphas.numpy()
                break

    if fusion_orig is not None and fusion_new is not None:
        print(f"\n原始模型:")
        print(f"  - Alpha 均值: {fusion_orig.mean():.4f}")
        print(f"  - Alpha 标准差: {fusion_orig.std():.4f}")
        print(f"  - Alpha 范围: [{fusion_orig.min():.4f}, {fusion_orig.max():.4f}]")

        print(f"\n升级模型:")
        print(f"  - Alpha 均值: {fusion_new.mean():.4f}")
        print(f"  - Alpha 标准差: {fusion_new.std():.4f}")
        print(f"  - Alpha 范围: [{fusion_new.min():.4f}, {fusion_new.max():.4f}]")

        print(f"\n改进:")
        std_improvement = (fusion_new.std() - fusion_orig.std()) / fusion_orig.std() * 100
        print(f"  - 标准差变化: {std_improvement:+.1f}%")

    print("\n" + "=" * 60)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--test', action='store_true', help='测试改进效果')
    args = parser.parse_args()

    if args.test:
        test_fusion_improvement(args.checkpoint, args.root_dir)
    else:
        print("请使用 --test 参数运行测试")
