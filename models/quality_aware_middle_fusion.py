"""
质量感知的中期融合模块

解决原始MiddleFusionModule在文本删除/损坏时的鲁棒性问题：
- 原始中期融合：删除全部文本后 MAE = 0.747
- 无中期融合：删除全部文本后 MAE = 0.536

改进方案：在中期融合阶段添加文本质量检测

作者：Claude
日期：2025-12-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityAwareMiddleFusion(nn.Module):
    """质量感知的中期融合模块

    关键改进：
    1. 添加文本质量检测器
    2. 根据质量动态调整文本贡献
    3. 当文本质量低时，自动降低文本影响

    预期效果：
    - 干净文本：性能保持（MAE ≈ 0.25）
    - 删除文本：鲁棒性改善（MAE ≈ 0.58-0.62）
    """

    def __init__(self, node_dim=64, text_dim=64, hidden_dim=128,
                 num_heads=2, dropout=0.1, quality_hidden_dim=64):
        """初始化质量感知中期融合

        Args:
            node_dim: 图节点特征维度
            text_dim: 文本特征维度
            hidden_dim: 融合隐藏层维度
            num_heads: 注意力头数（预留）
            dropout: Dropout率
            quality_hidden_dim: 质量检测隐藏层维度
        """
        super().__init__()
        self.node_dim = node_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim

        # 文本质量检测器（简化版）
        self.quality_detector = nn.Sequential(
            nn.Linear(text_dim, quality_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(quality_hidden_dim, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )

        # 文本转换
        self.text_transform = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )

        # 门控机制（原始）
        self.gate = nn.Sequential(
            nn.Linear(node_dim + node_dim, node_dim),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat, text_feat, batch_num_nodes=None,
                return_diagnostics=False):
        """前向传播

        Args:
            node_feat: 节点特征 [total_nodes, node_dim]
            text_feat: 文本特征 [batch_size, text_dim]
            batch_num_nodes: 每个图的节点数列表
            return_diagnostics: 是否返回诊断信息

        Returns:
            enhanced: 增强的节点特征
            diagnostics: (可选) 诊断信息字典
        """
        batch_size = text_feat.size(0)
        num_nodes = node_feat.size(0)

        # 步骤1: 检测文本质量
        quality_score = self.quality_detector(text_feat)  # [batch_size, 1]

        # 步骤2: 转换文本特征
        text_transformed = self.text_transform(text_feat)  # [batch_size, node_dim]

        # 步骤3: 广播文本特征到所有节点
        if num_nodes != batch_size:
            # 批次图：需要广播
            if batch_num_nodes is not None:
                text_expanded = []
                quality_expanded = []
                for i, num in enumerate(batch_num_nodes):
                    # 广播文本
                    text_expanded.append(
                        text_transformed[i].unsqueeze(0).repeat(num, 1)
                    )
                    # 广播质量分数
                    quality_expanded.append(
                        quality_score[i].unsqueeze(0).repeat(num, 1)
                    )
                text_broadcasted = torch.cat(text_expanded, dim=0)
                quality_broadcasted = torch.cat(quality_expanded, dim=0)
            else:
                # 后备方案
                text_pooled = text_transformed.mean(dim=0, keepdim=True)
                text_broadcasted = text_pooled.repeat(num_nodes, 1)
                quality_broadcasted = quality_score.mean().view(1, 1).repeat(num_nodes, 1)
        else:
            # 已池化特征
            text_broadcasted = text_transformed
            quality_broadcasted = quality_score

        # 步骤4: 计算原始gate值
        gate_input = torch.cat([node_feat, text_broadcasted], dim=-1)
        gate_values = self.gate(gate_input)  # [*, node_dim]

        # 步骤5: 关键改进 - 质量调制
        # 原始：enhanced = node_feat + gate_values * text_broadcasted
        # 改进：enhanced = node_feat + quality * gate_values * text_broadcasted
        effective_gate = quality_broadcasted * gate_values  # [*, node_dim]

        # 步骤6: 融合
        enhanced = node_feat + effective_gate * text_broadcasted
        enhanced = self.layer_norm(enhanced)
        enhanced = self.dropout(enhanced)

        if return_diagnostics:
            diagnostics = {
                'quality_score': quality_score.detach(),  # [batch_size, 1]
                'quality_mean': quality_score.mean().item(),
                'quality_std': quality_score.std().item(),
                'quality_min': quality_score.min().item(),
                'quality_max': quality_score.max().item(),
                'gate_mean': gate_values.mean().item(),
                'effective_gate_mean': effective_gate.mean().item(),
            }
            return enhanced, diagnostics

        return enhanced


class AdaptiveMiddleFusion(nn.Module):
    """自适应中期融合（更激进的质量控制）

    特点：
    1. 基于范数的质量检测（无需训练）
    2. 自动阈值调整
    3. 平滑的质量衰减

    适用场景：
    - 文本质量差异很大
    - 需要更强的鲁棒性
    """

    def __init__(self, node_dim=64, text_dim=64, hidden_dim=128,
                 dropout=0.1, quality_threshold=3.0):
        """初始化自适应中期融合

        Args:
            node_dim: 图节点特征维度
            text_dim: 文本特征维度
            hidden_dim: 融合隐藏层维度
            dropout: Dropout率
            quality_threshold: 质量判断阈值（可学习）
        """
        super().__init__()
        self.node_dim = node_dim
        self.text_dim = text_dim

        # 文本转换
        self.text_transform = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(node_dim + node_dim, node_dim),
            nn.Sigmoid()
        )

        # 可学习的质量阈值
        self.quality_threshold = nn.Parameter(
            torch.tensor(quality_threshold)
        )

        self.layer_norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def detect_quality_by_norm(self, text_feat):
        """基于范数检测文本质量（无需训练）

        直觉：
        - 正常文本：范数较大（多样化的特征）
        - 删除/mask文本：范数较小（特殊token的固定embedding）

        Args:
            text_feat: 文本特征 [batch_size, text_dim]

        Returns:
            quality_score: 质量分数 [batch_size, 1] ∈ [0, 1]
        """
        # 计算L2范数
        feat_norm = torch.norm(text_feat, p=2, dim=-1, keepdim=True)  # [batch, 1]

        # 使用sigmoid将范数映射到[0, 1]
        # 当norm < threshold时，质量分数接近0
        # 当norm > threshold时，质量分数接近1
        quality_score = torch.sigmoid(feat_norm - self.quality_threshold)

        return quality_score

    def forward(self, node_feat, text_feat, batch_num_nodes=None,
                return_diagnostics=False):
        """前向传播

        Args:
            node_feat: 节点特征 [total_nodes, node_dim]
            text_feat: 文本特征 [batch_size, text_dim]
            batch_num_nodes: 每个图的节点数列表
            return_diagnostics: 是否返回诊断信息

        Returns:
            enhanced: 增强的节点特征
            diagnostics: (可选) 诊断信息字典
        """
        batch_size = text_feat.size(0)
        num_nodes = node_feat.size(0)

        # 步骤1: 基于范数检测质量
        quality_score = self.detect_quality_by_norm(text_feat)  # [batch_size, 1]

        # 步骤2: 转换文本特征
        text_transformed = self.text_transform(text_feat)

        # 步骤3: 广播
        if num_nodes != batch_size:
            if batch_num_nodes is not None:
                text_expanded = []
                quality_expanded = []
                for i, num in enumerate(batch_num_nodes):
                    text_expanded.append(
                        text_transformed[i].unsqueeze(0).repeat(num, 1)
                    )
                    quality_expanded.append(
                        quality_score[i].unsqueeze(0).repeat(num, 1)
                    )
                text_broadcasted = torch.cat(text_expanded, dim=0)
                quality_broadcasted = torch.cat(quality_expanded, dim=0)
            else:
                text_pooled = text_transformed.mean(dim=0, keepdim=True)
                text_broadcasted = text_pooled.repeat(num_nodes, 1)
                quality_broadcasted = quality_score.mean().view(1, 1).repeat(num_nodes, 1)
        else:
            text_broadcasted = text_transformed
            quality_broadcasted = quality_score

        # 步骤4: 计算gate
        gate_input = torch.cat([node_feat, text_broadcasted], dim=-1)
        gate_values = self.gate(gate_input)

        # 步骤5: 质量调制
        effective_gate = quality_broadcasted * gate_values

        # 步骤6: 融合
        enhanced = node_feat + effective_gate * text_broadcasted
        enhanced = self.layer_norm(enhanced)
        enhanced = self.dropout(enhanced)

        if return_diagnostics:
            diagnostics = {
                'quality_score': quality_score.detach(),
                'quality_mean': quality_score.mean().item(),
                'quality_min': quality_score.min().item(),
                'quality_max': quality_score.max().item(),
                'quality_threshold': self.quality_threshold.item(),
                'text_norm_mean': torch.norm(text_feat, p=2, dim=-1).mean().item(),
            }
            return enhanced, diagnostics

        return enhanced


# ============================================================================
# 工厂函数
# ============================================================================

def create_quality_aware_middle_fusion(variant='quality_aware', **kwargs):
    """创建质量感知的中期融合模块

    Args:
        variant: 变体类型
            - 'quality_aware': 使用神经网络检测质量
            - 'adaptive': 使用范数检测质量（无需训练）
        **kwargs: 传递给构造函数的参数

    Returns:
        module: 中期融合模块实例
    """
    variants = {
        'quality_aware': QualityAwareMiddleFusion,
        'adaptive': AdaptiveMiddleFusion,
    }

    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. "
                        f"Available: {list(variants.keys())}")

    return variants[variant](**kwargs)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    print("测试质量感知中期融合模块\n")

    # 测试参数
    batch_size = 4
    num_nodes_per_graph = [10, 15, 12, 8]  # 不同图的节点数
    total_nodes = sum(num_nodes_per_graph)
    node_dim = 64
    text_dim = 64

    # 创建测试数据
    node_feat = torch.randn(total_nodes, node_dim)
    text_feat_clean = torch.randn(batch_size, text_dim)  # 干净文本
    text_feat_bad = torch.randn(batch_size, text_dim) * 0.1  # 低质量文本（小范数）

    print(f"输入形状:")
    print(f"  node_feat: {node_feat.shape}")
    print(f"  text_feat_clean: {text_feat_clean.shape}")
    print(f"  batch_num_nodes: {num_nodes_per_graph}")
    print(f"  total_nodes: {total_nodes}")

    # 测试1: QualityAwareMiddleFusion
    print("\n" + "="*80)
    print("测试1: QualityAwareMiddleFusion (神经网络检测)")
    print("="*80)

    fusion1 = QualityAwareMiddleFusion(node_dim=node_dim, text_dim=text_dim)
    fusion1.eval()

    with torch.no_grad():
        # 干净文本
        enhanced_clean, diag_clean = fusion1(
            node_feat, text_feat_clean,
            batch_num_nodes=num_nodes_per_graph,
            return_diagnostics=True
        )
        print(f"\n✅ 干净文本:")
        print(f"  Quality均值: {diag_clean['quality_mean']:.4f}")
        print(f"  Gate均值: {diag_clean['gate_mean']:.4f}")
        print(f"  Effective gate均值: {diag_clean['effective_gate_mean']:.4f}")

        # 低质量文本
        enhanced_bad, diag_bad = fusion1(
            node_feat, text_feat_bad,
            batch_num_nodes=num_nodes_per_graph,
            return_diagnostics=True
        )
        print(f"\n❌ 低质量文本:")
        print(f"  Quality均值: {diag_bad['quality_mean']:.4f}")
        print(f"  Gate均值: {diag_bad['gate_mean']:.4f}")
        print(f"  Effective gate均值: {diag_bad['effective_gate_mean']:.4f}")

    # 测试2: AdaptiveMiddleFusion
    print("\n" + "="*80)
    print("测试2: AdaptiveMiddleFusion (范数检测)")
    print("="*80)

    fusion2 = AdaptiveMiddleFusion(node_dim=node_dim, text_dim=text_dim)
    fusion2.eval()

    with torch.no_grad():
        # 干净文本
        enhanced_clean2, diag_clean2 = fusion2(
            node_feat, text_feat_clean,
            batch_num_nodes=num_nodes_per_graph,
            return_diagnostics=True
        )
        print(f"\n✅ 干净文本:")
        print(f"  Text范数均值: {diag_clean2['text_norm_mean']:.4f}")
        print(f"  Quality均值: {diag_clean2['quality_mean']:.4f}")
        print(f"  Quality阈值: {diag_clean2['quality_threshold']:.4f}")

        # 低质量文本
        enhanced_bad2, diag_bad2 = fusion2(
            node_feat, text_feat_bad,
            batch_num_nodes=num_nodes_per_graph,
            return_diagnostics=True
        )
        print(f"\n❌ 低质量文本:")
        print(f"  Text范数均值: {diag_bad2['text_norm_mean']:.4f}")
        print(f"  Quality均值: {diag_bad2['quality_mean']:.4f}")
        print(f"  Quality阈值: {diag_bad2['quality_threshold']:.4f}")

    print("\n" + "="*80)
    print("✅ 测试完成！")
    print("\n推荐使用：")
    print("  - AdaptiveMiddleFusion: 无需训练质量检测器，即插即用")
    print("  - QualityAwareMiddleFusion: 可学习的质量检测，更灵活")
