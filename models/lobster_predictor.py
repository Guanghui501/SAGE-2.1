"""
LOBSTER特征预测器

功能：从晶体结构预测化学键合特征（ICOHP, ICOBI），无需运行LOBSTER计算

架构：
1. ICOHPPredictor: 预测每条边的ICOHP值
2. 使用图神经网络直接从结构学习键合性质
3. 支持不确定性量化（用于质量控制）

应用场景：
- 为所有JARVIS数据生成伪LOBSTER特征
- 快速估计材料的化学键强度
- 材料筛选和高通量预测

作者：Claude
日期：2025-12-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import EdgeGatedGraphConv
import numpy as np
from typing import Dict, Tuple, Optional


class EdgeFeatureEncoder(nn.Module):
    """边特征编码器

    将原子对的几何和化学信息编码为边特征
    """

    def __init__(self, atom_feature_dim=92, edge_hidden_dim=128, dropout=0.1):
        """初始化

        Args:
            atom_feature_dim: 原子特征维度（CGCNN特征）
            edge_hidden_dim: 边特征隐藏层维度
            dropout: Dropout率
        """
        super().__init__()

        # 键长编码（RBF expansion）
        self.rbf_expansion = RBFExpansion(
            vmin=0,
            vmax=8.0,
            bins=40
        )

        # 原子对特征编码
        # 输入：atom_i特征 + atom_j特征 + 距离RBF
        input_dim = atom_feature_dim * 2 + 40

        self.edge_encoder = nn.Sequential(
            nn.Linear(input_dim, edge_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(edge_hidden_dim)
        )

    def forward(self, g, node_features):
        """前向传播

        Args:
            g: DGL图
            node_features: 节点特征 [num_nodes, atom_feature_dim]

        Returns:
            edge_features: 边特征 [num_edges, edge_hidden_dim]
        """
        src, dst = g.edges()

        # 获取原子对特征
        atom_i_feat = node_features[src]
        atom_j_feat = node_features[dst]

        # 获取距离并编码
        distances = torch.norm(g.edata['r'], dim=-1, keepdim=True)
        distance_rbf = self.rbf_expansion(distances)

        # 拼接
        edge_input = torch.cat([
            atom_i_feat,
            atom_j_feat,
            distance_rbf
        ], dim=-1)

        # 编码
        edge_features = self.edge_encoder(edge_input)

        return edge_features


class RBFExpansion(nn.Module):
    """径向基函数展开"""

    def __init__(self, vmin=0, vmax=8, bins=40):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers",
            torch.linspace(self.vmin, self.vmax, self.bins)
        )
        self.width = (self.vmax - self.vmin) / self.bins

    def forward(self, distance):
        """
        Args:
            distance: [num_edges, 1]
        Returns:
            rbf: [num_edges, bins]
        """
        distance = distance.squeeze(-1)  # [num_edges]
        distance = distance.unsqueeze(-1)  # [num_edges, 1]

        # Gaussian RBF
        rbf = torch.exp(
            -((distance - self.centers) ** 2) / (self.width ** 2)
        )

        return rbf


class ICOHPPredictor(nn.Module):
    """ICOHP预测器

    从晶体结构预测每条键的ICOHP值

    预测目标：
    - ICOHP: Crystal Orbital Hamilton Population
      - 负值：成键（越负越强）
      - 正值：反键
      - 典型范围：-6.0 到 +2.0 eV
    """

    def __init__(self,
                 atom_feature_dim=92,
                 edge_hidden_dim=128,
                 graph_hidden_dim=256,
                 num_layers=4,
                 dropout=0.1,
                 predict_uncertainty=True):
        """初始化

        Args:
            atom_feature_dim: 原子特征维度
            edge_hidden_dim: 边特征隐藏层维度
            graph_hidden_dim: GNN隐藏层维度
            num_layers: GNN层数
            dropout: Dropout率
            predict_uncertainty: 是否预测不确定性
        """
        super().__init__()

        self.predict_uncertainty = predict_uncertainty

        # 原子特征嵌入
        self.atom_embedding = nn.Sequential(
            nn.Linear(atom_feature_dim, graph_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(graph_hidden_dim)
        )

        # 边特征编码器
        self.edge_encoder = EdgeFeatureEncoder(
            atom_feature_dim=atom_feature_dim,
            edge_hidden_dim=edge_hidden_dim,
            dropout=dropout
        )

        # GNN层（用于学习全局上下文）
        self.gnn_layers = nn.ModuleList([
            EdgeGatedGraphConv(
                in_feats=graph_hidden_dim,
                out_feats=graph_hidden_dim,
                edge_feats=edge_hidden_dim
            )
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(graph_hidden_dim)
            for _ in range(num_layers)
        ])

        # ICOHP预测头
        # 输入：边特征 + 增强后的节点对特征
        predictor_input_dim = edge_hidden_dim + graph_hidden_dim * 2

        self.icohp_predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.SiLU(),
        )

        # 输出层
        if predict_uncertainty:
            # 同时预测均值和标准差
            self.output_layer = nn.Linear(64, 2)  # [mean, log_std]
        else:
            # 只预测均值
            self.output_layer = nn.Linear(64, 1)

    def forward(self, g, return_uncertainty=False):
        """前向传播

        Args:
            g: DGL图
            return_uncertainty: 是否返回不确定性

        Returns:
            如果 return_uncertainty=False:
                icohp_pred: [num_edges, 1] 预测的ICOHP值
            如果 return_uncertainty=True:
                icohp_pred: [num_edges, 1] 预测的ICOHP均值
                uncertainty: [num_edges, 1] 预测的不确定性（标准差）
        """
        # 获取节点特征
        x = g.ndata['atom_features']  # [num_nodes, atom_feature_dim]

        # 嵌入
        h = self.atom_embedding(x)  # [num_nodes, graph_hidden_dim]

        # 编码边特征
        edge_feat = self.edge_encoder(g, x)  # [num_edges, edge_hidden_dim]

        # GNN传播（学习全局上下文）
        for i, (gnn, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            h_new = gnn(g, h, edge_feat)
            h = norm(h_new + h)  # 残差连接

        # 获取边的源和目标节点的增强特征
        src, dst = g.edges()
        h_src = h[src]  # [num_edges, graph_hidden_dim]
        h_dst = h[dst]  # [num_edges, graph_hidden_dim]

        # 拼接边特征和节点对特征
        edge_input = torch.cat([
            edge_feat,
            h_src,
            h_dst
        ], dim=-1)  # [num_edges, predictor_input_dim]

        # 预测ICOHP
        edge_hidden = self.icohp_predictor(edge_input)  # [num_edges, 64]
        output = self.output_layer(edge_hidden)  # [num_edges, 1 or 2]

        if self.predict_uncertainty and return_uncertainty:
            # 分离均值和log标准差
            icohp_mean = output[:, 0:1]
            log_std = output[:, 1:2]

            # 转换为标准差
            uncertainty = torch.exp(log_std)

            return icohp_mean, uncertainty
        else:
            # 只返回ICOHP预测值
            if self.predict_uncertainty:
                icohp_pred = output[:, 0:1]
            else:
                icohp_pred = output

            return icohp_pred


class ICOBIPredictor(nn.Module):
    """ICOBI预测器

    从晶体结构预测每条键的ICOBI值

    ICOBI (Integrated Crystal Orbital Bond Index):
    - 正值：成键
    - 典型范围：0.0 到 1.0
    """

    def __init__(self,
                 atom_feature_dim=92,
                 edge_hidden_dim=128,
                 graph_hidden_dim=256,
                 num_layers=4,
                 dropout=0.1):
        """初始化（架构与ICOHPPredictor相似）"""
        super().__init__()

        # 与ICOHPPredictor相同的架构
        self.atom_embedding = nn.Sequential(
            nn.Linear(atom_feature_dim, graph_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(graph_hidden_dim)
        )

        self.edge_encoder = EdgeFeatureEncoder(
            atom_feature_dim=atom_feature_dim,
            edge_hidden_dim=edge_hidden_dim,
            dropout=dropout
        )

        self.gnn_layers = nn.ModuleList([
            EdgeGatedGraphConv(
                in_feats=graph_hidden_dim,
                out_feats=graph_hidden_dim,
                edge_feats=edge_hidden_dim
            )
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(graph_hidden_dim)
            for _ in range(num_layers)
        ])

        predictor_input_dim = edge_hidden_dim + graph_hidden_dim * 2

        self.icobi_predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # ICOBI在[0, 1]范围
        )

    def forward(self, g):
        """前向传播

        Returns:
            icobi_pred: [num_edges, 1] 预测的ICOBI值
        """
        x = g.ndata['atom_features']
        h = self.atom_embedding(x)
        edge_feat = self.edge_encoder(g, x)

        # GNN传播
        for gnn, norm in zip(self.gnn_layers, self.layer_norms):
            h_new = gnn(g, h, edge_feat)
            h = norm(h_new + h)

        # 获取边的节点对特征
        src, dst = g.edges()
        h_src = h[src]
        h_dst = h[dst]

        # 预测ICOBI
        edge_input = torch.cat([edge_feat, h_src, h_dst], dim=-1)
        icobi_pred = self.icobi_predictor(edge_input)

        return icobi_pred


class LOBSTERPredictorEnsemble(nn.Module):
    """LOBSTER预测器集成

    同时预测ICOHP和ICOBI
    支持多任务学习和不确定性量化
    """

    def __init__(self,
                 atom_feature_dim=92,
                 edge_hidden_dim=128,
                 graph_hidden_dim=256,
                 num_layers=4,
                 dropout=0.1,
                 shared_encoder=True):
        """初始化

        Args:
            shared_encoder: 是否共享底层编码器（减少参数）
        """
        super().__init__()

        self.shared_encoder = shared_encoder

        if shared_encoder:
            # 共享编码器
            self.atom_embedding = nn.Sequential(
                nn.Linear(atom_feature_dim, graph_hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(graph_hidden_dim)
            )

            self.edge_encoder = EdgeFeatureEncoder(
                atom_feature_dim=atom_feature_dim,
                edge_hidden_dim=edge_hidden_dim,
                dropout=dropout
            )

            self.gnn_layers = nn.ModuleList([
                EdgeGatedGraphConv(
                    in_feats=graph_hidden_dim,
                    out_feats=graph_hidden_dim,
                    edge_feats=edge_hidden_dim
                )
                for _ in range(num_layers)
            ])

            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(graph_hidden_dim)
                for _ in range(num_layers)
            ])

            # 独立的预测头
            predictor_input_dim = edge_hidden_dim + graph_hidden_dim * 2

            self.icohp_head = nn.Sequential(
                nn.Linear(predictor_input_dim, 128),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, 2)  # [mean, log_std]
            )

            self.icobi_head = nn.Sequential(
                nn.Linear(predictor_input_dim, 128),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        else:
            # 独立的预测器
            self.icohp_predictor = ICOHPPredictor(
                atom_feature_dim=atom_feature_dim,
                edge_hidden_dim=edge_hidden_dim,
                graph_hidden_dim=graph_hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                predict_uncertainty=True
            )

            self.icobi_predictor = ICOBIPredictor(
                atom_feature_dim=atom_feature_dim,
                edge_hidden_dim=edge_hidden_dim,
                graph_hidden_dim=graph_hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )

    def forward(self, g, return_uncertainty=False):
        """前向传播

        Returns:
            如果 return_uncertainty=False:
                icohp_pred: [num_edges, 1]
                icobi_pred: [num_edges, 1]
            如果 return_uncertainty=True:
                icohp_pred: [num_edges, 1]
                icohp_std: [num_edges, 1]
                icobi_pred: [num_edges, 1]
        """
        if self.shared_encoder:
            # 共享编码
            x = g.ndata['atom_features']
            h = self.atom_embedding(x)
            edge_feat = self.edge_encoder(g, x)

            # GNN传播
            for gnn, norm in zip(self.gnn_layers, self.layer_norms):
                h_new = gnn(g, h, edge_feat)
                h = norm(h_new + h)

            # 获取边特征
            src, dst = g.edges()
            h_src = h[src]
            h_dst = h[dst]
            edge_input = torch.cat([edge_feat, h_src, h_dst], dim=-1)

            # 预测ICOHP
            icohp_output = self.icohp_head(edge_input)
            icohp_pred = icohp_output[:, 0:1]

            # 预测ICOBI
            icobi_pred = self.icobi_head(edge_input)

            if return_uncertainty:
                log_std = icohp_output[:, 1:2]
                icohp_std = torch.exp(log_std)
                return icohp_pred, icohp_std, icobi_pred
            else:
                return icohp_pred, icobi_pred
        else:
            # 独立预测
            if return_uncertainty:
                icohp_pred, icohp_std = self.icohp_predictor(
                    g, return_uncertainty=True
                )
                icobi_pred = self.icobi_predictor(g)
                return icohp_pred, icohp_std, icobi_pred
            else:
                icohp_pred = self.icohp_predictor(g)
                icobi_pred = self.icobi_predictor(g)
                return icohp_pred, icobi_pred


# ============================================================================
# 损失函数
# ============================================================================

class ICOHPLoss(nn.Module):
    """ICOHP预测的损失函数

    考虑不确定性的负对数似然损失
    """

    def __init__(self, use_uncertainty=True, reduction='mean'):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.reduction = reduction

    def forward(self, pred, target, std=None):
        """
        Args:
            pred: [num_edges, 1] 预测的ICOHP
            target: [num_edges, 1] 真实的ICOHP
            std: [num_edges, 1] 预测的标准差（可选）

        Returns:
            loss: 标量损失
        """
        if self.use_uncertainty and std is not None:
            # 负对数似然损失（高斯分布）
            # NLL = 0.5 * log(2π) + log(std) + (pred - target)^2 / (2 * std^2)
            mse = (pred - target) ** 2
            loss = torch.log(std) + mse / (2 * std ** 2)
        else:
            # 简单的MSE
            loss = (pred - target) ** 2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiTaskLOBSTERLoss(nn.Module):
    """多任务LOBSTER损失

    同时优化ICOHP和ICOBI预测
    """

    def __init__(self,
                 icohp_weight=1.0,
                 icobi_weight=1.0,
                 use_uncertainty=True):
        super().__init__()

        self.icohp_weight = icohp_weight
        self.icobi_weight = icobi_weight

        self.icohp_loss = ICOHPLoss(use_uncertainty=use_uncertainty)
        self.icobi_loss = nn.MSELoss()  # ICOBI用简单的MSE

    def forward(self, icohp_pred, icobi_pred, icohp_target, icobi_target,
                icohp_std=None):
        """
        Args:
            icohp_pred: [num_edges, 1]
            icobi_pred: [num_edges, 1]
            icohp_target: [num_edges, 1]
            icobi_target: [num_edges, 1]
            icohp_std: [num_edges, 1] (可选)

        Returns:
            total_loss: 加权总损失
            loss_dict: 各损失分量
        """
        # ICOHP损失
        loss_icohp = self.icohp_loss(icohp_pred, icohp_target, icohp_std)

        # ICOBI损失
        loss_icobi = self.icobi_loss(icobi_pred, icobi_target)

        # 总损失
        total_loss = (
            self.icohp_weight * loss_icohp +
            self.icobi_weight * loss_icobi
        )

        loss_dict = {
            'total': total_loss.item(),
            'icohp': loss_icohp.item(),
            'icobi': loss_icobi.item()
        }

        return total_loss, loss_dict


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    print("测试LOBSTER预测器\n")

    # 创建测试数据
    from dgl import DGLGraph

    # 模拟一个小图（10个原子，30条边）
    num_nodes = 10
    num_edges = 30

    g = dgl.graph((
        torch.randint(0, num_nodes, (num_edges,)),
        torch.randint(0, num_nodes, (num_edges,))
    ))

    # 添加特征
    g.ndata['atom_features'] = torch.randn(num_nodes, 92)  # CGCNN特征
    g.edata['r'] = torch.randn(num_edges, 3)  # 位移向量

    print(f"测试图: {num_nodes} 个节点, {num_edges} 条边")

    # 测试1: ICOHP预测器
    print("\n" + "="*60)
    print("测试1: ICOHPPredictor")
    print("="*60)

    icohp_predictor = ICOHPPredictor(
        atom_feature_dim=92,
        predict_uncertainty=True
    )

    # 前向传播
    icohp_pred, uncertainty = icohp_predictor(g, return_uncertainty=True)

    print(f"ICOHP预测形状: {icohp_pred.shape}")
    print(f"ICOHP范围: [{icohp_pred.min():.3f}, {icohp_pred.max():.3f}]")
    print(f"不确定性范围: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")

    # 测试2: 集成预测器
    print("\n" + "="*60)
    print("测试2: LOBSTERPredictorEnsemble")
    print("="*60)

    ensemble = LOBSTERPredictorEnsemble(
        atom_feature_dim=92,
        shared_encoder=True
    )

    icohp_pred, icohp_std, icobi_pred = ensemble(g, return_uncertainty=True)

    print(f"ICOHP预测: {icohp_pred.shape}, 范围: [{icohp_pred.min():.3f}, {icohp_pred.max():.3f}]")
    print(f"ICOHP标准差: {icohp_std.shape}, 范围: [{icohp_std.min():.3f}, {icohp_std.max():.3f}]")
    print(f"ICOBI预测: {icobi_pred.shape}, 范围: [{icobi_pred.min():.3f}, {icobi_pred.max():.3f}]")

    # 测试3: 损失计算
    print("\n" + "="*60)
    print("测试3: MultiTaskLOBSTERLoss")
    print("="*60)

    # 创建假的目标
    icohp_target = torch.randn(num_edges, 1) * 2 - 3  # 典型ICOHP范围
    icobi_target = torch.rand(num_edges, 1)  # [0, 1]

    loss_fn = MultiTaskLOBSTERLoss()
    total_loss, loss_dict = loss_fn(
        icohp_pred, icobi_pred,
        icohp_target, icobi_target,
        icohp_std
    )

    print(f"总损失: {loss_dict['total']:.4f}")
    print(f"ICOHP损失: {loss_dict['icohp']:.4f}")
    print(f"ICOBI损失: {loss_dict['icobi']:.4f}")

    # 统计参数量
    total_params = sum(p.numel() for p in ensemble.parameters())
    print(f"\n模型参数量: {total_params:,}")

    print("\n✅ 所有测试通过！")
