"""
训练LOBSTER预测器

训练流程：
1. 加载JARVIS和LOBSTER的重叠样本
2. 训练预测器学习ICOHP和ICOBI
3. 验证预测质量
4. 保存模型用于特征生成

使用方法：
    python train_lobster_predictor.py \
        --lobster_dir data/lobster_database \
        --overlap_map data/jarvis_mp_overlap.json \
        --output_dir models/lobster_predictor

作者：Claude
日期：2025-12-10
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from models.lobster_predictor import (
    LOBSTERPredictorEnsemble,
    MultiTaskLOBSTERLoss,
    ICOHPPredictor
)
from utils.lobster_features import LobsterFeatureExtractor
from data import get_torch_dataset
from jarvis.db.figshare import data as jarvis_data


class LOBSTERPredictorDataset(torch.utils.data.Dataset):
    """LOBSTER预测器训练数据集

    每个样本包含：
    - 晶体结构图（DGL graph）
    - 真实的LOBSTER特征（从JSON提取）
    """

    def __init__(self, jarvis_dataset, lobster_dir, overlap_map,
                 dataset_name='dft_3d', atom_features='cgcnn'):
        """初始化

        Args:
            jarvis_dataset: JARVIS数据
            lobster_dir: LOBSTER JSON文件目录
            overlap_map: {jarvis_id: mp_id} 映射
            dataset_name: JARVIS数据集名称
            atom_features: 原子特征类型
        """
        self.jarvis_dataset = jarvis_dataset
        self.lobster_dir = Path(lobster_dir)
        self.overlap_map = overlap_map
        self.atom_features = atom_features

        # 只保留有LOBSTER数据的样本
        self.valid_samples = []
        self.lobster_cache = {}

        print("加载LOBSTER数据...")
        for entry in tqdm(jarvis_dataset):
            jid = entry['jid']

            if jid in overlap_map:
                mp_id = overlap_map[jid]
                lobster_path = self.lobster_dir / f"{mp_id}.json"

                if lobster_path.exists():
                    # 加载LOBSTER特征
                    lobster_extractor = LobsterFeatureExtractor(
                        str(lobster_path)
                    )
                    self.lobster_cache[jid] = lobster_extractor
                    self.valid_samples.append(entry)

        print(f"✅ 加载了 {len(self.valid_samples)} 个有效样本")

        # 构建图
        from graphs import Graph

        print("构建晶体图...")
        self.graphs = []
        for entry in tqdm(self.valid_samples):
            atoms = entry['atoms']

            # 构建DGL图
            g, _ = Graph.atom_dgl_multigraph(
                atoms,
                cutoff=8.0,
                max_neighbors=12,
                atom_features=atom_features,
                compute_line_graph=False
            )

            self.graphs.append(g)

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        """获取一个样本

        Returns:
            g: DGL图
            lobster_targets: {
                'icohp': [num_edges, 1] 真实ICOHP
                'icobi': [num_edges, 1] 真实ICOBI
            }
        """
        entry = self.valid_samples[idx]
        jid = entry['jid']
        g = self.graphs[idx]

        # 获取LOBSTER真实值
        lobster = self.lobster_cache[jid]

        # 为每条边提取LOBSTER特征
        src, dst = g.edges()
        num_edges = g.num_edges()

        icohp_values = []
        icobi_values = []

        for i, j in zip(src.numpy(), dst.numpy()):
            # 计算距离
            pos_i = g.ndata.get('pos', None)
            if pos_i is not None:
                pos_j = g.ndata['pos'][j]
                distance = torch.norm(pos_i - pos_j).item()
            else:
                # 从边的位移向量计算
                r = g.edata['r'][len(icohp_values)]
                distance = torch.norm(r).item()

            # 获取LOBSTER特征
            lobster_feat = lobster.get_edge_features(i, j, distance)
            icohp_values.append(lobster_feat[0])
            icobi_values.append(lobster_feat[1])

        lobster_targets = {
            'icohp': torch.FloatTensor(icohp_values).unsqueeze(-1),
            'icobi': torch.FloatTensor(icobi_values).unsqueeze(-1)
        }

        return g, lobster_targets


def collate_fn(batch):
    """批次整理函数"""
    import dgl

    graphs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # 批次图
    batched_graph = dgl.batch(graphs)

    # 合并目标
    batched_targets = {
        'icohp': torch.cat([t['icohp'] for t in targets], dim=0),
        'icobi': torch.cat([t['icobi'] for t in targets], dim=0)
    }

    return batched_graph, batched_targets


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()

    total_loss = 0
    total_icohp_loss = 0
    total_icobi_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (g, targets) in enumerate(pbar):
        g = g.to(device)
        icohp_target = targets['icohp'].to(device)
        icobi_target = targets['icobi'].to(device)

        # 前向传播
        optimizer.zero_grad()

        icohp_pred, icohp_std, icobi_pred = model(g, return_uncertainty=True)

        # 计算损失
        loss, loss_dict = criterion(
            icohp_pred, icobi_pred,
            icohp_target, icobi_target,
            icohp_std
        )

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 统计
        total_loss += loss_dict['total']
        total_icohp_loss += loss_dict['icohp']
        total_icobi_loss += loss_dict['icobi']
        num_batches += 1

        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'icohp': f"{loss_dict['icohp']:.4f}",
            'icobi': f"{loss_dict['icobi']:.4f}"
        })

    return {
        'total': total_loss / num_batches,
        'icohp': total_icohp_loss / num_batches,
        'icobi': total_icobi_loss / num_batches
    }


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()

    total_loss = 0
    total_icohp_loss = 0
    total_icobi_loss = 0
    num_batches = 0

    # 用于计算MAE和相关系数
    all_icohp_pred = []
    all_icohp_target = []
    all_icobi_pred = []
    all_icobi_target = []

    with torch.no_grad():
        for g, targets in tqdm(val_loader, desc="Validating"):
            g = g.to(device)
            icohp_target = targets['icohp'].to(device)
            icobi_target = targets['icobi'].to(device)

            # 前向传播
            icohp_pred, icohp_std, icobi_pred = model(
                g, return_uncertainty=True
            )

            # 计算损失
            loss, loss_dict = criterion(
                icohp_pred, icobi_pred,
                icohp_target, icobi_target,
                icohp_std
            )

            total_loss += loss_dict['total']
            total_icohp_loss += loss_dict['icohp']
            total_icobi_loss += loss_dict['icobi']
            num_batches += 1

            # 收集预测和目标
            all_icohp_pred.append(icohp_pred.cpu())
            all_icohp_target.append(icohp_target.cpu())
            all_icobi_pred.append(icobi_pred.cpu())
            all_icobi_target.append(icobi_target.cpu())

    # 合并结果
    all_icohp_pred = torch.cat(all_icohp_pred)
    all_icohp_target = torch.cat(all_icohp_target)
    all_icobi_pred = torch.cat(all_icobi_pred)
    all_icobi_target = torch.cat(all_icobi_target)

    # 计算指标
    icohp_mae = torch.abs(all_icohp_pred - all_icohp_target).mean().item()
    icobi_mae = torch.abs(all_icobi_pred - all_icobi_target).mean().item()

    # 计算相关系数
    icohp_corr = np.corrcoef(
        all_icohp_pred.numpy().flatten(),
        all_icohp_target.numpy().flatten()
    )[0, 1]

    icobi_corr = np.corrcoef(
        all_icobi_pred.numpy().flatten(),
        all_icobi_target.numpy().flatten()
    )[0, 1]

    return {
        'loss': {
            'total': total_loss / num_batches,
            'icohp': total_icohp_loss / num_batches,
            'icobi': total_icobi_loss / num_batches
        },
        'mae': {
            'icohp': icohp_mae,
            'icobi': icobi_mae
        },
        'correlation': {
            'icohp': icohp_corr,
            'icobi': icobi_corr
        }
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='训练LOBSTER预测器'
    )

    parser.add_argument('--lobster_dir', type=str,
                       default='data/lobster_database',
                       help='LOBSTER数据目录')
    parser.add_argument('--overlap_map', type=str,
                       default='data/jarvis_mp_overlap.json',
                       help='JARVIS-MP重叠映射文件')
    parser.add_argument('--dataset', type=str, default='dft_3d',
                       help='JARVIS数据集名称')
    parser.add_argument('--atom_features', type=str, default='cgcnn',
                       help='原子特征类型')

    # 模型参数
    parser.add_argument('--edge_hidden_dim', type=int, default=128,
                       help='边特征隐藏层维度')
    parser.add_argument('--graph_hidden_dim', type=int, default=256,
                       help='GNN隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='GNN层数')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout率')
    parser.add_argument('--shared_encoder', action='store_true',
                       help='使用共享编码器')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')

    # 输出
    parser.add_argument('--output_dir', type=str,
                       default='models/lobster_predictor',
                       help='输出目录')
    parser.add_argument('--save_every', type=int, default=10,
                       help='每N个epoch保存一次')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir / 'logs'))

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  使用设备: {device}")

    # 加载JARVIS数据
    print("\n📊 加载JARVIS数据...")
    jarvis_db = jarvis_data(args.dataset)

    # 加载重叠映射
    print(f"📂 加载重叠映射: {args.overlap_map}")
    with open(args.overlap_map) as f:
        overlap_map = json.load(f)

    print(f"   重叠样本数: {len(overlap_map)}")

    # 创建数据集
    print("\n🔨 构建训练数据集...")
    full_dataset = LOBSTERPredictorDataset(
        jarvis_dataset=jarvis_db,
        lobster_dir=args.lobster_dir,
        overlap_map=overlap_map,
        dataset_name=args.dataset,
        atom_features=args.atom_features
    )

    # 划分训练/验证集
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    print(f"   训练集: {len(train_dataset)} 样本")
    print(f"   验证集: {len(val_dataset)} 样本")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # 创建模型
    print("\n🏗️  创建模型...")
    model = LOBSTERPredictorEnsemble(
        atom_feature_dim=92,
        edge_hidden_dim=args.edge_hidden_dim,
        graph_hidden_dim=args.graph_hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        shared_encoder=args.shared_encoder
    )

    model = model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数量: {total_params:,}")

    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )

    # 损失函数
    criterion = MultiTaskLOBSTERLoss(
        icohp_weight=1.0,
        icobi_weight=1.0,
        use_uncertainty=True
    )

    # 训练循环
    print("\n🚀 开始训练...\n")

    best_val_mae = float('inf')

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # 验证
        val_metrics = validate(model, val_loader, criterion, device)

        # 学习率调度
        scheduler.step(val_metrics['loss']['total'])

        # 记录到TensorBoard
        writer.add_scalar('train/loss', train_metrics['total'], epoch)
        writer.add_scalar('train/icohp_loss', train_metrics['icohp'], epoch)
        writer.add_scalar('train/icobi_loss', train_metrics['icobi'], epoch)

        writer.add_scalar('val/loss', val_metrics['loss']['total'], epoch)
        writer.add_scalar('val/icohp_mae', val_metrics['mae']['icohp'], epoch)
        writer.add_scalar('val/icobi_mae', val_metrics['mae']['icobi'], epoch)
        writer.add_scalar('val/icohp_corr', val_metrics['correlation']['icohp'], epoch)
        writer.add_scalar('val/icobi_corr', val_metrics['correlation']['icobi'], epoch)

        # 打印结果
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['total']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']['total']:.4f}")
        print(f"  ICOHP MAE: {val_metrics['mae']['icohp']:.4f} | "
              f"Corr: {val_metrics['correlation']['icohp']:.4f}")
        print(f"  ICOBI MAE: {val_metrics['mae']['icobi']:.4f} | "
              f"Corr: {val_metrics['correlation']['icobi']:.4f}")

        # 保存最佳模型
        avg_mae = (val_metrics['mae']['icohp'] + val_metrics['mae']['icobi']) / 2
        if avg_mae < best_val_mae:
            best_val_mae = avg_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': args
            }, output_dir / 'best_model.pt')
            print(f"  ✅ 保存最佳模型 (MAE: {avg_mae:.4f})")

        # 定期保存
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': args
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')

    # 训练完成
    print("\n" + "="*60)
    print("✅ 训练完成！")
    print(f"最佳验证MAE: {best_val_mae:.4f}")
    print(f"模型保存至: {output_dir}")
    print("="*60)

    writer.close()


if __name__ == '__main__':
    main()
