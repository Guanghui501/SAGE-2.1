"""
为所有JARVIS数据生成伪LOBSTER特征

使用训练好的LOBSTER预测器为所有JARVIS样本生成ICOHP和ICOBI特征

使用方法：
    python generate_pseudo_lobster_features.py \
        --model_path models/lobster_predictor/best_model.pt \
        --dataset dft_3d \
        --output_file data/pseudo_lobster_features.pkl

作者：Claude
日期：2025-12-10
"""

import os
import sys
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import dgl

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from models.lobster_predictor import LOBSTERPredictorEnsemble
from jarvis.db.figshare import data as jarvis_data
from data import load_jarvis_data_smart
from graphs import Graph


def load_model(checkpoint_path, device):
    """加载训练好的LOBSTER预测器

    Args:
        checkpoint_path: checkpoint路径
        device: 设备

    Returns:
        model: 加载的模型
        args: 训练时的参数
    """
    print(f"📂 加载模型: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 获取模型参数
    args = checkpoint.get('args', None)

    # 创建模型
    model = LOBSTERPredictorEnsemble(
        atom_feature_dim=92,
        edge_hidden_dim=getattr(args, 'edge_hidden_dim', 128),
        graph_hidden_dim=getattr(args, 'graph_hidden_dim', 256),
        num_layers=getattr(args, 'num_layers', 4),
        dropout=getattr(args, 'dropout', 0.1),
        shared_encoder=getattr(args, 'shared_encoder', True)
    )

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("✅ 模型加载完成")

    # 打印验证指标
    if 'val_metrics' in checkpoint:
        val_metrics = checkpoint['val_metrics']
        print(f"\n模型性能:")
        print(f"  ICOHP MAE: {val_metrics['mae']['icohp']:.4f}")
        print(f"  ICOBI MAE: {val_metrics['mae']['icobi']:.4f}")
        print(f"  ICOHP相关系数: {val_metrics['correlation']['icohp']:.4f}")
        print(f"  ICOBI相关系数: {val_metrics['correlation']['icobi']:.4f}")

    return model, args


def generate_features_for_sample(model, g, device, return_uncertainty=True):
    """为单个样本生成LOBSTER特征

    Args:
        model: LOBSTER预测器
        g: DGL图
        device: 设备
        return_uncertainty: 是否返回不确定性

    Returns:
        features: {
            'icohp_mean': [num_edges],
            'icohp_std': [num_edges] (可选),
            'icobi': [num_edges],
            'icohp_global_mean': float,
            'icohp_global_min': float,
            'num_bonds': int
        }
    """
    g = g.to(device)

    with torch.no_grad():
        if return_uncertainty:
            icohp_pred, icohp_std, icobi_pred = model(
                g, return_uncertainty=True
            )
        else:
            icohp_pred, icobi_pred = model(g, return_uncertainty=False)
            icohp_std = None

    # 转换为numpy
    icohp_mean = icohp_pred.cpu().numpy().flatten()
    icobi = icobi_pred.cpu().numpy().flatten()

    # 计算全局统计特征
    icohp_global_mean = float(icohp_mean.mean())
    icohp_global_min = float(icohp_mean.min())  # 最强键
    icohp_global_max = float(icohp_mean.max())
    icohp_global_std = float(icohp_mean.std())
    num_bonds = len(icohp_mean)

    features = {
        # 边级特征（用于GNN）
        'icohp_mean': icohp_mean,
        'icobi': icobi,

        # 全局特征（用于特征工程或MLP）
        'icohp_global_mean': icohp_global_mean,
        'icohp_global_min': icohp_global_min,
        'icohp_global_max': icohp_global_max,
        'icohp_global_std': icohp_global_std,
        'num_bonds': num_bonds,
        'icobi_mean': float(icobi.mean()),
        'icobi_max': float(icobi.max()),
    }

    if icohp_std is not None:
        icohp_uncertainty = icohp_std.cpu().numpy().flatten()
        features['icohp_std'] = icohp_uncertainty
        features['icohp_uncertainty_mean'] = float(icohp_uncertainty.mean())

    return features


def generate_features_batch(model, graphs, device, batch_size=32,
                            return_uncertainty=True):
    """批量生成LOBSTER特征

    Args:
        model: LOBSTER预测器
        graphs: DGL图列表
        device: 设备
        batch_size: 批次大小
        return_uncertainty: 是否返回不确定性

    Returns:
        features_list: 特征列表
    """
    features_list = []

    # 分批处理
    num_batches = (len(graphs) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="生成特征"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(graphs))

        batch_graphs = graphs[start_idx:end_idx]

        # 批次图
        batched_g = dgl.batch(batch_graphs)

        # 生成特征
        with torch.no_grad():
            if return_uncertainty:
                icohp_pred, icohp_std, icobi_pred = model(
                    batched_g.to(device), return_uncertainty=True
                )
            else:
                icohp_pred, icobi_pred = model(
                    batched_g.to(device), return_uncertainty=False
                )
                icohp_std = None

        # 转换为CPU
        icohp_pred = icohp_pred.cpu().numpy()
        icobi_pred = icobi_pred.cpu().numpy()

        if icohp_std is not None:
            icohp_std = icohp_std.cpu().numpy()

        # 分解批次结果
        edge_offset = 0
        for j, g in enumerate(batch_graphs):
            num_edges = g.num_edges()

            # 提取该图的特征
            icohp_edges = icohp_pred[edge_offset:edge_offset+num_edges].flatten()
            icobi_edges = icobi_pred[edge_offset:edge_offset+num_edges].flatten()

            features = {
                'icohp_mean': icohp_edges,
                'icobi': icobi_edges,
                'icohp_global_mean': float(icohp_edges.mean()),
                'icohp_global_min': float(icohp_edges.min()),
                'icohp_global_max': float(icohp_edges.max()),
                'icohp_global_std': float(icohp_edges.std()),
                'num_bonds': num_edges,
                'icobi_mean': float(icobi_edges.mean()),
                'icobi_max': float(icobi_edges.max()),
            }

            if icohp_std is not None:
                icohp_unc = icohp_std[edge_offset:edge_offset+num_edges].flatten()
                features['icohp_std'] = icohp_unc
                features['icohp_uncertainty_mean'] = float(icohp_unc.mean())

            features_list.append(features)

            edge_offset += num_edges

    return features_list


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='为JARVIS数据生成伪LOBSTER特征'
    )

    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的LOBSTER预测器路径')
    parser.add_argument('--dataset', type=str, default='dft_3d',
                       help='JARVIS数据集名称')
    parser.add_argument('--atom_features', type=str, default='cgcnn',
                       help='原子特征类型')
    parser.add_argument('--output_file', type=str,
                       default='data/pseudo_lobster_features.pkl',
                       help='输出文件路径')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大样本数（用于测试）')
    parser.add_argument('--return_uncertainty', action='store_true',
                       help='是否返回不确定性')

    args = parser.parse_args()

    # 创建输出目录
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  使用设备: {device}")

    # 加载模型
    model, train_args = load_model(args.model_path, device)

    # 加载JARVIS数据
    print(f"\n📊 加载JARVIS数据集: {args.dataset}")
    jarvis_db = load_jarvis_data_smart(args.dataset)

    # 限制样本数（用于测试）
    if args.max_samples is not None:
        jarvis_db = jarvis_db[:args.max_samples]
        print(f"   限制样本数: {args.max_samples}")

    print(f"   总样本数: {len(jarvis_db)}")

    # 构建图
    print("\n🔨 构建晶体图...")
    graphs = []
    jids = []

    for entry in tqdm(jarvis_db):
        try:
            atoms = entry['atoms']
            jid = entry['jid']

            # 构建DGL图
            g, _ = Graph.atom_dgl_multigraph(
                atoms,
                cutoff=8.0,
                max_neighbors=12,
                atom_features=args.atom_features,
                compute_line_graph=False
            )

            graphs.append(g)
            jids.append(jid)

        except Exception as e:
            print(f"⚠️  跳过样本 {entry.get('jid', 'unknown')}: {e}")
            continue

    print(f"✅ 成功构建 {len(graphs)} 个图")

    # 生成特征
    print(f"\n🚀 生成伪LOBSTER特征...")
    print(f"   批次大小: {args.batch_size}")
    print(f"   返回不确定性: {args.return_uncertainty}")

    features_list = generate_features_batch(
        model=model,
        graphs=graphs,
        device=device,
        batch_size=args.batch_size,
        return_uncertainty=args.return_uncertainty
    )

    # 构建最终数据结构
    print("\n📦 整理数据...")
    pseudo_lobster_db = {}

    for jid, features in zip(jids, features_list):
        pseudo_lobster_db[jid] = features

    # 保存
    print(f"\n💾 保存到: {args.output_file}")
    with open(args.output_file, 'wb') as f:
        pickle.dump(pseudo_lobster_db, f)

    # 统计信息
    print("\n" + "="*60)
    print("📊 统计信息")
    print("="*60)

    all_icohp = [f['icohp_global_mean'] for f in features_list]
    all_icohp_min = [f['icohp_global_min'] for f in features_list]

    print(f"总样本数: {len(pseudo_lobster_db)}")
    print(f"\nICOHP统计:")
    print(f"  平均ICOHP范围: [{np.min(all_icohp):.3f}, {np.max(all_icohp):.3f}]")
    print(f"  最强键ICOHP范围: [{np.min(all_icohp_min):.3f}, {np.max(all_icohp_min):.3f}]")

    if args.return_uncertainty:
        all_unc = [f['icohp_uncertainty_mean'] for f in features_list]
        print(f"\n不确定性统计:")
        print(f"  平均不确定性: {np.mean(all_unc):.3f}")
        print(f"  不确定性范围: [{np.min(all_unc):.3f}, {np.max(all_unc):.3f}]")

    print("="*60)
    print("✅ 特征生成完成！")


if __name__ == '__main__':
    main()
