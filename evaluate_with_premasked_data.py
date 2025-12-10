#!/usr/bin/env python3
"""
使用预生成的遮挡数据进行评估

这个脚本使用 pregenerate_masked_dataset.py 生成的遮挡数据，
确保不同模型在评估时使用完全相同的遮挡数据，实现公平对比。

用法:
    # 评估模型1
    python evaluate_with_premasked_data.py \
        --checkpoint /path/to/model1.pt \
        --masked_data ./masked_datasets/random_token_0.5.pkl \
        --output_file ./results_model1.json

    # 评估模型2（使用相同的遮挡数据）
    python evaluate_with_premasked_data.py \
        --checkpoint /path/to/model2.pt \
        --masked_data ./masked_datasets/random_token_0.5.pkl \
        --output_file ./results_model2.json
"""

import os
import sys
import json
import pickle
import argparse
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import dgl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))

from models.alignn import ALIGNN, ALIGNNConfig


class PremaskedDataset(Dataset):
    """使用预生成遮挡数据的Dataset"""

    def __init__(self, masked_data):
        self.data = masked_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_premasked(samples):
    """Collate function for premasked data"""
    graphs = []
    for s in samples:
        g = s['graph']
        if isinstance(g, tuple):
            graphs.append(g[0])
        else:
            graphs.append(g)

    line_graphs = [s['line_graph'] for s in samples]
    texts = [s['text'] for s in samples]  # 已经遮挡过的文本
    targets = torch.tensor([s['target'] for s in samples], dtype=torch.float32)

    batched_graph = dgl.batch(graphs)
    batched_line_graph = dgl.batch(line_graphs)

    return batched_graph, batched_line_graph, targets, texts


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    评估模型

    Returns:
        metrics: {'mae': float, 'rmse': float, 'r2': float}
    """
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            g, lg, target, text_list = batch

            g = g.to(device)
            lg = lg.to(device)
            target = target.to(device)

            # 前向传播（文本已经遮挡过了）
            out_data = model([g, lg, text_list])
            prediction = out_data.cpu().numpy()

            all_predictions.extend(prediction.flatten())
            all_targets.extend(target.cpu().numpy().flatten())

    # 计算指标
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'num_samples': len(predictions)
    }


def main():
    parser = argparse.ArgumentParser(
        description='使用预生成的遮挡数据评估模型'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='模型checkpoint路径'
    )
    parser.add_argument(
        '--masked_data',
        type=str,
        required=True,
        help='预生成的遮挡数据路径（.pkl文件）'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='输出结果文件路径（.json）'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default=None,
        help='模型配置文件路径（可选）'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='批大小'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='设备'
    )

    args = parser.parse_args()

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    # 加载遮挡数据
    print(f"加载遮挡数据: {args.masked_data}")
    with open(args.masked_data, 'rb') as f:
        masked_data = pickle.load(f)

    print(f"✓ 加载了 {len(masked_data)} 个样本")

    # 检查遮挡信息
    if len(masked_data) > 0:
        sample = masked_data[0]
        masking_strategy = sample.get('masking_strategy', 'unknown')
        masking_ratio = sample.get('masking_ratio', 0.0)
        print(f"  遮挡策略: {masking_strategy}")
        print(f"  遮挡率: {masking_ratio*100:.1f}%\n")

    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # 获取模型配置
    model_config = None

    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        if isinstance(model_config, dict):
            model_config = ALIGNNConfig(**model_config)
    elif 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            model_config = ALIGNNConfig(**config)
        else:
            model_config = ALIGNNConfig(**vars(config))
    elif args.config_file:
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        model_config = ALIGNNConfig(**config_dict)
    else:
        raise ValueError("无法加载模型配置，请提供 --config_file")

    # 创建模型
    model = ALIGNN(model_config)

    # 加载权重
    try:
        model.load_state_dict(checkpoint['model'], strict=True)
        print("✓ 模型加载成功\n")
    except RuntimeError:
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint['model'], strict=False
        )
        print(f"⚠ 宽松加载: missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}\n")

    model = model.to(device)
    model.eval()

    # 创建数据加载器
    dataset = PremaskedDataset(masked_data)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_premasked,
        num_workers=4,
        pin_memory=True
    )

    # 评估
    print("开始评估...")
    metrics = evaluate_model(model, data_loader, device)

    print(f"\n评估结果:")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  样本数: {metrics['num_samples']}\n")

    # 添加元数据
    results = {
        'checkpoint': args.checkpoint,
        'masked_data': args.masked_data,
        'masking_strategy': masking_strategy,
        'masking_ratio': float(masking_ratio),
        'metrics': metrics
    }

    # 保存结果
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ 结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()
