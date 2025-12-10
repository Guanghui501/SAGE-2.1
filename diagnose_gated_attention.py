#!/usr/bin/env python
"""
Gated Cross-Attention 诊断工具

用于诊断和分析带有门控跨模态注意力机制的模型。

用法:
    python diagnose_gated_attention.py --checkpoint path/to/model.pt [--dataset jarvis/mbj_bandgap]
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))

from data import get_train_val_loaders
from models.alignn import ALIGNN, ALIGNNConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Gated Cross-Attention 诊断工具')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='模型checkpoint路径')
    parser.add_argument('--dataset', type=str, default='jarvis/mbj_bandgap',
                      help='数据集名称 (如: jarvis/mbj_bandgap, jarvis/formation_energy_peratom)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批次大小')
    parser.add_argument('--n_samples', type=int, default=10,
                      help='要分析的样本数')

    return parser.parse_args()


def load_model(checkpoint_path):
    """加载模型checkpoint"""
    print(f"📂 加载模型: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到checkpoint文件: {checkpoint_path}")

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 从checkpoint中提取配置
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # 如果没有保存配置，使用默认配置
        print("⚠️  Checkpoint中未找到配置，使用默认配置")
        config = ALIGNNConfig(
            name="alignn",
            alignn_layers=4,
            gcn_layers=4,
            hidden_features=256,
            use_cross_modal_attention=True,
            cross_modal_num_heads=4,
            cross_modal_hidden_dim=256,
            cross_modal_dropout=0.1
        )

    # 创建模型
    model = ALIGNN(config)

    # 加载权重
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✅ 模型加载完成")

    return model, config


def analyze_model_config(model, config):
    """分析模型配置"""
    print("\n" + "="*80)
    print("📊 模型配置分析")
    print("="*80)

    print(f"\n🔧 基础配置:")
    print(f"   - ALIGNN层数: {config.alignn_layers}")
    print(f"   - GCN层数: {config.gcn_layers}")
    print(f"   - 隐藏层维度: {config.hidden_features}")
    print(f"   - 输出维度: {config.output_features}")

    print(f"\n🔀 跨模态注意力配置:")
    print(f"   - 使用Gated Cross-Attention: {config.use_cross_modal_attention}")

    if config.use_cross_modal_attention:
        print(f"   - 注意力头数: {config.cross_modal_num_heads}")
        print(f"   - 隐藏层维度: {config.cross_modal_hidden_dim}")
        print(f"   - Dropout率: {config.cross_modal_dropout}")

        # 检查cross_modal_attention模块
        if hasattr(model, 'cross_modal_attention'):
            print(f"\n   ✅ Cross-Modal Attention 模块已启用")
            print(f"      - Graph维度: 64")
            print(f"      - Text维度: 64")
        else:
            print(f"\n   ⚠️  模型中未找到 cross_modal_attention 模块")

    print(f"\n🔬 细粒度注意力配置:")
    print(f"   - 使用Fine-Grained Attention: {getattr(config, 'use_fine_grained_attention', False)}")

    if getattr(config, 'use_fine_grained_attention', False):
        print(f"   - 注意力头数: {config.fine_grained_num_heads}")
        print(f"   - 隐藏层维度: {config.fine_grained_hidden_dim}")

    print(f"\n🔄 中期融合配置:")
    print(f"   - 使用Middle Fusion: {getattr(config, 'use_middle_fusion', False)}")

    if getattr(config, 'use_middle_fusion', False):
        print(f"   - 融合层索引: {config.middle_fusion_layers}")
        print(f"   - 注意力头数: {config.middle_fusion_num_heads}")


def load_data(dataset_name, batch_size=32):
    """加载数据集"""
    print(f"\n📊 加载数据集: {dataset_name}")

    try:
        # 解析数据集名称
        if '/' in dataset_name:
            # 格式: jarvis/property_name
            parts = dataset_name.split('/')
            dataset = parts[0]
            target = parts[1] if len(parts) > 1 else 'formation_energy_peratom'
        else:
            dataset = dataset_name
            target = 'formation_energy_peratom'

        # 调用 get_train_val_loaders (不传入 root_dir!)
        train_loader, val_loader, test_loader, prepare_batch = get_train_val_loaders(
            dataset=dataset,
            target=target,
            batch_size=batch_size,
            atom_features="cgcnn",
            neighbor_strategy="k-nearest",
            id_tag="jid",
            pin_memory=False,
            workers=0,
            save_dataloader=False,
            use_canonize=True,
            filename=f"temp_{dataset}_{target}",
            cutoff=8.0,
            max_neighbors=12,
            val_ratio=0.1,
            test_ratio=0.1,
        )

        print(f"✅ 数据加载完成")
        print(f"   - 训练集: {len(train_loader.dataset)} 样本")
        print(f"   - 验证集: {len(val_loader.dataset)} 样本")
        print(f"   - 测试集: {len(test_loader.dataset)} 样本")

        return train_loader, val_loader, test_loader, prepare_batch

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def analyze_attention_weights(model, test_loader, device, n_samples=10):
    """分析注意力权重"""
    print("\n" + "="*80)
    print("🔍 分析注意力权重")
    print("="*80)

    if not hasattr(model, 'cross_modal_attention'):
        print("⚠️  模型未使用跨模态注意力")
        return

    model.eval()
    all_g2t_weights = []
    all_t2g_weights = []

    print(f"\n📦 收集 {n_samples} 个样本的注意力权重...")

    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if count >= n_samples:
                break

            g, lg, text, labels = batch

            # Forward pass with attention tracking
            output = model(
                [g.to(device), lg.to(device), text],
                return_attention=True,
                return_features=True
            )

            if isinstance(output, dict) and 'attention_weights' in output:
                attn = output['attention_weights']

                if attn is not None:
                    if 'graph_to_text' in attn and attn['graph_to_text'] is not None:
                        all_g2t_weights.append(attn['graph_to_text'].cpu())

                    if 'text_to_graph' in attn and attn['text_to_graph'] is not None:
                        all_t2g_weights.append(attn['text_to_graph'].cpu())

            count += len(labels)
            print(f"   进度: {count}/{n_samples}", end='\r')

    print(f"\n✅ 收集完成: {count} 个样本")

    # 分析统计
    if all_g2t_weights:
        g2t = torch.cat(all_g2t_weights, dim=0)
        print(f"\n📊 Graph→Text 注意力统计:")
        print(f"   - 形状: {g2t.shape}")
        print(f"   - 平均值: {g2t.mean():.4f}")
        print(f"   - 标准差: {g2t.std():.4f}")
        print(f"   - 最小值: {g2t.min():.4f}")
        print(f"   - 最大值: {g2t.max():.4f}")

        # 按注意力头分析
        if g2t.dim() >= 2:
            num_heads = g2t.shape[1]
            print(f"\n   各注意力头统计:")
            for head_idx in range(num_heads):
                head_weights = g2t[:, head_idx]
                print(f"      Head {head_idx}: 均值={head_weights.mean():.4f}, "
                      f"标准差={head_weights.std():.4f}")

    if all_t2g_weights:
        t2g = torch.cat(all_t2g_weights, dim=0)
        print(f"\n📊 Text→Graph 注意力统计:")
        print(f"   - 形状: {t2g.shape}")
        print(f"   - 平均值: {t2g.mean():.4f}")
        print(f"   - 标准差: {t2g.std():.4f}")
        print(f"   - 最小值: {t2g.min():.4f}")
        print(f"   - 最大值: {t2g.max():.4f}")

        # 按注意力头分析
        if t2g.dim() >= 2:
            num_heads = t2g.shape[1]
            print(f"\n   各注意力头统计:")
            for head_idx in range(num_heads):
                head_weights = t2g[:, head_idx]
                print(f"      Head {head_idx}: 均值={head_weights.mean():.4f}, "
                      f"标准差={head_weights.std():.4f}")

    return all_g2t_weights, all_t2g_weights


def analyze_predictions(model, test_loader, device, n_samples=10):
    """分析模型预测"""
    print("\n" + "="*80)
    print("🎯 分析模型预测")
    print("="*80)

    model.eval()
    all_predictions = []
    all_labels = []
    all_errors = []

    print(f"\n📦 收集 {n_samples} 个样本的预测...")

    count = 0
    with torch.no_grad():
        for batch in test_loader:
            if count >= n_samples:
                break

            g, lg, text, labels = batch

            # Forward pass
            output = model([g.to(device), lg.to(device), text])

            if isinstance(output, dict):
                predictions = output['predictions']
            else:
                predictions = output

            predictions = predictions.cpu().squeeze()
            labels = labels.cpu().squeeze()

            all_predictions.append(predictions)
            all_labels.append(labels)
            all_errors.append(torch.abs(predictions - labels))

            count += len(labels)
            print(f"   进度: {count}/{n_samples}", end='\r')

    print(f"\n✅ 收集完成: {count} 个样本")

    # 合并结果
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    errors = torch.cat(all_errors)

    # 计算指标
    mae = errors.mean().item()
    rmse = torch.sqrt((errors ** 2).mean()).item()

    print(f"\n📊 预测统计:")
    print(f"   - MAE (平均绝对误差): {mae:.4f}")
    print(f"   - RMSE (均方根误差): {rmse:.4f}")
    print(f"   - 预测范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"   - 真实值范围: [{labels.min():.4f}, {labels.max():.4f}]")

    # 显示一些样本
    print(f"\n📋 样本预测 (前10个):")
    print(f"{'='*60}")
    print(f"{'样本':<8} {'真实值':<12} {'预测值':<12} {'误差':<12}")
    print(f"{'-'*60}")
    for i in range(min(10, len(predictions))):
        print(f"{i:<8} {labels[i]:<12.4f} {predictions[i]:<12.4f} {errors[i]:<12.4f}")
    print(f"{'='*60}")


def main():
    """主函数"""
    args = parse_args()

    print("\n" + "="*80)
    print("Gated Cross-Attention 诊断工具")
    print("="*80)

    # 1. 加载模型
    model, config = load_model(args.checkpoint)

    # 2. 分析模型配置
    analyze_model_config(model, config)

    # 3. 加载数据
    train_loader, val_loader, test_loader, prepare_batch = load_data(
        args.dataset,
        batch_size=args.batch_size
    )

    if test_loader is None:
        print("\n❌ 无法加载数据，诊断终止")
        return

    # 4. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\n🖥️  使用设备: {device}")

    # 5. 分析注意力权重
    if config.use_cross_modal_attention:
        analyze_attention_weights(model, test_loader, device, n_samples=args.n_samples)

    # 6. 分析预测
    analyze_predictions(model, test_loader, device, n_samples=args.n_samples)

    print("\n" + "="*80)
    print("✅ 诊断完成")
    print("="*80)


if __name__ == '__main__':
    main()
