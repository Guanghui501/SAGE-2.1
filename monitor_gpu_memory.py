"""GPU显存使用监控工具

用于诊断不同数据集的显存占用差异
"""

import torch
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from data import get_train_val_loaders


def monitor_memory_usage(dataset_name, target, batch_size=64, num_batches=10):
    """监控数据加载的显存使用

    Args:
        dataset_name: 数据集名称
        target: 目标属性
        batch_size: 批次大小
        num_batches: 监控的批次数量
    """

    print(f"\n{'='*80}")
    print(f"GPU显存监控 - {dataset_name}/{target}")
    print(f"{'='*80}\n")

    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法监控GPU显存")
        return

    # 清空显存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("📂 加载数据...")
    try:
        train_loader, val_loader, test_loader, _ = get_train_val_loaders(
            dataset=dataset_name,
            target=target,
            batch_size=batch_size,
            pin_memory=False,  # 统一用False避免干扰
            workers=0  # 统一用0避免多进程干扰
        )
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    print(f"✅ 数据加载成功")
    print(f"   训练样本: {len(train_loader.dataset)}")
    print(f"   验证样本: {len(val_loader.dataset)}")
    print(f"   测试样本: {len(test_loader.dataset)}")
    print(f"   Batch size: {batch_size}")
    print()

    # 统计图大小
    node_counts = []
    edge_counts = []
    memory_usage = []

    print(f"🔍 分析前{num_batches}个batch...\n")

    device = torch.device('cuda')

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break

        # 解包batch
        if len(batch) == 4:
            g, lg, text, labels = batch
        elif len(batch) == 3:
            g, lg, labels = batch
            text = None
        else:
            g, labels = batch
            lg = None
            text = None

        # 记录节点和边数
        nodes = g.num_nodes()
        edges = g.num_edges()
        node_counts.append(nodes)
        edge_counts.append(edges)

        # 清空显存重新开始
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_mem = torch.cuda.memory_allocated() / 1024**2

        # 模拟移到GPU
        g_gpu = g.to(device)
        if lg is not None:
            lg_gpu = lg.to(device)
        labels_gpu = labels.to(device)

        # 记录显存
        current_mem = torch.cuda.memory_allocated() / 1024**2
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        batch_mem = current_mem - baseline_mem

        memory_usage.append(batch_mem)

        print(f"Batch {i+1}/{num_batches}:")
        print(f"  节点数: {nodes:>6,} ({nodes/batch_size:>5.1f} 节点/样本)")
        print(f"  边数:   {edges:>6,} ({edges/batch_size:>5.1f} 边/样本)")
        print(f"  批次显存: {batch_mem:>6.1f} MB")
        print(f"  峰值显存: {peak_mem:>6.1f} MB")
        print()

        # 清理
        del g_gpu, labels_gpu
        if lg is not None:
            del lg_gpu
        torch.cuda.empty_cache()

    # 统计摘要
    print(f"\n{'='*80}")
    print("📊 统计摘要")
    print(f"{'='*80}\n")

    print(f"数据集信息:")
    print(f"  总样本数: {len(train_loader.dataset):,}")
    print(f"  Batch size: {batch_size}")
    print(f"  总batch数: {len(train_loader)}")
    print()

    print(f"图结构统计 (基于{num_batches}个batch):")
    print(f"  平均节点数/batch:  {np.mean(node_counts):>8.1f} ± {np.std(node_counts):>6.1f}")
    print(f"  平均边数/batch:    {np.mean(edge_counts):>8.1f} ± {np.std(edge_counts):>6.1f}")
    print(f"  平均节点数/样本:   {np.mean(node_counts)/batch_size:>8.1f}")
    print(f"  平均边数/样本:     {np.mean(edge_counts)/batch_size:>8.1f}")
    print(f"  节点数范围:        {min(node_counts):>8,} ~ {max(node_counts):>8,}")
    print(f"  边数范围:          {min(edge_counts):>8,} ~ {max(edge_counts):>8,}")
    print()

    print(f"显存使用统计:")
    print(f"  平均批次显存: {np.mean(memory_usage):>8.1f} MB ± {np.std(memory_usage):>6.1f} MB")
    print(f"  最小批次显存: {min(memory_usage):>8.1f} MB")
    print(f"  最大批次显存: {max(memory_usage):>8.1f} MB")
    print()

    # 估算完整训练显存
    model_size = 500  # MB，粗略估计
    optimizer_size = 1000  # MB，AdamW约2倍参数
    gradient_size = 500  # MB
    activation_size = 800  # MB，粗略估计

    fixed_overhead = model_size + optimizer_size + gradient_size + activation_size
    avg_batch_mem = np.mean(memory_usage)
    total_estimated = fixed_overhead + avg_batch_mem

    print(f"预估完整训练显存:")
    print(f"  固定开销 (模型+优化器+梯度+激活): ~{fixed_overhead:.0f} MB")
    print(f"  批次数据开销:                     ~{avg_batch_mem:.0f} MB")
    print(f"  总计:                             ~{total_estimated:.0f} MB ({total_estimated/1024:.2f} GB)")

    print(f"\n{'='*80}\n")

    return {
        'dataset': dataset_name,
        'target': target,
        'samples': len(train_loader.dataset),
        'batch_size': batch_size,
        'avg_nodes_per_batch': np.mean(node_counts),
        'avg_edges_per_batch': np.mean(edge_counts),
        'avg_nodes_per_sample': np.mean(node_counts) / batch_size,
        'avg_edges_per_sample': np.mean(edge_counts) / batch_size,
        'avg_batch_memory': np.mean(memory_usage),
        'estimated_total_memory': total_estimated,
    }


def compare_datasets(datasets, batch_size=64):
    """比较多个数据集的显存使用

    Args:
        datasets: [(dataset_name, target, display_name), ...]
        batch_size: 批次大小
    """
    results = []

    for dataset, target, name in datasets:
        result = monitor_memory_usage(dataset, target, batch_size)
        if result:
            result['display_name'] = name
            results.append(result)

    if len(results) < 2:
        return

    # 打印对比
    print(f"\n{'='*120}")
    print(f"📊 数据集对比")
    print(f"{'='*120}\n")

    header = f"{'数据集':<25} {'样本数':<10} {'节点/batch':<15} {'边/batch':<15} {'节点/样本':<12} {'批次显存':<12} {'预估总显存':<12}"
    print(header)
    print("-" * 120)

    for r in results:
        print(f"{r['display_name']:<25} {r['samples']:<10} "
              f"{r['avg_nodes_per_batch']:<15.1f} {r['avg_edges_per_batch']:<15.1f} "
              f"{r['avg_nodes_per_sample']:<12.1f} "
              f"{r['avg_batch_memory']:<12.1f} {r['estimated_total_memory']/1024:<12.2f}")

    print()

    # 计算差异
    if len(results) == 2:
        node_diff = (results[0]['avg_nodes_per_batch'] - results[1]['avg_nodes_per_batch'])
        node_diff_pct = node_diff / results[1]['avg_nodes_per_batch'] * 100

        mem_diff = (results[0]['avg_batch_memory'] - results[1]['avg_batch_memory'])
        mem_diff_pct = mem_diff / results[1]['avg_batch_memory'] * 100

        total_diff = (results[0]['estimated_total_memory'] - results[1]['estimated_total_memory'])
        total_diff_pct = total_diff / results[1]['estimated_total_memory'] * 100

        print("📈 差异分析:")
        print(f"  {results[0]['display_name']} vs {results[1]['display_name']}:")
        print(f"    节点数/batch: {node_diff:+.1f} ({node_diff_pct:+.1f}%)")
        print(f"    批次显存:     {mem_diff:+.1f} MB ({mem_diff_pct:+.1f}%)")
        print(f"    预估总显存:   {total_diff:+.1f} MB ({total_diff_pct:+.1f}%)")
        print()

        if abs(node_diff_pct) > 10:
            print("💡 结论: 数据集的平均图大小差异显著，这是显存差异的主要原因！")
        elif abs(mem_diff_pct) < 5:
            print("💡 结论: 显存差异较小，可能是配置参数（pin_memory/num_workers）导致。")
        else:
            print("💡 结论: 显存差异中等，建议检查完整的训练配置。")

    print(f"{'='*120}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GPU显存监控工具')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'compare'],
                       help='运行模式: single=监控单个数据集, compare=对比多个数据集')
    parser.add_argument('--dataset', type=str, default='dft_3d',
                       help='数据集名称')
    parser.add_argument('--target', type=str, default='hse_bandgap',
                       help='目标属性')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--num_batches', type=int, default=10,
                       help='监控的批次数量')

    args = parser.parse_args()

    if args.mode == 'single':
        monitor_memory_usage(
            args.dataset,
            args.target,
            args.batch_size,
            args.num_batches
        )
    elif args.mode == 'compare':
        # 预定义的对比组
        datasets = [
            ('dft_3d', 'hse_bandgap', 'HSE带隙 (1.6K样本)'),
            ('dft_3d', 'formation_energy_peratom', '形成能 (10K+样本)'),
        ]
        compare_datasets(datasets, args.batch_size)
