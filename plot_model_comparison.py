#!/usr/bin/env python3
"""
对比两个模型在不同遮挡率下的性能

用法:
    python plot_model_comparison.py \
        --model1_results ./results_model1_*.json \
        --model2_results ./results_model2_*.json \
        --model1_name "中期融合+跨模态+细粒度" \
        --model2_name "跨模态+细粒度" \
        --output_dir ./comparison_plots
"""

import os
import json
import glob
import argparse
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set font for better display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def load_results(pattern: str) -> List[Dict]:
    """加载匹配模式的所有结果文件"""
    files = glob.glob(pattern)
    results = []

    for file in sorted(files):
        with open(file, 'r') as f:
            data = json.load(f)
            results.append(data)

    return results


def organize_by_strategy(results: List[Dict]) -> Dict[str, List[Dict]]:
    """按策略组织结果"""
    organized = {}

    for result in results:
        strategy = result.get('masking_strategy', 'unknown')
        if strategy not in organized:
            organized[strategy] = []
        organized[strategy].append(result)

    # 按遮挡率排序
    for strategy in organized:
        organized[strategy].sort(key=lambda x: x.get('masking_ratio', 0))

    return organized


def plot_comparison(
    model1_results: Dict[str, List[Dict]],
    model2_results: Dict[str, List[Dict]],
    model1_name: str,
    model2_name: str,
    output_dir: str
):
    """生成对比图表"""

    strategies = list(set(model1_results.keys()) & set(model2_results.keys()))

    if not strategies:
        print("Warning: No common masking strategies found")
        return

    print(f"Found {len(strategies)} common strategies: {strategies}")

    # 为每个策略生成对比图
    for strategy in strategies:
        plot_strategy_comparison(
            model1_results[strategy],
            model2_results[strategy],
            strategy,
            model1_name,
            model2_name,
            output_dir
        )

    # 生成综合对比图
    plot_comprehensive_comparison(
        model1_results,
        model2_results,
        strategies,
        model1_name,
        model2_name,
        output_dir
    )


def plot_strategy_comparison(
    model1_data: List[Dict],
    model2_data: List[Dict],
    strategy: str,
    model1_name: str,
    model2_name: str,
    output_dir: str
):
    """为单个策略生成对比图"""

    # 提取数据
    ratios1 = [d['masking_ratio'] * 100 for d in model1_data]
    mae1 = [d['metrics']['mae'] for d in model1_data]
    rmse1 = [d['metrics']['rmse'] for d in model1_data]
    r2_1 = [d['metrics']['r2'] for d in model1_data]

    ratios2 = [d['masking_ratio'] * 100 for d in model2_data]
    mae2 = [d['metrics']['mae'] for d in model2_data]
    rmse2 = [d['metrics']['rmse'] for d in model2_data]
    r2_2 = [d['metrics']['r2'] for d in model2_data]

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 颜色
    color1 = '#2E86AB'  # 蓝色
    color2 = '#A23B72'  # 紫色

    # 1. MAE Comparison
    ax1 = axes[0, 0]
    ax1.plot(ratios1, mae1, 'o-', linewidth=2.5, markersize=8,
             label=model1_name, color=color1, alpha=0.8)
    ax1.plot(ratios2, mae2, 's-', linewidth=2.5, markersize=8,
             label=model2_name, color=color2, alpha=0.8)
    ax1.set_xlabel('Masking Ratio (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax1.set_title(f'MAE Comparison ({strategy})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 标注关键点（100%遮挡）
    if len(mae1) > 0 and len(mae2) > 0:
        mae1_100 = mae1[-1]
        mae2_100 = mae2[-1]
        improvement = (mae1_100 - mae2_100) / mae1_100 * 100

        ax1.annotate(
            f'{improvement:+.1f}%',
            xy=(ratios1[-1], mae2_100),
            xytext=(ratios1[-1] - 15, (mae1_100 + mae2_100) / 2),
            arrowprops=dict(arrowstyle='->', color='green' if improvement > 0 else 'red', lw=2),
            fontsize=11,
            fontweight='bold',
            color='green' if improvement > 0 else 'red',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3)
        )

    # 2. RMSE Comparison
    ax2 = axes[0, 1]
    ax2.plot(ratios1, rmse1, 'o-', linewidth=2.5, markersize=8,
             label=model1_name, color=color1, alpha=0.8)
    ax2.plot(ratios2, rmse2, 's-', linewidth=2.5, markersize=8,
             label=model2_name, color=color2, alpha=0.8)
    ax2.set_xlabel('Masking Ratio (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax2.set_title(f'RMSE Comparison ({strategy})', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 3. R² Comparison
    ax3 = axes[1, 0]
    ax3.plot(ratios1, r2_1, 'o-', linewidth=2.5, markersize=8,
             label=model1_name, color=color1, alpha=0.8)
    ax3.plot(ratios2, r2_2, 's-', linewidth=2.5, markersize=8,
             label=model2_name, color=color2, alpha=0.8)
    ax3.set_xlabel('Masking Ratio (%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax3.set_title(f'R² Comparison ({strategy})', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # 4. MAE Improvement Percentage
    ax4 = axes[1, 1]

    # Calculate improvement percentage
    improvements = []
    common_ratios = []
    for r1, m1 in zip(ratios1, mae1):
        for r2, m2 in zip(ratios2, mae2):
            if abs(r1 - r2) < 0.1:  # Same masking ratio
                improvement = (m1 - m2) / m1 * 100
                improvements.append(improvement)
                common_ratios.append(r1)
                break

    colors = ['green' if x > 0 else 'red' for x in improvements]
    ax4.bar(common_ratios, improvements, color=colors, alpha=0.6, edgecolor='black', linewidth=1.5)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Masking Ratio (%)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('MAE Improvement (%)', fontsize=12, fontweight='bold')
    ax4.set_title(f'MAE Improvement ({strategy})', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--', axis='y')

    # 添加数值标签
    for r, imp in zip(common_ratios, improvements):
        ax4.text(r, imp, f'{imp:+.1f}%',
                ha='center', va='bottom' if imp > 0 else 'top',
                fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save
    output_file = os.path.join(output_dir, f'comparison_{strategy}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_comprehensive_comparison(
    model1_results: Dict[str, List[Dict]],
    model2_results: Dict[str, List[Dict]],
    strategies: List[str],
    model1_name: str,
    model2_name: str,
    output_dir: str
):
    """生成综合对比图"""

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    color1 = '#2E86AB'
    color2 = '#A23B72'

    # 1. 所有策略的MAE对比
    ax1 = axes[0, 0]
    for i, strategy in enumerate(strategies):
        data1 = model1_results[strategy]
        data2 = model2_results[strategy]

        ratios1 = [d['masking_ratio'] * 100 for d in data1]
        mae1 = [d['metrics']['mae'] for d in data1]

        ratios2 = [d['masking_ratio'] * 100 for d in data2]
        mae2 = [d['metrics']['mae'] for d in data2]

        # 模型1
        ax1.plot(ratios1, mae1, 'o-', linewidth=2, markersize=6,
                label=f'{model1_name} - {strategy}', alpha=0.7)

        # 模型2
        ax1.plot(ratios2, mae2, 's--', linewidth=2, markersize=6,
                label=f'{model2_name} - {strategy}', alpha=0.7)

    ax1.set_xlabel('Masking Ratio (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax1.set_title('MAE Comparison Across All Strategies', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8, loc='best', ncol=2)
    ax1.grid(True, alpha=0.3)

    # 2. Comparison at 100% masking (bar chart)
    ax2 = axes[0, 1]

    x = np.arange(len(strategies))
    width = 0.35

    mae1_100 = []
    mae2_100 = []

    for strategy in strategies:
        data1 = model1_results[strategy]
        data2 = model2_results[strategy]

        # 找100%遮挡的数据
        mae1 = [d['metrics']['mae'] for d in data1 if abs(d['masking_ratio'] - 1.0) < 0.01]
        mae2 = [d['metrics']['mae'] for d in data2 if abs(d['masking_ratio'] - 1.0) < 0.01]

        mae1_100.append(mae1[0] if mae1 else 0)
        mae2_100.append(mae2[0] if mae2 else 0)

    bars1 = ax2.bar(x - width/2, mae1_100, width, label=model1_name,
                    color=color1, alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x + width/2, mae2_100, width, label=model2_name,
                    color=color2, alpha=0.8, edgecolor='black')

    ax2.set_xlabel('Masking Strategy', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MAE (100% Masking)', fontsize=12, fontweight='bold')
    ax2.set_title('Strategy Comparison at 100% Masking', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=15, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 3. Average improvement percentage
    ax3 = axes[1, 0]

    avg_improvements = []

    for strategy in strategies:
        data1 = model1_results[strategy]
        data2 = model2_results[strategy]

        improvements = []
        for d1 in data1:
            for d2 in data2:
                if abs(d1['masking_ratio'] - d2['masking_ratio']) < 0.01:
                    mae1 = d1['metrics']['mae']
                    mae2 = d2['metrics']['mae']
                    improvement = (mae1 - mae2) / mae1 * 100
                    improvements.append(improvement)
                    break

        avg_improvements.append(np.mean(improvements) if improvements else 0)

    colors = ['green' if x > 0 else 'red' for x in avg_improvements]
    bars = ax3.bar(strategies, avg_improvements, color=colors, alpha=0.6,
                   edgecolor='black', linewidth=1.5)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Masking Strategy', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average MAE Improvement (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Average Improvement by Strategy', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}%', ha='center',
                va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

    # 4. Statistical summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Calculate overall statistics
    summary_text = "STATISTICAL SUMMARY\n" + "="*60 + "\n\n"

    for strategy in strategies:
        data1 = model1_results[strategy]
        data2 = model2_results[strategy]

        # 0% masking
        mae1_0 = [d['metrics']['mae'] for d in data1 if d['masking_ratio'] < 0.01]
        mae2_0 = [d['metrics']['mae'] for d in data2 if d['masking_ratio'] < 0.01]

        # 100% masking
        mae1_100 = [d['metrics']['mae'] for d in data1 if abs(d['masking_ratio'] - 1.0) < 0.01]
        mae2_100 = [d['metrics']['mae'] for d in data2 if abs(d['masking_ratio'] - 1.0) < 0.01]

        summary_text += f"{strategy}:\n"

        if mae1_0 and mae2_0:
            imp_0 = (mae1_0[0] - mae2_0[0]) / mae1_0[0] * 100
            summary_text += f"  0% mask:   M1={mae1_0[0]:.3f}, "
            summary_text += f"M2={mae2_0[0]:.3f} ({imp_0:+.1f}%)\n"

        if mae1_100 and mae2_100:
            imp_100 = (mae1_100[0] - mae2_100[0]) / mae1_100[0] * 100
            summary_text += f"  100% mask: M1={mae1_100[0]:.3f}, "
            summary_text += f"M2={mae2_100[0]:.3f} ({imp_100:+.1f}%)\n"

        summary_text += "\n"

    # Overall conclusion
    summary_text += "="*60 + "\n"
    summary_text += "OVERALL ASSESSMENT:\n\n"

    if np.mean(avg_improvements) > 0:
        summary_text += f"{model2_name} performs better on average\n"
        summary_text += f"  Avg improvement: {np.mean(avg_improvements):.2f}%\n"
    else:
        summary_text += f"{model1_name} performs better on average\n"
        summary_text += f"  Avg improvement: {-np.mean(avg_improvements):.2f}%\n"

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save
    output_file = os.path.join(output_dir, 'comprehensive_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comprehensive comparison: {output_file}")
    plt.close()


def print_summary_table(
    model1_results: Dict[str, List[Dict]],
    model2_results: Dict[str, List[Dict]],
    strategies: List[str],
    model1_name: str,
    model2_name: str
):
    """Print comparison table"""

    print("\n" + "="*100)
    print("MODEL COMPARISON SUMMARY")
    print("="*100)

    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        print("-"*100)
        print(f"{'Masking':<10} {model1_name+' MAE':<20} {model2_name+' MAE':<20} {'Improvement':<15} {'Status':<20}")
        print("-"*100)

        data1 = model1_results[strategy]
        data2 = model2_results[strategy]

        for d1 in data1:
            ratio = d1['masking_ratio']
            mae1 = d1['metrics']['mae']

            # Find corresponding model2 data
            mae2 = None
            for d2 in data2:
                if abs(d2['masking_ratio'] - ratio) < 0.01:
                    mae2 = d2['metrics']['mae']
                    break

            if mae2 is not None:
                improvement = (mae1 - mae2) / mae1 * 100
                status = "✓ Better" if improvement > 0 else "✗ Worse"

                print(f"{ratio*100:<10.0f}% {mae1:<20.4f} {mae2:<20.4f} "
                      f"{improvement:<14.2f}% {status:<20}")

    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description='Compare evaluation results of two models')

    parser.add_argument(
        '--model1_results',
        type=str,
        required=True,
        help='Model 1 result files (supports wildcards, e.g., ./results_model1_*.json)'
    )
    parser.add_argument(
        '--model2_results',
        type=str,
        required=True,
        help='Model 2 result files (supports wildcards, e.g., ./results_model2_*.json)'
    )
    parser.add_argument(
        '--model1_name',
        type=str,
        default='Model 1',
        help='Name of model 1'
    )
    parser.add_argument(
        '--model2_name',
        type=str,
        default='Model 2',
        help='Name of model 2'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./comparison_plots',
        help='Output directory for plots'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    print(f"Loading Model 1 results: {args.model1_results}")
    model1_data = load_results(args.model1_results)
    print(f"  Found {len(model1_data)} result files")

    print(f"\nLoading Model 2 results: {args.model2_results}")
    model2_data = load_results(args.model2_results)
    print(f"  Found {len(model2_data)} result files")

    if not model1_data or not model2_data:
        print("\nError: No result files found")
        return

    # Organize by strategy
    model1_by_strategy = organize_by_strategy(model1_data)
    model2_by_strategy = organize_by_strategy(model2_data)

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_comparison(
        model1_by_strategy,
        model2_by_strategy,
        args.model1_name,
        args.model2_name,
        args.output_dir
    )

    # Print summary table
    strategies = list(set(model1_by_strategy.keys()) & set(model2_by_strategy.keys()))
    print_summary_table(
        model1_by_strategy,
        model2_by_strategy,
        strategies,
        args.model1_name,
        args.model2_name
    )

    print(f"\n✓ All plots saved to: {args.output_dir}")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
