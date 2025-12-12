"""监控训练过程中 text_scale 的演变

这个脚本可以：
1. 从训练 checkpoints 中提取 text_scale 值
2. 绘制 text_scale 随训练进程的变化
3. 分析 text_scale 是否收敛

使用方法:
python monitor_text_scale.py \
    --checkpoint_dir ./outputs/improved_fusion \
    --output_plot text_scale_evolution.png
"""

import argparse
import os
import glob
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def extract_text_scale_from_checkpoint(checkpoint_path):
    """从 checkpoint 中提取 text_scale 值"""
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_state = ckpt['model']

        text_scales = {}
        for key, value in model_state.items():
            if 'text_scale' in key and 'middle_fusion' in key:
                # 提取层号，例如 'middle_fusion_modules.layer_2.text_scale' -> 'layer_2'
                layer_name = key.split('.')[1]  # 'layer_2'
                text_scales[layer_name] = value.item()

        # 获取 epoch 信息（如果有）
        epoch = ckpt.get('epoch', None)

        return text_scales, epoch
    except Exception as e:
        print(f"⚠️  无法加载 {checkpoint_path}: {e}")
        return None, None


def find_checkpoints(checkpoint_dir):
    """查找所有 checkpoint 文件"""
    # 查找常见的 checkpoint 命名模式
    patterns = [
        'checkpoint_*.pt',
        'model_epoch_*.pt',
        'best_model.pt',
        'best_test_model.pt'
    ]

    checkpoint_files = []
    for pattern in patterns:
        checkpoint_files.extend(glob.glob(os.path.join(checkpoint_dir, pattern)))

    return sorted(checkpoint_files)


def main():
    parser = argparse.ArgumentParser(description="监控 text_scale 演变")
    parser.add_argument('--checkpoint_dir', required=True, help='checkpoint 目录')
    parser.add_argument('--output_plot', default='text_scale_evolution.png', help='输出图表路径')
    parser.add_argument('--checkpoint_pattern', default=None, help='checkpoint 文件模式（可选）')
    args = parser.parse_args()

    print("=" * 80)
    print("📊 监控 text_scale 演变")
    print("=" * 80)

    # 查找 checkpoints
    if args.checkpoint_pattern:
        checkpoint_files = glob.glob(os.path.join(args.checkpoint_dir, args.checkpoint_pattern))
        checkpoint_files = sorted(checkpoint_files)
    else:
        checkpoint_files = find_checkpoints(args.checkpoint_dir)

    print(f"\n📂 找到 {len(checkpoint_files)} 个 checkpoints")

    if len(checkpoint_files) == 0:
        print("❌ 未找到任何 checkpoint 文件")
        print(f"   请检查目录: {args.checkpoint_dir}")
        return

    # 提取数据
    all_data = []
    layer_names = set()

    for i, ckpt_path in enumerate(checkpoint_files):
        text_scales, epoch = extract_text_scale_from_checkpoint(ckpt_path)

        if text_scales:
            layer_names.update(text_scales.keys())
            all_data.append({
                'checkpoint': os.path.basename(ckpt_path),
                'index': i,
                'epoch': epoch if epoch is not None else i,
                'text_scales': text_scales
            })

            # 打印进度
            if (i + 1) % 10 == 0:
                print(f"   处理进度: {i + 1}/{len(checkpoint_files)}")

    if len(all_data) == 0:
        print("\n❌ 所有 checkpoints 都无法提取 text_scale")
        return

    print(f"\n✅ 成功提取 {len(all_data)} 个 checkpoints 的数据")
    print(f"   找到的层: {sorted(layer_names)}")

    # 打印统计信息
    print("\n" + "=" * 80)
    print("📈 text_scale 统计")
    print("=" * 80)

    for layer_name in sorted(layer_names):
        scales = [d['text_scales'].get(layer_name, None) for d in all_data]
        scales = [s for s in scales if s is not None]

        if scales:
            print(f"\n{layer_name}:")
            print(f"   初始值: {scales[0]:.4f}")
            print(f"   最终值: {scales[-1]:.4f}")
            print(f"   变化: {scales[-1] - scales[0]:+.4f} ({(scales[-1] - scales[0]) / scales[0] * 100:+.2f}%)")
            print(f"   最小值: {min(scales):.4f}")
            print(f"   最大值: {max(scales):.4f}")
            print(f"   平均值: {np.mean(scales):.4f}")
            print(f"   标准差: {np.std(scales):.4f}")

    # 绘制图表
    print(f"\n📊 生成图表: {args.output_plot}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 图 1: text_scale 随训练进程的变化
    ax1 = axes[0]
    for layer_name in sorted(layer_names):
        epochs = [d['epoch'] for d in all_data if layer_name in d['text_scales']]
        scales = [d['text_scales'][layer_name] for d in all_data if layer_name in d['text_scales']]

        ax1.plot(epochs, scales, marker='o', label=layer_name, linewidth=2, markersize=4)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('text_scale Value', fontsize=12)
    ax1.set_title('Evolution of Learnable Text Scaling Factor', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=12.0, color='red', linestyle='--', alpha=0.5, label='Initial Value (12.0)')

    # 图 2: text_scale 的变化率
    ax2 = axes[1]
    for layer_name in sorted(layer_names):
        scales = [d['text_scales'][layer_name] for d in all_data if layer_name in d['text_scales']]

        if len(scales) > 1:
            # 计算变化率（相邻 epoch 的差值）
            change_rates = np.diff(scales)
            epochs = [d['epoch'] for d in all_data if layer_name in d['text_scales']][1:]

            ax2.plot(epochs, change_rates, marker='o', label=layer_name, linewidth=2, markersize=4)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Change Rate (Δtext_scale)', fontsize=12)
    ax2.set_title('Rate of Change in Text Scaling Factor', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存")

    # 分析收敛性
    print("\n" + "=" * 80)
    print("🔍 收敛性分析")
    print("=" * 80)

    for layer_name in sorted(layer_names):
        scales = [d['text_scales'][layer_name] for d in all_data if layer_name in d['text_scales']]

        if len(scales) > 10:
            # 分析最后 10 个 epoch 的变化
            recent_scales = scales[-10:]
            recent_std = np.std(recent_scales)
            recent_mean = np.mean(recent_scales)

            print(f"\n{layer_name} (最后 10 个 epochs):")
            print(f"   平均值: {recent_mean:.4f}")
            print(f"   标准差: {recent_std:.4f}")
            print(f"   变化范围: [{min(recent_scales):.4f}, {max(recent_scales):.4f}]")

            # 判断收敛
            if recent_std < 0.5:
                print(f"   ✅ 已收敛（标准差 < 0.5）")
            elif recent_std < 1.0:
                print(f"   ⚠️  接近收敛（标准差 < 1.0）")
            else:
                print(f"   ❌ 仍在变化（标准差 > 1.0），建议继续训练")

    # 提供建议
    print("\n" + "=" * 80)
    print("💡 建议")
    print("=" * 80)

    for layer_name in sorted(layer_names):
        scales = [d['text_scales'][layer_name] for d in all_data if layer_name in d['text_scales']]
        final_scale = scales[-1]

        print(f"\n{layer_name}:")
        if final_scale > 20.0:
            print(f"   ⚠️  最终缩放值 {final_scale:.2f} 较高")
            print(f"      • text_transform 输出可能仍然较弱")
            print(f"      • 建议增加 middle_fusion_initial_scale 到 {final_scale * 1.2:.1f}")
        elif final_scale < 5.0:
            print(f"   ⚠️  最终缩放值 {final_scale:.2f} 较低")
            print(f"      • text_transform 输出可能过强")
            print(f"      • 建议检查 text_transform 初始化")
        else:
            print(f"   ✅ 最终缩放值 {final_scale:.2f} 在合理范围内")

    print("\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
