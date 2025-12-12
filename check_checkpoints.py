"""检查训练目录中所有checkpoint的详细信息

使用方法:
python check_checkpoints.py --checkpoint_dir ./output_100epochs_42_bs128_sw_ju_hse/hse_bandgap-2
"""

import os
import argparse
import torch
from pathlib import Path


def check_checkpoint(checkpoint_path):
    """检查单个checkpoint的信息"""
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        info = {
            'file': os.path.basename(checkpoint_path),
            'epoch': ckpt.get('epoch', 'N/A'),
            'best_val_mae': ckpt.get('best_val_mae', 'N/A'),
            'best_test_mae': ckpt.get('best_test_mae', 'N/A'),
            'learning_rate': ckpt.get('learning_rate', 'N/A'),
        }

        # 尝试从trainer获取信息
        if 'trainer' in ckpt and hasattr(ckpt['trainer'], 'state_dict'):
            trainer_state = ckpt['trainer'].state_dict()
            info['training_iteration'] = trainer_state.get('iteration', 'N/A')

        return info

    except Exception as e:
        return {
            'file': os.path.basename(checkpoint_path),
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='检查checkpoint信息')
    parser.add_argument('--checkpoint_dir', required=True, help='Checkpoint目录')
    args = parser.parse_args()

    print(f"\n{'='*100}")
    print(f"检查目录: {args.checkpoint_dir}")
    print(f"{'='*100}\n")

    # 查找所有 .pt 文件
    checkpoint_files = sorted(Path(args.checkpoint_dir).glob('*.pt'))

    if not checkpoint_files:
        print("❌ 未找到任何 .pt 文件")
        return

    print(f"找到 {len(checkpoint_files)} 个checkpoint文件\n")

    # 检查每个文件
    all_info = []
    for ckpt_path in checkpoint_files:
        info = check_checkpoint(str(ckpt_path))
        all_info.append(info)

    # 打印表格
    print(f"{'文件名':<30} {'Epoch':<10} {'Val MAE':<15} {'Test MAE':<15} {'LR':<15}")
    print("-" * 100)

    for info in all_info:
        if 'error' in info:
            print(f"{info['file']:<30} ERROR: {info['error']}")
        else:
            epoch_str = str(info['epoch']) if info['epoch'] != 'N/A' else 'N/A'
            val_mae_str = f"{info['best_val_mae']:.4f}" if info['best_val_mae'] != 'N/A' else 'N/A'
            test_mae_str = f"{info['best_test_mae']:.4f}" if info['best_test_mae'] != 'N/A' else 'N/A'
            lr_str = f"{info['learning_rate']:.6f}" if info['learning_rate'] != 'N/A' else 'N/A'

            print(f"{info['file']:<30} {epoch_str:<10} {val_mae_str:<15} {test_mae_str:<15} {lr_str:<15}")

    # 找出最好的模型
    print(f"\n{'='*100}")
    print("推荐用于继续训练的checkpoint:")
    print(f"{'='*100}\n")

    # 按 Val MAE 排序
    valid_info = [i for i in all_info if 'error' not in i and i['best_val_mae'] != 'N/A']

    if valid_info:
        best_val = min(valid_info, key=lambda x: x['best_val_mae'])
        print(f"✅ 验证集最佳: {best_val['file']}")
        print(f"   Epoch: {best_val['epoch']}")
        print(f"   Val MAE: {best_val['best_val_mae']:.4f}")
        print()

        # 检查哪个会被 resume 加载
        checkpoint_files_numbered = [i for i in all_info if i['file'].startswith('checkpoint_')]
        if checkpoint_files_numbered:
            # 提取编号
            numbered = []
            for info in checkpoint_files_numbered:
                try:
                    num = int(info['file'].split('_')[1].split('.')[0])
                    numbered.append((num, info))
                except:
                    pass

            if numbered:
                max_num, max_info = max(numbered, key=lambda x: x[0])
                print(f"🔄 --resume 1 会加载: {max_info['file']} (编号最大: {max_num})")

                if max_info['file'] != best_val['file']:
                    print(f"\n⚠️  警告: resume会加载 {max_info['file']}，但验证集最佳是 {best_val['file']}")
                    print(f"\n💡 建议操作:")
                    print(f"   cp {best_val['file']} checkpoint_{max_num + 1}.pt")
                    print(f"   然后运行 --resume 1")
    else:
        print("❌ 未找到有效的性能指标")

    print(f"\n{'='*100}\n")


if __name__ == '__main__':
    main()
