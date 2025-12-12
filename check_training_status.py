"""检查训练状态和使用的文件

使用方法:
python check_training_status.py --log_file training.log
"""

import argparse
import os
import re
import subprocess


def check_running_processes():
    """检查正在运行的训练进程"""
    print(f"\n{'='*80}")
    print("正在运行的训练进程:")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )

        lines = result.stdout.split('\n')
        training_processes = [
            line for line in lines
            if 'train_with_cross_modal_attention.py' in line or 'train.py' in line
        ]

        if training_processes:
            for proc in training_processes:
                print(proc)
        else:
            print("❌ 没有找到正在运行的训练进程")

    except Exception as e:
        print(f"⚠️  无法检查进程: {e}")


def parse_log_file(log_file):
    """解析日志文件，找出加载的checkpoint"""
    print(f"\n{'='*80}")
    print(f"分析日志文件: {log_file}")
    print(f"{'='*80}\n")

    if not os.path.exists(log_file):
        print(f"❌ 日志文件不存在: {log_file}")
        return

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # 查找checkpoint相关信息
    checkpoint_info = []
    resume_info = []

    for i, line in enumerate(lines[:200]):  # 只看前200行
        if 'checkpoint' in line.lower() or 'resume' in line.lower():
            checkpoint_info.append((i+1, line.strip()))
        if 'loading' in line.lower() or '加载' in line:
            resume_info.append((i+1, line.strip()))

    if checkpoint_info:
        print("📂 Checkpoint相关信息:")
        for line_num, content in checkpoint_info[:10]:  # 只显示前10条
            print(f"   第{line_num}行: {content}")

    if resume_info:
        print("\n🔄 加载信息:")
        for line_num, content in resume_info[:5]:
            print(f"   第{line_num}行: {content}")

    # 查找配置信息
    print("\n⚙️  训练配置:")
    config_keywords = ['epoch', 'batch', 'learning_rate', 'dataset', 'property']
    for i, line in enumerate(lines[:100]):
        for keyword in config_keywords:
            if keyword in line.lower() and ':' in line:
                print(f"   {line.strip()}")
                break

    # 查找最新的训练进度
    print("\n📊 最新训练进度:")
    epoch_lines = [line for line in lines if 'Epoch' in line or 'epoch' in line]
    if epoch_lines:
        print(f"   {epoch_lines[-1].strip()}")

    loss_lines = [line for line in lines if 'loss' in line.lower() or 'mae' in line.lower()]
    if loss_lines:
        print(f"   {loss_lines[-1].strip()}")


def check_checkpoint_dir(checkpoint_dir):
    """检查checkpoint目录"""
    print(f"\n{'='*80}")
    print(f"Checkpoint目录: {checkpoint_dir}")
    print(f"{'='*80}\n")

    if not os.path.exists(checkpoint_dir):
        print(f"❌ 目录不存在: {checkpoint_dir}")
        return

    # 列出所有.pt文件
    pt_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])

    if not pt_files:
        print("❌ 没有找到checkpoint文件")
        return

    print("找到的checkpoint文件:")
    for f in pt_files:
        file_path = os.path.join(checkpoint_dir, f)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        mtime = os.path.getmtime(file_path)

        # 格式化时间
        from datetime import datetime
        mod_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

        print(f"   {f:<30} {size_mb:>8.1f} MB    {mod_time}")

    # 检查哪个会被resume加载
    checkpoint_files = [f for f in pt_files if f.startswith('checkpoint_')]
    if checkpoint_files:
        numbers = []
        for f in checkpoint_files:
            try:
                num = int(f.split('_')[1].split('.')[0])
                numbers.append((num, f))
            except:
                pass

        if numbers:
            max_num, max_file = max(numbers, key=lambda x: x[0])
            print(f"\n🔄 --resume 1 会加载: {max_file}")


def main():
    parser = argparse.ArgumentParser(description='检查训练状态')
    parser.add_argument('--log_file', default='training.log', help='训练日志文件')
    parser.add_argument('--checkpoint_dir',
                       default='./output_100epochs_42_bs128_sw_ju_hse/hse_bandgap-2',
                       help='Checkpoint目录')
    parser.add_argument('--check_processes', action='store_true',
                       help='检查正在运行的进程')
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("训练状态检查工具")
    print(f"{'='*80}")

    # 检查进程
    if args.check_processes:
        check_running_processes()

    # 解析日志
    if os.path.exists(args.log_file):
        parse_log_file(args.log_file)
    else:
        print(f"\n⚠️  日志文件不存在: {args.log_file}")

    # 检查checkpoint目录
    check_checkpoint_dir(args.checkpoint_dir)

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
