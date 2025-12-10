"""调试数据加载问题 - 分析为什么大量样本被跳过"""

import sys
import json
import pickle as pk
from pathlib import Path
import math

def analyze_jarvis_data(file_path, target='hse_bandgap'):
    """分析JARVIS数据文件，统计被跳过的样本及原因

    Args:
        file_path: 数据文件路径
        target: 目标属性名称
    """
    print("=" * 80)
    print(f"调试数据加载问题")
    print("=" * 80)
    print(f"文件: {file_path}")
    print(f"目标属性: {target}")
    print()

    # 加载数据
    file_path = Path(file_path)
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
    elif file_path.suffix in ['.pkl', '.pickle']:
        with open(file_path, 'rb') as f:
            data = pk.load(f)
    else:
        print(f"❌ 不支持的文件格式: {file_path.suffix}")
        return

    print(f"📊 总样本数: {len(data)}")
    print()

    # 统计各种情况
    stats = {
        'valid': 0,           # 有效样本
        'missing_target': 0,  # 缺少目标字段
        'target_none': 0,     # 目标为None
        'target_na': 0,       # 目标为"na"
        'target_nan': 0,      # 目标为NaN
        'not_dict': 0,        # 样本不是字典
        'list_target': 0,     # 目标是列表
        'other_errors': 0,    # 其他错误
    }

    error_samples = []
    valid_samples = []

    for idx, sample in enumerate(data):
        try:
            # 检查样本是否是字典
            if not isinstance(sample, dict):
                stats['not_dict'] += 1
                error_samples.append({
                    'index': idx,
                    'reason': 'not_dict',
                    'type': str(type(sample)),
                    'sample': str(sample)[:100]
                })
                continue

            # 检查是否有目标字段
            if target not in sample:
                stats['missing_target'] += 1
                error_samples.append({
                    'index': idx,
                    'reason': 'missing_target',
                    'fields': list(sample.keys())[:10],
                    'jid': sample.get('jid', 'N/A')
                })
                continue

            target_value = sample[target]

            # 检查目标值类型
            if isinstance(target_value, list):
                stats['list_target'] += 1
                stats['valid'] += 1
                valid_samples.append(sample)
                continue

            # 检查目标值是否为None
            if target_value is None:
                stats['target_none'] += 1
                error_samples.append({
                    'index': idx,
                    'reason': 'target_none',
                    'jid': sample.get('jid', 'N/A')
                })
                continue

            # 检查目标值是否为"na"
            if target_value == "na":
                stats['target_na'] += 1
                error_samples.append({
                    'index': idx,
                    'reason': 'target_na',
                    'jid': sample.get('jid', 'N/A')
                })
                continue

            # 检查目标值是否为NaN
            try:
                if math.isnan(target_value):
                    stats['target_nan'] += 1
                    error_samples.append({
                        'index': idx,
                        'reason': 'target_nan',
                        'jid': sample.get('jid', 'N/A')
                    })
                    continue
            except (TypeError, ValueError):
                # 如果不能检查isnan，说明类型不对
                pass

            # 有效样本
            stats['valid'] += 1
            valid_samples.append(sample)

        except Exception as e:
            stats['other_errors'] += 1
            error_samples.append({
                'index': idx,
                'reason': 'exception',
                'error': str(e),
                'error_type': type(e).__name__
            })

    # 打印统计结果
    print("📈 样本统计:")
    print("-" * 80)
    print(f"  ✅ 有效样本:           {stats['valid']:>6} ({stats['valid']/len(data)*100:.1f}%)")
    print(f"  ❌ 跳过样本:           {len(data) - stats['valid']:>6} ({(len(data) - stats['valid'])/len(data)*100:.1f}%)")
    print()
    print("跳过原因分布:")
    print(f"  - 缺少目标字段:       {stats['missing_target']:>6}")
    print(f"  - 目标值为None:       {stats['target_none']:>6}")
    print(f"  - 目标值为'na':       {stats['target_na']:>6}")
    print(f"  - 目标值为NaN:        {stats['target_nan']:>6}")
    print(f"  - 样本不是字典:       {stats['not_dict']:>6}")
    print(f"  - 目标值是列表:       {stats['list_target']:>6} (有效)")
    print(f"  - 其他错误:           {stats['other_errors']:>6}")
    print()

    # 显示一些错误样本的详细信息
    if error_samples:
        print("🔍 错误样本示例 (前10个):")
        print("-" * 80)
        for i, err in enumerate(error_samples[:10]):
            print(f"\n样本 #{err['index']}:")
            print(f"  原因: {err['reason']}")
            for key, value in err.items():
                if key not in ['index', 'reason']:
                    print(f"  {key}: {value}")

    # 显示一些有效样本
    if valid_samples:
        print("\n✅ 有效样本示例 (前3个):")
        print("-" * 80)
        for i, sample in enumerate(valid_samples[:3]):
            print(f"\n样本 #{i}:")
            print(f"  jid: {sample.get('jid', 'N/A')}")
            print(f"  {target}: {sample.get(target, 'N/A')}")
            print(f"  字段数: {len(sample)}")
            print(f"  字段: {list(sample.keys())[:10]}")

    # 返回统计信息
    return stats, error_samples, valid_samples


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='调试JARVIS数据加载问题')
    parser.add_argument('--data_path', type=str,
                       default='/public/home/ghzhang/crysmmnet-main/dataset/jarvis/hse_bandgap',
                       help='数据文件或目录路径')
    parser.add_argument('--target', type=str, default='hse_bandgap',
                       help='目标属性名称')

    args = parser.parse_args()

    # 如果是目录，查找JSON或Pickle文件
    data_path = Path(args.data_path)
    if data_path.is_dir():
        json_files = list(data_path.glob('*.json'))
        pkl_files = list(data_path.glob('*.pkl')) + list(data_path.glob('*.pickle'))

        if json_files:
            data_file = json_files[0]
            print(f"找到JSON文件: {data_file}")
        elif pkl_files:
            data_file = pkl_files[0]
            print(f"找到Pickle文件: {data_file}")
        else:
            print(f"❌ 在 {data_path} 中没有找到JSON或Pickle文件")
            sys.exit(1)
    else:
        data_file = data_path

    # 分析数据
    stats, errors, valid = analyze_jarvis_data(data_file, args.target)

    print("\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)
