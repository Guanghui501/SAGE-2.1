#!/usr/bin/env python
"""
从训练预测CSV文件提取正确的测试集

这个脚本读取训练时生成的predictions CSV文件，提取测试集ID，
然后从预处理数据中筛选出对应的测试集。

用法:
    python extract_test_set_from_csv.py \
        --predictions_csv /path/to/predictions_best_test_model_test.csv \
        --preprocessed_dir /path/to/preprocessed_data \
        --dataset jarvis \
        --property mbj_bandgap \
        --output_dir ./corrected_test_set
"""

import sys
import os
import pickle
import json
import argparse
import pandas as pd
from pathlib import Path

def load_test_ids_from_csv(csv_path):
    """从predictions CSV文件加载测试集ID"""
    print(f"读取CSV文件: {csv_path}")

    # 读取CSV
    df = pd.read_csv(csv_path)

    print(f"\nCSV文件信息:")
    print(f"  列名: {list(df.columns)}")
    print(f"  行数: {len(df)}")
    print(f"\n前5行:")
    print(df.head())

    # 提取ID列
    if 'id' in df.columns:
        test_ids = df['id'].tolist()
    elif 'ID' in df.columns:
        test_ids = df['ID'].tolist()
    else:
        print(f"\n✗ 错误: CSV中没有找到'id'或'ID'列")
        print(f"  可用列: {list(df.columns)}")
        return None

    # 转换为字符串（有些ID可能是整数）
    test_ids = [str(tid).strip() for tid in test_ids]

    print(f"\n✓ 成功提取 {len(test_ids)} 个测试集ID")
    print(f"  前5个ID: {test_ids[:5]}")
    print(f"  后5个ID: {test_ids[-5:]}")

    return test_ids


def load_preprocessed_data(preprocessed_dir, dataset, property_name):
    """加载所有预处理数据"""
    data_dir = Path(preprocessed_dir) / dataset / property_name

    all_data = []

    for split in ['train', 'val', 'test']:
        pkl_file = data_dir / f"{split}.pkl"
        if pkl_file.exists():
            with open(pkl_file, 'rb') as f:
                split_data = pickle.load(f)
            print(f"✓ 加载 {split} 集: {len(split_data)} 个样本")
            all_data.extend(split_data)
        else:
            print(f"⚠ 找不到 {split} 集: {pkl_file}")

    print(f"\n总数据量: {len(all_data)} 个样本")

    return all_data


def filter_test_set(all_data, test_ids):
    """从预处理数据中筛选出测试集"""
    print(f"\n开始筛选测试集...")
    print(f"  目标ID数量: {len(test_ids)}")

    # 创建ID到数据的映射
    id_to_data = {}
    for item in all_data:
        item_id = str(item['id']).strip()
        id_to_data[item_id] = item

    print(f"  预处理数据中的唯一ID数: {len(id_to_data)}")

    # 检查ID格式
    sample_preprocessed_ids = list(id_to_data.keys())[:5]
    print(f"\n  预处理数据ID示例: {sample_preprocessed_ids}")
    print(f"  CSV中的ID示例: {test_ids[:5]}")

    # 筛选测试集
    filtered_test_set = []
    missing_ids = []

    for test_id in test_ids:
        test_id_str = str(test_id).strip()

        # 尝试多种ID格式匹配
        matched = False

        # 1. 直接匹配
        if test_id_str in id_to_data:
            filtered_test_set.append(id_to_data[test_id_str])
            matched = True
        # 2. 尝试添加前缀 (如 JVASP-)
        elif f"JVASP-{test_id_str}" in id_to_data:
            filtered_test_set.append(id_to_data[f"JVASP-{test_id_str}"])
            matched = True
        # 3. 尝试去掉前缀
        elif test_id_str.startswith('JVASP-'):
            clean_id = test_id_str.replace('JVASP-', '')
            if clean_id in id_to_data:
                filtered_test_set.append(id_to_data[clean_id])
                matched = True

        if not matched:
            missing_ids.append(test_id_str)

    print(f"\n✓ 成功匹配: {len(filtered_test_set)} 个样本 ({len(filtered_test_set)/len(test_ids)*100:.1f}%)")

    if missing_ids:
        print(f"⚠ 缺失: {len(missing_ids)} 个样本 ({len(missing_ids)/len(test_ids)*100:.1f}%)")
        print(f"  前5个缺失ID: {missing_ids[:5]}")

    return filtered_test_set, missing_ids


def verify_targets(filtered_test_set, predictions_csv):
    """验证目标值是否匹配"""
    print(f"\n验证目标值...")

    df = pd.read_csv(predictions_csv)

    # 创建CSV中的ID到target的映射
    csv_targets = {}
    for _, row in df.iterrows():
        csv_targets[str(row['id']).strip()] = float(row['target'])

    # 检查前10个样本
    mismatches = 0
    for i, item in enumerate(filtered_test_set[:10]):
        item_id = str(item['id']).strip()
        pkl_target = float(item['target'])

        # 尝试匹配ID
        csv_target = None
        if item_id in csv_targets:
            csv_target = csv_targets[item_id]
        elif f"JVASP-{item_id}" in csv_targets:
            csv_target = csv_targets[f"JVASP-{item_id}"]
        elif item_id.startswith('JVASP-'):
            clean_id = item_id.replace('JVASP-', '')
            if clean_id in csv_targets:
                csv_target = csv_targets[clean_id]

        if csv_target is not None:
            diff = abs(pkl_target - csv_target)
            if diff > 1e-5:
                print(f"  样本 {i+1} (ID: {item_id}): PKL={pkl_target:.6f}, CSV={csv_target:.6f}, diff={diff:.6f}")
                mismatches += 1

    if mismatches == 0:
        print(f"✓ 前10个样本的目标值完全匹配")
    else:
        print(f"⚠ 发现 {mismatches} 个目标值不匹配")

    return mismatches == 0


def main():
    parser = argparse.ArgumentParser(description='从predictions CSV提取正确的测试集')
    parser.add_argument('--predictions_csv', type=str, required=True,
                       help='训练生成的predictions CSV文件路径')
    parser.add_argument('--preprocessed_dir', type=str, required=True,
                       help='预处理数据目录')
    parser.add_argument('--dataset', type=str, default='jarvis',
                       help='数据集名称')
    parser.add_argument('--property', type=str, default='mbj_bandgap',
                       help='属性名称')
    parser.add_argument('--output_dir', type=str, default='./corrected_test_set',
                       help='输出目录')

    args = parser.parse_args()

    print("="*80)
    print("从Predictions CSV提取正确的测试集")
    print("="*80)
    print(f"\nPredictions CSV: {args.predictions_csv}")
    print(f"预处理数据目录: {args.preprocessed_dir}")
    print(f"数据集: {args.dataset}")
    print(f"属性: {args.property}")
    print()

    # 步骤1: 从CSV加载测试集ID
    print("="*80)
    print("步骤1: 从CSV加载测试集ID")
    print("="*80)
    print()

    test_ids = load_test_ids_from_csv(args.predictions_csv)

    if not test_ids:
        print("\n✗ 无法从CSV提取测试集ID")
        return

    # 步骤2: 加载预处理数据
    print()
    print("="*80)
    print("步骤2: 加载预处理数据")
    print("="*80)
    print()

    all_data = load_preprocessed_data(
        args.preprocessed_dir,
        args.dataset,
        args.property
    )

    if not all_data:
        print("\n✗ 无法加载预处理数据")
        return

    # 步骤3: 筛选测试集
    print()
    print("="*80)
    print("步骤3: 筛选测试集")
    print("="*80)

    filtered_test_set, missing_ids = filter_test_set(all_data, test_ids)

    if not filtered_test_set:
        print("\n✗ 无法筛选出测试集")
        return

    # 步骤4: 验证目标值
    print()
    print("="*80)
    print("步骤4: 验证目标值")
    print("="*80)

    targets_match = verify_targets(filtered_test_set, args.predictions_csv)

    # 步骤5: 保存测试集
    print()
    print("="*80)
    print("步骤5: 保存测试集")
    print("="*80)
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存测试集pickle
    output_file = output_dir / 'test.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(filtered_test_set, f)
    print(f"✓ 测试集已保存: {output_file}")
    print(f"  样本数: {len(filtered_test_set)}")

    # 保存测试集ID
    ids_file = output_dir / 'test_ids.json'
    with open(ids_file, 'w') as f:
        json.dump([str(item['id']) for item in filtered_test_set], f, indent=2)
    print(f"✓ 测试集ID已保存: {ids_file}")

    # 保存提取信息
    info_file = output_dir / 'extraction_info.json'
    info = {
        'source_csv': args.predictions_csv,
        'csv_samples': len(test_ids),
        'matched_samples': len(filtered_test_set),
        'missing_samples': len(missing_ids),
        'match_rate': len(filtered_test_set) / len(test_ids) * 100,
        'targets_verified': targets_match,
        'missing_ids': missing_ids[:20] if missing_ids else []
    }
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"✓ 提取信息已保存: {info_file}")

    # 总结
    print()
    print("="*80)
    print("总结")
    print("="*80)
    print()
    print(f"CSV中的测试样本数: {len(test_ids)}")
    print(f"成功匹配: {len(filtered_test_set)} ({len(filtered_test_set)/len(test_ids)*100:.1f}%)")

    if missing_ids:
        print(f"缺失: {len(missing_ids)} ({len(missing_ids)/len(test_ids)*100:.1f}%)")

    print(f"目标值验证: {'✓ 通过' if targets_match else '⚠ 存在不匹配'}")

    print()
    print("="*80)
    print("下一步")
    print("="*80)
    print()
    print("使用提取的测试集运行评估:")
    print()
    print("python evaluate_text_masking.py \\")
    print(f"    --checkpoint /path/to/best_test_model.pt \\")
    print(f"    --test_data {output_file} \\")
    print(f"    --preprocessed_dir {args.preprocessed_dir} \\")
    print(f"    --dataset {args.dataset} \\")
    print(f"    --property {args.property} \\")
    print("    --masking_strategy random_token \\")
    print("    --masking_ratios 0.0 0.5 1.0 \\")
    print("    --output_dir ./test_output")
    print()


if __name__ == "__main__":
    main()
