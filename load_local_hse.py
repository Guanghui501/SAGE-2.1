"""
从本地文件加载HSE带隙数据

使用方法：
    python train_with_cross_modal_attention.py \
        --dataset hse_bandgap \
        --local_data_path /public/home/ghzhang/crysmmnet-main/dataset/jarvis/hse_bandgap \
        --target hse_bandgap

作者：Claude
日期：2025-12-10
"""

import json
import pickle
import os
from pathlib import Path


def load_local_jarvis_data(data_path):
    """从本地路径加载JARVIS数据

    支持的格式：
    1. JSON文件 (.json)
    2. Pickle文件 (.pkl, .pickle)
    3. 目录（自动检测文件格式）

    Args:
        data_path: 本地数据路径（文件或目录）

    Returns:
        data: JARVIS格式的数据列表
    """
    data_path = Path(data_path)

    # 情况1: 目录（查找数据文件）
    if data_path.is_dir():
        print(f"📂 加载目录: {data_path}")

        # 查找JSON文件
        json_files = list(data_path.glob("*.json"))
        if json_files:
            print(f"   找到 {len(json_files)} 个JSON文件")
            return load_json_file(json_files[0])

        # 查找Pickle文件
        pkl_files = list(data_path.glob("*.pkl")) + list(data_path.glob("*.pickle"))
        if pkl_files:
            print(f"   找到 {len(pkl_files)} 个Pickle文件")
            return load_pickle_file(pkl_files[0])

        raise FileNotFoundError(f"在 {data_path} 中没有找到JSON或Pickle文件")

    # 情况2: 单个文件
    elif data_path.is_file():
        print(f"📄 加载文件: {data_path}")

        if data_path.suffix == '.json':
            return load_json_file(data_path)
        elif data_path.suffix in ['.pkl', '.pickle']:
            return load_pickle_file(data_path)
        else:
            raise ValueError(f"不支持的文件格式: {data_path.suffix}")

    else:
        raise FileNotFoundError(f"路径不存在: {data_path}")


def load_json_file(file_path):
    """加载JSON文件"""
    print(f"   加载JSON: {file_path.name}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"   ✅ 加载了 {len(data)} 个样本")

    # 验证数据格式
    validate_data_format(data)

    return data


def load_pickle_file(file_path):
    """加载Pickle文件"""
    print(f"   加载Pickle: {file_path.name}")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print(f"   ✅ 加载了 {len(data)} 个样本")

    # 验证数据格式
    validate_data_format(data)

    return data


def validate_data_format(data):
    """验证数据格式是否符合JARVIS标准

    必需字段：
    - atoms: 晶体结构
    - jid: 样本ID
    - 至少一个目标属性（如hse_bandgap）
    """
    if not isinstance(data, list):
        raise ValueError("数据应该是列表格式")

    if len(data) == 0:
        raise ValueError("数据为空")

    # 检查第一个样本
    sample = data[0]

    # 必需字段
    required_fields = ['atoms', 'jid']
    missing_fields = [f for f in required_fields if f not in sample]

    if missing_fields:
        raise ValueError(f"缺少必需字段: {missing_fields}")

    # 检查目标属性
    common_targets = [
        'hse_bandgap',
        'formation_energy_peratom',
        'optb88vdw_bandgap',
        'mbj_bandgap'
    ]

    available_targets = [t for t in common_targets if t in sample]

    if available_targets:
        print(f"   可用目标: {', '.join(available_targets)}")
    else:
        print(f"   ⚠️  警告：未找到常见目标属性")

    print(f"   样本字段: {list(sample.keys())[:10]}{'...' if len(sample.keys()) > 10 else ''}")


def load_hse_bandgap_data(base_path="/public/home/ghzhang/crysmmnet-main/dataset/jarvis"):
    """便捷函数：加载HSE带隙数据

    Args:
        base_path: JARVIS数据集基础路径

    Returns:
        data: HSE带隙数据
    """
    hse_path = Path(base_path) / "hse_bandgap"

    if not hse_path.exists():
        raise FileNotFoundError(
            f"HSE带隙数据路径不存在: {hse_path}\n"
            f"请检查路径是否正确"
        )

    return load_local_jarvis_data(hse_path)


# ============================================================================
# 测试和使用示例
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='测试本地数据加载')
    parser.add_argument('--data_path', type=str,
                       default='/public/home/ghzhang/crysmmnet-main/dataset/jarvis/hse_bandgap',
                       help='本地数据路径')
    parser.add_argument('--show_samples', type=int, default=3,
                       help='显示前N个样本')

    args = parser.parse_args()

    print("="*80)
    print("测试本地JARVIS数据加载")
    print("="*80)

    try:
        # 加载数据
        data = load_local_jarvis_data(args.data_path)

        print(f"\n📊 数据统计:")
        print(f"   总样本数: {len(data)}")

        # 统计HSE带隙
        hse_gaps = [d['hse_bandgap'] for d in data
                    if 'hse_bandgap' in d and d['hse_bandgap'] is not None]

        if hse_gaps:
            import numpy as np
            print(f"\n🎯 HSE带隙统计:")
            print(f"   有效样本: {len(hse_gaps)}")
            print(f"   范围: [{np.min(hse_gaps):.3f}, {np.max(hse_gaps):.3f}] eV")
            print(f"   均值: {np.mean(hse_gaps):.3f} eV")
            print(f"   中位数: {np.median(hse_gaps):.3f} eV")

            # 统计材料类型
            metals = sum(1 for gap in hse_gaps if gap < 0.01)
            semiconductors = sum(1 for gap in hse_gaps if 0.01 <= gap <= 3.0)
            insulators = sum(1 for gap in hse_gaps if gap > 3.0)

            print(f"\n   材料分布:")
            print(f"     金属 (gap < 0.01): {metals} ({metals/len(hse_gaps)*100:.1f}%)")
            print(f"     半导体 (0.01-3.0): {semiconductors} ({semiconductors/len(hse_gaps)*100:.1f}%)")
            print(f"     绝缘体 (> 3.0): {insulators} ({insulators/len(hse_gaps)*100:.1f}%)")

        # 显示示例
        print(f"\n📋 示例数据（前{args.show_samples}个）:")
        for i, sample in enumerate(data[:args.show_samples]):
            print(f"\n样本 {i+1}:")
            print(f"   JID: {sample.get('jid', 'N/A')}")
            print(f"   HSE带隙: {sample.get('hse_bandgap', 'N/A')} eV")
            if 'atoms' in sample:
                from jarvis.core.atoms import Atoms
                atoms = Atoms.from_dict(sample['atoms'])
                print(f"   化学式: {atoms.composition.reduced_formula}")
                print(f"   原子数: {atoms.num_atoms}")

        print("\n" + "="*80)
        print("✅ 数据加载成功！")
        print("="*80)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
