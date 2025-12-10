"""改进的晶体结构文件加载 - 支持CIF和POSCAR格式

解决CIF文件跳过问题的方案：
1. 自动检测并支持多种格式（CIF, POSCAR, VASP）
2. 详细的错误日志
3. 格式转换建议
"""

import os
from pathlib import Path
from jarvis.core.atoms import Atoms


def load_structure_file(file_path, fallback_formats=None):
    """智能加载晶体结构文件，支持多种格式

    Args:
        file_path: 文件路径（可以不带扩展名）
        fallback_formats: 备选格式列表，默认 ['cif', 'poscar', 'vasp']

    Returns:
        Atoms: JARVIS Atoms对象
        str: 实际使用的文件路径

    Raises:
        FileNotFoundError: 所有格式都找不到文件
        ValueError: 所有格式都无法解析
    """
    if fallback_formats is None:
        fallback_formats = ['cif', 'poscar', 'vasp', 'POSCAR']

    file_path = Path(file_path)
    base_path = file_path.parent / file_path.stem  # 去掉扩展名

    # 尝试的文件路径和加载方法
    attempts = []

    # 1. 如果指定了完整路径，先尝试它
    if file_path.suffix:
        ext = file_path.suffix.lower().lstrip('.')
        if ext == 'cif':
            attempts.append((file_path, 'cif', Atoms.from_cif))
        elif ext in ['poscar', 'vasp']:
            attempts.append((file_path, 'poscar', Atoms.from_poscar))

    # 2. 尝试所有备选格式
    for fmt in fallback_formats:
        if fmt.lower() == 'cif':
            test_path = Path(str(base_path) + '.cif')
            attempts.append((test_path, 'cif', Atoms.from_cif))
        elif fmt.lower() in ['poscar', 'vasp']:
            # POSCAR文件可能没有扩展名
            test_paths = [
                Path(str(base_path) + '.poscar'),
                Path(str(base_path) + '.vasp'),
                Path(str(base_path) + '.POSCAR'),
                base_path / 'POSCAR',  # 目录下的POSCAR文件
            ]
            for test_path in test_paths:
                attempts.append((test_path, 'poscar', Atoms.from_poscar))

    # 尝试加载
    errors = []
    for test_path, fmt_name, loader_func in attempts:
        if not test_path.exists():
            continue

        try:
            atoms = loader_func(str(test_path))
            return atoms, str(test_path)
        except Exception as e:
            errors.append({
                'path': str(test_path),
                'format': fmt_name,
                'error': str(e)
            })

    # 所有尝试都失败
    if not errors:
        raise FileNotFoundError(
            f"找不到结构文件: {file_path}\n"
            f"尝试了以下扩展名: {fallback_formats}"
        )
    else:
        error_msg = f"无法加载结构文件 {file_path}，尝试了以下格式:\n"
        for err in errors:
            error_msg += f"  - {err['format']}: {err['path']}\n    错误: {err['error']}\n"
        raise ValueError(error_msg)


def convert_cif_to_poscar(cif_file, poscar_file=None):
    """将CIF文件转换为POSCAR格式

    Args:
        cif_file: CIF文件路径
        poscar_file: 输出POSCAR文件路径（如果为None，则使用同名的.poscar文件）

    Returns:
        str: POSCAR文件路径
    """
    atoms = Atoms.from_cif(cif_file)

    if poscar_file is None:
        poscar_file = Path(cif_file).with_suffix('.poscar')

    atoms.write_poscar(str(poscar_file))
    return str(poscar_file)


def batch_convert_structures(input_dir, output_dir=None, input_format='cif', output_format='poscar'):
    """批量转换结构文件格式

    Args:
        input_dir: 输入目录
        output_dir: 输出目录（如果为None，使用输入目录）
        input_format: 输入格式 ('cif', 'poscar')
        output_format: 输出格式 ('cif', 'poscar')

    Returns:
        dict: 转换统计信息
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'success': 0,
        'failed': 0,
        'errors': []
    }

    # 查找输入文件
    if input_format.lower() == 'cif':
        files = list(input_dir.glob('*.cif'))
        loader = Atoms.from_cif
        writer = lambda atoms, path: atoms.write_poscar(str(path))
        out_ext = '.poscar'
    elif input_format.lower() == 'poscar':
        files = list(input_dir.glob('*.poscar')) + list(input_dir.glob('*.vasp'))
        loader = Atoms.from_poscar
        writer = lambda atoms, path: atoms.write_cif(str(path))
        out_ext = '.cif'
    else:
        raise ValueError(f"不支持的输入格式: {input_format}")

    print(f"找到 {len(files)} 个 {input_format.upper()} 文件")

    for i, file in enumerate(files):
        try:
            # 加载
            atoms = loader(str(file))

            # 保存
            output_file = output_dir / (file.stem + out_ext)
            writer(atoms, output_file)

            stats['success'] += 1

            if (i + 1) % 100 == 0:
                print(f"进度: {i+1}/{len(files)}")

        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append({
                'file': str(file),
                'error': str(e)
            })

    return stats


# ==================== 改进的train_with_cross_modal_attention.py数据加载 ====================

def load_external_dataset_robust(dataset, cif_dir, property_name):
    """改进版本的外部数据集加载 - 支持多种结构文件格式

    相比原版的改进：
    1. 支持CIF和POSCAR格式自动切换
    2. 详细的错误日志
    3. 统计不同类型的跳过原因
    """
    import pandas as pd
    import numpy as np

    # 读取CSV
    data = pd.read_csv(f'data/{dataset}.csv').values

    dataset_array = []
    skip_stats = {
        'file_not_found': 0,
        'parse_error': 0,
        'invalid_data': 0,
        'other': 0
    }
    error_examples = []

    for j in range(len(data)):
        try:
            # 解析CSV行
            if dataset.lower() == 'mp':
                id, composition, target, crys_desc_full = data[j]
            elif dataset.lower() == 'class':
                id, target, crys_desc_full = data[j]
                composition = ''
            elif dataset.lower() == 'toy':
                id, composition, target, crys_desc_full, _ = data[j]

            # 智能加载结构文件
            base_path = os.path.join(cif_dir, str(id))
            try:
                atoms, actual_path = load_structure_file(base_path)
            except FileNotFoundError as e:
                skip_stats['file_not_found'] += 1
                if skip_stats['file_not_found'] <= 3:
                    error_examples.append(f"文件不存在: {id}")
                continue
            except ValueError as e:
                skip_stats['parse_error'] += 1
                if skip_stats['parse_error'] <= 3:
                    error_examples.append(f"解析失败: {id} - {str(e)[:100]}")
                continue

            # 构建样本
            info = {
                "atoms": atoms.to_dict(),
                "jid": id,
                "text": crys_desc_full,
                "target": float(target)
            }

            # MP数据集的特殊处理
            if dataset.lower() == 'mp' and property_name in ['shear', 'bulk', 'bulk_modulus', 'shear_modulus']:
                info["target"] = np.log10(float(target))

            dataset_array.append(info)

        except Exception as e:
            skip_stats['other'] += 1
            if skip_stats['other'] <= 3:
                error_examples.append(f"其他错误: {id} - {str(e)}")

    # 打印统计
    total_skipped = sum(skip_stats.values())
    print(f"\n{'='*80}")
    print(f"数据加载统计 - {dataset}")
    print(f"{'='*80}")
    print(f"总样本数:   {len(data)}")
    print(f"成功加载:   {len(dataset_array)} ({len(dataset_array)/len(data)*100:.1f}%)")
    print(f"跳过样本:   {total_skipped} ({total_skipped/len(data)*100:.1f}%)")
    print()
    print("跳过原因:")
    print(f"  - 文件不存在:  {skip_stats['file_not_found']}")
    print(f"  - 解析错误:    {skip_stats['parse_error']}")
    print(f"  - 数据无效:    {skip_stats['invalid_data']}")
    print(f"  - 其他错误:    {skip_stats['other']}")

    if error_examples:
        print()
        print("错误示例:")
        for err in error_examples[:5]:
            print(f"  {err}")
    print(f"{'='*80}\n")

    return dataset_array


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='晶体结构文件格式转换和测试')
    parser.add_argument('--mode', type=str, default='test',
                       choices=['test', 'convert', 'batch_convert'],
                       help='运行模式')
    parser.add_argument('--input', type=str, help='输入文件或目录')
    parser.add_argument('--output', type=str, help='输出文件或目录')
    parser.add_argument('--input_format', type=str, default='cif',
                       choices=['cif', 'poscar'],
                       help='输入格式')
    parser.add_argument('--output_format', type=str, default='poscar',
                       choices=['cif', 'poscar'],
                       help='输出格式')

    args = parser.parse_args()

    if args.mode == 'test':
        # 测试单个文件加载
        if args.input:
            print(f"测试加载: {args.input}")
            try:
                atoms, actual_path = load_structure_file(args.input)
                print(f"✅ 成功加载!")
                print(f"   实际路径: {actual_path}")
                print(f"   原子数: {len(atoms)}")
                print(f"   化学式: {atoms.composition.reduced_formula}")
            except Exception as e:
                print(f"❌ 加载失败: {e}")
        else:
            print("请使用 --input 指定文件路径")

    elif args.mode == 'convert':
        # 单文件转换
        if not args.input or not args.output:
            print("请指定 --input 和 --output")
        else:
            print(f"转换: {args.input} -> {args.output}")
            if args.input_format == 'cif':
                convert_cif_to_poscar(args.input, args.output)
            print("✅ 转换完成")

    elif args.mode == 'batch_convert':
        # 批量转换
        if not args.input:
            print("请使用 --input 指定输入目录")
        else:
            print(f"批量转换: {args.input}")
            stats = batch_convert_structures(
                args.input,
                args.output,
                args.input_format,
                args.output_format
            )
            print(f"\n{'='*80}")
            print("转换统计:")
            print(f"  成功: {stats['success']}")
            print(f"  失败: {stats['failed']}")
            if stats['errors']:
                print("\n失败示例:")
                for err in stats['errors'][:5]:
                    print(f"  {err['file']}: {err['error']}")
            print(f"{'='*80}")
