"""RoboCrystallographer 文本分层工具

将 RoboCrystallographer 生成的描述分为全局、半全局、局部三个层次。
支持自动识别和手动规则两种方式。

使用方法:
python split_robocrys_text.py \
    --input description.csv \
    --output description_hierarchical.csv \
    --mode auto  # 或 manual
"""

import re
import argparse
import pandas as pd
from typing import Tuple, Dict


class RoboCrysTextSplitter:
    """RoboCrystallographer 文本分层器"""

    def __init__(self):
        # 定义全局信息的关键词
        self.global_keywords = [
            'space group', 'crystal system', 'symmetry',
            'lattice', 'unit cell', 'structure type',
            'prototype', 'Pearson symbol', 'Wyckoff',
            'cubic', 'tetragonal', 'orthorhombic', 'hexagonal',
            'monoclinic', 'triclinic', 'trigonal'
        ]

        # 定义半全局信息的关键词
        self.semi_global_keywords = [
            'consists of', 'contains', 'composition',
            'coordination number', 'average', 'overall',
            'primarily', 'mainly', 'forms', 'bonded to',
            'species', 'oxidation state'
        ]

        # 定义局部信息的关键词
        self.local_keywords = [
            'site', 'atom', 'position', 'coordinates',
            'each', 'every', 'individual', 'specific',
            'bond length', 'bond angle', 'distance',
            'nearest neighbor'
        ]

    def split_by_sentences(self, text: str) -> Tuple[str, str, str]:
        """基于句子内容分离文本

        Args:
            text: RoboCrystallographer 生成的完整描述

        Returns:
            global_text: 全局信息
            semi_global_text: 半全局信息
            local_text: 局部信息
        """
        # 分割句子
        sentences = re.split(r'[.!?]\s+', text)

        global_sents = []
        semi_global_sents = []
        local_sents = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            sent_lower = sent.lower()

            # 判断句子类型
            is_global = any(kw in sent_lower for kw in self.global_keywords)
            is_semi_global = any(kw in sent_lower for kw in self.semi_global_keywords)
            is_local = any(kw in sent_lower for kw in self.local_keywords)

            # 优先级：全局 > 半全局 > 局部
            if is_global:
                global_sents.append(sent)
            elif is_semi_global:
                semi_global_sents.append(sent)
            elif is_local:
                local_sents.append(sent)
            else:
                # 默认归为半全局
                semi_global_sents.append(sent)

        global_text = '. '.join(global_sents) + '.' if global_sents else ''
        semi_global_text = '. '.join(semi_global_sents) + '.' if semi_global_sents else ''
        local_text = '. '.join(local_sents) + '.' if local_sents else ''

        return global_text, semi_global_text, local_text

    def extract_global_only(self, text: str) -> str:
        """仅提取全局信息（推荐用于提升性能）

        Args:
            text: 完整描述

        Returns:
            global_text: 仅包含全局信息的文本
        """
        global_text, _, _ = self.split_by_sentences(text)
        return global_text

    def extract_global_and_semi_global(self, text: str) -> str:
        """提取全局+半全局信息（推荐配置）

        Args:
            text: 完整描述

        Returns:
            combined_text: 全局+半全局信息
        """
        global_text, semi_global_text, _ = self.split_by_sentences(text)

        # 合并并确保全局信息在前
        combined = []
        if global_text:
            combined.append(global_text)
        if semi_global_text:
            combined.append(semi_global_text)

        return ' '.join(combined)

    def enhance_global_info(self, text: str, boost_repetition: int = 2) -> str:
        """增强全局信息的重要性（通过重复）

        Args:
            text: 原始文本
            boost_repetition: 全局信息重复次数

        Returns:
            enhanced_text: 增强后的文本
        """
        global_text, semi_global_text, local_text = self.split_by_sentences(text)

        # 将全局信息重复多次，放在开头和结尾
        enhanced_parts = []

        # 开头：全局信息（重复）
        for _ in range(boost_repetition):
            if global_text:
                enhanced_parts.append(global_text)

        # 中间：半全局信息
        if semi_global_text:
            enhanced_parts.append(semi_global_text)

        # 结尾：全局信息（再次强调）
        if global_text:
            enhanced_parts.append(global_text)

        return ' '.join(enhanced_parts)

    def analyze_text_composition(self, text: str) -> Dict[str, float]:
        """分析文本的组成比例

        Args:
            text: 完整描述

        Returns:
            composition: {'global': 0.3, 'semi_global': 0.5, 'local': 0.2}
        """
        global_text, semi_global_text, local_text = self.split_by_sentences(text)

        total_len = len(text)
        if total_len == 0:
            return {'global': 0.0, 'semi_global': 0.0, 'local': 0.0}

        return {
            'global': len(global_text) / total_len,
            'semi_global': len(semi_global_text) / total_len,
            'local': len(local_text) / total_len
        }


def process_dataset(
    input_file: str,
    output_file: str,
    text_column: str = 'description',
    mode: str = 'split'
):
    """处理整个数据集

    Args:
        input_file: 输入CSV文件
        output_file: 输出CSV文件
        text_column: 文本列名
        mode: 处理模式
            - 'split': 分离为三列
            - 'global_only': 仅保留全局信息
            - 'global_semi': 保留全局+半全局（推荐）
            - 'enhanced': 增强全局信息
    """
    print(f"\n{'='*80}")
    print(f"RoboCrystallographer 文本分层处理")
    print(f"{'='*80}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"处理模式: {mode}")
    print(f"{'='*80}\n")

    # 读取数据
    df = pd.read_csv(input_file)
    print(f"加载了 {len(df)} 条数据\n")

    # 初始化分离器
    splitter = RoboCrysTextSplitter()

    # 统计信息
    total_global_ratio = 0
    total_semi_global_ratio = 0
    total_local_ratio = 0

    if mode == 'split':
        # 分离为三列
        print("分离文本为三个层次...\n")

        global_texts = []
        semi_global_texts = []
        local_texts = []

        for idx, row in df.iterrows():
            text = row[text_column]
            global_text, semi_global_text, local_text = splitter.split_by_sentences(text)

            global_texts.append(global_text)
            semi_global_texts.append(semi_global_text)
            local_texts.append(local_text)

            # 统计
            comp = splitter.analyze_text_composition(text)
            total_global_ratio += comp['global']
            total_semi_global_ratio += comp['semi_global']
            total_local_ratio += comp['local']

            if (idx + 1) % 100 == 0:
                print(f"  处理进度: {idx + 1}/{len(df)}")

        df['global_description'] = global_texts
        df['semi_global_description'] = semi_global_texts
        df['local_description'] = local_texts

        # 打印统计
        print(f"\n文本组成统计:")
        print(f"  全局信息占比: {total_global_ratio / len(df) * 100:.1f}%")
        print(f"  半全局信息占比: {total_semi_global_ratio / len(df) * 100:.1f}%")
        print(f"  局部信息占比: {total_local_ratio / len(df) * 100:.1f}%")

    elif mode == 'global_only':
        # 仅保留全局信息
        print("提取全局信息...\n")

        new_texts = []
        for idx, row in df.iterrows():
            text = row[text_column]
            global_text = splitter.extract_global_only(text)
            new_texts.append(global_text)

            if (idx + 1) % 100 == 0:
                print(f"  处理进度: {idx + 1}/{len(df)}")

        df['description'] = new_texts

    elif mode == 'global_semi':
        # 保留全局+半全局（推荐）
        print("提取全局+半全局信息（推荐配置）...\n")

        new_texts = []
        for idx, row in df.iterrows():
            text = row[text_column]
            combined_text = splitter.extract_global_and_semi_global(text)
            new_texts.append(combined_text)

            if (idx + 1) % 100 == 0:
                print(f"  处理进度: {idx + 1}/{len(df)}")

        df['description'] = new_texts

    elif mode == 'enhanced':
        # 增强全局信息
        print("增强全局信息（通过重复）...\n")

        new_texts = []
        for idx, row in df.iterrows():
            text = row[text_column]
            enhanced_text = splitter.enhance_global_info(text, boost_repetition=2)
            new_texts.append(enhanced_text)

            if (idx + 1) % 100 == 0:
                print(f"  处理进度: {idx + 1}/{len(df)}")

        df['description'] = new_texts

    # 保存
    df.to_csv(output_file, index=False)
    print(f"\n✅ 处理完成！")
    print(f"   输出文件: {output_file}")
    print(f"{'='*80}\n")


def show_examples(input_file: str, text_column: str = 'description', n_samples: int = 3):
    """展示分层示例

    Args:
        input_file: 输入CSV文件
        text_column: 文本列名
        n_samples: 展示样本数
    """
    df = pd.read_csv(input_file)
    splitter = RoboCrysTextSplitter()

    print(f"\n{'='*80}")
    print(f"文本分层示例")
    print(f"{'='*80}\n")

    for idx in range(min(n_samples, len(df))):
        text = df.iloc[idx][text_column]

        print(f"样本 {idx + 1}:")
        print(f"-" * 80)

        global_text, semi_global_text, local_text = splitter.split_by_sentences(text)

        print(f"\n【原始文本】")
        print(f"{text[:300]}..." if len(text) > 300 else text)

        print(f"\n【全局信息】")
        print(f"{global_text[:200]}..." if len(global_text) > 200 else global_text)

        print(f"\n【半全局信息】")
        print(f"{semi_global_text[:200]}..." if len(semi_global_text) > 200 else semi_global_text)

        print(f"\n【局部信息】")
        print(f"{local_text[:200]}..." if len(local_text) > 200 else local_text)

        # 统计
        comp = splitter.analyze_text_composition(text)
        print(f"\n【组成比例】")
        print(f"  全局: {comp['global']*100:.1f}%")
        print(f"  半全局: {comp['semi_global']*100:.1f}%")
        print(f"  局部: {comp['local']*100:.1f}%")

        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="RoboCrystallographer 文本分层工具"
    )
    parser.add_argument('--input', required=True, help='输入CSV文件')
    parser.add_argument('--output', required=True, help='输出CSV文件')
    parser.add_argument('--text_column', default='description', help='文本列名')
    parser.add_argument('--mode', default='global_semi',
                        choices=['split', 'global_only', 'global_semi', 'enhanced'],
                        help='处理模式')
    parser.add_argument('--show_examples', action='store_true',
                        help='展示分层示例后退出')
    parser.add_argument('--n_examples', type=int, default=3,
                        help='展示的示例数量')

    args = parser.parse_args()

    if args.show_examples:
        show_examples(args.input, args.text_column, args.n_examples)
    else:
        process_dataset(args.input, args.output, args.text_column, args.mode)


if __name__ == '__main__':
    main()
