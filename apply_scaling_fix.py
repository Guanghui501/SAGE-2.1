"""应用缩放修复并保存模型

这个脚本会加载模型，应用缩放修复，然后保存。

使用方法:
python apply_scaling_fix.py \
    --input_checkpoint best_test_model.pt \
    --output_checkpoint best_test_model_scaled_12.0.pt \
    --scale_factor 12.0
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))
from models.alignn import ALIGNN, MiddleFusionModule


class ScaledTextTransform(torch.nn.Module):
    """Wrapper that scales text_transform output"""

    def __init__(self, original_transform, scale_factor):
        super().__init__()
        self.original_transform = original_transform
        self.scale_factor = scale_factor

    def forward(self, x):
        output = self.original_transform(x)
        return output * self.scale_factor

    def load_state_dict(self, *args, **kwargs):
        return self.original_transform.load_state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.original_transform.state_dict(*args, **kwargs)


def apply_scaling_fix(model, scale_factor=12.0):
    """应用缩放修复到模型的所有 MiddleFusionModule"""

    fixed_count = 0

    for name, module in model.named_modules():
        if isinstance(module, MiddleFusionModule):
            print(f"🔧 对 {name} 应用缩放 (factor={scale_factor})")

            # 包装 text_transform
            original_transform = module.text_transform
            module.text_transform = ScaledTextTransform(original_transform, scale_factor)

            fixed_count += 1

    print(f"✅ 成功应用缩放到 {fixed_count} 个融合模块")
    return model


def main():
    parser = argparse.ArgumentParser(description="应用缩放修复并保存模型")
    parser.add_argument('--input_checkpoint', required=True, help='输入模型路径')
    parser.add_argument('--output_checkpoint', required=True, help='输出模型路径')
    parser.add_argument('--scale_factor', type=float, default=12.0, help='缩放因子')
    args = parser.parse_args()

    print("=" * 80)
    print("🔧 应用缩放修复")
    print("=" * 80)

    # 加载模型
    print(f"\n📂 加载模型: {args.input_checkpoint}")
    ckpt = torch.load(args.input_checkpoint, map_location='cpu', weights_only=False)
    config = ckpt['config']

    model = ALIGNN(config)
    model.load_state_dict(ckpt['model'])

    # 应用修复
    print(f"\n🔧 应用缩放因子: {args.scale_factor}")
    model = apply_scaling_fix(model, scale_factor=args.scale_factor)

    # 保存
    print(f"\n💾 保存到: {args.output_checkpoint}")
    torch.save({
        'config': config,
        'model': model.state_dict()
    }, args.output_checkpoint)

    print("\n" + "=" * 80)
    print("✅ 完成")
    print("=" * 80)

    print(f"\n现在可以使用修复后的模型进行分析：")
    print(f"\n1. 重新运行融合诊断：")
    print(f"   python diagnose_fusion_effectiveness.py \\")
    print(f"       --checkpoint {args.output_checkpoint} \\")
    print(f"       --root_dir <your-root-dir>")

    print(f"\n2. 重新运行文本流分析：")
    print(f"   python 3.analyze_text_flow_v2.py \\")
    print(f"       --checkpoint {args.output_checkpoint} \\")
    print(f"       --root_dir <your-root-dir>")

    print(f"\n3. 提取 Alpha 值并生成图表：")
    print(f"   python 1.extract_alpha_final.py \\")
    print(f"       --checkpoint {args.output_checkpoint} \\")
    print(f"       --root_dir <your-root-dir> \\")
    print(f"       --dataset jarvis \\")
    print(f"       --property hse_bandgap-2 \\")
    print(f"       --n_samples 500")
    print(f"\n   python 2.create_paper_alpha_figures.py")


if __name__ == '__main__':
    main()
