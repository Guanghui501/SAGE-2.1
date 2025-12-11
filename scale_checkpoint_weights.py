"""直接修改 checkpoint 中的 text_transform 权重

这个脚本会直接缩放 text_transform 的权重，使其输出放大 N 倍。

原理：
如果 y = W*x + b，那么要让 y' = scale * y，只需：
W' = scale * W
b' = scale * b

使用方法:
python scale_checkpoint_weights.py \
    --input_checkpoint best_test_model.pt \
    --output_checkpoint best_test_model_scaled_12.0.pt \
    --scale_factor 12.0
"""

import argparse
import torch
import copy

def scale_text_transform_weights(checkpoint, scale_factor):
    """直接缩放 text_transform 的权重"""

    state_dict = checkpoint['model']
    modified_keys = []

    # 查找所有 text_transform 的权重
    for key in state_dict.keys():
        # 匹配 middle_fusion_modules.layer_X.text_transform.0.weight 等
        if 'text_transform' in key and ('weight' in key or 'bias' in key):
            # 只缩放最后一层的权重（输出层）
            # text_transform 结构：0.weight, 0.bias (第一层), 3.weight, 3.bias (第二层)
            if '.3.' in key:  # 第二层 Linear（输出层）
                original_value = state_dict[key].clone()
                state_dict[key] = original_value * scale_factor
                modified_keys.append(key)
                print(f"✅ 缩放 {key}: shape={state_dict[key].shape}, scale={scale_factor}")

    if not modified_keys:
        print("⚠️  警告: 未找到 text_transform 权重！")
        return checkpoint, False

    checkpoint['model'] = state_dict
    return checkpoint, True


def main():
    parser = argparse.ArgumentParser(description="直接修改 checkpoint 权重以缩放 text_transform 输出")
    parser.add_argument('--input_checkpoint', required=True, help='输入模型路径')
    parser.add_argument('--output_checkpoint', required=True, help='输出模型路径')
    parser.add_argument('--scale_factor', type=float, default=12.0, help='缩放因子')
    args = parser.parse_args()

    print("=" * 80)
    print("🔧 直接修改 Checkpoint 权重")
    print("=" * 80)

    # 加载 checkpoint
    print(f"\n📂 加载 checkpoint: {args.input_checkpoint}")
    checkpoint = torch.load(args.input_checkpoint, map_location='cpu', weights_only=False)

    print(f"   包含的键: {list(checkpoint.keys())}")

    # 复制一份，避免修改原始数据
    checkpoint_scaled = copy.deepcopy(checkpoint)

    # 应用缩放
    print(f"\n🔧 应用缩放因子: {args.scale_factor}")
    checkpoint_scaled, success = scale_text_transform_weights(checkpoint_scaled, args.scale_factor)

    if not success:
        print("\n❌ 缩放失败！请检查 checkpoint 结构。")
        return

    # 保存
    print(f"\n💾 保存到: {args.output_checkpoint}")
    torch.save(checkpoint_scaled, args.output_checkpoint)

    print("\n" + "=" * 80)
    print("✅ 完成")
    print("=" * 80)

    # 验证
    print("\n🔍 验证缩放效果...")
    ckpt_orig = torch.load(args.input_checkpoint, map_location='cpu', weights_only=False)
    ckpt_scaled = torch.load(args.output_checkpoint, map_location='cpu', weights_only=False)

    for key in ckpt_orig['model'].keys():
        if 'text_transform.3' in key and 'weight' in key:
            orig_norm = ckpt_orig['model'][key].norm().item()
            scaled_norm = ckpt_scaled['model'][key].norm().item()
            ratio = scaled_norm / orig_norm
            print(f"   {key}:")
            print(f"      原始范数: {orig_norm:.4f}")
            print(f"      缩放范数: {scaled_norm:.4f}")
            print(f"      比例: {ratio:.2f} (预期: {args.scale_factor:.2f})")

            if abs(ratio - args.scale_factor) < 0.01:
                print(f"      ✅ 验证通过")
            else:
                print(f"      ⚠️  比例不匹配！")

    print(f"\n现在可以使用缩放后的模型进行分析：")
    print(f"\npython diagnose_fusion_effectiveness.py \\")
    print(f"    --checkpoint {args.output_checkpoint} \\")
    print(f"    --root_dir <your-root-dir>")
    print(f"\npython 3.analyze_text_flow_v2.py \\")
    print(f"    --checkpoint {args.output_checkpoint} \\")
    print(f"    --root_dir <your-root-dir>")


if __name__ == '__main__':
    main()
