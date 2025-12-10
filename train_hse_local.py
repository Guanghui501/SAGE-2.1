"""
使用本地HSE带隙数据训练

直接使用本地数据文件，无需从JARVIS在线数据库下载

使用方法：
    python train_hse_local.py \
        --local_data_path /public/home/ghzhang/crysmmnet-main/dataset/jarvis/hse_bandgap \
        --target hse_bandgap \
        --epochs 400 \
        --output_dir runs/hse_local

作者：Claude
日期：2025-12-10
"""

import os
import sys
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from load_local_hse import load_local_jarvis_data
from data import get_train_val_loaders
from train import train_dgl
from config import TrainingConfig
from models.alignn import ALIGNNConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='使用本地HSE数据训练模型'
    )

    # 数据参数
    parser.add_argument('--local_data_path', type=str, required=True,
                       help='本地数据路径')
    parser.add_argument('--target', type=str, default='hse_bandgap',
                       help='目标属性')
    parser.add_argument('--atom_features', type=str, default='cgcnn',
                       help='原子特征类型')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=400,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                       help='学习率预热步数')

    # 数据划分
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='测试集比例')

    # 模型参数
    parser.add_argument('--use_cross_modal', action='store_true',
                       help='使用跨模态注意力')
    parser.add_argument('--use_fine_grained', action='store_true',
                       help='使用细粒度注意力')
    parser.add_argument('--use_middle_fusion', action='store_true',
                       help='使用中期融合')

    # 输出
    parser.add_argument('--output_dir', type=str, default='runs/hse_local',
                       help='输出目录')
    parser.add_argument('--log_tensorboard', action='store_true',
                       help='启用TensorBoard日志')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("="*80)
    print("使用本地HSE带隙数据训练")
    print("="*80)

    # 步骤1: 加载本地数据
    print(f"\n📂 加载本地数据: {args.local_data_path}")
    try:
        dataset_array = load_local_jarvis_data(args.local_data_path)
    except Exception as e:
        print(f"\n❌ 数据加载失败: {e}")
        sys.exit(1)

    # 步骤2: 创建配置
    print(f"\n🔧 创建训练配置...")

    model_config = ALIGNNConfig(
        name="alignn",
        alignn_layers=4,
        gcn_layers=4,
        hidden_features=256,
        output_features=1,

        # 跨模态设置
        use_cross_modal_attention=args.use_cross_modal,
        cross_modal_attention_type="bidirectional" if args.use_cross_modal else None,
        cross_modal_num_heads=4,
        cross_modal_hidden_dim=256,

        # 细粒度注意力
        use_fine_grained_attention=args.use_fine_grained,
        fine_grained_num_heads=8,

        # 中期融合
        use_middle_fusion=args.use_middle_fusion,
    )

    config = TrainingConfig(
        # 数据集设置（使用虚拟名称，因为我们直接提供数据）
        dataset="hse_bandgap_local",
        target=args.target,
        atom_features=args.atom_features,

        # 训练参数
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        optimizer="adamw",
        scheduler="onecycle",

        # 数据划分
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,

        # 输出
        output_dir=args.output_dir,
        write_checkpoint=True,
        store_outputs=True,
        log_tensorboard=args.log_tensorboard,

        # 模型
        model=model_config
    )

    print(f"   目标属性: {args.target}")
    print(f"   样本数: {len(dataset_array)}")
    print(f"   训练轮数: {args.epochs}")
    print(f"   批次大小: {args.batch_size}")

    # 步骤3: 创建数据加载器
    print(f"\n🔨 创建数据加载器...")

    train_loader, val_loader, test_loader, prepare_batch = get_train_val_loaders(
        dataset="hse_bandgap_local",  # 虚拟名称
        dataset_array=dataset_array,  # ⭐ 关键：直接提供数据
        target=args.target,
        atom_features=args.atom_features,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        workers=4,
        pin_memory=True,
        line_graph=True,
        cutoff=8.0,
        max_neighbors=12,
        use_canonize=True,
    )

    print(f"   ✅ 训练集: {len(train_loader.dataset)} 样本")
    print(f"   ✅ 验证集: {len(val_loader.dataset)} 样本")
    print(f"   ✅ 测试集: {len(test_loader.dataset)} 样本")

    # 步骤4: 开始训练
    print(f"\n🚀 开始训练...")
    print(f"   输出目录: {args.output_dir}")

    history = train_dgl(
        config=config,
        model=None,  # 将自动创建
        train_val_test_loaders=[train_loader, val_loader, test_loader],
        prepare_batch=prepare_batch,
        output_dir=args.output_dir
    )

    # 步骤5: 显示结果
    print("\n" + "="*80)
    print("✅ 训练完成！")
    print("="*80)

    if history:
        best_val_mae = min(history.get('val_mae', [float('inf')]))
        print(f"\n📊 最佳验证MAE: {best_val_mae:.4f} eV")

    print(f"\n📁 结果保存在: {args.output_dir}")
    print("   - best_model.pt: 最佳模型")
    print("   - train_log.txt: 训练日志")

    if args.log_tensorboard:
        print(f"   - logs/: TensorBoard日志")
        print(f"\n查看TensorBoard:")
        print(f"   tensorboard --logdir {args.output_dir}/logs")


if __name__ == '__main__':
    main()
