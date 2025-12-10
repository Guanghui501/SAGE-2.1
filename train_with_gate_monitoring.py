"""
带门控监控的训练脚本

扩展自 train_with_cross_modal_attention.py，添加：
1. 实时门控权重监控
2. Gate统计记录
3. 异常检测和警告
4. 可视化工具

使用方法：
    python train_with_gate_monitoring.py --config config.json

作者：Claude
日期：2025-12-10
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))


class GateMonitor:
    """门控权重监控器

    功能：
    1. 收集gate统计信息
    2. 检测异常（权重过低/过高）
    3. 记录到tensorboard
    4. 生成可视化报告
    """

    def __init__(self, log_dir='runs', check_interval=100,
                 warn_threshold_low=0.3, warn_threshold_high=0.9):
        """初始化监控器

        Args:
            log_dir: TensorBoard日志目录
            check_interval: 检查间隔（步数）
            warn_threshold_low: 低权重警告阈值
            warn_threshold_high: 高权重警告阈值
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.check_interval = check_interval
        self.warn_threshold_low = warn_threshold_low
        self.warn_threshold_high = warn_threshold_high

        # 统计历史
        self.history = {
            'steps': [],
            'quality_mean': [],
            'fusion_mean': [],
            'effective_mean': [],
            'gate_mean': [],  # 用于简化版
        }

        self.step_count = 0
        self.warnings = []

    def update(self, diagnostics, step=None):
        """更新统计信息

        Args:
            diagnostics: 诊断信息字典
            step: 当前步数（可选）
        """
        if step is None:
            step = self.step_count
            self.step_count += 1

        # 记录到tensorboard
        if 'quality_mean' in diagnostics:
            # GatedCrossAttention 或 BalancedGatedCrossAttention
            self.writer.add_scalar('gate/quality_mean',
                                  diagnostics['quality_mean'], step)
            self.writer.add_scalar('gate/fusion_mean',
                                  diagnostics['fusion_mean'], step)
            self.writer.add_scalar('gate/effective_mean',
                                  diagnostics['effective_mean'], step)

            # 记录到历史
            self.history['steps'].append(step)
            self.history['quality_mean'].append(diagnostics['quality_mean'])
            self.history['fusion_mean'].append(diagnostics['fusion_mean'])
            self.history['effective_mean'].append(diagnostics['effective_mean'])

            # 检查异常
            if step % self.check_interval == 0:
                self._check_anomalies(diagnostics, step, gate_type='gated')

        elif 'gate_mean' in diagnostics:
            # SimplifiedGatedCrossAttention
            self.writer.add_scalar('gate/gate_mean',
                                  diagnostics['gate_mean'], step)

            # 记录到历史
            self.history['steps'].append(step)
            self.history['gate_mean'].append(diagnostics['gate_mean'])

            # 检查异常
            if step % self.check_interval == 0:
                self._check_anomalies(diagnostics, step, gate_type='simplified')

        # 记录warmup因子（如果有）
        if 'warmup_factor' in diagnostics:
            self.writer.add_scalar('gate/warmup_factor',
                                  diagnostics['warmup_factor'], step)

    def _check_anomalies(self, diagnostics, step, gate_type='gated'):
        """检查异常情况

        Args:
            diagnostics: 诊断信息
            step: 当前步数
            gate_type: 门控类型 ('gated' 或 'simplified')
        """
        warnings = []

        if gate_type == 'gated':
            # 检查 effective_weight
            effective_mean = diagnostics.get('effective_mean', 0)

            if effective_mean < self.warn_threshold_low:
                msg = (f"⚠️  [Step {step}] Effective weight过低: "
                      f"{effective_mean:.3f} < {self.warn_threshold_low}")
                warnings.append(msg)
                print(msg)

            # 检查 quality_score
            quality_mean = diagnostics.get('quality_mean', 0)

            if quality_mean < 0.5:
                msg = (f"⚠️  [Step {step}] Quality score过低: "
                      f"{quality_mean:.3f} (文本质量检测可能有问题)")
                warnings.append(msg)
                print(msg)

        elif gate_type == 'simplified':
            # 检查 gate_weight
            gate_mean = diagnostics.get('gate_mean', 0)

            if gate_mean < self.warn_threshold_low:
                msg = (f"⚠️  [Step {step}] Gate weight过低: "
                      f"{gate_mean:.3f} < {self.warn_threshold_low}")
                warnings.append(msg)
                print(msg)

            if gate_mean > self.warn_threshold_high:
                msg = (f"⚠️  [Step {step}] Gate weight过高: "
                      f"{gate_mean:.3f} > {self.warn_threshold_high}")
                warnings.append(msg)
                print(msg)

        # 保存警告
        self.warnings.extend(warnings)

    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "="*80)
        print("门控监控摘要")
        print("="*80)

        if not self.history['steps']:
            print("⚠️  没有收集到数据")
            return

        print(f"\n总步数: {len(self.history['steps'])}")

        if self.history['effective_mean']:
            # GatedCrossAttention
            print(f"\n📊 Effective Weight统计:")
            print(f"  均值: {np.mean(self.history['effective_mean']):.4f}")
            print(f"  标准差: {np.std(self.history['effective_mean']):.4f}")
            print(f"  最小值: {np.min(self.history['effective_mean']):.4f}")
            print(f"  最大值: {np.max(self.history['effective_mean']):.4f}")

            print(f"\n📊 Quality Score统计:")
            print(f"  均值: {np.mean(self.history['quality_mean']):.4f}")
            print(f"  标准差: {np.std(self.history['quality_mean']):.4f}")

            print(f"\n📊 Fusion Weight统计:")
            print(f"  均值: {np.mean(self.history['fusion_mean']):.4f}")
            print(f"  标准差: {np.std(self.history['fusion_mean']):.4f}")

        elif self.history['gate_mean']:
            # SimplifiedGatedCrossAttention
            print(f"\n📊 Gate Weight统计:")
            print(f"  均值: {np.mean(self.history['gate_mean']):.4f}")
            print(f"  标准差: {np.std(self.history['gate_mean']):.4f}")
            print(f"  最小值: {np.min(self.history['gate_mean']):.4f}")
            print(f"  最大值: {np.max(self.history['gate_mean']):.4f}")

        if self.warnings:
            print(f"\n⚠️  总警告数: {len(self.warnings)}")
            print("最近的警告:")
            for warning in self.warnings[-5:]:
                print(f"  {warning}")

        print("="*80)

    def save_plots(self, save_dir=None):
        """保存可视化图表

        Args:
            save_dir: 保存目录（默认为log_dir）
        """
        if save_dir is None:
            save_dir = self.log_dir

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        if not self.history['steps']:
            print("⚠️  没有数据可绘制")
            return

        steps = np.array(self.history['steps'])

        # 绘制门控权重曲线
        if self.history['effective_mean']:
            # GatedCrossAttention
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))

            # Effective Weight
            axes[0].plot(steps, self.history['effective_mean'], 'b-', alpha=0.7)
            axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5,
                          label='Middle (0.5)')
            axes[0].axhline(y=self.warn_threshold_low, color='orange',
                          linestyle='--', alpha=0.5,
                          label=f'Low threshold ({self.warn_threshold_low})')
            axes[0].set_xlabel('Training Steps')
            axes[0].set_ylabel('Effective Weight')
            axes[0].set_title('Effective Weight over Training')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Quality Score
            axes[1].plot(steps, self.history['quality_mean'], 'g-', alpha=0.7)
            axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5,
                          label='Threshold (0.5)')
            axes[1].set_xlabel('Training Steps')
            axes[1].set_ylabel('Quality Score')
            axes[1].set_title('Text Quality Score over Training')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Fusion Weight
            axes[2].plot(steps, self.history['fusion_mean'], 'purple', alpha=0.7)
            axes[2].axhline(y=0.5, color='r', linestyle='--', alpha=0.5,
                          label='Middle (0.5)')
            axes[2].set_xlabel('Training Steps')
            axes[2].set_ylabel('Fusion Weight')
            axes[2].set_title('Fusion Weight over Training')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = save_dir / 'gate_weights_gated.png'
            plt.savefig(save_path, dpi=150)
            print(f"📊 保存图表: {save_path}")
            plt.close()

        elif self.history['gate_mean']:
            # SimplifiedGatedCrossAttention
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(steps, self.history['gate_mean'], 'b-', alpha=0.7,
                   label='Gate Weight')
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5,
                      label='Middle (0.5)')
            ax.axhline(y=self.warn_threshold_low, color='orange',
                      linestyle='--', alpha=0.5,
                      label=f'Low threshold ({self.warn_threshold_low})')
            ax.axhline(y=self.warn_threshold_high, color='orange',
                      linestyle='--', alpha=0.5,
                      label=f'High threshold ({self.warn_threshold_high})')

            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Gate Weight')
            ax.set_title('Gate Weight over Training (Simplified)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = save_dir / 'gate_weights_simplified.png'
            plt.savefig(save_path, dpi=150)
            print(f"📊 保存图表: {save_path}")
            plt.close()

    def save_statistics(self, save_path=None):
        """保存统计数据到JSON

        Args:
            save_path: 保存路径（默认为log_dir/gate_statistics.json）
        """
        if save_path is None:
            save_path = self.log_dir / 'gate_statistics.json'

        save_path = Path(save_path)

        # 计算统计摘要
        summary = {
            'total_steps': len(self.history['steps']),
            'warnings_count': len(self.warnings),
            'warnings': self.warnings,
        }

        if self.history['effective_mean']:
            summary['effective_weight'] = {
                'mean': float(np.mean(self.history['effective_mean'])),
                'std': float(np.std(self.history['effective_mean'])),
                'min': float(np.min(self.history['effective_mean'])),
                'max': float(np.max(self.history['effective_mean'])),
            }
            summary['quality_score'] = {
                'mean': float(np.mean(self.history['quality_mean'])),
                'std': float(np.std(self.history['quality_mean'])),
                'min': float(np.min(self.history['quality_mean'])),
                'max': float(np.max(self.history['quality_mean'])),
            }
            summary['fusion_weight'] = {
                'mean': float(np.mean(self.history['fusion_mean'])),
                'std': float(np.std(self.history['fusion_mean'])),
                'min': float(np.min(self.history['fusion_mean'])),
                'max': float(np.max(self.history['fusion_mean'])),
            }

        elif self.history['gate_mean']:
            summary['gate_weight'] = {
                'mean': float(np.mean(self.history['gate_mean'])),
                'std': float(np.std(self.history['gate_mean'])),
                'min': float(np.min(self.history['gate_mean'])),
                'max': float(np.max(self.history['gate_mean'])),
            }

        # 保存
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"💾 保存统计数据: {save_path}")

    def close(self):
        """关闭监控器"""
        self.writer.close()


# ============================================================================
# 训练钩子函数
# ============================================================================

def add_gate_monitoring_to_trainer(trainer, model, gate_monitor,
                                   check_every_n_steps=10):
    """为训练器添加门控监控

    Args:
        trainer: Ignite训练器
        model: 模型
        gate_monitor: GateMonitor实例
        check_every_n_steps: 检查间隔（步数）
    """
    from ignite.engine import Events

    @trainer.on(Events.ITERATION_COMPLETED(every=check_every_n_steps))
    def log_gate_weights(engine):
        """记录门控权重"""
        # 这需要在训练循环中保存diagnostics
        if hasattr(engine.state, 'gate_diagnostics'):
            gate_monitor.update(
                engine.state.gate_diagnostics,
                step=engine.state.iteration
            )

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_summary(engine):
        """每个epoch结束时打印摘要"""
        print(f"\n[Epoch {engine.state.epoch}] Gate监控摘要")
        gate_monitor.print_summary()

    @trainer.on(Events.COMPLETED)
    def training_completed(engine):
        """训练完成时保存报告"""
        print("\n训练完成，生成门控监控报告...")
        gate_monitor.print_summary()
        gate_monitor.save_plots()
        gate_monitor.save_statistics()
        gate_monitor.close()


# ============================================================================
# 修改后的训练步骤（示例）
# ============================================================================

def train_step_with_monitoring(engine, batch, model, optimizer, criterion, device):
    """带监控的训练步骤

    这是一个示例函数，展示如何在训练中收集diagnostics
    """
    model.train()

    # 解包batch
    g, lg, text, labels = batch
    g = g.to(device)
    lg = lg.to(device)
    labels = labels.to(device)

    # Forward pass (启用diagnostics)
    optimizer.zero_grad()

    # 检查模型是否支持return_diagnostics
    if hasattr(model, 'return_diagnostics_supported'):
        output = model([g, lg, text], return_diagnostics=True)

        if isinstance(output, dict):
            predictions = output['predictions']
            # 保存diagnostics供监控器使用
            if 'gate_diagnostics' in output:
                engine.state.gate_diagnostics = output['gate_diagnostics']
        else:
            predictions = output
    else:
        predictions = model([g, lg, text])

    # 计算损失
    loss = criterion(predictions.squeeze(), labels.squeeze())

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


# ============================================================================
# 命令行接口
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='带门控监控的训练脚本'
    )

    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='恢复训练的checkpoint路径')
    parser.add_argument('--log_dir', type=str, default='runs/gate_monitoring',
                       help='TensorBoard日志目录')
    parser.add_argument('--check_interval', type=int, default=100,
                       help='门控检查间隔（步数）')

    return parser.parse_args()


def main():
    """主函数（示例）"""
    args = parse_args()

    print("="*80)
    print("带门控监控的训练脚本")
    print("="*80)

    # 创建监控器
    gate_monitor = GateMonitor(
        log_dir=args.log_dir,
        check_interval=args.check_interval
    )

    print(f"\n✅ 门控监控器已初始化")
    print(f"   - 日志目录: {args.log_dir}")
    print(f"   - 检查间隔: {args.check_interval} 步")

    # TODO: 这里添加实际的训练代码
    # 1. 加载配置
    # 2. 创建模型、优化器、数据加载器
    # 3. 创建训练器（Ignite Engine）
    # 4. 添加监控钩子
    # 5. 开始训练

    print("\n⚠️  这是一个示例脚本")
    print("请将其集成到您的 train_with_cross_modal_attention.py 中")

    # 示例：模拟一些数据
    print("\n运行模拟数据测试...")
    for step in range(1, 501):
        # 模拟gate统计
        diagnostics = {
            'quality_mean': 0.7 + 0.1 * np.random.randn(),
            'fusion_mean': 0.6 + 0.1 * np.random.randn(),
            'effective_mean': 0.42 + 0.08 * np.random.randn(),
        }
        gate_monitor.update(diagnostics, step=step)

    # 生成报告
    gate_monitor.print_summary()
    gate_monitor.save_plots()
    gate_monitor.save_statistics()
    gate_monitor.close()

    print("\n✅ 测试完成！请查看生成的图表和统计数据。")


if __name__ == '__main__':
    main()
