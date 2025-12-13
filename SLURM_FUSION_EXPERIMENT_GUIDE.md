# 🚀 SLURM融合方式消融实验快速指南

## 📋 概述

本指南帮助您使用SLURM作业管理系统批量测试5种后期融合方式的性能。

### 实验设计

| # | 融合方式 | 描述 | 预期提升 |
|---|---------|------|---------|
| 1 | **concat** | 简单拼接（基线） | 0% |
| 2 | **gated** | 门控融合（自适应权重） | +2-5% |
| 3 | **bilinear** | 双线性池化（二阶交互，Rank=16） | +3-7% |
| 4 | **adaptive** | 自适应融合（多策略组合） | +4-8% |
| 5 | **tucker** | Tucker分解（高阶张量，Rank=16） | +5-10% |

---

## 🎯 快速开始

### 方式1: 串行执行（推荐，节省资源）

逐个运行实验，每个实验完成后自动启动下一个：

```bash
# 提交作业链
./submit_fusion_ablation.sh
```

**优点**：
- ✅ 只需1张GPU
- ✅ 自动排队，无需手动管理
- ✅ 适合GPU资源有限的情况

**缺点**：
- ⏱️ 总时间 = 单个实验时间 × 5

---

### 方式2: 并行执行（快速，需要多GPU）

同时运行所有实验：

```bash
# 提交所有作业（并行）
./submit_fusion_ablation_parallel.sh
```

**优点**：
- ⚡ 总时间 = 单个实验时间
- ⚡ 5倍加速

**缺点**：
- 需要5张GPU（或等待资源）

**修改GPU分配**：
编辑 `submit_fusion_ablation_parallel.sh`：
```bash
# 如果有5张GPU (0-4)
CUDA_DEVICES=("0" "1" "2" "3" "4")

# 如果只有2张GPU (0-1)，会排队使用
CUDA_DEVICES=("0" "1" "0" "1" "0")

# 如果只有1张GPU (3)，全部排队
CUDA_DEVICES=("3" "3" "3" "3" "3")
```

---

## 📊 监控训练进度

### 1. 查看所有作业状态
```bash
# 查看您的所有作业
squeue -u $USER

# 详细信息
squeue -u $USER -o '%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R'
```

**状态说明**：
- `PD` (Pending): 等待资源
- `R` (Running): 正在运行
- `CG` (Completing): 即将完成
- `CD` (Completed): 已完成

### 2. 实时监控
```bash
# 每10秒刷新一次
watch -n 10 'squeue -u $USER'
```

### 3. 查看训练日志
```bash
# Concat基线
tail -f ./hse_fusion_concat/train_*.out

# Gated融合
tail -f ./hse_fusion_gated/train_*.out

# Tucker融合
tail -f ./hse_fusion_tucker_r16/train_*.out
```

### 4. 检查训练进度（快速）
```bash
# 查看所有实验的当前epoch
for dir in hse_fusion_*/; do
    echo "=== $dir ==="
    grep -oP 'Epoch \K[0-9]+' ${dir}hse_bandgap-2/train_*.out 2>/dev/null | tail -n 1
done
```

---

## 📈 收集和分析结果

### 实验完成后

```bash
# 运行结果收集脚本
./collect_fusion_results.sh
```

**生成的文件**：
1. `fusion_ablation_results.txt` - 完整文本报告
2. `fusion_ablation_results.csv` - CSV数据（可导入Excel）

### 快速查看结果

```bash
# 查看文本报告
cat fusion_ablation_results.txt

# 查看性能对比表格
tail -n 20 fusion_ablation_results.txt
```

### 示例输出

```
================================================================================
性能对比表格
================================================================================

融合方式                   Val MAE      Test MAE     Val改进         Test改进
--------------------------------------------------------------------------------
Concat (Baseline)         0.0850       0.0920       N/A             N/A
Gated                     0.0820       0.0890       +3.53%          +3.26%
Bilinear (R=16)           0.0805       0.0875       +5.29%          +4.89%
Adaptive                  0.0795       0.0865       +6.47%          +5.98%
Tucker (R=16)             0.0780       0.0850       +8.24%          +7.61%
================================================================================
```

---

## 🛠️ 管理作业

### 查看作业依赖关系（串行模式）
```bash
squeue -u $USER -o '%.18i %.30j %.8T %.10r'
```

### 取消作业

```bash
# 取消所有您的作业
scancel -u $USER

# 取消特定作业
scancel <JOB_ID>

# 取消整个作业链（脚本会输出作业ID列表）
scancel <JOB_ID_1> <JOB_ID_2> <JOB_ID_3> <JOB_ID_4> <JOB_ID_5>
```

---

## 📁 输出目录结构

```
.
├── hse_fusion_concat/              # Concat基线
│   ├── hse_bandgap-2/
│   │   ├── best_val_model.pt       # 最佳验证集模型
│   │   ├── checkpoint_*.pt         # 训练checkpoint
│   │   └── config.json             # 配置文件
│   └── train_*.out                 # SLURM输出日志
│
├── hse_fusion_gated/               # Gated融合
│   └── ...
│
├── hse_fusion_bilinear_r16/        # Bilinear融合
│   └── ...
│
├── hse_fusion_adaptive/            # Adaptive融合
│   └── ...
│
└── hse_fusion_tucker_r16/          # Tucker融合
    └── ...
```

---

## ⚙️ 自定义配置

### 修改训练参数

编辑 `submit_fusion_ablation.sh` 或 `submit_fusion_ablation_parallel.sh`：

```bash
# 1. 修改数据集路径
DATA_ROOT="/your/path/to/dataset"

# 2. 修改GPU设备
CUDA_DEVICE="0"  # 使用GPU 0

# 3. 修改训练参数（在submit_job函数中）
--batch_size 128          # 批次大小
--epochs 150              # 训练轮数
--learning_rate 1e-3      # 学习率
--early_stopping_patience 50  # Early stopping patience

# 4. 修改融合参数
--middle_fusion_initial_scale 12.0  # 中期融合初始缩放（根据您的诊断）
--late_fusion_output_dim 64         # 融合输出维度
```

### 添加更多融合配置

例如，测试Tucker融合的不同Rank值：

编辑 `submit_fusion_ablation_parallel.sh`：

```bash
FUSION_CONFIGS=(
    "concat:Baseline: Concat Fusion:fusion_concat:concat:16:64"
    "gated:Gated Fusion:fusion_gated:gated:16:64"
    "tucker_r8:Tucker Fusion (Rank=8):fusion_tucker_r8:tucker:8:64"
    "tucker_r16:Tucker Fusion (Rank=16):fusion_tucker_r16:tucker:16:64"
    "tucker_r32:Tucker Fusion (Rank=32):fusion_tucker_r32:tucker:32:64"
)
```

---

## 🔍 故障排除

### 问题1: 作业一直处于PD状态

**原因**: GPU资源不足

**解决**:
```bash
# 查看队列状态
squeue

# 查看可用GPU
sinfo -o "%.20N %.10P %.11T %.4c %.8z %.6m %.8d %.6w %.8f %20E"

# 取消并重新提交到其他分区
scancel <JOB_ID>
# 编辑脚本，设置 SLURM_PARTITION="your_partition"
```

### 问题2: 作业失败，退出码非0

**检查错误日志**:
```bash
# 查看SLURM错误输出
cat ./hse_fusion_*/train_*-<JOB_ID>.err

# 查看训练日志
cat ./hse_fusion_*/train_*-<JOB_ID>.out
```

**常见错误**:
- CUDA out of memory → 减小batch_size
- 找不到数据集 → 检查DATA_ROOT路径
- 环境问题 → 检查CONDA_ENV名称

### 问题3: Python命令找不到

**解决**:
```bash
# 确保在脚本中使用绝对路径
python train_with_cross_modal_attention.py

# 或者指定完整路径
/path/to/conda/envs/sganet/bin/python train_with_cross_modal_attention.py
```

---

## 📊 结果分析建议

### 1. 基本对比
收集所有Val MAE，找出最佳融合方式

### 2. 时间效率对比
```bash
# 检查每个实验的训练时间
for dir in hse_fusion_*/; do
    echo "=== $dir ==="
    grep "Training Complete" ${dir}hse_bandgap-2/train_*.out -A 5
done
```

### 3. 参数量对比
不同融合方式的参数量：
- Concat: 最少
- Gated: +10-15%
- Bilinear (R=16): +20-25%
- Adaptive: +30-35%
- Tucker (R=16): +25-30%

### 4. 生成对比图表

使用Python脚本（可选）：
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV
df = pd.read_csv('fusion_ablation_results.csv')

# 绘制Val MAE对比
plt.figure(figsize=(10, 6))
plt.bar(df['Fusion_Type'], df['Val_MAE'])
plt.xlabel('Fusion Type')
plt.ylabel('Val MAE')
plt.title('Fusion Method Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('fusion_comparison.png')
```

---

## 💡 最佳实践

### 1. 第一次运行
```bash
# 使用串行模式，先跑完整流程
./submit_fusion_ablation.sh

# 监控第一个任务（Concat基线）
tail -f ./hse_fusion_concat/train_*.out
```

### 2. 验证配置正确
检查训练日志，确认：
- ✅ 融合类型正确
- ✅ 数据加载成功
- ✅ 模型初始化正常
- ✅ 训练正常进行

### 3. 资源充足时
```bash
# 使用并行模式加速
./submit_fusion_ablation_parallel.sh
```

### 4. 定期检查
```bash
# 每小时检查一次进度
watch -n 3600 './collect_fusion_results.sh'
```

---

## 🎯 预期时间线

假设单个实验训练100 epochs，每个epoch约2分钟：

**串行模式**:
- 单个实验: ~200分钟（3.3小时）
- 5个实验: ~1000分钟（16.7小时）

**并行模式**（5张GPU）:
- 总时间: ~200分钟（3.3小时）

**Early Stopping（patience=30）**:
- 可能在50-80 epochs停止
- 单个实验: ~100-160分钟
- 串行总时间: ~8-13小时

---

## 📞 支持

遇到问题？检查：
1. SLURM日志: `./hse_fusion_*/train_*.err`
2. 训练日志: `./hse_fusion_*/train_*.out`
3. 配置文件: `./hse_fusion_*/hse_bandgap-2/config.json`

---

## 🎉 完成后

1. 收集结果: `./collect_fusion_results.sh`
2. 查看最佳模型: `hse_fusion_*/hse_bandgap-2/best_val_model.pt`
3. 分析对比报告: `fusion_ablation_results.txt`
4. 选择最佳融合方式用于后续实验

**Good luck!** 🚀
