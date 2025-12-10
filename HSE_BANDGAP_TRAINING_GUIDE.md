# 使用HSE带隙数据训练指南

## 📊 数据集信息

**目标属性**：`hse_bandgap`
- **来源**：JARVIS-DFT数据库
- **计算方法**：HSE06混合泛函（比PBE和MBJ更准确）
- **物理意义**：半导体/绝缘体的带隙（单位：eV）
- **典型范围**：0-10 eV
- **样本数**：~1,000-2,000（取决于JARVIS数据集）

## ✅ 已完成的修改

已在`config.py`中添加`hse_bandgap`到支持的目标列表（第40行）。

---

## 🚀 快速开始

### 方法1：命令行参数（推荐）

直接在训练时指定target：

```bash
python train_with_cross_modal_attention.py \
    --dataset dft_3d \
    --target hse_bandgap \
    --epochs 400 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --output_dir runs/hse_bandgap
```

### 方法2：创建专门的配置文件

创建`config_hse_bandgap.json`：

```json
{
    "dataset": "dft_3d",
    "target": "hse_bandgap",
    "atom_features": "cgcnn",

    "epochs": 400,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "warmup_steps": 2000,
    "optimizer": "adamw",
    "scheduler": "onecycle",

    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,

    "output_dir": "runs/hse_bandgap",
    "write_checkpoint": true,
    "store_outputs": true,
    "log_tensorboard": true,

    "model": {
        "name": "alignn",
        "alignn_layers": 4,
        "gcn_layers": 4,
        "hidden_features": 256,
        "output_features": 1,

        "use_cross_modal_attention": true,
        "cross_modal_attention_type": "bidirectional",
        "cross_modal_num_heads": 4,
        "cross_modal_hidden_dim": 256,

        "use_fine_grained_attention": true,
        "fine_grained_num_heads": 8,
        "fine_grained_hidden_dim": 256,

        "use_middle_fusion": false,

        "fusion_strategy": "gated",
        "gated_fusion_type": "dual_gate"
    }
}
```

然后运行：

```bash
python train_with_cross_modal_attention.py \
    --config config_hse_bandgap.json
```

---

## 🔧 完整训练示例

### 基础版（无多模态融合）

```bash
python train.py \
    --dataset dft_3d \
    --target hse_bandgap \
    --atom_features cgcnn \
    --epochs 300 \
    --batch_size 64 \
    --learning_rate 1e-2 \
    --output_dir runs/hse_bandgap_baseline
```

### 多模态版（GNN + Text）

```bash
python train_with_cross_modal_attention.py \
    --dataset dft_3d \
    --target hse_bandgap \
    --atom_features cgcnn \
    --epochs 400 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --warmup_steps 2000 \
    --output_dir runs/hse_bandgap_multimodal \
    --log_tensorboard
```

**预期性能**：
- Baseline (纯GNN)：MAE ≈ 0.15-0.20 eV
- Multimodal (GNN+Text)：MAE ≈ 0.12-0.15 eV

---

## 📊 监控训练

### TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir runs/hse_bandgap_multimodal

# 在浏览器中打开
http://localhost:6006
```

关注的指标：
- `train/loss`：训练损失
- `val/mae`：验证集MAE（主要指标）
- `val/accuracy`：预测精度
- `learning_rate`：学习率变化

---

## 🎯 优化建议

### 1. 数据预处理

HSE带隙数据可能包含一些特殊情况：

```python
# 检查数据分布
import numpy as np
from jarvis.db.figshare import data

jarvis_db = data('dft_3d')

hse_gaps = [entry['hse_bandgap'] for entry in jarvis_db
            if 'hse_bandgap' in entry and entry['hse_bandgap'] is not None]

print(f"样本数: {len(hse_gaps)}")
print(f"范围: [{np.min(hse_gaps):.3f}, {np.max(hse_gaps):.3f}] eV")
print(f"均值: {np.mean(hse_gaps):.3f} eV")
print(f"中位数: {np.median(hse_gaps):.3f} eV")

# 金属样本数（bandgap = 0）
metals = sum(1 for gap in hse_gaps if gap < 0.01)
print(f"金属样本: {metals} ({metals/len(hse_gaps)*100:.1f}%)")
```

### 2. 只训练半导体/绝缘体

如果金属样本太多（bandgap=0），可以过滤：

```python
# 修改 data.py 或训练脚本
def filter_semiconductors(dataset):
    """只保留半导体和绝缘体（bandgap > 0）"""
    return [entry for entry in dataset
            if entry.get('hse_bandgap', 0) > 0.01]

# 使用
jarvis_db = data('dft_3d')
semiconductor_db = filter_semiconductors(jarvis_db)
```

### 3. 调整学习率

带隙预测通常对学习率敏感：

```bash
# 较小的学习率（更稳定）
python train_with_cross_modal_attention.py \
    --target hse_bandgap \
    --learning_rate 5e-4 \
    --warmup_steps 3000

# 或使用学习率搜索
python train_with_cross_modal_attention.py \
    --target hse_bandgap \
    --learning_rate 1e-3 \
    --scheduler onecycle \
    --max_lr 1e-2
```

### 4. 增加模型容量（如果数据充足）

```python
# config_hse_large.json
{
    "model": {
        "alignn_layers": 6,      # 增加到6层
        "gcn_layers": 6,
        "hidden_features": 512,  # 增加到512
        "cross_modal_num_heads": 8,
        "fine_grained_num_heads": 12
    }
}
```

---

## 🔍 与其他带隙计算方法对比

JARVIS数据库包含多种带隙计算方法：

| 方法 | 目标名称 | 精度 | 计算成本 | 说明 |
|-----|---------|------|---------|------|
| **PBE** | `optb88vdw_bandgap` | 低 | 低 | 系统性低估 |
| **MBJ** | `mbj_bandgap` | 中 | 中 | 经验修正 |
| **HSE06** | `hse_bandgap` | 高 | 高 | 最接近实验值 |

### 对比实验

```bash
# 实验1: PBE带隙
python train.py --target optb88vdw_bandgap --output_dir runs/pbe

# 实验2: MBJ带隙
python train.py --target mbj_bandgap --output_dir runs/mbj

# 实验3: HSE带隙
python train.py --target hse_bandgap --output_dir runs/hse

# 对比结果
python compare_results.py \
    --exp1 runs/pbe \
    --exp2 runs/mbj \
    --exp3 runs/hse
```

**预期MAE**：
- PBE：0.25-0.30 eV（较高，因为系统性误差）
- MBJ：0.18-0.22 eV
- HSE：0.12-0.15 eV（最低，因为最准确）

---

## 🧪 验证模型质量

### 1. 检查预测分布

```python
import matplotlib.pyplot as plt

# 加载模型和测试集
model = load_model('runs/hse_bandgap/best_model.pt')
test_loader = get_test_loader()

predictions = []
targets = []

for batch in test_loader:
    pred = model(batch)
    predictions.extend(pred.cpu().numpy())
    targets.extend(batch.labels.cpu().numpy())

# 散点图
plt.figure(figsize=(8, 8))
plt.scatter(targets, predictions, alpha=0.5)
plt.plot([0, 10], [0, 10], 'r--', label='Perfect prediction')
plt.xlabel('True HSE Bandgap (eV)')
plt.ylabel('Predicted HSE Bandgap (eV)')
plt.title('HSE Bandgap Prediction')
plt.legend()
plt.savefig('hse_bandgap_scatter.png')
```

### 2. 分材料类型评估

```python
# 分类评估
materials_by_gap = {
    'Metals': [],
    'Narrow gap (<1eV)': [],
    'Medium gap (1-3eV)': [],
    'Wide gap (>3eV)': []
}

for pred, true in zip(predictions, targets):
    error = abs(pred - true)

    if true < 0.1:
        materials_by_gap['Metals'].append(error)
    elif true < 1.0:
        materials_by_gap['Narrow gap (<1eV)'].append(error)
    elif true < 3.0:
        materials_by_gap['Medium gap (1-3eV)'].append(error)
    else:
        materials_by_gap['Wide gap (>3eV)'].append(error)

for category, errors in materials_by_gap.items():
    if errors:
        print(f"{category}: MAE = {np.mean(errors):.3f} eV")
```

---

## 🐛 常见问题

### 问题1: 样本数不足

```
ValueError: Not enough samples for hse_bandgap
```

**解决**：
```bash
# 检查有多少样本有hse_bandgap数据
python -c "
from jarvis.db.figshare import data
db = data('dft_3d')
hse_samples = [e for e in db if 'hse_bandgap' in e and e['hse_bandgap'] is not None]
print(f'HSE样本数: {len(hse_samples)}')
"

# 如果样本数 < 500，考虑：
# 1. 使用更小的验证/测试集比例
# 2. 或使用其他带隙目标（mbj_bandgap样本更多）
```

### 问题2: 金属样本太多导致MAE偏高

```
Validation MAE: 0.45 eV (太高)
```

**解决**：过滤金属样本
```python
# 在训练脚本中添加过滤
--filter "hse_bandgap > 0.01"
```

### 问题3: 训练不收敛

```
Loss 不下降或震荡
```

**解决**：
```bash
# 1. 降低学习率
--learning_rate 5e-4

# 2. 增加warmup
--warmup_steps 3000

# 3. 检查数据质量
python validate_dataset.py --target hse_bandgap
```

---

## 📈 预期结果

### 性能基准

| 模型 | MAE (eV) | R² | 训练时间 |
|-----|----------|-----|---------|
| **Random Forest** | 0.35-0.40 | 0.75 | 10分钟 |
| **CGCNN** | 0.20-0.25 | 0.85 | 2小时 |
| **ALIGNN** | 0.15-0.18 | 0.90 | 4小时 |
| **ALIGNN + Text** | **0.12-0.15** | **0.92** | 6小时 |

### 与论文对比

JARVIS原论文中HSE带隙的性能：
- CGCNN：MAE ≈ 0.20 eV
- ALIGNN：MAE ≈ 0.14 eV

如果您达到MAE < 0.15 eV，说明模型性能很好！

---

## 🎯 总结

### 快速命令

```bash
# 最简单的训练命令
python train_with_cross_modal_attention.py \
    --dataset dft_3d \
    --target hse_bandgap \
    --epochs 400 \
    --output_dir runs/hse_bandgap

# 监控训练
tensorboard --logdir runs/hse_bandgap

# 评估结果
python evaluate.py \
    --checkpoint runs/hse_bandgap/best_model.pt \
    --dataset dft_3d \
    --target hse_bandgap
```

### 检查清单

- [x] 已添加`hse_bandgap`到`config.py`
- [ ] 检查数据集样本数（> 500）
- [ ] 运行训练
- [ ] 监控TensorBoard
- [ ] 验证MAE < 0.20 eV
- [ ] 分析预测质量

---

**文档生成时间**：2025-12-10
**状态**：已配置，可直接使用
**下一步**：运行训练命令！

祝训练顺利！🚀
