# 数据加载问题调试指南

## 🔍 问题现象

您在训练时看到：
```
list index out of range
list index out of range
...
成功加载: 1639 样本
跳过: 8825 样本
```

**大量样本被跳过（约84%）！**

---

## 📊 可能的原因

### 1. **缺少目标字段** ⭐ 最常见
您的数据文件中大部分样本可能没有 `hse_bandgap` 字段。

**例如**：如果您加载的是 `dft_3d` 完整数据集（~40,000样本），但只有少数样本（~1,600）计算了HSE带隙。

### 2. **目标值为None或"na"**
样本有 `hse_bandgap` 字段，但值是 `None` 或 `"na"`。

### 3. **目标值为NaN**
样本的目标值是 `float('nan')`。

### 4. **数据文件不匹配**
您可能加载了错误的数据文件（如 `dft_3d.json` 而不是 `hse_bandgap.json`）。

---

## 🛠️ 诊断步骤

### 步骤1: 使用调试脚本分析数据

```bash
python debug_data_loading.py \
    --data_path /public/home/ghzhang/crysmmnet-main/dataset/jarvis/hse_bandgap \
    --target hse_bandgap
```

这将显示：
- 总样本数
- 有效样本数和跳过样本数
- 跳过原因的详细统计
- 错误样本示例

**预期输出示例**：
```
================================================================================
调试数据加载问题
================================================================================
文件: /public/home/ghzhang/.../hse_bandgap/data.json
目标属性: hse_bandgap

📊 总样本数: 10464

📈 样本统计:
--------------------------------------------------------------------------------
  ✅ 有效样本:            1639 (15.7%)
  ❌ 跳过样本:            8825 (84.3%)

跳过原因分布:
  - 缺少目标字段:          0
  - 目标值为None:       8825      ← 主要原因！
  - 目标值为'na':          0
  - 目标值为NaN:           0
  - 样本不是字典:          0
  - 目标值是列表:          0 (有效)
  - 其他错误:              0

🔍 错误样本示例 (前10个):

样本 #1:
  原因: target_none
  jid: JVASP-1234
```

### 步骤2: 检查数据文件

```bash
# 查看数据文件
ls -lh /public/home/ghzhang/crysmmnet-main/dataset/jarvis/hse_bandgap/

# 如果是JSON，查看前几行
head -100 /public/home/ghzhang/crysmmnet-main/dataset/jarvis/hse_bandgap/*.json

# 统计有效的hse_bandgap值
python -c "
import json
data = json.load(open('/public/home/ghzhang/crysmmnet-main/dataset/jarvis/hse_bandgap/data.json'))
valid = sum(1 for d in data if d.get('hse_bandgap') not in [None, 'na'])
print(f'总样本: {len(data)}')
print(f'有效hse_bandgap: {valid}')
print(f'有效率: {valid/len(data)*100:.1f}%')
"
```

### 步骤3: 改进的训练日志

现在训练时会自动显示详细的跳过统计：

```bash
python train_with_cross_modal_attention.py \
    --dataset dft_3d \
    --target hse_bandgap \
    --epochs 400 \
    --output_dir runs/hse_bandgap
```

**新的输出示例**：
```
✅ 从本地加载JARVIS数据: .../hse_bandgap/data.json
   加载了 10464 个样本

⚠️  样本 #0 (jid=JVASP-1001) 缺少字段 'hse_bandgap'，跳过
⚠️  样本 #1 (jid=JVASP-1002) 缺少字段 'hse_bandgap'，跳过
...

⚠️  数据加载统计:
   总样本数: 10464
   成功加载: 1639 (15.7%)
   跳过样本: 8825 (84.3%)
   跳过原因:
     - 目标值为None: 8825

继续训练使用1639个有效样本...
```

---

## ✅ 解决方案

### 方案1: 使用正确的数据文件 ⭐ 推荐

**问题**：您可能加载了包含所有JARVIS材料的文件，但只有部分计算了HSE带隙。

**解决**：

1. **检查您是否有专门的HSE数据集文件**：
   ```bash
   # 查找所有可能的文件
   find /public/home/ghzhang/crysmmnet-main/dataset/jarvis/ -name "*hse*" -type f
   ```

2. **如果有单独的HSE文件**（如 `hse_bandgap_only.json`），使用它：
   ```bash
   # 将其重命名或链接为主文件
   mv .../hse_bandgap/hse_bandgap_only.json .../hse_bandgap/data.json
   ```

3. **或者从JARVIS在线下载纯HSE数据集**：
   ```python
   from jarvis.db.figshare import data as jdata
   import json

   # 下载HSE数据
   hse_data = jdata('dft_3d')

   # 只保留有HSE值的样本
   hse_filtered = [d for d in hse_data
                   if d.get('hse_bandgap') not in [None, 'na']]

   print(f"过滤后: {len(hse_filtered)} 样本")

   # 保存
   with open('hse_bandgap_filtered.json', 'w') as f:
       json.dump(hse_filtered, f)
   ```

### 方案2: 接受较低的有效样本率

**如果1639个样本足够训练**，您可以直接继续：

```bash
python train_with_cross_modal_attention.py \
    --dataset dft_3d \
    --target hse_bandgap \
    --epochs 400 \
    --output_dir runs/hse_bandgap
```

**优点**：
- 简单直接
- 1639个样本对于DFT性质预测通常足够

**缺点**：
- 训练集较小
- 可能泛化性能受限

### 方案3: 创建过滤后的数据集

创建一个只包含有效HSE样本的新数据集：

```bash
# 使用调试脚本导出有效样本
python -c "
import json
import pickle as pk
from pathlib import Path

# 加载原始数据
data_path = Path('/public/home/ghzhang/crysmmnet-main/dataset/jarvis/hse_bandgap')
json_files = list(data_path.glob('*.json'))
if json_files:
    with open(json_files[0], 'r') as f:
        data = json.load(f)
else:
    pkl_files = list(data_path.glob('*.pkl'))
    with open(pkl_files[0], 'rb') as f:
        data = pk.load(f)

# 过滤有效样本
target = 'hse_bandgap'
valid_data = []
for d in data:
    if isinstance(d, dict) and target in d:
        val = d[target]
        if val is not None and val != 'na':
            try:
                import math
                if not math.isnan(val):
                    valid_data.append(d)
            except (TypeError, ValueError):
                pass

print(f'原始样本: {len(data)}')
print(f'有效样本: {len(valid_data)}')

# 保存过滤后的数据
output_file = data_path / 'hse_bandgap_filtered.json'
with open(output_file, 'w') as f:
    json.dump(valid_data, f, indent=2)

print(f'已保存到: {output_file}')
"
```

然后将过滤后的文件设为主文件：
```bash
cd /public/home/ghzhang/crysmmnet-main/dataset/jarvis/hse_bandgap
mv data.json data_original.json.bak
mv hse_bandgap_filtered.json data.json
```

---

## 🎯 预期结果

### 使用正确数据后

```
✅ 从本地加载JARVIS数据: .../hse_bandgap/data.json
   加载了 1639 个样本

⚠️  数据加载统计:
   总样本数: 1639
   成功加载: 1639 (100.0%)
   跳过样本: 0 (0.0%)

✅ 所有样本有效！
```

### 训练配置建议

对于1639个HSE样本：

```bash
python train_with_cross_modal_attention.py \
    --dataset dft_3d \
    --target hse_bandgap \
    --atom_features cgcnn \
    --epochs 400 \
    --batch_size 32 \              # 小数据集用小batch
    --learning_rate 1e-3 \
    --train_ratio 0.8 \            # 80% 训练
    --val_ratio 0.1 \              # 10% 验证
    --test_ratio 0.1 \             # 10% 测试
    --output_dir runs/hse_bandgap
```

这将得到：
- 训练集: ~1311 样本
- 验证集: ~164 样本
- 测试集: ~164 样本

---

## 🐛 常见错误

### 错误1: "list index out of range"

**原因**：数据格式问题，可能某些样本不是字典格式

**解决**：运行调试脚本检查数据格式：
```bash
python debug_data_loading.py --data_path YOUR_PATH --target hse_bandgap
```

### 错误2: 训练集太小

**症状**：
```
ValueError: Train ratio is too low, no samples in training set
```

**解决**：
1. 确保有足够的有效样本（>100）
2. 调整分割比例：`--train_ratio 0.9 --val_ratio 0.05 --test_ratio 0.05`

### 错误3: 数据路径不存在

**症状**：
```
📡 本地未找到数据，从figshare下载: hse_bandgap
```

**解决**：
1. 检查路径是否正确
2. 检查文件权限
3. 检查文件名（必须是 `.json` 或 `.pkl`）

---

## 📚 更多信息

- **数据加载机制**：参见 `LOCAL_DATA_USAGE_GUIDE.md`
- **HSE训练指南**：参见 `HSE_BANDGAP_TRAINING_GUIDE.md`
- **本地数据工具**：参见 `load_local_hse.py`

---

## 🚀 快速诊断

运行此命令快速诊断您的数据：

```bash
python debug_data_loading.py \
    --data_path /public/home/ghzhang/crysmmnet-main/dataset/jarvis/hse_bandgap \
    --target hse_bandgap
```

然后根据输出选择上述解决方案！

---

**文档生成时间**：2025-12-10
**状态**：✅ 已添加详细日志和调试工具
