# 🌍 全局信息增强指南

## 问题背景

使用 RoboCrystallographer 生成的文本描述时，发现：
- ✅ **全局 + 半全局信息**：性能最佳
- ❌ **全局 + 半全局 + 局部信息**：性能下降

**原因分析**：
1. **局部信息稀释了关键的全局特征**
2. **局部细节对于全局性质预测（如带隙）可能是噪声**
3. **文本过长导致 BERT 难以捕捉关键信息**
4. **局部信息在图网络中已被编码，文本重复编码降低效率**

---

## 🚀 改进方案

### **方案 1：直接过滤 - 仅使用全局+半全局信息** ⭐ 最简单

#### 步骤 1：分离文本层次

```bash
# 查看分层示例
python split_robocrys_text.py \
    --input dataset/jarvis/hse_bandgap-2/description.csv \
    --output dataset/jarvis/hse_bandgap-2/description_filtered.csv \
    --show_examples \
    --n_examples 5

# 提取全局+半全局信息（推荐）
python split_robocrys_text.py \
    --input dataset/jarvis/hse_bandgap-2/description.csv \
    --output dataset/jarvis/hse_bandgap-2/description_global_semi.csv \
    --mode global_semi
```

#### 步骤 2：使用过滤后的数据训练

```bash
# 将原始 description.csv 备份
cp dataset/jarvis/hse_bandgap-2/description.csv \
   dataset/jarvis/hse_bandgap-2/description_full.csv

# 使用过滤后的数据
cp dataset/jarvis/hse_bandgap-2/description_global_semi.csv \
   dataset/jarvis/hse_bandgap-2/description.csv

# 正常训练
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property hse_bandgap-2 \
    --use_middle_fusion True \
    --middle_fusion_use_learnable_scale True \
    --middle_fusion_initial_scale 12.0 \
    --batch_size 64 \
    --epochs 500
```

**优点**：
- ✅ 最简单，无需修改代码
- ✅ 立即见效
- ✅ 文本更短，训练更快

**缺点**：
- ⚠️ 需要为每个数据集手动处理
- ⚠️ 信息完全丢弃，无法动态调整

---

### **方案 2：层次化编码 - 自动学习最优权重** ⭐ 最智能

使用 `HierarchicalTextEncoder` 分别编码不同层次，自动学习最优权重。

#### 步骤 1：准备分层数据

```bash
# 分离为三个独立的列
python split_robocrys_text.py \
    --input dataset/jarvis/hse_bandgap-2/description.csv \
    --output dataset/jarvis/hse_bandgap-2/description_hierarchical.csv \
    --mode split
```

输出将包含三列：
- `global_description`：全局信息
- `semi_global_description`：半全局信息
- `local_description`：局部信息

#### 步骤 2：修改 ALIGNN 模型集成层次化编码器

在 `models/alignn.py` 中：

```python
from hierarchical_text_encoding import HierarchicalTextEncoder

class ALIGNN(nn.Module):
    def __init__(self, config):
        # ... 现有代码 ...

        # 替换原始的 text_encoder
        if config.use_hierarchical_text:
            self.text_encoder = HierarchicalTextEncoder(
                use_global=True,
                use_semi_global=True,
                use_local=False,  # 根据实验结果关闭局部
                global_weight_init=1.0,
                semi_global_weight_init=0.5,
                learnable_weights=True,
                pooling='cls'
            )
        else:
            # 原始的 BERT 编码器
            self.text_encoder = AutoModel.from_pretrained('m3rg-iitd/matscibert')
```

#### 步骤 3：修改数据加载器

在 `data.py` 或数据加载代码中：

```python
def collate_fn(batch):
    # ... 现有代码 ...

    # 加载三个层次的文本
    global_texts = [item['global_description'] for item in batch]
    semi_global_texts = [item['semi_global_description'] for item in batch]
    local_texts = [item['local_description'] for item in batch]

    # Tokenize
    global_encoded = tokenizer(global_texts, padding=True, truncation=True, return_tensors='pt')
    semi_global_encoded = tokenizer(semi_global_texts, padding=True, truncation=True, return_tensors='pt')

    return (
        batched_graph,
        batched_line_graph,
        global_encoded['input_ids'],
        global_encoded['attention_mask'],
        semi_global_encoded['input_ids'],
        semi_global_encoded['attention_mask'],
        targets
    )
```

#### 步骤 4：修改 forward 方法

```python
def forward(self, input_tuple):
    g, lg, global_ids, global_mask, semi_ids, semi_mask, targets = input_tuple

    # 层次化文本编码
    text_emb, weights = self.text_encoder(
        global_input_ids=global_ids,
        global_attention_mask=global_mask,
        semi_global_input_ids=semi_ids,
        semi_global_attention_mask=semi_mask
    )

    # 打印权重（监控学习过程）
    if self.training and random.random() < 0.01:  # 1% 概率打印
        print(f"层次权重: {weights}")

    # 后续处理与原来相同
    # ...
```

#### 步骤 5：训练并监控权重

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property hse_bandgap-2 \
    --use_hierarchical_text True \
    --batch_size 64 \
    --epochs 500
```

训练过程中会自动输出权重：
```
层次权重: {'global': 0.72, 'semi_global': 0.28}
```

**优点**：
- ✅ 自动学习最优权重
- ✅ 可以动态调整不同层次的重要性
- ✅ 提供可解释性（可以查看学到的权重）

**缺点**：
- ⚠️ 需要修改较多代码
- ⚠️ 训练时需要加载三份文本（内存占用增加）

---

### **方案 3：全局信息增强 - 通过重复强调** ⭐ 折中方案

不修改模型，通过数据增强的方式突出全局信息。

#### 步骤 1：增强全局信息

```bash
# 将全局信息重复2次，放在开头和结尾
python split_robocrys_text.py \
    --input dataset/jarvis/hse_bandgap-2/description.csv \
    --output dataset/jarvis/hse_bandgap-2/description_enhanced.csv \
    --mode enhanced
```

生成的文本格式：
```
[全局信息] [全局信息] [半全局信息] [全局信息]
```

通过重复，BERT 会自动学习到全局信息的高权重。

#### 步骤 2：正常训练

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property hse_bandgap-2 \
    --batch_size 64 \
    --epochs 500
```

**优点**：
- ✅ 无需修改代码
- ✅ 简单有效
- ✅ 利用 BERT 的位置编码特性（开头和结尾权重高）

**缺点**：
- ⚠️ 文本长度增加（但比包含局部信息短）
- ⚠️ 需要为每个数据集处理

---

### **方案 4：组合策略** ⭐ 推荐生产环境

结合多个方案的优点。

#### 配置 1：过滤 + 中期融合增强

```bash
# 1. 过滤文本
python split_robocrys_text.py \
    --input dataset/jarvis/hse_bandgap-2/description.csv \
    --output dataset/jarvis/hse_bandgap-2/description_filtered.csv \
    --mode global_semi

# 2. 训练时使用 LayerNorm + 可学习缩放
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property hse_bandgap-2 \
    --use_middle_fusion True \
    --middle_fusion_use_gate_norm True \
    --middle_fusion_use_learnable_scale True \
    --middle_fusion_initial_scale 12.0 \
    --batch_size 64 \
    --epochs 500
```

#### 配置 2：增强 + 细粒度注意力

```bash
# 1. 增强全局信息
python split_robocrys_text.py \
    --input dataset/jarvis/hse_bandgap-2/description.csv \
    --output dataset/jarvis/hse_bandgap-2/description_enhanced.csv \
    --mode enhanced

# 2. 训练时使用细粒度注意力
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property hse_bandgap-2 \
    --use_middle_fusion True \
    --use_fine_grained_attention True \
    --fine_grained_num_heads 8 \
    --batch_size 64 \
    --epochs 500
```

---

## 📊 预期效果对比

| 方案 | 实现难度 | 预期MAE改进 | 可解释性 | 训练速度 |
|------|---------|------------|---------|---------|
| **过滤（方案1）** | ⭐ 简单 | +3-5% | ⭐⭐ | ⬆️ 提升20% |
| **层次化编码（方案2）** | ⭐⭐⭐ 复杂 | +5-8% | ⭐⭐⭐⭐ | ⬇️ 降低10% |
| **增强（方案3）** | ⭐⭐ 中等 | +2-4% | ⭐⭐ | ➡️ 持平 |
| **组合（方案4）** | ⭐⭐ 中等 | +6-10% | ⭐⭐⭐ | ⬆️ 提升10% |

---

## 🔬 实验验证流程

### 第一阶段：快速验证（1-2天）

```bash
# 1. 备份原始数据
cp description.csv description_original.csv

# 2. 生成三个版本
python split_robocrys_text.py --input description.csv --output description_full.csv --mode split
python split_robocrys_text.py --input description.csv --output description_filtered.csv --mode global_semi
python split_robocrys_text.py --input description.csv --output description_enhanced.csv --mode enhanced

# 3. 训练三个模型（使用较少的 epochs 快速测试）
for mode in original filtered enhanced; do
    cp description_${mode}.csv description.csv
    python train_with_cross_modal_attention.py \
        --property hse_bandgap-2 \
        --epochs 100 \
        --output_dir ./output_${mode}
done

# 4. 对比结果
python compare_results.py \
    --model1 output_original/best_model.pt \
    --model2 output_filtered/best_model.pt \
    --model3 output_enhanced/best_model.pt
```

### 第二阶段：完整训练（3-5天）

选择第一阶段表现最好的方案，进行完整训练：

```bash
# 假设 filtered 表现最好
python train_with_cross_modal_attention.py \
    --property hse_bandgap-2 \
    --epochs 500 \
    --use_middle_fusion True \
    --middle_fusion_use_gate_norm True \
    --middle_fusion_use_learnable_scale True \
    --middle_fusion_initial_scale 12.0 \
    --use_cross_modal True \
    --output_dir ./output_final
```

---

## 💡 关键建议

### 1. **文本长度分析**

先分析您的数据中不同层次的比例：

```bash
python split_robocrys_text.py \
    --input description.csv \
    --output temp.csv \
    --show_examples \
    --n_examples 10
```

查看输出的组成比例，例如：
```
全局信息占比: 25%
半全局信息占比: 45%
局部信息占比: 30%
```

如果局部信息占比很高（>40%），说明过滤效果会很明显。

### 2. **渐进式改进**

不要一次性使用所有方案，建议顺序：
1. **先过滤**（方案1）：验证局部信息确实有害
2. **再增强**（方案3）：如果过滤效果好，尝试进一步增强
3. **最后层次化**（方案2）：如果需要最大性能和可解释性

### 3. **监控文本质量**

```python
# 在训练前检查文本
import pandas as pd

df = pd.read_csv('description_filtered.csv')
print(f"平均文本长度: {df['description'].str.len().mean():.1f} 字符")
print(f"最短文本: {df['description'].str.len().min()} 字符")
print(f"最长文本: {df['description'].str.len().max()} 字符")

# 如果有很多空文本，说明过滤太激进
empty_count = (df['description'].str.len() < 10).sum()
print(f"空文本数量: {empty_count} ({empty_count/len(df)*100:.1f}%)")
```

### 4. **调整分离规则**

如果自动分离效果不好，修改 `split_robocrys_text.py` 中的关键词：

```python
# 根据您的数据调整
self.global_keywords = [
    'space group', 'crystal system',
    # 添加您观察到的全局特征关键词
    'band gap', 'formation energy',  # 如果这些在全局描述中
]
```

---

## 🎯 推荐配置（基于您的情况）

您已经发现全局+半全局效果最好，**推荐从方案1开始**：

```bash
# 步骤1：过滤文本（仅保留全局+半全局）
python split_robocrys_text.py \
    --input /public/home/ghzhang/crysmmnet-main/dataset/jarvis/hse_bandgap-2/description.csv \
    --output /public/home/ghzhang/crysmmnet-main/dataset/jarvis/hse_bandgap-2/description_filtered.csv \
    --mode global_semi

# 步骤2：备份并替换
cd /public/home/ghzhang/crysmmnet-main/dataset/jarvis/hse_bandgap-2/
cp description.csv description_original.csv
cp description_filtered.csv description.csv

# 步骤3：训练（使用您当前的配置）
python train_with_cross_modal_attention.py \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --property hse_bandgap-2 \
    --use_middle_fusion True \
    --middle_fusion_use_gate_norm True \
    --middle_fusion_use_learnable_scale True \
    --middle_fusion_initial_scale 12.0 \
    --use_fine_grained_attention True \
    --batch_size 64 \
    --epochs 100
```

**预期改进**：
- ✅ MAE 降低 3-7%
- ✅ 训练速度提升 15-25%（文本更短）
- ✅ 内存占用减少（文本更短）

---

## 📚 相关工具和文档

1. **hierarchical_text_encoding.py** - 层次化文本编码器实现
2. **split_robocrys_text.py** - 文本分层工具
3. **IMPROVED_MIDDLE_FUSION_GUIDE.md** - 中期融合改进指南
4. **monitor_text_scale.py** - 监控可学习缩放因子

---

## ❓ 常见问题

### Q1: 如果过滤后文本太短怎么办？

**A**: 调整为 `enhanced` 模式，重复全局信息：
```bash
python split_robocrys_text.py --mode enhanced
```

### Q2: 如何确定哪些信息是全局的？

**A**: 查看示例：
```bash
python split_robocrys_text.py --show_examples --n_examples 10
```
根据输出调整关键词列表。

### Q3: 所有性质都适合这个方案吗？

**A**: 不一定。对于：
- ✅ **全局性质**（带隙、形成能、体积模量）：非常有效
- ⚠️ **局部性质**（特定原子的磁矩）：可能需要局部信息

### Q4: 能否保留少量重要的局部信息？

**A**: 可以！修改 `split_robocrys_text.py`，添加"重要局部信息"的规则：
```python
# 例如：保留异常键长的信息（可能影响性质）
important_local_keywords = ['unusually', 'significantly', 'distorted']
```

---

## 🎉 总结

您的发现非常有价值！建议：

1. **立即行动**：使用方案1过滤文本，验证改进
2. **渐进优化**：如果效果好，尝试方案3增强
3. **长期方案**：考虑实现方案2获得最佳性能和可解释性

祝实验顺利！ 🚀
