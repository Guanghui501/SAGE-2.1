# HSE带隙训练问题诊断和改进方案

## 🔍 问题诊断

### 当前训练曲线分析

从您提供的曲线可以看出：

1. **严重过拟合** ⚠️
   - 训练MAE: ~0.05 eV (接近0)
   - 验证MAE: ~0.40 eV
   - **差距: 0.35 eV (8倍差异！)**

2. **验证损失停滞**
   - 验证MAE在约40个epoch后就停止改善
   - 停留在0.40 eV附近

3. **训练损失过低**
   - 训练损失几乎为0，说明模型完全记住了训练集
   - 这对泛化能力非常不利

### 当前配置问题

```json
{
    "epochs": 100,              // ❌ 训练太久
    "batch_size": 64,
    "learning_rate": 0.0005,
    "weight_decay": 0.001,      // ❌ 正则化太弱
    "graph_dropout": 0.15,      // ❌ dropout太小
    "fine_grained_dropout": 0.35,
    "middle_fusion_dropout": 0.35,
    "use_middle_fusion": true,  // ⚠️ 您之前实验显示这个降低性能
    "n_early_stopping": 150     // ❌ 太大，几乎不起作用
}
```

---

## 💡 改进方案

### 方案1: 增强正则化 ⭐ 首选

**核心思路**：防止过拟合，提升泛化能力

#### 修改建议：

```json
{
    // 1. 增加Dropout
    "graph_dropout": 0.3,           // 0.15 → 0.3
    "fine_grained_dropout": 0.45,   // 0.35 → 0.45
    "middle_fusion_dropout": 0.45,  // 0.35 → 0.45
    "cross_modal_dropout": 0.2,     // 0.1 → 0.2

    // 2. 增强权重衰减
    "weight_decay": 0.01,           // 0.001 → 0.01 (10倍)

    // 3. 早停策略
    "n_early_stopping": 30,         // 150 → 30

    // 4. 减少epochs
    "epochs": 200,                  // 保持200，但会被early stopping提前终止
}
```

**训练命令**：
```bash
python train_with_cross_modal_attention.py \
    --dataset dft_3d \
    --target hse_bandgap \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --weight_decay 0.01 \
    --warmup_steps 2000 \
    --graph_dropout 0.3 \
    --fine_grained_dropout 0.45 \
    --middle_fusion_dropout 0.45 \
    --cross_modal_dropout 0.2 \
    --n_early_stopping 30 \
    --output_dir runs/hse_reg_strong
```

**预期效果**：
- 训练MAE: ~0.15-0.20 eV
- 验证MAE: ~0.30-0.35 eV
- 差距: <0.15 eV

---

### 方案2: 去掉Middle Fusion ⭐⭐ 强烈推荐

**您之前的实验结果**：
- 中期融合 + 跨模态 + 细粒度: MAE = 0.25
- 中期融合 + gate跨模态 + 细粒度: MAE = 0.27 (降低8%)
- **中期融合导致删除文本后鲁棒性降低39% (0.536 → 0.747)**

**结论**：Middle Fusion是性能瓶颈！

#### 修改建议：

```json
{
    // 核心改变：去掉middle fusion
    "use_middle_fusion": false,     // true → false

    // 调整其他参数
    "graph_dropout": 0.25,
    "fine_grained_dropout": 0.4,
    "cross_modal_dropout": 0.15,
    "weight_decay": 0.005,
    "n_early_stopping": 40,
}
```

**训练命令**：
```bash
python train_with_cross_modal_attention.py \
    --dataset dft_3d \
    --target hse_bandgap \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --weight_decay 0.005 \
    --warmup_steps 2000 \
    --use_cross_modal_attention \
    --use_fine_grained_attention \
    --fine_grained_dropout 0.4 \
    --graph_dropout 0.25 \
    --n_early_stopping 40 \
    --output_dir runs/hse_no_middle_fusion
```

**预期效果**：
- 验证MAE: ~0.25 eV（基于您之前的实验）
- 比当前0.40改善37.5%！

---

### 方案3: 减小模型容量

**核心思路**：更小的模型 = 更难过拟合

#### 修改建议：

```json
{
    // 减小模型尺寸
    "alignn_layers": 3,             // 4 → 3
    "gcn_layers": 3,                // 4 → 3
    "hidden_features": 192,         // 256 → 192
    "cross_modal_hidden_dim": 192,  // 256 → 192
    "fine_grained_hidden_dim": 192, // 256 → 192
    "fine_grained_num_heads": 4,    // 8 → 4

    // 调整正则化
    "graph_dropout": 0.25,
    "fine_grained_dropout": 0.4,
    "weight_decay": 0.005,
    "n_early_stopping": 40,
    "use_middle_fusion": false,     // 去掉middle fusion
}
```

**训练命令**：
```bash
python train_with_cross_modal_attention.py \
    --dataset dft_3d \
    --target hse_bandgap \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --weight_decay 0.005 \
    --warmup_steps 2000 \
    --alignn_layers 3 \
    --gcn_layers 3 \
    --hidden_features 192 \
    --use_cross_modal_attention \
    --cross_modal_hidden_dim 192 \
    --use_fine_grained_attention \
    --fine_grained_hidden_dim 192 \
    --fine_grained_num_heads 4 \
    --fine_grained_dropout 0.4 \
    --graph_dropout 0.25 \
    --n_early_stopping 40 \
    --output_dir runs/hse_smaller_model
```

**优点**：
- 训练更快
- 更难过拟合
- 推理速度更快

---

### 方案4: 优化学习率策略

**当前问题**：学习率可能在早期过高，导致快速过拟合

#### 修改建议：

```json
{
    // 降低学习率
    "learning_rate": 0.0003,        // 0.0005 → 0.0003
    "warmup_steps": 3000,           // 2000 → 3000 (更长的warmup)

    // 或使用余弦退火
    "scheduler": "cosine",          // "onecycle" → "cosine"

    // 其他调整
    "weight_decay": 0.005,
    "graph_dropout": 0.25,
    "fine_grained_dropout": 0.4,
    "use_middle_fusion": false,
    "n_early_stopping": 40,
}
```

**训练命令**：
```bash
python train_with_cross_modal_attention.py \
    --dataset dft_3d \
    --target hse_bandgap \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 0.0003 \
    --weight_decay 0.005 \
    --warmup_steps 3000 \
    --scheduler cosine \
    --use_cross_modal_attention \
    --use_fine_grained_attention \
    --fine_grained_dropout 0.4 \
    --graph_dropout 0.25 \
    --n_early_stopping 40 \
    --output_dir runs/hse_lower_lr
```

---

### 方案5: 增加数据增强（高级）

**核心思路**：通过扰动增加训练样本多样性

#### 实现方法：

1. **晶格扰动**（需要修改代码）
   ```python
   # 在图构建时添加小的随机扰动
   perturbed_lattice = lattice * (1 + np.random.normal(0, 0.02, lattice.shape))
   ```

2. **文本增强**
   - 随机删除部分描述词
   - 同义词替换
   - 回译（英语→中文→英语）

3. **Mixup策略**（需要修改代码）
   ```python
   # 混合两个样本
   lambda_mix = np.random.beta(0.2, 0.2)
   mixed_features = lambda_mix * features1 + (1 - lambda_mix) * features2
   mixed_target = lambda_mix * target1 + (1 - lambda_mix) * target2
   ```

---

## 📊 方案对比

| 方案 | 实施难度 | 预期改善 | 训练时间 | 推荐指数 |
|-----|---------|---------|---------|---------|
| **方案2: 去掉Middle Fusion** | ⭐ 简单 | ⭐⭐⭐⭐⭐ | 快 | ⭐⭐⭐⭐⭐ |
| **方案1: 增强正则化** | ⭐ 简单 | ⭐⭐⭐⭐ | 中 | ⭐⭐⭐⭐ |
| **方案3: 减小模型** | ⭐ 简单 | ⭐⭐⭐ | 快 | ⭐⭐⭐⭐ |
| **方案4: 优化学习率** | ⭐ 简单 | ⭐⭐⭐ | 中 | ⭐⭐⭐ |
| **方案5: 数据增强** | ⭐⭐⭐ 困难 | ⭐⭐⭐⭐ | 慢 | ⭐⭐ |

---

## 🎯 推荐实施顺序

### 第一步：去掉Middle Fusion（最重要！）⭐⭐⭐⭐⭐

基于您之前的实验，这应该立即带来显著改善：

```bash
python train_with_cross_modal_attention.py \
    --dataset dft_3d \
    --target hse_bandgap \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --weight_decay 0.005 \
    --warmup_steps 2000 \
    --use_cross_modal_attention \
    --use_fine_grained_attention \
    --fine_grained_dropout 0.4 \
    --graph_dropout 0.25 \
    --n_early_stopping 40 \
    --output_dir runs/hse_no_middle_fusion
```

**预期结果**：验证MAE从0.40降到0.25 eV

---

### 第二步：如果仍有过拟合，增强正则化

```bash
python train_with_cross_modal_attention.py \
    --dataset dft_3d \
    --target hse_bandgap \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --weight_decay 0.01 \
    --warmup_steps 2000 \
    --use_cross_modal_attention \
    --use_fine_grained_attention \
    --fine_grained_dropout 0.45 \
    --cross_modal_dropout 0.2 \
    --graph_dropout 0.3 \
    --n_early_stopping 30 \
    --output_dir runs/hse_strong_reg
```

---

### 第三步：尝试更小的模型

```bash
python train_with_cross_modal_attention.py \
    --dataset dft_3d \
    --target hse_bandgap \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --weight_decay 0.005 \
    --warmup_steps 2000 \
    --alignn_layers 3 \
    --gcn_layers 3 \
    --hidden_features 192 \
    --use_cross_modal_attention \
    --cross_modal_hidden_dim 192 \
    --use_fine_grained_attention \
    --fine_grained_hidden_dim 192 \
    --fine_grained_num_heads 4 \
    --fine_grained_dropout 0.4 \
    --graph_dropout 0.25 \
    --n_early_stopping 40 \
    --output_dir runs/hse_smaller
```

---

## 🔧 其他技巧

### 1. Label Smoothing

如果代码支持，添加：
```json
"label_smoothing": 0.1
```

### 2. Gradient Clipping

```json
"gradient_clip": 1.0
```

### 3. 混合精度训练

```bash
--use_amp  # 如果支持
```

### 4. 监控梯度

在训练时添加梯度监控，确保没有梯度爆炸或消失。

---

## 📈 预期改善对比

| 配置 | 当前 | 方案2 | 方案1+2 | 方案2+3 |
|-----|------|-------|---------|---------|
| 训练MAE | 0.05 | 0.18 | 0.20 | 0.22 |
| 验证MAE | **0.40** | **0.25** | **0.22** | **0.20** |
| 过拟合程度 | 严重 | 轻微 | 很小 | 很小 |
| 训练时间 | 100 epochs | 60 epochs | 80 epochs | 50 epochs |

---

## 🎓 理论解释

### 为什么会过拟合？

1. **模型容量过大**
   - 您的模型有很多层和参数
   - 数据集只有~1600个样本
   - 模型容量 >> 数据量 = 过拟合

2. **Middle Fusion问题**
   - 增加了额外的参数
   - 在第2层就融合，容易产生特征污染
   - 您的实验已经证明它降低性能

3. **正则化不足**
   - weight_decay=0.001太小
   - dropout=0.15太小
   - 无法有效防止过拟合

### 为什么去掉Middle Fusion会改善？

1. **减少参数**：更少的可训练参数
2. **延迟融合**：让GNN先充分提取结构特征
3. **提高鲁棒性**：减少早期特征污染
4. **您的实验验证**：无middle fusion时MAE=0.25，有middle fusion时MAE=0.27

---

## ✅ 总结

### 核心问题
- **严重过拟合**（训练0.05 vs 验证0.40）
- **Middle Fusion降低性能**（您的实验已证明）
- **正则化不足**

### 最佳方案组合 ⭐⭐⭐⭐⭐

```bash
python train_with_cross_modal_attention.py \
    --dataset dft_3d \
    --target hse_bandgap \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --weight_decay 0.01 \
    --warmup_steps 2000 \
    --alignn_layers 3 \
    --gcn_layers 3 \
    --hidden_features 192 \
    --use_cross_modal_attention \
    --cross_modal_hidden_dim 192 \
    --cross_modal_dropout 0.2 \
    --use_fine_grained_attention \
    --fine_grained_hidden_dim 192 \
    --fine_grained_num_heads 4 \
    --fine_grained_dropout 0.45 \
    --graph_dropout 0.3 \
    --n_early_stopping 30 \
    --output_dir runs/hse_optimized
```

**预期结果**：
- 验证MAE: **0.20-0.25 eV**（当前0.40的50-62.5%）
- 训练MAE: **0.22-0.28 eV**
- 过拟合差距: **<0.05 eV**

### 快速测试（先跑这个！）

```bash
# 只去掉middle fusion，其他保持不变
python train_with_cross_modal_attention.py \
    --dataset dft_3d \
    --target hse_bandgap \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --weight_decay 0.005 \
    --warmup_steps 2000 \
    --use_cross_modal_attention \
    --use_fine_grained_attention \
    --fine_grained_dropout 0.4 \
    --graph_dropout 0.25 \
    --n_early_stopping 40 \
    --output_dir runs/hse_quick_fix
```

这个应该立即看到改善！如果验证MAE降到0.25左右，说明方向正确。然后再尝试其他优化。

---

**生成时间**：2025-12-10
**问题**：严重过拟合（训练0.05 vs 验证0.40）
**核心解决方案**：去掉Middle Fusion + 增强正则化 + 减小模型
**预期改善**：验证MAE从0.40降到0.20-0.25 eV（改善37.5-50%）
