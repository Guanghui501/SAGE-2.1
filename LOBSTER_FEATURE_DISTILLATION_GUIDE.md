# LOBSTER特征蒸馏完整指南

**策略4：训练LOBSTER预测器，为所有JARVIS数据生成伪特征**

---

## 🎯 方案概述

### 核心思路

训练一个GNN模型学习从晶体结构预测ICOHP和ICOBI，然后用它为所有40,000个JARVIS样本生成伪LOBSTER特征。

```
Phase 1: 训练LOBSTER预测器
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入：晶体结构（重叠样本 ~500-1000个）
监督：真实LOBSTER特征（ICOHP, ICOBI）
输出：训练好的预测器模型

Phase 2: 生成伪特征
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入：所有JARVIS晶体结构（40,000个）
模型：训练好的LOBSTER预测器
输出：伪LOBSTER特征数据库

Phase 3: 主模型训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入：JARVIS数据 + 伪LOBSTER特征
模型：ALIGNN + 增强的边特征
输出：改进的性质预测
```

---

## 📊 预期效果

| 指标 | 估计值 | 说明 |
|-----|--------|------|
| **预测器训练数据** | 500-1000样本 | 重叠样本数 |
| **ICOHP预测MAE** | 0.3-0.5 eV | 基于论文经验 |
| **ICOBI预测MAE** | 0.05-0.1 | Bond index范围[0,1] |
| **覆盖率** | 100% | 所有JARVIS数据 |
| **主模型MAE改善** | 5-10% | 预期提升 |

**与其他策略对比**：

| 策略 | 覆盖率 | 特征质量 | 改善幅度 | 实施难度 |
|-----|--------|---------|---------|---------|
| 策略1: 辅助特征 | 2.5% | 100%真实 | 2-5% | 低 |
| 策略2: 子任务模型 | 30% | 100%真实 | 10-15% | 中 |
| **策略4: 特征蒸馏** | **100%** | **60-70%伪** | **5-10%** | 高 |

---

## 🏗️ 架构设计

### 1. LOBSTER预测器架构

```
输入：晶体结构图 (DGL Graph)
   ├─ 节点：原子 (CGCNN 92-dim特征)
   ├─ 边：化学键 (RBF展开的距离)
   └─ 全局：晶格参数

编码器：
   ├─ 原子特征嵌入 (92 → 256)
   ├─ 边特征编码 (atom_i + atom_j + distance → 128)
   └─ GNN传播 (4层 EdgeGatedGraphConv)

预测头：
   ├─ ICOHP预测器
   │  ├─ MLP (edge_feat + node_pair → 64 → 2)
   │  └─ 输出: [mean, log_std]  # 带不确定性
   │
   └─ ICOBI预测器
      ├─ MLP (edge_feat + node_pair → 64 → 1)
      └─ 输出: [0, 1]  # Sigmoid激活

输出：每条边的 [ICOHP, ICOBI, Uncertainty]
```

### 2. 关键创新点

#### A. 不确定性量化

```python
# 同时预测均值和标准差
output = model(graph)
icohp_mean = output[:, 0]
icohp_log_std = output[:, 1]
icohp_std = exp(icohp_log_std)

# 训练时使用负对数似然
loss = log(std) + (pred - target)^2 / (2 * std^2)
```

**好处**：
- ✅ 可以识别不确定的预测
- ✅ 在主模型中可以降低低置信度特征的权重

#### B. 多任务学习

```python
# 同时学习ICOHP和ICOBI
loss_total = loss_icohp + loss_icobi

# 共享底层编码器（减少参数）
shared_encoder → [icohp_head, icobi_head]
```

**好处**：
- ✅ 两个任务相互促进
- ✅ 参数共享，更高效

#### C. 全局特征提取

```python
# 除了边级特征，还提取全局统计
features = {
    # 边级（用于GNN）
    'icohp_edges': [num_edges],
    'icobi_edges': [num_edges],

    # 全局（用于特征工程）
    'icohp_mean': float,
    'icohp_min': float,  # 最强键！
    'icohp_std': float,
    'num_bonds': int
}
```

**好处**：
- ✅ 论文证明 `icohp_min` 是最重要特征
- ✅ 可用于随机森林等baseline

---

## 🛠️ 实施流程（3周）

### Week 1: 准备和训练预测器

#### Day 1-2: 数据准备

```bash
# 1. 对齐JARVIS和Materials Project样本
python utils/mp_jarvis_alignment.py \
    --lobster_dir data/lobster_database \
    --jarvis_dataset dft_3d \
    --output data/jarvis_mp_overlap.json

# 预期输出：
# 找到 XXX 个重叠样本
# 保存映射到 data/jarvis_mp_overlap.json

# 2. 验证LOBSTER数据质量
python utils/validate_lobster_data.py \
    --lobster_dir data/lobster_database \
    --overlap_map data/jarvis_mp_overlap.json
```

**检查点**：
- [ ] 重叠样本 > 300个（否则考虑策略2）
- [ ] LOBSTER数据完整（无损坏的JSON）
- [ ] 特征范围合理（ICOHP: -6~+2, ICOBI: 0~1）

#### Day 3-5: 训练预测器

```bash
# 训练命令
python train_lobster_predictor.py \
    --lobster_dir data/lobster_database \
    --overlap_map data/jarvis_mp_overlap.json \
    --dataset dft_3d \
    --atom_features cgcnn \
    --edge_hidden_dim 128 \
    --graph_hidden_dim 256 \
    --num_layers 4 \
    --dropout 0.1 \
    --shared_encoder \
    --epochs 200 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --output_dir models/lobster_predictor
```

**训练监控**：

```python
# 在TensorBoard中查看
tensorboard --logdir models/lobster_predictor/logs

# 关键指标：
# - ICOHP MAE: 期望 < 0.5 eV
# - ICOBI MAE: 期望 < 0.1
# - ICOHP相关系数: 期望 > 0.7
# - ICOBI相关系数: 期望 > 0.8
```

**检查点**：
- [ ] ICOHP MAE < 0.5 eV
- [ ] ICOHP相关系数 > 0.7
- [ ] 训练收敛（loss不再下降）

#### Day 6-7: 验证预测质量

```python
# 验证脚本
python validate_lobster_predictor.py \
    --model_path models/lobster_predictor/best_model.pt \
    --lobster_dir data/lobster_database \
    --overlap_map data/jarvis_mp_overlap.json \
    --n_samples 100

# 输出：
# 1. 预测vs真实的散点图
# 2. 残差分析
# 3. 不确定性校准曲线
# 4. 案例分析（最好/最差预测）
```

**质量标准**：
- [ ] 散点图R² > 0.5
- [ ] 残差无系统性偏差
- [ ] 不确定性与误差相关（高不确定性 → 大误差）

---

### Week 2: 生成伪特征

#### Day 8-10: 批量生成特征

```bash
# 为所有JARVIS数据生成伪LOBSTER特征
python generate_pseudo_lobster_features.py \
    --model_path models/lobster_predictor/best_model.pt \
    --dataset dft_3d \
    --atom_features cgcnn \
    --output_file data/pseudo_lobster_features.pkl \
    --batch_size 32 \
    --return_uncertainty

# 时间估算：
# 40,000样本 / 32 batch / 2秒 ≈ 40分钟（GPU）
```

**输出文件结构**：

```python
# data/pseudo_lobster_features.pkl
{
    'jid-123': {
        # 边级特征
        'icohp_mean': array([num_edges]),  # 每条边的ICOHP
        'icohp_std': array([num_edges]),   # 不确定性
        'icobi': array([num_edges]),       # 每条边的ICOBI

        # 全局特征
        'icohp_global_mean': float,
        'icohp_global_min': float,  # 最强键
        'icohp_global_std': float,
        'num_bonds': int,
        'icobi_mean': float
    },
    'jid-456': {...},
    ...
}
```

#### Day 11-12: 质量控制

```python
# 分析生成的伪特征
python analyze_pseudo_features.py \
    --pseudo_features data/pseudo_lobster_features.pkl \
    --true_lobster_dir data/lobster_database \
    --overlap_map data/jarvis_mp_overlap.json

# 检查：
# 1. 特征分布是否合理？
# 2. 与真实LOBSTER的差异？
# 3. 不确定性分布？
# 4. 异常样本识别
```

**质量检查**：
- [ ] ICOHP分布：均值≈-2, 范围[-6, +2]
- [ ] ICOBI分布：均值≈0.3, 范围[0, 1]
- [ ] 无NaN或Inf值
- [ ] 不确定性合理（不全是0或1）

#### Day 13-14: 特征筛选（可选）

```python
# 根据不确定性筛选高质量特征
python filter_pseudo_features.py \
    --input data/pseudo_lobster_features.pkl \
    --output data/pseudo_lobster_features_filtered.pkl \
    --uncertainty_threshold 0.5  # 只保留不确定性<0.5的

# 或者：加权使用
# 高不确定性特征 → 降低权重
# 低不确定性特征 → 保持权重
```

---

### Week 3: 集成到主模型

#### Day 15-17: 修改主模型

```python
# 修改 data.py 加载伪特征
class StructureDatasetWithPseudoLobster(StructureDataset):
    def __init__(self, ..., pseudo_lobster_file):
        super().__init__(...)

        # 加载伪特征
        with open(pseudo_lobster_file, 'rb') as f:
            self.pseudo_lobster = pickle.load(f)

    def __getitem__(self, idx):
        g, text, label = super().__getitem__(idx)

        # 添加伪LOBSTER边特征
        jid = self.ids[idx]
        if jid in self.pseudo_lobster:
            pseudo_feat = self.pseudo_lobster[jid]

            # 添加到边特征
            g.edata['lobster'] = torch.FloatTensor([
                pseudo_feat['icohp_mean'],
                pseudo_feat['icobi'],
                pseudo_feat['icohp_std']  # 不确定性
            ]).T  # [num_edges, 3]
        else:
            g.edata['lobster'] = torch.zeros(g.num_edges(), 3)

        return g, text, label

# 修改 models/alignn.py 使用伪特征
class ALIGNNWithPseudoLobster(ALIGNN):
    def __init__(self, config):
        super().__init__(config)

        # 边特征编码器
        self.edge_rbf = RBFExpansion(...)  # 80-dim
        self.lobster_encoder = nn.Linear(3, 64)  # 伪LOBSTER特征

        # 不确定性加权
        self.uncertainty_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, g, lg, text):
        # RBF特征
        rbf_feat = self.edge_rbf(g.edata['r'])

        # 伪LOBSTER特征
        lobster_raw = g.edata['lobster']
        icohp = lobster_raw[:, 0:1]
        icobi = lobster_raw[:, 1:2]
        uncertainty = lobster_raw[:, 2:3]

        # 不确定性加权
        # 高不确定性 → 低权重
        conf_weight = 1.0 - self.uncertainty_gate(uncertainty)

        lobster_feat = self.lobster_encoder(lobster_raw[:, 0:2])
        lobster_feat = lobster_feat * conf_weight  # 加权

        # 融合边特征
        edge_feat = torch.cat([rbf_feat, lobster_feat], dim=-1)

        # ... 后续ALIGNN层
```

#### Day 18-20: 训练和验证

```bash
# 训练主模型（带伪LOBSTER特征）
python train_with_cross_modal_attention.py \
    --config config_with_pseudo_lobster.json \
    --pseudo_lobster_file data/pseudo_lobster_features.pkl \
    --output_dir runs/pseudo_lobster_augmented

# 对比实验
# Baseline: 无LOBSTER特征
# Experiment: 有伪LOBSTER特征

# 预期改善：
# - 干净文本MAE: 0.25 → 0.24 (4%改善)
# - 删除文本MAE: 0.75 → 0.70 (7%改善)
```

#### Day 21: 分析和报告

```python
# 分析哪些样本受益最多
python analyze_pseudo_lobster_impact.py \
    --baseline_model runs/baseline/best_model.pt \
    --augmented_model runs/pseudo_lobster_augmented/best_model.pt \
    --test_loader test_loader

# 输出：
# 1. 改善最大的样本（top 100）
# 2. 无改善或变差的样本
# 3. 特征重要性分析（伪LOBSTER vs 其他特征）
# 4. 不确定性vs改善的关系
```

---

## 📊 质量验证

### 1. 预测器质量指标

| 指标 | 目标值 | 说明 |
|-----|--------|------|
| **ICOHP MAE** | < 0.5 eV | 平均绝对误差 |
| **ICOHP R²** | > 0.5 | 预测vs真实相关性 |
| **ICOBI MAE** | < 0.1 | Bond index误差 |
| **不确定性校准** | 0.7-0.9 | 校准曲线下面积 |

### 2. 伪特征质量检查

```python
def validate_pseudo_features(pseudo_db, true_db, overlap_map):
    """验证伪特征质量"""

    # 对比真实vs伪特征（重叠样本）
    errors = []

    for jid, mp_id in overlap_map.items():
        if jid in pseudo_db and mp_id in true_db:
            pseudo_icohp_mean = pseudo_db[jid]['icohp_global_mean']
            true_icohp_mean = true_db[mp_id].get_global_features()['icohp_mean']

            error = abs(pseudo_icohp_mean - true_icohp_mean)
            errors.append(error)

    avg_error = np.mean(errors)

    print(f"伪特征vs真实特征误差: {avg_error:.3f} eV")

    # 期望：< 0.5 eV
    if avg_error < 0.5:
        print("✅ 伪特征质量良好")
    else:
        print("⚠️ 伪特征质量可能不足，考虑：")
        print("   1. 增加训练数据")
        print("   2. 改进模型架构")
        print("   3. 只使用高置信度样本")

    return avg_error
```

### 3. 主模型性能评估

```python
# A/B测试
baseline_mae = evaluate(baseline_model, test_loader)
augmented_mae = evaluate(augmented_model, test_loader)

improvement = (baseline_mae - augmented_mae) / baseline_mae * 100

print(f"Baseline MAE: {baseline_mae:.4f}")
print(f"Augmented MAE: {augmented_mae:.4f}")
print(f"Improvement: {improvement:.1f}%")

# 期望：5-10%改善
if improvement >= 5:
    print("✅ 伪LOBSTER特征有效！")
elif improvement >= 2:
    print("⚠️ 改善有限，考虑：")
    print("   1. 提高预测器质量")
    print("   2. 优化特征融合方式")
    print("   3. 增加特征工程")
else:
    print("❌ 几乎无改善，可能原因：")
    print("   1. 预测器质量不足")
    print("   2. 伪特征噪声太大")
    print("   3. 主模型未能利用新特征")
```

---

## ⚠️ 风险和缓解

### 风险1: 预测器质量不足

**症状**：
- ICOHP MAE > 0.8 eV
- 相关系数 < 0.5

**原因**：
- 训练数据太少（< 300样本）
- 模型欠拟合或过拟合
- 特征表达能力不足

**缓解**：
1. **数据增强**
   ```python
   # 旋转、扰动结构
   from jarvis.core.atoms import Atoms

   def augment_structure(atoms, sigma=0.1):
       """添加小幅随机扰动"""
       coords = atoms.cart_coords
       noise = np.random.randn(*coords.shape) * sigma
       new_coords = coords + noise
       return Atoms(lattice=atoms.lattice, coords=new_coords,
                    elements=atoms.elements)
   ```

2. **迁移学习**
   ```python
   # 先在大规模分子数据上预训练
   # 然后在LOBSTER数据上fine-tune
   ```

3. **集成学习**
   ```python
   # 训练多个预测器，取平均
   ensemble = [ICOHPPredictor(...) for _ in range(5)]
   icohp_pred = torch.mean([m(g) for m in ensemble], dim=0)
   ```

---

### 风险2: 伪特征引入噪声

**症状**：
- 主模型性能下降
- 高不确定性样本很多

**原因**：
- 预测器在某些类型材料上失效
- 外推到训练分布之外

**缓解**：
1. **质量过滤**
   ```python
   # 只使用低不确定性样本
   if uncertainty < threshold:
       use_pseudo_feature()
   else:
       use_zero_feature()  # 回退
   ```

2. **软加权**
   ```python
   # 根据不确定性调整权重
   weight = sigmoid(-(uncertainty - threshold) / scale)
   feature_weighted = weight * pseudo_feature
   ```

3. **域自适应**
   ```python
   # 检测测试样本是否在分布内
   if is_in_distribution(sample):
       use_pseudo_feature()
   else:
       use_baseline_feature()
   ```

---

### 风险3: 计算资源不足

**症状**：
- 特征生成时间 > 2小时
- GPU内存不足

**缓解**：
1. **批处理优化**
   ```python
   # 增大batch size（如果GPU允许）
   batch_size = 64  # 从32增加到64
   ```

2. **混合精度**
   ```python
   from torch.cuda.amp import autocast

   with autocast():
       icohp_pred = model(g)
   ```

3. **分阶段处理**
   ```python
   # 分成多个子集，逐个处理
   for subset in split_dataset(jarvis_db, n_splits=10):
       generate_features(subset)
       save_checkpoint()
   ```

---

## 📈 预期成果

### 定量指标

| 阶段 | 指标 | 目标值 |
|-----|------|--------|
| **Phase 1: 预测器** | ICOHP MAE | < 0.5 eV |
| | ICOBI MAE | < 0.1 |
| | 训练时间 | < 2天 |
| **Phase 2: 特征生成** | 覆盖率 | 100% |
| | 生成时间 | < 1小时 |
| | 平均不确定性 | < 0.4 |
| **Phase 3: 主模型** | MAE改善 | 5-10% |
| | 训练时间 | ≈ baseline |

### 定性收获

1. **科学洞察**
   - 理解哪些材料的化学键容易预测
   - 发现结构-键合的规律

2. **方法论贡献**
   - 特征蒸馏的通用方法
   - 不确定性量化的应用

3. **可发表成果**
   - 伪特征生成方法
   - 多模态学习改进
   - 化学可解释性分析

---

## 💻 完整代码示例

### 端到端流程

```bash
#!/bin/bash
# 完整的LOBSTER特征蒸馏流程

echo "=========================================="
echo "LOBSTER特征蒸馏流程"
echo "=========================================="

# Step 1: 数据对齐
echo "\n[Step 1] 对齐JARVIS和Materials Project样本..."
python utils/mp_jarvis_alignment.py \
    --lobster_dir data/lobster_database \
    --jarvis_dataset dft_3d \
    --output data/jarvis_mp_overlap.json

# Step 2: 训练LOBSTER预测器
echo "\n[Step 2] 训练LOBSTER预测器..."
python train_lobster_predictor.py \
    --lobster_dir data/lobster_database \
    --overlap_map data/jarvis_mp_overlap.json \
    --epochs 200 \
    --batch_size 32 \
    --output_dir models/lobster_predictor

# Step 3: 验证预测器
echo "\n[Step 3] 验证预测器质量..."
python validate_lobster_predictor.py \
    --model_path models/lobster_predictor/best_model.pt \
    --lobster_dir data/lobster_database \
    --overlap_map data/jarvis_mp_overlap.json

# Step 4: 生成伪特征
echo "\n[Step 4] 为所有JARVIS数据生成伪LOBSTER特征..."
python generate_pseudo_lobster_features.py \
    --model_path models/lobster_predictor/best_model.pt \
    --dataset dft_3d \
    --output_file data/pseudo_lobster_features.pkl \
    --return_uncertainty

# Step 5: 质量检查
echo "\n[Step 5] 检查伪特征质量..."
python analyze_pseudo_features.py \
    --pseudo_features data/pseudo_lobster_features.pkl \
    --true_lobster_dir data/lobster_database

# Step 6: 训练主模型
echo "\n[Step 6] 训练增强的主模型..."
python train_with_cross_modal_attention.py \
    --config config_with_pseudo_lobster.json \
    --pseudo_lobster_file data/pseudo_lobster_features.pkl \
    --output_dir runs/pseudo_lobster_augmented

# Step 7: 评估改善
echo "\n[Step 7] 评估性能改善..."
python evaluate_improvement.py \
    --baseline_dir runs/baseline \
    --augmented_dir runs/pseudo_lobster_augmented

echo "\n=========================================="
echo "✅ 流程完成！"
echo "=========================================="
```

---

## 📚 参考资料

### 论文

1. **LOBSTER数据库论文**
   - "Quantum-Chemical Bonding Database for Solid-State Materials"
   - 证明了ICOHP特征在声子频率预测中的重要性

2. **知识蒸馏**
   - "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)
   - 特征蒸馏的理论基础

3. **不确定性量化**
   - "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" (Kendall & Gal, 2017)
   - 预测不确定性的方法

### 代码实现

- **已创建文件**：
  1. `models/lobster_predictor.py` - 预测器模型
  2. `train_lobster_predictor.py` - 训练脚本
  3. `generate_pseudo_lobster_features.py` - 特征生成脚本

- **需要补充**：
  1. `validate_lobster_predictor.py` - 验证脚本
  2. `analyze_pseudo_features.py` - 特征分析
  3. `evaluate_improvement.py` - 性能评估

---

## 🎯 成功标准

### 最低标准（必须达到）
- [ ] LOBSTER预测器ICOHP MAE < 0.8 eV
- [ ] 成功为所有JARVIS数据生成特征
- [ ] 主模型性能不下降

### 目标标准（期望达到）
- [ ] LOBSTER预测器ICOHP MAE < 0.5 eV
- [ ] ICOHP相关系数 > 0.7
- [ ] 主模型MAE改善 > 5%

### 优秀标准（超出期望）
- [ ] LOBSTER预测器ICOHP MAE < 0.3 eV
- [ ] ICOHP相关系数 > 0.8
- [ ] 主模型MAE改善 > 10%
- [ ] 发表论文级别的分析

---

**文档生成时间**：2025-12-10
**预计总耗时**：3周
**推荐起始条件**：重叠样本 > 500个
**预期改善**：主模型MAE ↓ 5-10%

祝实验顺利！🚀
