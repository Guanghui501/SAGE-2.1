# 🚀 Improved Middle Fusion Module Guide

## 概述

已成功将 **LayerNorm** 和 **可学习缩放因子** 添加到 MiddleFusionModule，用于解决文本特征过弱和特征尺度不平衡的问题。

---

## 新增功能

### 1. **Gate LayerNorm** (`use_gate_norm`)

**作用**: 在 Gate 网络之前对输入进行归一化，平衡节点特征和文本特征的尺度。

**原理**:
```python
gate_input = torch.cat([node_feat, text_broadcasted], dim=-1)  # [N, 512]
if use_gate_norm:
    gate_input = self.gate_norm(gate_input)  # 归一化到相同尺度
gate_values = self.gate(gate_input)
```

**优势**:
- 消除特征尺度差异（如 27.43:2.17 的不平衡）
- 使 Gate 网络能够平等考虑两种特征
- 稳定训练过程

---

### 2. **可学习缩放因子** (`use_learnable_scale`)

**作用**: 自动学习最优的文本特征缩放比例，无需手动调整。

**原理**:
```python
text_transformed = self.text_transform(text_feat)  # [B, 256]
text_transformed = text_transformed * self.text_scale  # 可学习缩放
```

**优势**:
- 模型自动找到最优缩放比例
- 从诊断结果的 12.0 开始，训练中自适应调整
- 不同层可以学习不同的缩放比例

---

### 3. **初始缩放值** (`initial_scale`)

**作用**: 设置可学习缩放因子的初始值。

**建议值**:
- 基于诊断结果：`12.0`（已验证能将特征比例从 12.63:1 降到 1.05:1）
- 保守值：`8.0-10.0`
- 激进值：`15.0-20.0`

---

## 配置选项

### ALIGNNConfig 新增参数

```python
class ALIGNNConfig(BaseSettings):
    # ... 现有参数 ...

    # Middle fusion settings
    use_middle_fusion: bool = False
    middle_fusion_layers: str = "2"  # 或 "2,3" 用于多层融合
    middle_fusion_hidden_dim: int = 128
    middle_fusion_num_heads: int = 2
    middle_fusion_dropout: float = 0.1

    # === 新增参数 ===
    middle_fusion_use_gate_norm: bool = False  # 启用 Gate LayerNorm
    middle_fusion_use_learnable_scale: bool = False  # 启用可学习缩放
    middle_fusion_initial_scale: float = 1.0  # 缩放初始值
```

---

## 使用方法

### 方案 1：仅使用 LayerNorm（推荐用于快速改进）

**配置文件** (`config.json`):
```json
{
  "name": "alignn",
  "use_middle_fusion": true,
  "middle_fusion_layers": "2",
  "middle_fusion_use_gate_norm": true,
  "middle_fusion_use_learnable_scale": false,
  "middle_fusion_initial_scale": 1.0
}
```

**预期效果**:
- 特征尺度自动平衡
- 余弦相似度：0.12 → 0.35-0.45
- Alpha 标准差：0.029 → 0.045-0.065
- 训练稳定性提升

---

### 方案 2：仅使用可学习缩放（推荐用于最大灵活性）

**配置文件** (`config.json`):
```json
{
  "name": "alignn",
  "use_middle_fusion": true,
  "middle_fusion_layers": "2",
  "middle_fusion_use_gate_norm": false,
  "middle_fusion_use_learnable_scale": true,
  "middle_fusion_initial_scale": 12.0
}
```

**预期效果**:
- 模型自动学习最优缩放
- 初始效果接近手动缩放（12.0x）
- 训练后可能进一步优化
- 余弦相似度：0.12 → 0.45-0.55

---

### 方案 3：同时使用两者（推荐用于最佳效果）⭐

**配置文件** (`config.json`):
```json
{
  "name": "alignn",
  "use_middle_fusion": true,
  "middle_fusion_layers": "2",
  "middle_fusion_use_gate_norm": true,
  "middle_fusion_use_learnable_scale": true,
  "middle_fusion_initial_scale": 12.0
}
```

**预期效果**:
- **最佳特征融合效果**
- LayerNorm 提供稳定的特征尺度
- 可学习缩放提供额外的适应性
- 余弦相似度：0.12 → 0.50-0.65
- Alpha 标准差：0.029 → 0.06-0.10
- Alpha 范围：0.116 → 0.20-0.30

---

### 方案 4：多层融合（实验性）

**配置文件** (`config.json`):
```json
{
  "name": "alignn",
  "use_middle_fusion": true,
  "middle_fusion_layers": "2,3",
  "middle_fusion_use_gate_norm": true,
  "middle_fusion_use_learnable_scale": true,
  "middle_fusion_initial_scale": 12.0
}
```

**说明**:
- 在 Layer 2 和 Layer 3 都注入文本信息
- 每层有独立的 MiddleFusionModule
- 每层学习独立的缩放因子

**预期效果**:
- 文本信息更深入融入图编码
- 可能进一步提升性能
- 计算成本增加（~10-15%）

---

## 训练示例

### 1. 准备配置文件

创建 `config_improved_fusion.json`:

```json
{
  "name": "alignn",
  "alignn_layers": 4,
  "gcn_layers": 4,
  "atom_input_features": 92,
  "edge_input_features": 80,
  "triplet_input_features": 40,
  "embedding_features": 64,
  "hidden_features": 256,
  "output_features": 1,

  "use_middle_fusion": true,
  "middle_fusion_layers": "2",
  "middle_fusion_hidden_dim": 128,
  "middle_fusion_num_heads": 2,
  "middle_fusion_dropout": 0.1,
  "middle_fusion_use_gate_norm": true,
  "middle_fusion_use_learnable_scale": true,
  "middle_fusion_initial_scale": 12.0,

  "use_cross_modal_attention": true,
  "cross_modal_hidden_dim": 256,
  "cross_modal_num_heads": 4,
  "cross_modal_dropout": 0.1
}
```

---

### 2. 训练命令

```bash
python train.py \
    --config config_improved_fusion.json \
    --root_dir <data_root> \
    --dataset jarvis \
    --property hse_bandgap-2 \
    --batch_size 32 \
    --epochs 500 \
    --learning_rate 0.001 \
    --output_dir ./outputs/improved_fusion
```

---

### 3. 监控训练过程

**关键指标**:
1. **text_scale 值的演变**
   - 初始值：12.0
   - 预期稳定范围：10.0-15.0
   - 如果持续增大（>20.0）：文本特征可能仍然太弱
   - 如果持续减小（<5.0）：文本特征可能过强

2. **训练损失曲线**
   - 应比原始模型更平滑
   - 收敛速度可能略快

3. **验证性能**
   - MAE 应持平或略有改善（±5%）
   - 主要改进体现在可解释性，而非预测性能

---

## 训练后分析

### 1. 提取并查看学到的缩放因子

```python
import torch

checkpoint = torch.load('best_model.pt')
model_state = checkpoint['model']

# 查找所有 text_scale 参数
for key, value in model_state.items():
    if 'text_scale' in key:
        print(f"{key}: {value.item():.4f}")

# 示例输出:
# middle_fusion_modules.layer_2.text_scale: 13.2451
```

**解读**:
- 如果 scale ≈ 12.0：初始值已经接近最优
- 如果 scale > 15.0：文本特征在训练中需要更强的权重
- 如果 scale < 8.0：文本特征在训练中被自动调弱

---

### 2. 重新运行诊断工具

```bash
# 诊断融合效果
python diagnose_fusion_effectiveness.py \
    --checkpoint outputs/improved_fusion/best_model.pt \
    --root_dir <data_root>

# 分析文本流
python 3.analyze_text_flow_v2.py \
    --checkpoint outputs/improved_fusion/best_model.pt \
    --root_dir <data_root>
```

**期望结果**:
```
text_transform 输入 L2:  8.30
text_transform 输出 L2:  28-35  ← 已缩放
节点特征 L2:            27-30
节点/文本比例:          0.9:1 - 1.2:1  ← 平衡！
余弦相似度:             0.50-0.65  ← 显著改善！
```

---

### 3. 提取并可视化 Alpha 值

```bash
# 提取 Alpha
python 1.extract_alpha_final.py \
    --checkpoint outputs/improved_fusion/best_model.pt \
    --root_dir <data_root> \
    --dataset jarvis \
    --property hse_bandgap-2 \
    --n_samples 500

# 生成图表
python 2.create_paper_alpha_figures.py
```

**期望改进**:
- Alpha 标准差：0.029 → 0.06-0.10 （提升 100-200%）
- Alpha 范围：0.116 → 0.20-0.30 （提升 70-160%）
- 元素间差异更明显
- 材料间差异更清晰

---

## 对模型性能的预期影响

### 预测性能（MAE / RMSE）

**可能的结果**:
1. **持平**（最常见）：±2%
   - 原因：融合机制改进主要提升可解释性
   - 预测性能已由深层网络和晚期融合保证

2. **小幅提升**（乐观）：3-5%
   - 原因：更好的文本融合帮助模型捕获化学直觉
   - 尤其在训练数据较少的情况下

3. **小幅下降**（罕见）：2-3%
   - 原因：引入更多可学习参数可能导致轻微过拟合
   - 解决方案：调整 dropout 或减少 middle_fusion_hidden_dim

---

### 可解释性（Alpha 多样性）

**显著提升**:
- ✅ Alpha 标准差：+100% 到 +200%
- ✅ 余弦相似度：+300% 到 +400%
- ✅ 元素间差异清晰可见
- ✅ 材料级模式更明显
- ✅ 论文图表质量大幅改善

---

### 训练效率

**影响**:
- 训练时间：+5-10%（由于额外的 LayerNorm 和缩放操作）
- 内存使用：+3-5%（LayerNorm 参数和缓存）
- 收敛速度：可能略快（由于特征尺度平衡）

---

## 调试和优化

### 问题 1：text_scale 训练中爆炸（>50.0）

**原因**:
- text_transform 输出仍然太弱
- 或者学习率过高

**解决方案**:
```python
# 方案 A: 增加初始缩放
middle_fusion_initial_scale: 20.0

# 方案 B: 添加缩放因子的约束
# 在 MiddleFusionModule.__init__ 中添加:
self.text_scale = nn.Parameter(torch.tensor(initial_scale).clamp(1.0, 30.0))

# 方案 C: 降低学习率
learning_rate: 0.0005  # 从 0.001 降低
```

---

### 问题 2：text_scale 训练中快速归零（<0.1）

**原因**:
- 文本信息对当前任务贡献不大
- 或者 text_transform 初始化不当

**解决方案**:
```python
# 方案 A: 增加文本信息的重要性（使用对比学习）
use_contrastive_loss: true
contrastive_loss_weight: 0.1

# 方案 B: 添加最小缩放约束
self.text_scale = nn.Parameter(torch.tensor(initial_scale).clamp(0.1, 30.0))

# 方案 C: 检查数据集是否包含有意义的文本描述
```

---

### 问题 3：余弦相似度仍然很低（<0.3）

**原因**:
- text_transform 架构需要改进
- 或者文本编码器（BERT）未正确加载

**诊断**:
```bash
python diagnose_fusion_effectiveness.py --checkpoint <path>
```

**检查**:
- text_transform 输入 L2 是否正常（~8-10）
- text_transform 输出 L2 是否已放大（~25-35）
- Gate 网络是否正常工作

**解决方案**:
```python
# 如果 text_transform 输出仍然弱：
# 1. 检查 text_transform 的初始化
# 2. 尝试更简单的架构（单层 Linear）
self.text_transform = nn.Linear(text_dim, node_dim)  # 移除 ReLU 和 Dropout

# 3. 使用更大的 initial_scale
middle_fusion_initial_scale: 20.0
```

---

## 与手动缩放的对比

### 手动缩放（scale_checkpoint_weights.py）

**优点**:
- 无需重训练，立即可用
- 精确控制缩放比例
- 适合快速实验和论文可视化

**缺点**:
- 缩放比例固定，无法适应
- 只修复了已训练的模型
- 不能从根本上改善训练过程

---

### 可学习缩放（本方案）

**优点**:
- 模型自动学习最优缩放
- 训练过程中动态调整
- 可能发现比 12.0 更好的值
- 从根本上解决特征不平衡问题

**缺点**:
- 需要重新训练（时间成本）
- 需要监控缩放因子的演变
- 可能需要调整超参数

---

## 推荐工作流程

### 阶段 1：快速验证（使用手动缩放）

```bash
# 1. 应用手动缩放到现有模型
python scale_checkpoint_weights.py \
    --input_checkpoint best_test_model.pt \
    --output_checkpoint best_test_model_scaled_12.0.pt \
    --scale_factor 12.0

# 2. 验证效果
python diagnose_fusion_effectiveness.py \
    --checkpoint best_test_model_scaled_12.0.pt \
    --root_dir <data_root>

# 3. 生成论文图表
python 1.extract_alpha_final.py --checkpoint best_test_model_scaled_12.0.pt ...
python 2.create_paper_alpha_figures.py
```

**时间**: 1-2 小时
**目的**: 确认缩放有效，生成初步结果

---

### 阶段 2：长期优化（使用可学习缩放）

```bash
# 1. 使用改进的配置重新训练
python train.py --config config_improved_fusion.json ...

# 2. 监控 text_scale 演变
# 在训练脚本中添加日志，或使用 TensorBoard

# 3. 训练完成后分析
python diagnose_fusion_effectiveness.py --checkpoint outputs/improved_fusion/best_model.pt ...
python 3.analyze_text_flow_v2.py --checkpoint outputs/improved_fusion/best_model.pt ...

# 4. 生成最终论文图表
python 1.extract_alpha_final.py --checkpoint outputs/improved_fusion/best_model.pt ...
python 2.create_paper_alpha_figures.py
```

**时间**: 1-2 天（取决于训练时间）
**目的**: 获得从头优化的最佳模型

---

## 代码改动总结

### 1. MiddleFusionModule.__init__

**新增参数**:
```python
def __init__(self, node_dim=64, text_dim=64, hidden_dim=128, num_heads=2, dropout=0.1,
             use_gate_norm=False, use_learnable_scale=False, initial_scale=1.0):
```

**新增组件**:
```python
# 可学习缩放因子
if use_learnable_scale:
    self.text_scale = nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32))
else:
    self.register_buffer('text_scale', torch.tensor(1.0, dtype=torch.float32))

# Gate LayerNorm
if use_gate_norm:
    self.gate_norm = nn.LayerNorm(node_dim * 2)
```

---

### 2. MiddleFusionModule.forward

**新增缩放**:
```python
text_transformed = self.text_transform(text_feat)
text_transformed = text_transformed * self.text_scale  # 应用缩放
```

**新增归一化**:
```python
gate_input = torch.cat([node_feat, text_broadcasted], dim=-1)
if self.use_gate_norm:
    gate_input = self.gate_norm(gate_input)  # 归一化
gate_values = self.gate(gate_input)
```

---

### 3. ALIGNNConfig

**新增配置项**:
```python
middle_fusion_use_gate_norm: bool = False
middle_fusion_use_learnable_scale: bool = False
middle_fusion_initial_scale: float = 1.0
```

---

### 4. ALIGNN.__init__

**更新模块创建**:
```python
self.middle_fusion_modules[f'layer_{layer_idx}'] = MiddleFusionModule(
    node_dim=config.hidden_features,
    text_dim=64,
    hidden_dim=config.middle_fusion_hidden_dim,
    num_heads=config.middle_fusion_num_heads,
    dropout=config.middle_fusion_dropout,
    use_gate_norm=config.middle_fusion_use_gate_norm,  # 新增
    use_learnable_scale=config.middle_fusion_use_learnable_scale,  # 新增
    initial_scale=config.middle_fusion_initial_scale  # 新增
)
```

---

## 总结

### ✅ 已完成

1. 添加 LayerNorm 用于特征尺度平衡
2. 添加可学习缩放因子用于自适应调整
3. 更新配置系统支持新参数
4. 保持向后兼容（默认关闭新功能）

---

### 📊 预期效果

**可解释性**（主要目标）:
- ✅ Alpha 多样性提升 100-200%
- ✅ 余弦相似度提升 300-400%
- ✅ 元素级和材料级模式更清晰

**预测性能**（次要目标）:
- ⚠️ 可能持平或小幅提升（±5%）
- ✅ 训练过程更稳定

---

### 🎯 推荐使用场景

1. **论文发表**: 使用手动缩放快速生成图表
2. **模型优化**: 使用可学习缩放重新训练
3. **实验研究**: 尝试不同配置组合
4. **生产部署**: 使用训练好的改进模型

---

### 📚 相关文档

- `FINAL_SUMMARY.md`: 完整的项目总结
- `FIX_TEXT_TRANSFORM_WEAK.md`: 问题分析和解决方案
- `CRITICAL_FINDINGS.md`: 诊断结果
- `diagnose_fusion_effectiveness.py`: 诊断工具
- `scale_checkpoint_weights.py`: 手动缩放工具

---

## 联系和支持

如有问题或需要帮助，请参考现有诊断脚本或查看训练日志。

**祝训练顺利！** 🚀
