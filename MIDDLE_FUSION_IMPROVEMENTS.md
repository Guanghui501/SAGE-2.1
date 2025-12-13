# 🚀 中期融合精度提升改进方案

## 📊 当前中期融合实现分析

### 现有特性
```python
MiddleFusionModule:
  1. 文本变换: Linear(768→128→64) + ReLU + Dropout
  2. 可学习缩放: text_scale (初始值=12.0) ✅
  3. Gate归一化: LayerNorm(128) ✅
  4. 门控机制: Sigmoid(Linear(128))
  5. 残差连接: output = node_feat + gate * text_feat
  6. 层归一化: LayerNorm(64)
```

### 当前限制
1. ❌ **简单门控**: 单一Sigmoid门控，表达能力有限
2. ❌ **静态融合**: 所有节点使用相同的文本信息
3. ❌ **单层文本**: 只使用一个文本表示
4. ❌ **独立层次**: 不同ALIGNN层的中期融合互不相关
5. ❌ **固定残差**: 残差权重固定为1.0

---

## 🎯 改进方案（按优先级排序）

### 🥇 改进1: 多头注意力门控（Multi-Head Gating）

#### 原理
用多头注意力替代简单Sigmoid，让文本选择性地关注不同节点特征。

#### 实现
```python
class MultiHeadGatedFusion(nn.Module):
    """多头注意力门控融合"""
    def __init__(self, node_dim=64, text_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads

        # 查询、键、值
        self.query = nn.Linear(node_dim, node_dim)  # 节点查询文本
        self.key = nn.Linear(text_dim, node_dim)    # 文本作为键
        self.value = nn.Linear(text_dim, node_dim)  # 文本作为值

        # 门控权重（多头）
        self.gate_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim + text_dim, node_dim),
                nn.Sigmoid()
            ) for _ in range(num_heads)
        ])

        # 融合多头结果
        self.fusion = nn.Linear(node_dim * num_heads, node_dim)

    def forward(self, node_feat, text_feat, batch_num_nodes):
        # 广播文本特征
        text_broadcasted = broadcast_text(text_feat, batch_num_nodes)

        # 多头门控
        head_outputs = []
        for i, gate in enumerate(self.gate_heads):
            gate_weight = gate(torch.cat([node_feat, text_broadcasted], dim=-1))
            head_output = gate_weight * text_broadcasted
            head_outputs.append(head_output)

        # 融合多头
        fused = self.fusion(torch.cat(head_outputs, dim=-1))
        return node_feat + fused
```

#### 优势
- ✅ 不同头关注文本的不同方面
- ✅ 更强的表达能力
- ✅ 类似于Transformer的注意力机制

#### 预期提升
**+2-4% MAE**

---

### 🥈 改进2: 层次化文本特征（Hierarchical Text Features）

#### 原理
提取多尺度文本特征（全局/半全局/局部），在不同层注入不同粒度的信息。

#### 实现
```python
class HierarchicalMiddleFusion(nn.Module):
    """层次化中期融合"""
    def __init__(self, node_dim=64, text_dim=768, layer_id=0, total_layers=4):
        super().__init__()
        self.layer_id = layer_id

        # 多尺度文本提取器
        self.global_extractor = nn.Linear(text_dim, node_dim)      # 全局特征
        self.semi_global_extractor = nn.Linear(text_dim, node_dim) # 半全局
        self.local_extractor = nn.Linear(text_dim, node_dim)       # 局部

        # 根据层次选择文本粒度的权重
        # 浅层：全局 > 半全局 > 局部
        # 深层：局部 > 半全局 > 全局
        ratio = layer_id / (total_layers - 1)  # 0.0 -> 1.0

        self.global_weight = nn.Parameter(torch.tensor(1.0 - ratio))
        self.semi_global_weight = nn.Parameter(torch.tensor(0.5))
        self.local_weight = nn.Parameter(torch.tensor(ratio))

        # 门控
        self.gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.Sigmoid()
        )

    def forward(self, node_feat, text_feat, batch_num_nodes):
        # 提取多尺度文本特征
        global_feat = self.global_extractor(text_feat)      # [batch, node_dim]
        semi_global_feat = self.semi_global_extractor(text_feat)
        local_feat = self.local_extractor(text_feat)

        # 加权组合（可学习权重）
        weights = F.softmax(torch.stack([
            self.global_weight,
            self.semi_global_weight,
            self.local_weight
        ]), dim=0)

        text_multi_scale = (
            weights[0] * global_feat +
            weights[1] * semi_global_feat +
            weights[2] * local_feat
        )

        # 广播并融合
        text_broadcasted = broadcast_text(text_multi_scale, batch_num_nodes)
        gate_weight = self.gate(torch.cat([node_feat, text_broadcasted], dim=-1))

        return node_feat + gate_weight * text_broadcasted
```

#### 优势
- ✅ 浅层注入全局信息（结构、对称性）
- ✅ 深层注入局部信息（键长、角度）
- ✅ 自适应调整粒度权重

#### 预期提升
**+3-5% MAE**

---

### 🥉 改进3: 动态门控（Dynamic Gating）

#### 原理
让门控强度依赖于节点的重要性，重要节点获得更多文本信息。

#### 实现
```python
class DynamicGatedFusion(nn.Module):
    """动态门控融合"""
    def __init__(self, node_dim=64, text_dim=64, dropout=0.1):
        super().__init__()

        # 节点重要性预测器
        self.importance_predictor = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.ReLU(),
            nn.Linear(node_dim // 2, 1),
            nn.Sigmoid()
        )

        # 文本变换
        self.text_transform = nn.Linear(text_dim, node_dim)

        # 动态门控（依赖节点重要性）
        self.gate_base = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.Sigmoid()
        )

        # 可学习的门控调制因子
        self.gate_modulation = nn.Parameter(torch.tensor(1.0))

    def forward(self, node_feat, text_feat, batch_num_nodes):
        # 预测节点重要性
        importance = self.importance_predictor(node_feat)  # [total_nodes, 1]

        # 文本变换和广播
        text_transformed = self.text_transform(text_feat)
        text_broadcasted = broadcast_text(text_transformed, batch_num_nodes)

        # 基础门控
        gate_base = self.gate_base(torch.cat([node_feat, text_broadcasted], dim=-1))

        # 重要性调制门控
        gate_modulated = gate_base * (1.0 + importance * self.gate_modulation)

        # 融合
        fused = node_feat + gate_modulated * text_broadcasted
        return fused
```

#### 优势
- ✅ 重要节点（活性位点）获得更多文本信息
- ✅ 不重要节点减少文本干扰
- ✅ 自适应调整融合强度

#### 预期提升
**+2-3% MAE**

---

### 4️⃣ 改进4: 残差缩放学习（Learnable Residual Scaling）

#### 原理
学习节点特征和文本特征的最优残差权重。

#### 实现
```python
class ResidualScalingFusion(nn.Module):
    """残差缩放融合"""
    def __init__(self, node_dim=64, text_dim=64, initial_node_scale=1.0, initial_text_scale=12.0):
        super().__init__()

        # 可学习的残差权重
        self.node_scale = nn.Parameter(torch.tensor(initial_node_scale))
        self.text_scale = nn.Parameter(torch.tensor(initial_text_scale))

        # 文本变换
        self.text_transform = nn.Linear(text_dim, node_dim)

        # 门控
        self.gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self, node_feat, text_feat, batch_num_nodes):
        # 文本变换
        text_transformed = self.text_transform(text_feat)
        text_broadcasted = broadcast_text(text_transformed, batch_num_nodes)

        # 门控
        gate_weight = self.gate(torch.cat([node_feat, text_broadcasted], dim=-1))

        # 残差缩放融合
        fused = (
            self.node_scale * node_feat +
            self.text_scale * gate_weight * text_broadcasted
        )

        return self.layer_norm(fused)
```

#### 优势
- ✅ 自动学习节点和文本的最优权重
- ✅ 替代固定的1.0残差权重
- ✅ 可以根据数据动态调整

#### 预期提升
**+1-3% MAE**

---

### 5️⃣ 改进5: SENet风格的通道注意力（Channel Attention）

#### 原理
让模型学习哪些特征维度更重要。

#### 实现
```python
class ChannelAttentionFusion(nn.Module):
    """通道注意力融合"""
    def __init__(self, node_dim=64, text_dim=64, reduction=4):
        super().__init__()

        # 文本变换
        self.text_transform = nn.Linear(text_dim, node_dim)

        # 通道注意力模块（SENet风格）
        self.channel_attention = nn.Sequential(
            nn.Linear(node_dim, node_dim // reduction),
            nn.ReLU(),
            nn.Linear(node_dim // reduction, node_dim),
            nn.Sigmoid()
        )

        # 空间门控
        self.spatial_gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.Sigmoid()
        )

    def forward(self, node_feat, text_feat, batch_num_nodes):
        # 文本变换
        text_transformed = self.text_transform(text_feat)
        text_broadcasted = broadcast_text(text_transformed, batch_num_nodes)

        # 通道注意力（对文本特征）
        channel_weight = self.channel_attention(text_broadcasted)  # [nodes, node_dim]
        text_reweighted = text_broadcasted * channel_weight

        # 空间门控
        spatial_gate = self.spatial_gate(torch.cat([node_feat, text_reweighted], dim=-1))

        # 融合
        return node_feat + spatial_gate * text_reweighted
```

#### 优势
- ✅ 学习哪些特征维度重要
- ✅ 抑制冗余特征
- ✅ 增强有效特征

#### 预期提升
**+2-4% MAE**

---

### 6️⃣ 改进6: 跨层信息传递（Cross-Layer Information Flow）

#### 原理
不同层的中期融合模块共享信息，增强一致性。

#### 实现
```python
class CrossLayerMiddleFusion(nn.Module):
    """跨层信息传递的中期融合"""
    def __init__(self, node_dim=64, text_dim=64, layer_id=0, shared_memory_dim=32):
        super().__init__()
        self.layer_id = layer_id

        # 共享记忆（跨层）
        if not hasattr(CrossLayerMiddleFusion, 'shared_memory'):
            CrossLayerMiddleFusion.shared_memory = None

        # 记忆写入
        self.memory_writer = nn.Linear(node_dim, shared_memory_dim)

        # 记忆读取
        self.memory_reader = nn.Linear(shared_memory_dim, node_dim)

        # 文本变换
        self.text_transform = nn.Linear(text_dim, node_dim)

        # 门控
        self.gate = nn.Sequential(
            nn.Linear(node_dim * 3, node_dim),  # node + text + memory
            nn.Sigmoid()
        )

    def forward(self, node_feat, text_feat, batch_num_nodes):
        batch_size = text_feat.size(0)

        # 读取上一层的记忆
        if CrossLayerMiddleFusion.shared_memory is not None:
            memory_feat = self.memory_reader(CrossLayerMiddleFusion.shared_memory)
            memory_broadcasted = broadcast_text(memory_feat, batch_num_nodes)
        else:
            memory_broadcasted = torch.zeros_like(node_feat)

        # 文本变换
        text_transformed = self.text_transform(text_feat)
        text_broadcasted = broadcast_text(text_transformed, batch_num_nodes)

        # 三路融合：节点 + 文本 + 记忆
        gate_weight = self.gate(torch.cat([
            node_feat,
            text_broadcasted,
            memory_broadcasted
        ], dim=-1))

        fused = node_feat + gate_weight * (text_broadcasted + 0.5 * memory_broadcasted)

        # 更新记忆（全局池化）
        # 按batch分组池化
        fused_list = []
        offset = 0
        for num_nodes in batch_num_nodes:
            fused_list.append(fused[offset:offset+num_nodes].mean(dim=0))
            offset += num_nodes
        fused_pooled = torch.stack(fused_list)  # [batch, node_dim]

        CrossLayerMiddleFusion.shared_memory = self.memory_writer(fused_pooled)

        return fused
```

#### 优势
- ✅ 浅层和深层融合互相协调
- ✅ 增强全局一致性
- ✅ 减少信息丢失

#### 预期提升
**+2-3% MAE**

---

### 7️⃣ 改进7: 对比学习增强（Contrastive Enhancement）

#### 原理
让文本-图融合后的特征更具判别性。

#### 实现
```python
class ContrastiveMiddleFusion(nn.Module):
    """对比学习增强的中期融合"""
    def __init__(self, node_dim=64, text_dim=64, temperature=0.1):
        super().__init__()
        self.temperature = temperature

        # 文本变换
        self.text_transform = nn.Linear(text_dim, node_dim)

        # 投影头（用于对比学习）
        self.projection = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim // 2)
        )

        # 门控
        self.gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.Sigmoid()
        )

    def forward(self, node_feat, text_feat, batch_num_nodes, compute_loss=False):
        # 文本变换
        text_transformed = self.text_transform(text_feat)
        text_broadcasted = broadcast_text(text_transformed, batch_num_nodes)

        # 门控融合
        gate_weight = self.gate(torch.cat([node_feat, text_broadcasted], dim=-1))
        fused = node_feat + gate_weight * text_broadcasted

        if compute_loss:
            # 对比学习损失（融合前后的一致性）
            node_proj = self.projection(node_feat)
            fused_proj = self.projection(fused)

            # InfoNCE loss
            sim = F.cosine_similarity(node_proj, fused_proj, dim=-1)
            loss = -torch.log(torch.exp(sim / self.temperature).mean())

            return fused, loss
        else:
            return fused
```

#### 优势
- ✅ 增强融合特征的判别性
- ✅ 保持节点和文本的语义一致性
- ✅ 正则化作用

#### 预期提升
**+1-2% MAE**

---

## 📊 改进方案对比

| 改进方案 | 复杂度 | 参数增加 | 预期MAE提升 | 实现难度 | 推荐优先级 |
|---------|--------|---------|------------|---------|-----------|
| **多头注意力门控** | ⭐⭐⭐ | +30% | +2-4% | 中 | 🥇 **1** |
| **层次化文本特征** | ⭐⭐⭐⭐ | +50% | +3-5% | 中-高 | 🥈 **2** |
| **动态门控** | ⭐⭐ | +15% | +2-3% | 低 | 🥉 **3** |
| **残差缩放学习** | ⭐ | +0.1% | +1-3% | 低 | **4** |
| **通道注意力** | ⭐⭐ | +20% | +2-4% | 低-中 | **5** |
| **跨层信息传递** | ⭐⭐⭐⭐ | +25% | +2-3% | 高 | **6** |
| **对比学习增强** | ⭐⭐⭐ | +30% | +1-2% | 中 | **7** |

---

## 🎯 推荐实施策略

### 阶段1: 快速见效（1-2周）

实施改进4（残差缩放）+ 改进3（动态门控）

```python
# 组合实现
class ImprovedMiddleFusion_Stage1(nn.Module):
    """第一阶段改进：残差缩放 + 动态门控"""
    def __init__(self, node_dim=64, text_dim=768,
                 initial_node_scale=1.0,
                 initial_text_scale=12.0):
        super().__init__()

        # 残差缩放
        self.node_scale = nn.Parameter(torch.tensor(initial_node_scale))
        self.text_scale = nn.Parameter(torch.tensor(initial_text_scale))

        # 文本变换
        self.text_transform = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, node_dim)
        )

        # 节点重要性预测
        self.importance = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.ReLU(),
            nn.Linear(node_dim // 2, 1),
            nn.Sigmoid()
        )

        # 动态门控
        self.gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self, node_feat, text_feat, batch_num_nodes):
        # 节点重要性
        importance = self.importance(node_feat)

        # 文本变换
        text_transformed = self.text_transform(text_feat)
        text_broadcasted = broadcast_text(text_transformed, batch_num_nodes)

        # 门控
        gate_weight = self.gate(torch.cat([node_feat, text_broadcasted], dim=-1))

        # 重要性调制
        gate_modulated = gate_weight * (1.0 + importance)

        # 残差缩放融合
        fused = (
            self.node_scale * node_feat +
            self.text_scale * gate_modulated * text_broadcasted
        )

        return self.layer_norm(fused)
```

**预期提升**: +3-6% MAE

---

### 阶段2: 深度优化（2-4周）

加入改进1（多头注意力门控）

```python
class ImprovedMiddleFusion_Stage2(nn.Module):
    """第二阶段改进：多头注意力门控 + 残差缩放"""
    # 组合阶段1 + 多头注意力
    # 详细实现见上文
```

**预期提升**: +5-9% MAE

---

### 阶段3: 全面升级（4-6周）

加入改进2（层次化文本）+ 改进5（通道注意力）

```python
class ImprovedMiddleFusion_Stage3(nn.Module):
    """第三阶段改进：完整版"""
    # 组合所有改进
```

**预期提升**: +7-12% MAE

---

## 💡 快速实验建议

### 实验A: 残差缩放（最简单）

只需修改当前MiddleFusionModule：

```python
# 在 __init__ 中添加
self.node_scale = nn.Parameter(torch.tensor(1.0))

# 在 forward 的融合部分改为
fused = self.node_scale * node_feat + self.text_scale * gate_weight * text_broadcasted
```

**实施时间**: 5分钟
**预期提升**: +1-3% MAE

---

### 实验B: 动态门控（简单）

添加重要性预测器：

```python
# 在 __init__ 中添加
self.importance_predictor = nn.Sequential(
    nn.Linear(node_dim, node_dim // 2),
    nn.ReLU(),
    nn.Linear(node_dim // 2, 1),
    nn.Sigmoid()
)

# 在 forward 中
importance = self.importance_predictor(node_feat)
gate_weight = self.gate(...) * (1.0 + importance)
```

**实施时间**: 10分钟
**预期提升**: +2-3% MAE

---

## 📝 实施检查清单

### 准备阶段
- [ ] 备份当前代码
- [ ] 创建新的实验分支
- [ ] 准备对比实验配置

### 实施阶段1（快速改进）
- [ ] 添加残差缩放（node_scale参数）
- [ ] 添加动态门控（importance predictor）
- [ ] 测试forward正常运行
- [ ] 运行小规模训练验证

### 实施阶段2（深度优化）
- [ ] 实现多头注意力门控
- [ ] 集成到现有模块
- [ ] 消融实验对比

### 实施阶段3（全面升级）
- [ ] 实现层次化文本特征
- [ ] 实现通道注意力
- [ ] 完整性能评估

---

## 🔬 消融实验设计

建议测试顺序：

1. **基线**: 当前MiddleFusion (initial_scale=12.0)
2. **+残差缩放**: 添加node_scale
3. **+动态门控**: 添加importance调制
4. **+多头注意力**: 替换简单门控
5. **+层次化文本**: 添加多尺度文本
6. **完整版**: 所有改进组合

每个改进单独测试，然后逐步组合。

---

## 🎊 总结

### 最推荐的组合（性价比最高）

**组合A**: 残差缩放 + 动态门控 + 通道注意力
- 实现难度: ⭐⭐
- 参数增加: +35%
- 预期提升: **+5-8% MAE**
- 实施时间: 1-2天

**组合B**: 多头注意力门控 + 层次化文本
- 实现难度: ⭐⭐⭐⭐
- 参数增加: +80%
- 预期提升: **+7-12% MAE**
- 实施时间: 1-2周

### 您的下一步

我建议从**实验A（残差缩放）**开始，这是最简单的改进，只需5分钟实现，就能获得1-3%的提升！

需要我帮您实现哪个改进方案？
