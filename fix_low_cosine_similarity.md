# 修复余弦相似度过低问题

## 问题诊断
如果 `3.analyze_text_flow_v2.py` 显示余弦相似度 < 0.15，说明文本特征在网络中丢失。

## 可能原因分析

### 原因 1: Hook 点不正确

**检查**: 确认 Hook 的是正确的文本特征

当前代码 (`analyze_text_flow_v2.py:61`):
```python
module.text_transform.register_forward_hook(get_text_emb)
```

这个 Hook 捕获的是 `text_transform` 的输出，即 [Batch, 64] 的变换后文本特征。

**问题**: 如果 `text_projection` (在 forward 中) 和 `text_transform` (在 fusion 中)
使用不同的变换，它们的向量空间可能不一致！

**验证**:
```python
# 检查两个 projection 的权重是否共享
print(model.text_projection is fusion_module.text_transform)  # 应该是 False
```

**修复**: 改为直接使用模型 forward 中的 `text_emb`（CLS token + projection）

### 原因 2: 维度不匹配导致所有层被跳过

**检查**: 运行修改后的 `3.analyze_text_flow_v2.py`，看是否有：
```
⚠️ 跳过了 X 个维度不匹配的层
```

**可能情况**:
- ALIGNN 层输出: [Total_Atoms, 256]  (hidden_features=256)
- GCN 层输出: [Total_Atoms, 256]
- 文本特征: [Batch, 64]  (projection_dim=64)
- 广播后文本: [Total_Atoms, 64]

❌ **256 ≠ 64** → 所有层都被跳过！

**修复**: 使用 `hidden_features=256` 而不是投影后的 64

### 原因 3: 文本特征确实没有融入

**检查中期融合配置**:
```python
# 查看 checkpoint 的 config
config.use_middle_fusion  # 应该是 True
config.middle_fusion_layers  # 例如 "2"
```

如果只在 Layer 2 融合，后续 4 层 GCN 会稀释文本信息。

**修复**: 尝试在多个层融合：
```python
config.middle_fusion_layers = "2,3"  # 在 ALIGNN Layer 2 和 3 都融合
```

## 推荐的修复方案

### 方案 1: 修改 `analyze_text_flow_v2.py` 使用正确维度

```python
class TextFlowAnalyzer:
    def __init__(self, model, use_hidden_dim=True):
        self.model = model
        self.layer_outputs = {}
        self.captured_text_emb = None
        self.use_hidden_dim = use_hidden_dim  # 新参数

    def register_hooks(self):
        # ... (保持不变)

        def get_text_emb(model, input, output):
            # output 是 [Batch, Node_Dim]，例如 64
            # 但我们需要匹配 layer 的 256 维
            if self.use_hidden_dim:
                # 直接Hook模型的text_emb（在forward中）
                # 或者添加一个投影层
                pass
            else:
                self.captured_text_emb = output.detach()
```

### 方案 2: 使用未投影的文本特征（BERT CLS）

修改 Hook 点，直接捕获 BERT 的 CLS token (768维)，
然后与相应维度的图层比较：

```python
# Hook BERT 的输出而不是 projection
def get_text_emb_bert(model, input, output):
    # output 是 (last_hidden_state, pooler_output)
    cls_token = output[0][:, 0, :]  # [Batch, 768]
    self.captured_text_emb = cls_token.detach()

# 在模型的 text_model 上注册
text_model.register_forward_hook(get_text_emb_bert)
```

但这需要修改更多代码。

### 方案 3: 添加投影层到匹配维度

最简单的方法：在计算相似度前，将文本特征投影到 layer 维度

```python
def compute_layer_similarity(self, g, lg, text_list, device):
    # ... (前面不变)

    for name, layer_feat in self.layer_outputs.items():
        layer_dim = layer_feat.shape[1]
        text_dim = text_emb.shape[1]

        if layer_dim != text_dim:
            # 动态投影
            projection = nn.Linear(text_dim, layer_dim).to(device)
            text_projected = projection(text_emb)

            # 广播
            text_expanded = []
            for i, n in enumerate(batch_num_nodes):
                text_expanded.append(text_projected[i].unsqueeze(0).repeat(n, 1))
            text_broadcasted = torch.cat(text_expanded, dim=0)
        else:
            # 原逻辑
            ...

        sim = F.cosine_similarity(layer_feat, text_broadcasted, dim=1)
        similarities[name] = sim.mean().item()
```

**缺点**: 随机初始化的投影层可能引入噪声

## 最佳实践

**推荐**: 只分析 **中期融合层之后** 的特征保留情况

因为中期融合只在特定层注入文本，只有这些层之后的特征才应该包含文本信息。

修改分析逻辑：
```python
# 只Hook融合后的层
fusion_layer_idx = 2  # 从config读取
for i, layer in enumerate(self.model.alignn_layers):
    if i >= fusion_layer_idx:  # 只Hook融合后的层
        layer.register_forward_hook(get_layer_output(f'Layer {i+1} (ALIGNN)'))
```
