# 修复 Gate 饱和问题

## 问题诊断
如果 `debug_alpha_extraction.py` 显示：
```
⚠️ 警告: Gate 值范围太窄 (0.3-0.5)，可能发生了饱和!
```

## 解决方案

### 方案 1: 使用 Tanh 替代 Sigmoid（推荐用于新训练）

在 `models/alignn.py:154-157` 修改：

```python
# 原始代码
self.gate = nn.Sequential(
    nn.Linear(node_dim + node_dim, node_dim),
    nn.Sigmoid()  # 输出范围 [0, 1]
)

# 修改为 Tanh + 缩放
self.gate = nn.Sequential(
    nn.Linear(node_dim + node_dim, node_dim),
    nn.Tanh()     # 输出范围 [-1, 1]
)

# 然后在 forward 中归一化到 [0, 1]:
# gate_values = (self.gate(gate_input) + 1) / 2  # 映射到 [0, 1]
```

### 方案 2: 添加温度参数（可用于已训练模型）

```python
class MiddleFusionModule(nn.Module):
    def __init__(self, ..., gate_temperature=1.0):
        ...
        self.gate = nn.Sequential(
            nn.Linear(node_dim + node_dim, node_dim),
            nn.Sigmoid()
        )
        self.gate_temperature = gate_temperature  # 温度参数

    def forward(self, ...):
        ...
        # 在 Sigmoid 之前除以温度
        gate_logits = self.gate[0](gate_input)  # 只过 Linear
        gate_values = torch.sigmoid(gate_logits / self.gate_temperature)
```

温度 > 1 会让分布更平坦，< 1 会让分布更尖锐。

### 方案 3: 检查输入特征归一化

确保输入到 gate 的特征有足够的方差：

```python
# 在 forward 中添加调试
gate_input = torch.cat([node_feat, text_broadcasted], dim=-1)
print(f"Gate input std: {gate_input.std(dim=1).mean()}")  # 应该 > 0.1
```

如果标准差太小，考虑添加 LayerNorm：

```python
self.gate = nn.Sequential(
    nn.LayerNorm(node_dim * 2),  # 归一化输入
    nn.Linear(node_dim * 2, node_dim),
    nn.Sigmoid()
)
```

## 训练建议

如果需要重新训练，考虑添加：

1. **多样性正则化**：鼓励不同原子有不同的 alpha 值
   ```python
   # 添加到损失函数
   alpha_diversity_loss = -gate_values.std()  # 负号：最大化标准差
   total_loss = task_loss + 0.01 * alpha_diversity_loss
   ```

2. **熵正则化**：鼓励 alpha 值接近 0 或 1
   ```python
   # 二元熵：p*log(p) + (1-p)*log(1-p)
   entropy = -(gate_values * torch.log(gate_values + 1e-8) +
               (1 - gate_values) * torch.log(1 - gate_values + 1e-8))
   entropy_loss = entropy.mean()
   total_loss = task_loss - 0.01 * entropy_loss  # 负号：最小化熵
   ```
