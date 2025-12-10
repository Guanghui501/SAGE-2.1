# GPU显存占用分析指南

## 🤔 您的问题

**现象**：相同的训练参数（batch size、模型结构等），但数据量多的数据集反而占用显存更少。

这确实违反直觉！让我详细解释原因。

---

## 📊 显存占用组成

训练时GPU显存主要由以下部分组成：

```
总显存占用 = 模型参数 + 优化器状态 + 梯度 + 激活值 + 批次数据
```

### 各部分详解

| 组件 | 大小 | 是否随数据量变化 |
|-----|------|----------------|
| **模型参数** | 固定 | ❌ 不变 |
| **优化器状态** | 2×参数（AdamW） | ❌ 不变 |
| **梯度** | 等于参数 | ❌ 不变 |
| **激活值** | 取决于batch内容 | ✅ **变化** |
| **批次数据** | 取决于batch内容 | ✅ **变化** |

**关键发现**：只有激活值和批次数据会因数据内容而变化！

---

## 🔍 关键因素：图的大小

### DGL图批处理机制

您的代码使用了DGL (Deep Graph Library)：

```python
# graphs.py:644
batched_graph = dgl.batch(graphs)
```

DGL的`batch()`函数将多个小图合并成一个大图：

```
图1: 24个原子, 288条边
图2: 32个原子, 384条边
图3: 16个原子, 192条边
图4: 28个原子, 336条边
...
batch_size=64个图

合并后的大图:
总节点数 = 24 + 32 + 16 + 28 + ... (64个图的总和)
总边数 = 288 + 384 + 192 + 336 + ... (64个图的总和)
```

### 显存占用与图大小的关系

```python
批次显存 ≈ batch_size × 平均节点数 × 节点特征维度
        + batch_size × 平均边数 × 边特征维度
```

**关键公式**：
```
显存占用 ∝ 总节点数 × 节点特征维度 + 总边数 × 边特征维度
```

---

## 💡 为什么数据量多反而显存少？

### 原因1: 平均晶体尺寸不同 ⭐⭐⭐⭐⭐

**最可能的原因！**

不同数据集的材料复杂度可能不同：

| 数据集 | 样本数 | 平均原子数 | batch总节点数 (bs=64) | 显存占用 |
|-------|-------|-----------|---------------------|---------|
| 小数据集 | 1,639 | **35原子** | 64×35=**2,240** | 高 |
| 大数据集 | 10,464 | **20原子** | 64×20=**1,280** | 低 |

**即使batch size相同，总节点数可以差75%！**

### 验证方法：

```bash
# 创建脚本查看数据集统计
python -c "
import json
from jarvis.core.atoms import Atoms
from pathlib import Path

# 加载数据
data_path = Path('/path/to/your/data.json')
data = json.load(open(data_path))

# 统计原子数
atom_counts = []
for sample in data[:1000]:  # 取前1000个样本
    atoms = Atoms.from_dict(sample['atoms'])
    atom_counts.append(len(atoms))

import numpy as np
print(f'平均原子数: {np.mean(atom_counts):.1f}')
print(f'中位数: {np.median(atom_counts):.1f}')
print(f'最小值: {min(atom_counts)}')
print(f'最大值: {max(atom_counts)}')
print(f'标准差: {np.std(atom_counts):.1f}')
"
```

---

### 原因2: 数据预加载策略

#### pin_memory的影响

```python
# data.py:386
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    num_workers=workers,
    pin_memory=pin_memory  # ← 关键参数
)
```

**您的当前配置**（从训练曲线推测）：
```json
{
    "pin_memory": false,
    "num_workers": 24
}
```

**pin_memory的显存影响**：

| pin_memory | CPU→GPU传输 | 显存占用 | 说明 |
|-----------|------------|---------|------|
| `True` | 快 | **高** | 在CPU锁页内存中预分配 |
| `False` | 慢 | **低** | 按需传输 |

如果小数据集用了`pin_memory=True`，大数据集用了`False`，会导致显存差异。

---

### 原因3: num_workers的影响

```python
"num_workers": 24  # 您的配置
```

**num_workers与显存的关系**：

- `num_workers=0`: 主进程加载，显存占用稳定
- `num_workers>0`: 多进程预加载，每个worker缓存1-2个batch

**显存占用**：
```
额外显存 ≈ num_workers × 预取batch数 × 单batch大小
```

如果数据集A用了24 workers，数据集B用了0 workers：
```
差异 = 24 × 2 × 单batch显存
```

---

### 原因4: 图缓存机制

DGL可能会缓存预处理的图：

```python
# data.py:141-146
if cachefile is not None and cachefile.is_file():
    graphs, labels = dgl.load_graphs(str(cachefile))  # 从缓存加载
else:
    graphs = df["atoms"].progress_apply(atoms_to_graph).values
    if cachefile is not None:
        dgl.save_graphs(str(cachefile), graphs.tolist())  # 保存缓存
```

**影响**：
- 有缓存：图已经在GPU显存中 → 高显存
- 无缓存：动态加载 → 低显存

---

### 原因5: 文本编码缓存

如果使用了文本模态：

```python
# graphs.py:668-670
if len(labels[0].shape) > 0:
    return batched_graph, batched_line_graph, batch_text, torch.stack(labels)
else:
    return batched_graph, batched_line_graph, batch_text, torch.tensor(labels)
```

文本编码可能被缓存：
- BERT编码: 768维向量 × batch_size × 序列长度
- 小数据集可能缓存了所有样本的编码
- 大数据集动态编码

---

## 🔧 如何诊断具体原因

### 方法1: 添加显存监控脚本

创建 `monitor_memory.py`：

```python
import torch
import numpy as np
from data import get_train_val_loaders

def monitor_memory_usage(dataset_name, target, batch_size=64):
    """监控数据加载的显存使用"""

    print(f"\n{'='*80}")
    print(f"显存监控 - {dataset_name}")
    print(f"{'='*80}\n")

    # 清空显存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 加载数据
    train_loader, val_loader, test_loader, _ = get_train_val_loaders(
        dataset=dataset_name,
        target=target,
        batch_size=batch_size,
        pin_memory=False,
        workers=0  # 先用0，隔离num_workers影响
    )

    # 统计图大小
    node_counts = []
    edge_counts = []

    print("分析前10个batch...")
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break

        g, lg, text, labels = batch

        # 记录节点和边数
        node_counts.append(g.num_nodes())
        edge_counts.append(g.num_edges())

        # 模拟移到GPU
        if torch.cuda.is_available():
            g_gpu = g.to('cuda')
            lg_gpu = lg.to('cuda')
            labels_gpu = labels.to('cuda')

            # 记录显存
            current_mem = torch.cuda.memory_allocated() / 1024**2  # MB
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2

            print(f"Batch {i+1}:")
            print(f"  节点数: {g.num_nodes():,}")
            print(f"  边数: {g.num_edges():,}")
            print(f"  当前显存: {current_mem:.1f} MB")
            print(f"  峰值显存: {peak_mem:.1f} MB")

            # 清理
            del g_gpu, lg_gpu, labels_gpu
            torch.cuda.empty_cache()

    # 统计
    print(f"\n数据集统计:")
    print(f"  总样本数: {len(train_loader.dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  平均节点数/batch: {np.mean(node_counts):.1f}")
    print(f"  平均边数/batch: {np.mean(edge_counts):.1f}")
    print(f"  平均节点数/样本: {np.mean(node_counts)/batch_size:.1f}")
    print(f"  平均边数/样本: {np.mean(edge_counts)/batch_size:.1f}")
    print(f"  节点数标准差: {np.std(node_counts):.1f}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dft_3d')
    parser.add_argument('--target', type=str, default='hse_bandgap')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    monitor_memory_usage(args.dataset, args.target, args.batch_size)
```

**使用方法**：

```bash
# 监控小数据集
python monitor_memory.py --dataset dft_3d --target hse_bandgap --batch_size 64

# 监控大数据集
python monitor_memory.py --dataset dft_3d --target formation_energy_peratom --batch_size 64
```

---

### 方法2: 比较训练日志

检查两次训练的完整配置：

```bash
# 查看训练配置
cat output_dir1/config.json
cat output_dir2/config.json

# 比较差异
diff output_dir1/config.json output_dir2/config.json
```

特别关注：
- `pin_memory`
- `num_workers`
- `batch_size`
- 数据集大小和平均样本复杂度

---

### 方法3: 实时显存监控

训练时使用`nvidia-smi`监控：

```bash
# 终端1: 启动训练
python train_with_cross_modal_attention.py --dataset dft_3d --target hse_bandgap

# 终端2: 实时监控显存
watch -n 1 nvidia-smi
```

或使用`gpustat`:

```bash
pip install gpustat
watch -n 1 gpustat -cpu
```

---

## 📈 典型显存占用示例

### 场景1: HSE数据集 (1,639样本)

假设平均每个材料35个原子：

```
模型参数:        500 MB  (固定)
优化器状态:      1000 MB (固定，AdamW = 2×参数)
梯度:            500 MB  (固定)
激活值:          800 MB  (取决于前向传播)
-------------------------------------------
固定开销:        2800 MB

批次数据 (batch_size=64):
  节点数: 64 × 35 = 2,240
  节点特征: 2,240 × 256 × 4 bytes = 2.2 MB
  边数: 2,240 × 12 = 26,880 (假设平均12条边/节点)
  边特征: 26,880 × 80 × 4 bytes = 8.4 MB
  Line graph: 约2×边特征 = 16.8 MB
  文本特征: 64 × 768 × 4 bytes = 0.2 MB
-------------------------------------------
批次开销:        27.6 MB

总显存:          ~2830 MB ≈ 2.8 GB
```

### 场景2: 完整DFT数据集 (10,464样本)

假设平均每个材料20个原子（更多简单材料）：

```
固定开销:        2800 MB (相同)

批次数据 (batch_size=64):
  节点数: 64 × 20 = 1,280 (-43%)
  节点特征: 1,280 × 256 × 4 bytes = 1.3 MB
  边数: 1,280 × 12 = 15,360
  边特征: 15,360 × 80 × 4 bytes = 4.8 MB
  Line graph: 9.6 MB
  文本特征: 0.2 MB
-------------------------------------------
批次开销:        15.9 MB (-42%)

总显存:          ~2816 MB ≈ 2.75 GB
```

**差异仅50 MB，但如果平均原子数差异更大，差异会更明显！**

---

## 🎯 实际测试

### 创建对比脚本

```python
# compare_datasets.py
import torch
import numpy as np
from data import get_train_val_loaders

def compare_datasets():
    datasets = [
        ('dft_3d', 'hse_bandgap', 'HSE数据集'),
        ('dft_3d', 'formation_energy_peratom', '形成能数据集'),
    ]

    results = []

    for dataset, target, name in datasets:
        print(f"\n分析: {name}")

        loader, _, _, _ = get_train_val_loaders(
            dataset=dataset,
            target=target,
            batch_size=64,
            pin_memory=False,
            workers=0
        )

        # 采样前100个batch
        node_counts = []
        edge_counts = []

        for i, batch in enumerate(loader):
            if i >= 100:
                break
            g, lg, _, _ = batch
            node_counts.append(g.num_nodes())
            edge_counts.append(g.num_edges())

        results.append({
            'name': name,
            'samples': len(loader.dataset),
            'avg_nodes_per_batch': np.mean(node_counts),
            'avg_edges_per_batch': np.mean(edge_counts),
            'avg_nodes_per_sample': np.mean(node_counts) / 64,
            'avg_edges_per_sample': np.mean(edge_counts) / 64,
            'std_nodes': np.std(node_counts),
        })

    # 打印对比
    print(f"\n{'='*100}")
    print(f"{'数据集':<20} {'样本数':<10} {'节点/batch':<15} {'边/batch':<15} {'节点/样本':<12} {'边/样本':<12}")
    print(f"{'='*100}")
    for r in results:
        print(f"{r['name']:<20} {r['samples']:<10} {r['avg_nodes_per_batch']:<15.1f} "
              f"{r['avg_edges_per_batch']:<15.1f} {r['avg_nodes_per_sample']:<12.1f} "
              f"{r['avg_edges_per_sample']:<12.1f}")

    # 计算差异
    if len(results) == 2:
        node_diff = (results[0]['avg_nodes_per_batch'] - results[1]['avg_nodes_per_batch']) / results[1]['avg_nodes_per_batch'] * 100
        print(f"\n节点数差异: {node_diff:+.1f}%")
        print(f"预期显存差异: 约 {abs(node_diff)/2:.1f}%")  # 粗略估计

if __name__ == '__main__':
    compare_datasets()
```

运行：
```bash
python compare_datasets.py
```

---

## 💡 结论

### 最可能的原因排序

1. **⭐⭐⭐⭐⭐ 平均晶体尺寸不同**
   - HSE数据集可能包含更复杂的材料（更多原子）
   - 完整DFT数据集包含很多简单材料（更少原子）
   - **这会直接影响批次显存占用**

2. **⭐⭐⭐⭐ pin_memory配置不同**
   - 小数据集可能用了`pin_memory=True`
   - 大数据集用了`pin_memory=False`

3. **⭐⭐⭐ num_workers不同**
   - 不同的预加载进程数

4. **⭐⭐ 图缓存机制**
   - 小数据集可能有预加载的图缓存

5. **⭐ 文本编码差异**
   - 文本描述长度不同

### 验证步骤

1. **运行`compare_datasets.py`** - 查看平均节点/边数
2. **检查配置文件** - 比较`pin_memory`和`num_workers`
3. **运行`monitor_memory.py`** - 实时监控显存

### 优化建议

如果需要统一显存占用：

```python
# 配置统一化
config = {
    "batch_size": 64,
    "pin_memory": False,      # 统一用False
    "num_workers": 0,         # 统一用0（或都用相同数字）
}
```

或者根据平均图大小调整batch size：

```python
# 如果数据集A平均35原子，数据集B平均20原子
# 可以调整batch size保持总节点数相近
batch_size_A = 64
batch_size_B = int(64 * 35 / 20) = 112  # 保持总节点数相近
```

---

**总结**：相同训练参数但显存占用不同，**几乎肯定是因为数据集的平均图大小不同**。运行上面的诊断脚本就能确认！

**文档生成时间**：2025-12-10
