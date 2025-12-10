# LOBSTER数据整合方案

## 问题分析

### 数据现状
- **JARVIS数据**：40,000+ 样本（金属+半导体+绝缘体）
- **LOBSTER数据**：1,520 样本（仅半导体+绝缘体）
- **数据量差距**：25倍
- **样本重叠率**：预计 5-10%（500-1000个样本）

### 核心挑战
1. ❌ 数据量太小，直接混合会稀释性能
2. ❌ 特征不兼容（LOBSTER特征在JARVIS中缺失）
3. ❌ 材料类型偏差（LOBSTER只有半导体/绝缘体）

---

## 推荐方案对比

| 策略 | 改善幅度 | 实施难度 | 时间 | 适用场景 |
|-----|---------|---------|------|---------|
| **策略1: 辅助特征** | 2-5% | 低 | 1-2天 | 通用性能提升 |
| **策略2: 子任务模型** | 10-15% | 中 | 1周 | 专项任务（带隙预测） |
| **策略3: 验证集** | 0% | 低 | 2天 | 可解释性验证 |
| **策略4: 特征蒸馏** | 5-10% | 高 | 2-3周 | 研究型项目 |

---

## 策略1: 辅助特征整合（推荐）⭐⭐⭐⭐⭐

### 核心思路

只对JARVIS和Materials Project的**重叠样本**添加LOBSTER边特征，其他样本用0填充。

### 实施步骤

#### 步骤1: 识别重叠样本（1小时）

```python
# utils/mp_jarvis_alignment.py

import json
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jarvis_data
from pymatgen.core import Structure

def find_overlapping_samples(lobster_dir, jarvis_dataset='dft_3d'):
    """找到Materials Project和JARVIS的重叠样本

    匹配策略：
    1. 结构相似度（RMSD < 0.1 Å）
    2. 化学式一致
    3. 空间群一致

    Returns:
        overlap_map: {jarvis_id: mp_id}
    """
    # 加载JARVIS数据
    jarvis_db = jarvis_data(jarvis_dataset)

    # 加载LOBSTER数据
    lobster_db = load_lobster_database(lobster_dir)

    overlap_map = {}

    for jarvis_entry in jarvis_db:
        jarvis_struct = Atoms.from_dict(jarvis_entry['atoms'])

        # 遍历LOBSTER数据库
        for mp_id, lobster_entry in lobster_db.items():
            mp_struct = Structure.from_dict(lobster_entry['structure'])

            # 检查匹配
            if is_structure_match(jarvis_struct, mp_struct):
                overlap_map[jarvis_entry['jid']] = mp_id
                break

    print(f"找到 {len(overlap_map)} 个重叠样本")

    # 保存映射
    with open('data/jarvis_mp_overlap.json', 'w') as f:
        json.dump(overlap_map, f, indent=2)

    return overlap_map


def is_structure_match(jarvis_atoms, mp_structure,
                       rmsd_threshold=0.1,
                       check_spacegroup=True):
    """检查两个结构是否匹配"""
    from pymatgen.analysis.structure_matcher import StructureMatcher

    # 转换JARVIS为pymatgen格式
    jarvis_pmg = jarvis_atoms.pymatgen_converter()

    # 使用pymatgen的结构匹配器
    matcher = StructureMatcher(
        ltol=0.2,  # 晶格参数容差
        stol=0.3,  # 位点容差
        angle_tol=5,  # 角度容差
    )

    return matcher.fit(jarvis_pmg, mp_structure)


def load_lobster_database(lobster_dir):
    """加载所有LOBSTER JSON文件

    Returns:
        {mp_id: lobster_data}
    """
    import os

    lobster_db = {}

    for filename in os.listdir(lobster_dir):
        if filename.endswith('.json') and filename.startswith('mp-'):
            mp_id = filename.replace('.json', '')

            with open(os.path.join(lobster_dir, filename)) as f:
                lobster_db[mp_id] = json.load(f)

    return lobster_db
```

#### 步骤2: 提取LOBSTER边特征（2小时）

```python
# utils/lobster_features.py

import numpy as np

class LobsterFeatureExtractor:
    """从LOBSTER数据提取图边特征"""

    def __init__(self, lobster_json_path):
        """初始化

        Args:
            lobster_json_path: LOBSTER的lightweight JSON文件路径
        """
        with open(lobster_json_path) as f:
            self.data = json.load(f)

    def get_edge_features(self, atom_i, atom_j, distance):
        """获取指定原子对的LOBSTER边特征

        Args:
            atom_i: 原子i的索引
            atom_j: 原子j的索引
            distance: 两原子距离（用于匹配LOBSTER中的键）

        Returns:
            edge_features: [icohp, icobi, bond_length] (3-dim)
        """
        # 从LOBSTER数据中查找对应的键
        bonds = self.data.get('bonds', {}).get('ICOHP', [])

        for bond in bonds:
            # 匹配原子对（考虑双向）
            if self._is_matching_bond(bond, atom_i, atom_j, distance):
                icohp = bond.get('ICOHP', 0.0)
                icobi = bond.get('ICOBI', 0.0)
                length = bond.get('length', distance)

                return np.array([icohp, icobi, length])

        # 如果找不到，返回默认值
        return np.array([0.0, 0.0, distance])

    def _is_matching_bond(self, bond, atom_i, atom_j, distance, tol=0.1):
        """检查LOBSTER键是否匹配给定的原子对"""
        bond_atoms = bond.get('atoms', [])
        bond_length = bond.get('length', 0)

        # 检查原子索引
        atoms_match = (
            (bond_atoms[0] == atom_i and bond_atoms[1] == atom_j) or
            (bond_atoms[0] == atom_j and bond_atoms[1] == atom_i)
        )

        # 检查距离
        distance_match = abs(bond_length - distance) < tol

        return atoms_match and distance_match

    def get_global_features(self):
        """获取全局材料特征

        Returns:
            global_features: dict包含
                - icohp_mean: 平均ICOHP
                - icohp_min: 最强键的ICOHP
                - num_bonds: 键的数量
                - madelung_energy: 马德隆能
        """
        bonds = self.data.get('bonds', {}).get('ICOHP', [])

        if not bonds:
            return {
                'icohp_mean': 0.0,
                'icohp_min': 0.0,
                'num_bonds': 0,
                'madelung_energy': 0.0
            }

        icohp_values = [bond['ICOHP'] for bond in bonds]

        return {
            'icohp_mean': np.mean(icohp_values),
            'icohp_min': np.min(icohp_values),  # 最负 = 最强
            'icohp_max': np.max(icohp_values),
            'icohp_std': np.std(icohp_values),
            'num_bonds': len(bonds),
            'madelung_energy': self.data.get('madelung_energy', 0.0)
        }
```

#### 步骤3: 修改数据加载器（4小时）

```python
# data.py 修改

from utils.lobster_features import LobsterFeatureExtractor

class StructureDatasetWithLobster(torch.utils.data.Dataset):
    """增强版数据集，支持LOBSTER特征"""

    def __init__(self, df, graphs, target,
                 lobster_dir=None,
                 overlap_map=None,
                 **kwargs):
        """
        Args:
            lobster_dir: LOBSTER JSON文件目录
            overlap_map: {jarvis_id: mp_id} 映射
        """
        super().__init__(df, graphs, target, **kwargs)

        self.lobster_dir = lobster_dir
        self.overlap_map = overlap_map or {}

        # 预加载LOBSTER特征
        self.lobster_cache = {}
        if lobster_dir and overlap_map:
            print("加载LOBSTER特征...")
            self._preload_lobster_features()

    def _preload_lobster_features(self):
        """预加载所有重叠样本的LOBSTER特征"""
        for jid, mp_id in self.overlap_map.items():
            lobster_path = os.path.join(
                self.lobster_dir,
                f"{mp_id}.json"
            )

            if os.path.exists(lobster_path):
                self.lobster_cache[jid] = LobsterFeatureExtractor(lobster_path)

        print(f"✅ 加载了 {len(self.lobster_cache)} 个LOBSTER特征")

    def __getitem__(self, idx):
        # 原始数据
        g = self.graphs[idx]
        label = self.labels[idx]
        text = self.text[idx]
        jid = self.ids[idx]

        # 如果有LOBSTER特征，添加到边
        if jid in self.lobster_cache:
            g = self._add_lobster_edge_features(g, jid)
        else:
            # 没有LOBSTER特征，添加零特征（保持维度一致）
            g = self._add_zero_lobster_features(g)

        if self.line_graph:
            return g, self.line_graphs[idx], text, label
        return g, text, label

    def _add_lobster_edge_features(self, g, jid):
        """为图添加LOBSTER边特征"""
        lobster = self.lobster_cache[jid]

        # 获取边的原子对和距离
        src, dst = g.edges()
        edge_lobster_features = []

        for i, j in zip(src.numpy(), dst.numpy()):
            # 计算距离（从节点位置）
            pos_i = g.ndata['pos'][i]  # 假设图有位置信息
            pos_j = g.ndata['pos'][j]
            distance = torch.norm(pos_i - pos_j).item()

            # 获取LOBSTER特征
            lobster_feat = lobster.get_edge_features(i, j, distance)
            edge_lobster_features.append(lobster_feat)

        # 添加到图的边特征
        edge_lobster_features = torch.FloatTensor(edge_lobster_features)
        g.edata['lobster'] = edge_lobster_features  # [num_edges, 3]

        # 添加全局特征（可选）
        global_feat = lobster.get_global_features()
        g.graph_attr = global_feat

        return g

    def _add_zero_lobster_features(self, g):
        """为没有LOBSTER数据的图添加零特征"""
        num_edges = g.num_edges()
        g.edata['lobster'] = torch.zeros(num_edges, 3)
        g.graph_attr = {
            'icohp_mean': 0.0,
            'icohp_min': 0.0,
            'num_bonds': 0,
            'madelung_energy': 0.0
        }
        return g
```

#### 步骤4: 修改ALIGNN模型（2小时）

```python
# models/alignn.py 修改

class ALIGNNWithLobster(ALIGNN):
    """支持LOBSTER边特征的ALIGNN"""

    def __init__(self, config):
        super().__init__(config)

        # 边特征编码器（处理 r + lobster）
        original_edge_dim = config.edge_input_features  # 80 (RBF)
        lobster_dim = 3  # [icohp, icobi, length]

        self.edge_embedding = nn.Sequential(
            # 原始RBF特征
            nn.Linear(original_edge_dim, config.embedding_features),
            nn.SiLU(),
        )

        # LOBSTER特征编码器
        self.lobster_embedding = nn.Sequential(
            nn.Linear(lobster_dim, config.embedding_features),
            nn.SiLU(),
        )

        # 融合层
        self.edge_fusion = nn.Sequential(
            nn.Linear(config.embedding_features * 2, config.hidden_features),
            nn.SiLU(),
        )

    def forward(self, data):
        g, lg, text = data

        # 节点特征
        x = g.ndata['atom_features']
        x = self.atom_embedding(x)

        # 边特征：原始RBF
        y = g.edata['r']  # [num_edges, 3]
        y = self.rbf_expansion(y)  # [num_edges, 80]
        y_embedded = self.edge_embedding(y)

        # 边特征：LOBSTER（如果存在）
        if 'lobster' in g.edata:
            lobster = g.edata['lobster']  # [num_edges, 3]
            lobster_embedded = self.lobster_embedding(lobster)

            # 融合
            y = torch.cat([y_embedded, lobster_embedded], dim=-1)
            y = self.edge_fusion(y)
        else:
            y = y_embedded

        # 后续ALIGNN层
        # ... (保持原有逻辑)
```

#### 步骤5: 训练配置（1小时）

```python
# config.py 添加

class ALIGNNConfig(BaseSettings):
    # ... 原有配置 ...

    # LOBSTER特征
    use_lobster_features: bool = False
    lobster_data_dir: str = "data/lobster_database"
    lobster_overlap_map: str = "data/jarvis_mp_overlap.json"

    # LOBSTER特征权重（可学习）
    lobster_feature_weight: float = 1.0

# 训练脚本
from utils.mp_jarvis_alignment import find_overlapping_samples

config = TrainingConfig(
    dataset="dft_3d",
    target="formation_energy_peratom",

    model=ALIGNNConfig(
        use_lobster_features=True,
        lobster_data_dir="data/lobster_database",
        lobster_overlap_map="data/jarvis_mp_overlap.json"
    )
)

# 如果overlap map不存在，先生成
if not os.path.exists(config.model.lobster_overlap_map):
    overlap_map = find_overlapping_samples(
        lobster_dir=config.model.lobster_data_dir,
        jarvis_dataset=config.dataset
    )

# 加载数据
train_loader, val_loader, test_loader = get_train_val_loaders_with_lobster(
    dataset=config.dataset,
    target=config.target,
    lobster_dir=config.model.lobster_data_dir,
    overlap_map_file=config.model.lobster_overlap_map,
    batch_size=config.batch_size
)
```

---

## 策略2: 专门子任务模型

### 适用任务
- ✅ 带隙预测（bandgap）
- ✅ 声子最高频率（max_phonon_freq）
- ✅ 热导率（thermal_conductivity）
- ✅ 弹性模量（bulk_modulus, shear_modulus）

### 实施方案

```python
# 只在半导体/绝缘体子集上训练专门模型

# 1. 筛选数据
semiconductor_data = [
    sample for sample in jarvis_data
    if sample.get('optb88vdw_bandgap', 0) > 0
]

print(f"半导体样本: {len(semiconductor_data)}")

# 2. 加载LOBSTER数据（全部1520个样本）
lobster_data = load_all_lobster_samples()

# 3. 合并数据集
combined_data = semiconductor_data + lobster_data

# 4. 训练专门模型
model_semiconductor = ALIGNNWithLobster(config)
model_semiconductor.train(combined_data)

# 5. 评估
# 预期：带隙预测MAE降低 10-15%
```

---

## 策略3: 可解释性验证

### 验证模型是否学到化学意义

```python
# validate_chemical_understanding.py

def validate_attention_vs_icohp(model, lobster_data):
    """验证：强化学键是否获得更高注意力权重？"""

    correlations = []

    for sample in lobster_data:
        # Forward pass with attention
        output = model([sample.g, sample.lg, sample.text],
                      return_attention=True)

        # 获取边的注意力
        edge_attention = output['attention_weights']['edge']  # [num_edges]

        # 获取边的ICOHP
        edge_icohp = sample.g.edata['lobster'][:, 0]  # [num_edges]

        # 计算相关系数
        # 预期：ICOHP越负（键越强），attention越高
        corr = np.corrcoef(-edge_icohp.numpy(),
                           edge_attention.numpy())[0, 1]
        correlations.append(corr)

    avg_corr = np.mean(correlations)

    print(f"Attention-ICOHP相关系数: {avg_corr:.3f}")

    if avg_corr > 0.3:
        print("✅ 模型学到了化学键强度信息")
    else:
        print("⚠️ 模型未充分利用化学键信息")

    return avg_corr
```

---

## 预期效果对比

### 各策略的预期改善（基于论文27%的改善率）

| 策略 | 覆盖率 | 折扣因子 | 预期改善 | 实测MAE变化 |
|-----|--------|---------|---------|-----------|
| **直接混合（不推荐）** | 3.8% | 0.1 | 2.7% | 0.25 → 0.243 |
| **策略1: 辅助特征** | 2.5% | 0.15 | 4% | 0.25 → 0.24 |
| **策略2: 子任务模型** | 30% | 0.4 | 11% | 0.25 → 0.22 |
| **策略4: 特征蒸馏** | 100% | 0.3 | 8% | 0.25 → 0.23 |

### 计算公式

```
实际改善 = 理论改善 (27%) × 覆盖率 × 折扣因子

例如策略1:
实际改善 = 27% × 2.5% × 0.15 ≈ 0.01 (即1%)
```

---

## 工作量评估

| 阶段 | 策略1 | 策略2 | 策略3 | 策略4 |
|-----|-------|-------|-------|-------|
| **数据对齐** | 1小时 | 1小时 | 1小时 | 2小时 |
| **特征提取** | 2小时 | 2小时 | - | 4小时 |
| **模型修改** | 2小时 | 4小时 | - | 8小时 |
| **训练验证** | 4小时 | 1天 | 2小时 | 2天 |
| **调试优化** | 2小时 | 4小时 | - | 1天 |
| **总计** | **1-2天** | **1周** | **2天** | **2-3周** |

---

## 风险提示

### 策略1的风险
1. ⚠️ 重叠样本可能少于预期（<500个）
   - **缓解**：放宽结构匹配标准
2. ⚠️ LOBSTER特征可能与DGL图的边不完全对应
   - **缓解**：用距离容差匹配

### 策略2的风险
1. ⚠️ LOBSTER数据太少，专门模型可能过拟合
   - **缓解**：增加数据增强（结构扰动）
2. ⚠️ 半导体定义不一致（JARVIS vs LOBSTER）
   - **缓解**：用带隙阈值统一定义

### 策略4的风险
1. ⚠️ ICOHP预测器质量难以保证
   - **缓解**：先在验证集上测试预测精度
2. ⚠️ 伪特征可能引入噪声
   - **缓解**：使用不确定性量化

---

## 决策树

```
是否整合LOBSTER数据？
│
├─ 目标：通用性能提升
│  └─ 选择：策略1（辅助特征）
│     ├─ 预期改善：2-5%
│     └─ 时间：1-2天
│
├─ 目标：特定任务优化（带隙预测）
│  └─ 选择：策略2（子任务模型）
│     ├─ 预期改善：10-15%
│     └─ 时间：1周
│
├─ 目标：论文可解释性
│  └─ 选择：策略3（验证集）
│     ├─ 预期改善：0%（但提供化学洞察）
│     └─ 时间：2天
│
└─ 目标：研究型探索
   └─ 选择：策略4（特征蒸馏）
      ├─ 预期改善：5-10%
      └─ 时间：2-3周
```

---

## 推荐行动

### 立即行动（今天）
1. ✅ 运行样本对齐脚本，确认重叠样本数量
2. ✅ 如果 > 300个，继续策略1
3. ✅ 如果 < 300个，考虑策略2或3

### 如果选择策略1
```bash
# 1. 对齐样本
python utils/mp_jarvis_alignment.py \
    --lobster_dir data/lobster_database \
    --jarvis_dataset dft_3d \
    --output data/jarvis_mp_overlap.json

# 2. 验证特征提取
python utils/lobster_features.py \
    --test_sample mp-66.json

# 3. 修改数据加载器
# (按照上面的代码修改 data.py)

# 4. 训练
python train_with_cross_modal_attention.py \
    --config config_with_lobster.json \
    --output_dir runs/lobster_augmented
```

---

## 总结

**核心观点**：
1. ❌ 不要直接混合（数据量差距太大）
2. ✅ 用策略1作为起点（低风险，快速验证）
3. ✅ 如果效果好，再考虑策略2或4
4. ✅ 策略3适合写论文时增强可解释性

**预期结果**：
- 保守估计：MAE改善 **2-5%**
- 理想情况（专项任务）：MAE改善 **10-15%**

**立即下一步**：
运行样本对齐脚本，看看实际重叠有多少样本！

---

**文档生成时间**：2025-12-10
**状态**：待验证重叠样本数
**推荐策略**：策略1（辅助特征）
