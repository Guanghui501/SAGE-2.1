# 使用改进中期融合训练模型 - 示例命令

本文档展示如何使用 `train_with_cross_modal_attention.py` 训练带有 LayerNorm 和可学习缩放因子的改进模型。

---

## 🚀 快速开始

### 示例 1：基础中期融合（仅使用 LayerNorm）

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --root_dir ../dataset/ \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --middle_fusion_use_gate_norm True \
    --middle_fusion_use_learnable_scale False \
    --batch_size 64 \
    --epochs 500 \
    --learning_rate 0.001 \
    --output_dir ./outputs/fusion_with_layernorm
```

**预期效果**:
- 特征尺度自动平衡
- 余弦相似度提升 150-250%
- 训练稳定性提升

---

### 示例 2：仅使用可学习缩放（推荐用于自适应）

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --root_dir ../dataset/ \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --middle_fusion_use_gate_norm False \
    --middle_fusion_use_learnable_scale True \
    --middle_fusion_initial_scale 12.0 \
    --batch_size 64 \
    --epochs 500 \
    --learning_rate 0.001 \
    --output_dir ./outputs/fusion_with_learnable_scale
```

**预期效果**:
- 模型自动学习最优缩放比例
- 初始缩放值 12.0 基于诊断结果
- 余弦相似度提升 300-400%
- Alpha 多样性提升 100-200%

**监控训练**:
```bash
# 在训练过程中或训练后监控 text_scale 演变
python monitor_text_scale.py \
    --checkpoint_dir ./outputs/fusion_with_learnable_scale/formation_energy_peratom \
    --output_plot text_scale_evolution.png
```

---

### 示例 3：同时使用 LayerNorm + 可学习缩放（推荐）⭐

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --root_dir ../dataset/ \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --middle_fusion_use_gate_norm True \
    --middle_fusion_use_learnable_scale True \
    --middle_fusion_initial_scale 12.0 \
    --batch_size 64 \
    --epochs 500 \
    --learning_rate 0.001 \
    --output_dir ./outputs/fusion_improved
```

**预期效果**（最佳配置）:
- LayerNorm 提供稳定的特征尺度
- 可学习缩放提供额外的自适应性
- 余弦相似度提升 350-450%
- Alpha 多样性提升 150-250%
- 训练最稳定

---

### 示例 4：多层中期融合（实验性）

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --root_dir ../dataset/ \
    --use_middle_fusion True \
    --middle_fusion_layers "2,3" \
    --middle_fusion_use_gate_norm True \
    --middle_fusion_use_learnable_scale True \
    --middle_fusion_initial_scale 12.0 \
    --batch_size 64 \
    --epochs 500 \
    --learning_rate 0.001 \
    --output_dir ./outputs/fusion_multilayer
```

**说明**:
- 在 Layer 2 和 Layer 3 都注入文本信息
- 每层有独立的 MiddleFusionModule
- 每层学习独立的缩放因子

**预期效果**:
- 文本信息更深入融入图编码
- 可能进一步提升性能（1-3%）
- 计算成本增加约 10-15%

---

### 示例 5：完整配置（所有融合机制）

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --root_dir ../dataset/ \
    \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --middle_fusion_use_gate_norm True \
    --middle_fusion_use_learnable_scale True \
    --middle_fusion_initial_scale 12.0 \
    \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --cross_modal_hidden_dim 256 \
    \
    --batch_size 64 \
    --epochs 500 \
    --learning_rate 0.001 \
    --output_dir ./outputs/fusion_complete
```

**包含**:
- ✅ 中期融合（Layer 2）with LayerNorm + 可学习缩放
- ✅ 晚期融合（CrossModalAttention）

---

## 📊 Material Project 数据集示例

### HSE Bandgap

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property hse_bandgap-2 \
    --root_dir ../dataset/ \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --middle_fusion_use_gate_norm True \
    --middle_fusion_use_learnable_scale True \
    --middle_fusion_initial_scale 12.0 \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --batch_size 32 \
    --epochs 500 \
    --learning_rate 0.001 \
    --output_dir ./outputs/hse_bandgap_improved
```

### Bulk Modulus

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property bulk_modulus_kv \
    --root_dir ../dataset/ \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --middle_fusion_use_gate_norm True \
    --middle_fusion_use_learnable_scale True \
    --middle_fusion_initial_scale 12.0 \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --batch_size 32 \
    --epochs 500 \
    --output_dir ./outputs/bulk_modulus_improved
```

---

## 🔧 参数调优指南

### initial_scale 的选择

| 场景 | 建议值 | 说明 |
|------|--------|------|
| **默认（保守）** | 1.0 | 让模型从头学习，适合探索性实验 |
| **基于诊断** | 12.0 | 根据 diagnose_fusion_effectiveness.py 结果 |
| **激进** | 15.0-20.0 | 如果已知文本特征极弱 |
| **微调** | 8.0-10.0 | 稍保守，适合不确定的情况 |

**如何确定最优值**:
```bash
# 1. 先用现有模型运行诊断
python diagnose_fusion_effectiveness.py \
    --checkpoint existing_model.pt \
    --root_dir ../dataset/

# 2. 查看输出的 "节点/文本比例"，例如 12.63:1
# 3. 将该比例作为 initial_scale（如 12.0 或 13.0）
```

---

### 训练超参数建议

| 参数 | 推荐值 | 备注 |
|------|--------|------|
| **learning_rate** | 0.001 | 标准值 |
| **batch_size** | 32-64 | 根据GPU内存调整 |
| **epochs** | 500-1000 | 中期融合收敛较快，500足够 |
| **warmup_steps** | 2000 | 帮助稳定训练初期 |
| **weight_decay** | 1e-5 | 轻度正则化 |

---

## 📈 训练后分析

### 1. 查看学到的缩放因子

```bash
python -c "
import torch
ckpt = torch.load('outputs/fusion_improved/formation_energy_peratom/best_model.pt')
for key, value in ckpt['model'].items():
    if 'text_scale' in key:
        print(f'{key}: {value.item():.4f}')
"
```

**示例输出**:
```
middle_fusion_modules.layer_2.text_scale: 13.2451
```

**解读**:
- 如果 ≈ 12.0: 初始值已经接近最优
- 如果 > 15.0: 文本特征需要更强权重
- 如果 < 8.0: 文本特征被自动调弱

---

### 2. 诊断融合效果

```bash
python diagnose_fusion_effectiveness.py \
    --checkpoint outputs/fusion_improved/formation_energy_peratom/best_model.pt \
    --root_dir ../dataset/
```

**期望结果**:
```
节点/文本比例: 0.9:1 - 1.2:1  ← 平衡！
余弦相似度:    0.50-0.65       ← 显著改善
```

---

### 3. 提取并可视化 Alpha

```bash
# 提取 Alpha 值
python 1.extract_alpha_final.py \
    --checkpoint outputs/fusion_improved/formation_energy_peratom/best_model.pt \
    --root_dir ../dataset/ \
    --dataset jarvis \
    --property formation_energy_peratom \
    --n_samples 500

# 生成图表
python 2.create_paper_alpha_figures.py
```

**期望改进**:
- Alpha 标准差: 0.029 → 0.06-0.10 (+100-200%)
- Alpha 范围: 0.116 → 0.20-0.30 (+70-160%)

---

### 4. 对比原始模型

```bash
python compare_fusion_models.py \
    --model1 best_test_model.pt \
    --model2 outputs/fusion_improved/formation_energy_peratom/best_model.pt \
    --root_dir ../dataset/ \
    --n_samples 50
```

---

## ⚠️ 常见问题

### 问题 1: text_scale 训练中爆炸（>50.0）

**原因**: text_transform 输出仍然太弱，或学习率过高

**解决**:
```bash
# 方案 A: 增加初始缩放
--middle_fusion_initial_scale 20.0

# 方案 B: 降低学习率
--learning_rate 0.0005
```

---

### 问题 2: text_scale 快速归零（<0.1）

**原因**: 文本信息对当前任务贡献不大

**解决**:
```bash
# 方案 A: 启用对比学习增强文本重要性
--use_contrastive True \
--contrastive_weight 0.1

# 方案 B: 检查数据集文本质量
```

---

### 问题 3: 余弦相似度仍然很低（<0.3）

**诊断**:
```bash
python diagnose_fusion_effectiveness.py \
    --checkpoint outputs/fusion_improved/.../best_model.pt \
    --root_dir ../dataset/
```

**检查**:
- text_transform 输入 L2 是否正常（~8-10）
- text_transform 输出 L2 是否已放大（~25-35）

**解决**:
```bash
# 如果输出仍然弱，尝试更大的 initial_scale
--middle_fusion_initial_scale 20.0
```

---

## 🎯 推荐工作流程

### 第一阶段：快速验证（1-2小时）

使用手动缩放快速验证改进效果：

```bash
# 应用手动缩放到现有模型
python scale_checkpoint_weights.py \
    --input_checkpoint best_test_model.pt \
    --output_checkpoint best_test_model_scaled_12.0.pt \
    --scale_factor 12.0

# 验证效果
python diagnose_fusion_effectiveness.py \
    --checkpoint best_test_model_scaled_12.0.pt \
    --root_dir ../dataset/

# 生成论文图表
python 1.extract_alpha_final.py --checkpoint best_test_model_scaled_12.0.pt ...
python 2.create_paper_alpha_figures.py
```

---

### 第二阶段：完整训练（1-2天）

使用改进配置从头训练：

```bash
# 训练改进模型
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --root_dir ../dataset/ \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --middle_fusion_use_gate_norm True \
    --middle_fusion_use_learnable_scale True \
    --middle_fusion_initial_scale 12.0 \
    --use_cross_modal True \
    --batch_size 64 \
    --epochs 500 \
    --output_dir ./outputs/fusion_improved

# 监控 text_scale 演变
python monitor_text_scale.py \
    --checkpoint_dir ./outputs/fusion_improved/formation_energy_peratom \
    --output_plot text_scale_evolution.png

# 训练完成后分析
python diagnose_fusion_effectiveness.py \
    --checkpoint ./outputs/fusion_improved/.../best_model.pt \
    --root_dir ../dataset/

# 对比两个模型
python compare_fusion_models.py \
    --model1 best_test_model_scaled_12.0.pt \
    --model2 ./outputs/fusion_improved/.../best_model.pt \
    --root_dir ../dataset/

# 生成最终论文图表
python 1.extract_alpha_final.py \
    --checkpoint ./outputs/fusion_improved/.../best_model.pt ...
python 2.create_paper_alpha_figures.py
```

---

## 📝 总结

### 推荐配置

**生产使用**（最佳性能）:
```bash
--use_middle_fusion True \
--middle_fusion_use_gate_norm True \
--middle_fusion_use_learnable_scale True \
--middle_fusion_initial_scale 12.0
```

**实验探索**（保守）:
```bash
--use_middle_fusion True \
--middle_fusion_use_gate_norm True \
--middle_fusion_use_learnable_scale True \
--middle_fusion_initial_scale 1.0  # 让模型从头学习
```

**激进尝试**（文本特征极弱时）:
```bash
--use_middle_fusion True \
--middle_fusion_use_learnable_scale True \
--middle_fusion_initial_scale 20.0
```

---

## 📚 相关文档

- `IMPROVED_MIDDLE_FUSION_GUIDE.md` - 完整的功能说明和理论背景
- `config_improved_fusion.json` - 配置文件示例
- `monitor_text_scale.py` - 训练监控工具
- `compare_fusion_models.py` - 模型对比工具
- `FINAL_SUMMARY.md` - Alpha 分析项目完整总结

---

## 💡 提示

1. **初始缩放值选择**: 建议先运行诊断工具确定最优值
2. **监控训练**: 使用 `monitor_text_scale.py` 跟踪缩放因子演变
3. **对比实验**: 使用 `compare_fusion_models.py` 量化改进效果
4. **调试**: 如果效果不佳，运行 `diagnose_fusion_effectiveness.py`

祝训练顺利！ 🚀
