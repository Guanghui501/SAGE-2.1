# python 3.analyze_text_flow_v2.py --checkpoint bulk-new/output_100epochs_42_bs64_sw_ju_onlymiddle_bulk_modulus_kv_quantext/bulk_modulus_kv/best_test_model.pt --root_dir ~/crysmmnet-main/dataset/
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

# 引入项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))
from models.alignn import ALIGNN, ALIGNNConfig, MiddleFusionModule

# 复用之前的数据加载逻辑
from extract_alpha_final import SimpleDataset, load_local_data, get_dataset_paths, collate_fn

class TextFlowAnalyzer:
    def __init__(self, model):
        self.model = model
        self.layer_outputs = {}
        self.captured_text_emb = None # 用于存储截获的文本特征
        
    def register_hooks(self):
        """注册钩子：既捕获层输出，也捕获文本特征"""
        
        # 1. 捕获 GCN/ALIGNN 层的节点特征
        def get_layer_output(name):
            def hook(model, input, output):
                # ALIGNNLayer 返回 (x, y, z)，取 x
                val = output[0] if isinstance(output, tuple) else output
                if isinstance(val, torch.Tensor):
                    self.layer_outputs[name] = val.detach()
            return hook

        # 注册 ALIGNN 层
        if hasattr(self.model, 'alignn_layers'):
            for i, layer in enumerate(self.model.alignn_layers):
                layer.register_forward_hook(get_layer_output(f'Layer {i+1} (ALIGNN)'))
                
        # 注册 GCN 层
        if hasattr(self.model, 'gcn_layers'):
            for i, layer in enumerate(self.model.gcn_layers):
                layer.register_forward_hook(get_layer_output(f'Layer {i+5} (GCN)'))

        # 2. [关键修复] 捕获 MiddleFusionModule 内部处理好的文本特征
        # 我们直接 Hook 它的 text_transform 层，拿到 [Batch, 64] 的特征
        
        def get_text_emb(model, input, output):
            # output 就是经过 Linear+ReLU 后的文本特征，维度一定是匹配的
            self.captured_text_emb = output.detach()

        # 自动查找并注册 fusion hook
        fusion_found = False
        for module in self.model.modules():
            if isinstance(module, MiddleFusionModule):
                # Hook 它的 text_transform 子模块
                if hasattr(module, 'text_transform'):
                    module.text_transform.register_forward_hook(get_text_emb)
                    fusion_found = True
                    # print("✅ 已Hook到 MiddleFusionModule.text_transform")
                    break # 只需要Hook第一个找到的即可
        
        if not fusion_found:
            print("⚠️ 警告: 未找到 MiddleFusionModule，无法捕获文本特征！")

    def compute_layer_similarity(self, g, lg, text_list, device):
        """计算相似度"""
        
        # 1. 运行模型 (这会自动触发所有 Hooks)
        # 不需要手动跑 BERT 了，模型自己会跑
        _ = self.model((g, lg, text_list))
        
        # 2. 检查是否捕获到了文本特征
        if self.captured_text_emb is None:
            return None

        # self.captured_text_emb 是 [Batch, Node_Dim] (例如 64)
        text_emb = self.captured_text_emb
        
        # 3. 广播文本特征到每个原子 [Total_Nodes, Node_Dim]
        batch_num_nodes = g.batch_num_nodes().cpu().numpy()
        text_expanded_list = []
        for i, n in enumerate(batch_num_nodes):
            # 确保索引不超过 batch 范围
            if i < len(text_emb):
                text_expanded_list.append(text_emb[i].unsqueeze(0).repeat(n, 1))
        
        if not text_expanded_list:
            return None
            
        text_broadcasted = torch.cat(text_expanded_list, dim=0)

        # 4. 计算相似度
        similarities = {}
        skipped_layers = []

        for name, layer_feat in self.layer_outputs.items():
            # 维度校验
            if layer_feat.shape[1] == text_broadcasted.shape[1]:
                sim = F.cosine_similarity(layer_feat, text_broadcasted, dim=1)
                similarities[name] = sim.mean().item()

                # 诊断信息（仅第一次）
                if len(similarities) == 1:
                    print(f"\n   ✅ 维度匹配: layer_feat={layer_feat.shape}, text={text_broadcasted.shape}")
                    print(f"   - 层特征 L2 范数: {layer_feat.norm(dim=1).mean():.4f}")
                    print(f"   - 文本特征 L2 范数: {text_broadcasted.norm(dim=1).mean():.4f}")
                    print(f"   - 余弦相似度: {sim.mean().item():.4f}")
            else:
                skipped_layers.append((name, layer_feat.shape[1], text_broadcasted.shape[1]))

        # 报告跳过的层
        if skipped_layers:
            print(f"\n   ⚠️  跳过了 {len(skipped_layers)} 个维度不匹配的层:")
            for name, layer_dim, text_dim in skipped_layers[:3]:  # 只显示前3个
                print(f"      - {name}: layer_dim={layer_dim}, text_dim={text_dim}")

        return similarities

def plot_layer_similarity(sim_records, output_file):
    if not sim_records:
        print("❌ 数据为空，无法绘图")
        return

    layers = list(sim_records[0].keys())
    # 排序优化
    try:
        layers.sort(key=lambda x: int(x.split()[1]))
    except:
        pass
    
    means = [np.mean([r.get(l, 0) for r in sim_records]) for l in layers]
    stds = [np.std([r.get(l, 0) for r in sim_records]) for l in layers]
    
    plt.figure(figsize=(10, 6))
    x = range(len(layers))
    
    # --- 颜色修改：使用深紫色 ---
    # #482878 是 Viridis 色带起始的深紫色
    deep_purple = '#482878' 
    
    plt.plot(x, means, 'o-', color=deep_purple, linewidth=2.5, label='Text Retention')
    plt.fill_between(x, np.array(means)-np.array(stds), np.array(means)+np.array(stds), 
                     color=deep_purple, alpha=0.15)
    
    plt.xticks(x, layers, rotation=45)
    plt.ylabel('Cosine Similarity (Features vs Text)', fontsize=12, fontweight='bold')
    plt.title('Deep Information Flow: Text Retention Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"✅ 图表已保存: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--output_plot', default='figure_text_flow.pdf')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f" 加载模型: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Config 加载逻辑
    if 'config' in ckpt:
        config = ckpt['config']
    else:
        config_path = os.path.join(os.path.dirname(args.checkpoint), 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as f: config = ALIGNNConfig(**json.load(f))
        else:
            print("❌ 缺少 Config")
            return
            
    model = ALIGNN(config)
    
    # 权重加载逻辑
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False) # strict=False 容错
    model.to(device)
    model.eval()
    
    # 初始化分析器 (不需要 Tokenizer 了)
    analyzer = TextFlowAnalyzer(model)
    analyzer.register_hooks()
    
    # 数据加载
    if args.root_dir.endswith('cif'):
        cif_dir = args.root_dir
        csv_file = os.path.join(os.path.dirname(args.root_dir.rstrip('/')), 'description.csv')
    else:
        # 默认 Jarvis
        cif_dir = os.path.join(args.root_dir, 'jarvis/hse_bandgap-2/cif/')
        csv_file = os.path.join(args.root_dir, 'jarvis/hse_bandgap-2/description.csv')
        
    raw_data = load_local_data(cif_dir, csv_file, max_samples=100)
    loader = DataLoader(SimpleDataset(raw_data, None), batch_size=8, collate_fn=collate_fn)
    
    all_sims = []
    print(" 追踪信息流...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            g, lg, text_list, _, _, _ = batch
            g, lg = g.to(device), lg.to(device)
            
            sims = analyzer.compute_layer_similarity(g, lg, text_list, device)
            if sims:
                all_sims.append(sims)
                
    plot_layer_similarity(all_sims, args.output_plot)

if __name__ == '__main__':
    main()
