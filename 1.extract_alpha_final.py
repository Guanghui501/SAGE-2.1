#python extract_alpha_local.py     --checkpoint bulk-new/output_100epochs_42_bs64_sw_ju_onlymiddle_bulk_modulus_kv_quantext/bulk_modulus_kv/best_test_model.pt         --root_dir /public/home/ghzhang/crysmmnet-main/dataset         --dataset jarvis         --property hse_bandgap-2         --n_samples 500         --output alpha_values.npz
import os
import sys
import csv
import argparse
import numpy as np
import torch
import dgl
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# 添加源代码路径 (请根据实际路径调整)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))

from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from models.alignn import ALIGNN, MiddleFusionModule
from tokenizers.normalizers import BertNormalizer

# ==========================================
# 1. 数据加载与处理
# ==========================================

def get_dataset_paths(root_dir, dataset, property_name):
    """获取本地数据集文件路径"""
    property_map = {
        'hse_bandgap': 'hse_bandgap', 'hse_bandgap-2': 'hse_bandgap-2',
        'bandgap_mbj': 'mbj_bandgap', 'formation_energy': 'formation_energy_peratom',
    }
    
    if dataset == 'jarvis':
        prop = property_map.get(property_name, property_name)
        return (os.path.join(root_dir, f'jarvis/{prop}/cif/'), 
                os.path.join(root_dir, f'jarvis/{prop}/description.csv'))
    elif dataset == 'mp':
        # 根据你的目录结构适配 MP
        return (os.path.join(root_dir, 'mp_2018_new/'), 
                os.path.join(root_dir, 'mp_2018_new/mat_text.csv'))
    elif dataset == 'class':
        return (os.path.join(root_dir, f'class/{property_name}/cif/'), 
                os.path.join(root_dir, f'class/{property_name}/description.csv'))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def load_local_data(cif_dir, csv_file, max_samples=None):
    """加载 CSV 和 CIF"""
    print(f"读取数据:\n  CIF: {cif_dir}\n  CSV: {csv_file}")
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        try: headings = next(reader) 
        except: return []
        rows = [r for r in reader]
        
    data_list = []
    # 简单的文本归一化
    norm = BertNormalizer(lowercase=True)
    
    print(f" 开始处理 {min(len(rows), max_samples if max_samples else 99999)} 个样本...")
    for row in tqdm(rows):
        if max_samples and len(data_list) >= max_samples: break
        
        try:
            # 解析 CSV (根据不同数据集调整索引)
            # 假设 Jarvis 格式: [id, formula, target, text, ...]
            # 假设 MP 格式: [id, formula, target, ..., text, ...]
            if len(row) == 5: # Jarvis
                jid, target, text = row[0], row[2], row[3]
            elif len(row) >= 6: # MP
                jid, target, text = row[0], row[2], row[4]
            else:
                continue

            # 读取 CIF
            cif_path = os.path.join(cif_dir, f"{jid}.cif")
            if not os.path.exists(cif_path): 
                cif_path = os.path.join(cif_dir, jid) # 尝试无后缀
                if not os.path.exists(cif_path): continue
            
            atoms = Atoms.from_cif(cif_path)
            if atoms.num_atoms > 400: continue # 过滤过大晶体

            data_list.append({
                'atoms': atoms,
                'text': norm.normalize_str(text),
                'target': float(target),
                'jid': jid
            })
        except Exception:
            continue
            
    return data_list

class SimpleDataset(Dataset):
    def __init__(self, data, tokenizer=None):
        self.data = data
        #self.tokenizer = tokenizer
        
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # 构图
        #g = Graph.atom_dgl_multigraph(item['atoms'], cutoff=8.0, max_neighbors=12)
        graph_output = Graph.atom_dgl_multigraph(item['atoms'], cutoff=8.0, max_neighbors=12)

        if isinstance(graph_output, tuple):
            g = graph_output[0]
            lg = graph_output[1]
        else:
            g = graph_output
            lg = g.line_graph(backtracking=False)

            # 确保有原子序数 Z (用于后续按元素分析)
        if 'atom_types' not in g.ndata:
            if hasattr(item['atoms'], 'atomic_numbers'):
                z_vals = item['atoms'].atomic_numbers
            else:
                from jarvis.core.specie import Specie
                z_vals = [Specie(e).Z for e in item['atoms'].elements]
            #z_vals = [e.Z for e in item['atoms'].elements]
            g.ndata['atom_types'] = torch.tensor(z_vals).long()
            
        #lg = g.line_graph(backtracking=False)
        # 文本
        #tokens = self.tokenizer(item['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        text = item['text']

        return g, lg, text, item['target'], item['jid']

def collate_fn(batch):
    gs, lgs, texts, targets, jids = zip(*batch)
    # 保存 batch 中每个图的原子序数，供后续拆分使用
    atom_types_list = [g.ndata['atom_types'] for g in gs]
    batched_g = dgl.batch(gs)
    batched_lg = dgl.batch(lgs)
    batched_text = list(texts)
    batched_targets = torch.tensor(targets)
    return batched_g, batched_lg, batched_text, batched_targets, jids, atom_types_list

# ==========================================
# 2. 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--output', default='alpha_values.npz')
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset', default='jarvis')
    parser.add_argument('--property', default='hse_bandgap-2')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载模型
    print(f" 加载模型: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ALIGNN(ckpt['config'])
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    # 2. 定位 Fusion 模块 (用于提取属性)
    fusion_module = None
    for m in model.modules():
        if isinstance(m, MiddleFusionModule):
            fusion_module = m
            break
            
    if fusion_module is None:
        print("❌ 错误: 模型中未找到 MiddleFusionModule！")
        return

    # 3. 准备数据
    cif_dir, csv_file = get_dataset_paths(args.root_dir, args.dataset, args.property)
    raw_data = load_local_data(cif_dir, csv_file, args.n_samples)
    if not raw_data:
        print("❌ 未加载到数据，请检查路径。")
        return
        
    #tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert', model_max_length=512)

    loader = DataLoader(SimpleDataset(raw_data, tokenizer=None), 
                        batch_size=args.batch_size, 
                        collate_fn=collate_fn, 
                        shuffle=False, num_workers=0)

    # 4. 推理提取 Loop
    results = {'alphas': [], 'atom_types': [], 'targets': [], 'jids': []}
    
    print(" 开始提取 Alpha 值...")
    with torch.no_grad():
        for batch in tqdm(loader):
            g, lg, text_list, targets, jids, atom_types_list = batch
            
            # Forward
            _ = model((g.to(device), lg.to(device), text_list))
            
            # === 关键：从模块属性中读取 Alpha ===
            batch_alphas = None
            
            # 优先尝试读取 stored_alphas (推荐版本)
            if hasattr(fusion_module, 'stored_alphas') and fusion_module.stored_alphas is not None:
                # 应该是 [Total_Atoms] 的 CPU Tensor
                batch_alphas = fusion_module.stored_alphas.numpy()
                
            # 兼容尝试读取 gate_values (原始版本)
            elif hasattr(fusion_module, 'gate_values') and fusion_module.gate_values is not None:
                # 可能是 [Total_Atoms, Hidden] 的 GPU Tensor
                val = fusion_module.gate_values
                if val.dim() > 1: val = val.mean(dim=1) # 取平均
                batch_alphas = val.detach().cpu().numpy()
            
            if batch_alphas is None:
                print("⚠️ 警告: 未在模块中找到 stored_alphas 或 gate_values，跳过此Batch")
                continue

            # === 拆分 Batch ===
            batch_num_nodes = g.batch_num_nodes().cpu().numpy()
            idx = 0
            for i, n_atoms in enumerate(batch_num_nodes):
                # 提取单个晶体的 alpha
                c_alpha = batch_alphas[idx : idx + n_atoms]
                # 提取单个晶体的 Z (原子序数)
                c_z = atom_types_list[i].numpy()
                
                results['alphas'].append(c_alpha)
                results['atom_types'].append(c_z)
                results['targets'].append(targets[i].item())
                results['jids'].append(jids[i])
                
                idx += n_atoms

    # 5. 保存
    if len(results['alphas']) > 0:
        #np.savez(args.output, **results)
        #print(f"\n✅ 成功保存 {len(results['alphas'])} 个样本的 Alpha 值到: {args.output}")
        print(f" 正在打包数据 (共 {len(results['alphas'])} 个样本)...")
        save_dict = {
            'alphas': np.array(results['alphas'], dtype=object),
            'atom_types': np.array(results['atom_types'], dtype=object),
            'targets': np.array(results['targets']), # 标量，不需要 object
            'jids': np.array(results['jids'])        # 字符串/ID，不需要 object
        }
        np.savez(args.output, **save_dict)
        print(f"\n✅ 成功保存到: {args.output}")
    else:
        print("\n❌ 提取失败: 结果为空。")

if __name__ == '__main__':
    main()
