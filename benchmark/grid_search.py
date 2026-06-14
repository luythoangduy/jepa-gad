import os
import torch
import warnings
import itertools
from collections import defaultdict
import numpy as np
import tqdm

# Bỏ qua các warning không cần thiết để log sạch hơn
warnings.filterwarnings('ignore')

from pygod.detector import GADJEPA
from pygod.utils import load_data
from pygod.metric import eval_roc_auc, eval_average_precision, eval_recall_at_k
from torch_geometric.nn import GAT

def run_grid_search():
    # 3 Dataset thực tế (Real-world datasets), không dùng bộ injected/synthetic
    datasets = ['reddit']
    num_trial = 5
    epoch = 100
    gpu = 1  # Đổi thành 0 nếu có GPU
    
    # Định nghĩa Grid (Không gian tìm kiếm)
    mask_rates = [0.1, 0.3, 0.5, 0.7]
    contrast_modes = ['none', 'linear', 'infonce']
    normal_weights = [0.0, 0.5]  # 0.0 tương đương với w/o Normal Projection
    
    grid = list(itertools.product(mask_rates, contrast_modes, normal_weights))
    print(f"Tổng số cấu hình Grid: {len(grid)}\n")
    
    results = defaultdict(dict)
    
    for dataset_name in datasets:
        print(f"{'='*50}\nEvaluating Dataset: {dataset_name}\n{'='*50}")
        try:
            data = load_data(dataset_name)
        except Exception as e:
            print(f"Không thể load dataset {dataset_name}. Lỗi: {e}")
            continue
            
        y = data.y.bool()
        k = int(sum(y))
        
        # Giữ nguyên logic batching từ utils.py
        if dataset_name in ['inj_flickr', 'dgraph']:
            batch_size = 64
            num_neigh = 3
        else:
            batch_size = 0
            num_neigh = -1
            
        for config_idx, (mask_rate, contrast_mode, normal_weight) in enumerate(grid):
            print(f"[{config_idx+1}/{len(grid)}] mask={mask_rate}, contrast={contrast_mode}, normal_w={normal_weight}")
            auc_list, ap_list, rec_list = [], [], []
            
            for trial in tqdm.tqdm(range(num_trial), desc="Trials", leave=False):
                # Cố định các hyperparam khác (hid_dim, lr) để đánh giá đúng ablation
                model = GADJEPA(
                    hid_dim=64, 
                    lr=0.01,
                    epoch=epoch, 
                    gpu=gpu,
                    batch_size=batch_size,
                    num_neigh=num_neigh,
                    backbone=GAT,
                    v2=True,   # Kích hoạt GATv2
                    mask_rate=mask_rate,
                    contrast_mode=contrast_mode,
                    contrast_weight=0.1 if contrast_mode != 'none' else 0.0,
                    normal_weight=normal_weight,
                    verbose=0  # Tắt log epoch
                )
                
                try:
                    model.fit(data)
                    score = model.decision_score_
                    
                    if torch.isnan(score).any():
                        print(f"  Trial {trial+1}: Bị lỗi NaN, bỏ qua.")
                        continue
                        
                    auc_list.append(eval_roc_auc(y, score))
                    ap_list.append(eval_average_precision(y, score))
                    rec_list.append(eval_recall_at_k(y, score, k))
                except Exception as e:
                    print(f"  Trial {trial+1} Lỗi: {e}")
            
            if len(auc_list) > 0:
                # Chuyển đổi tensor list thành số thực
                auc_tensors = [float(x) for x in auc_list]
                ap_tensors = [float(x) for x in ap_list]
                
                mean_auc = np.mean(auc_tensors)
                std_auc = np.std(auc_tensors)
                mean_ap = np.mean(ap_tensors)
                
                print(f"  -> AUC: {mean_auc:.4f} ± {std_auc:.4f} | AP: {mean_ap:.4f}")
                results[dataset_name][config_idx] = {
                    'config': (mask_rate, contrast_mode, normal_weight),
                    'auc': mean_auc,
                    'auc_std': std_auc,
                    'ap': mean_ap
                }
            else:
                print("  -> Tất cả các trials đều lỗi/NaN.")
                
        # Tổng kết cấu hình tốt nhất cho dataset
        if results[dataset_name]:
            best_idx = max(results[dataset_name], key=lambda x: results[dataset_name][x]['auc'])
            best_res = results[dataset_name][best_idx]
            print(f"\n>>> Best Config cho {dataset_name}:")
            print(f"mask_rate={best_res['config'][0]}, contrast_mode={best_res['config'][1]}, normal_weight={best_res['config'][2]}")
            print(f"AUC = {best_res['auc']:.4f} ± {best_res['auc_std']:.4f}\n")

if __name__ == '__main__':
    run_grid_search()
