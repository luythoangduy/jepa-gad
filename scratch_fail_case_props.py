import torch
import numpy as np
from pygod.utils import load_data
from torch_geometric.utils import degree

def main():
    print("Loading reddit dataset...")
    data = load_data('reddit')
    y = data.y.bool().cpu().numpy()
    
    top_10_fp = [7825, 7443, 10581, 1547, 602, 5635, 7591, 6030, 9448, 2197]
    top_10_fn = [6538, 290, 8125, 501, 3228, 7149, 3814, 400, 6893, 2241]
    
    # Calculate degrees
    row, col = data.edge_index
    deg = degree(row, data.num_nodes, dtype=torch.float).numpy()
    
    # Calculate feature norms
    feat_norm = torch.norm(data.x, p=2, dim=1).numpy()
    
    print("\n=== GLOBAL STATS ===")
    print(f"Normal nodes ({np.sum(~y)}) - Avg Degree: {np.mean(deg[~y]):.2f}, Avg Feat Norm: {np.mean(feat_norm[~y]):.2f}")
    print(f"Anomaly nodes ({np.sum(y)}) - Avg Degree: {np.mean(deg[y]):.2f}, Avg Feat Norm: {np.mean(feat_norm[y]):.2f}")
    
    print("\n=== TOP 10 FP STATS (Normal nodes predicted as Anomaly) ===")
    print(f"Avg Degree: {np.mean(deg[top_10_fp]):.2f} (min: {np.min(deg[top_10_fp])}, max: {np.max(deg[top_10_fp])})")
    print(f"Avg Feat Norm: {np.mean(feat_norm[top_10_fp]):.2f}")
    for node in top_10_fp:
        print(f"  Node {node}: Degree {deg[node]:.0f}, FeatNorm {feat_norm[node]:.2f}")
        
    print("\n=== TOP 10 FN STATS (Anomaly nodes predicted as Normal) ===")
    print(f"Avg Degree: {np.mean(deg[top_10_fn]):.2f} (min: {np.min(deg[top_10_fn])}, max: {np.max(deg[top_10_fn])})")
    print(f"Avg Feat Norm: {np.mean(feat_norm[top_10_fn]):.2f}")
    for node in top_10_fn:
        print(f"  Node {node}: Degree {deg[node]:.0f}, FeatNorm {feat_norm[node]:.2f}")

if __name__ == '__main__':
    main()
