# CONAD-JEPA: Contrastive Attributed Network Anomaly Detection
#             with JEPA-style Latent Prediction
# Gap fixes: PPR dual-view target, uncertainty weighting,
#            adaptive margin, z-score score normalization

"""Personalized PageRank utilities for CONAD-JEPA."""

import torch
from torch_geometric.utils import subgraph


def compute_ppr(edge_index, num_nodes, alpha=0.15, num_iter=10):
    """Compute a dense Personalized PageRank matrix by power iteration.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices with shape ``[2, num_edges]``.
    num_nodes : int
        Number of nodes in the graph.
    alpha : float, optional
        Propagation probability. Default: ``0.15``.
    num_iter : int, optional
        Number of power-iteration steps. Default: ``10``.

    Returns
    -------
    torch.Tensor
        Dense PPR matrix with shape ``[num_nodes, num_nodes]``.
    """
    device = edge_index.device
    dtype = torch.float32
    adj = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = 1.0

    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
    norm_adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

    eye = torch.eye(num_nodes, dtype=dtype, device=device)
    pi = eye.clone()
    for _ in range(num_iter):
        pi = (1.0 - alpha) * eye + alpha * norm_adj.t().matmul(pi)
    return pi


def get_topk_ppr_subgraph(v, Pi, edge_index, x, k=32):
    """Build the PPR top-k induced subgraph for a center node.

    Parameters
    ----------
    v : int or torch.Tensor
        Center node id.
    Pi : torch.Tensor
        Dense PPR matrix returned by :func:`compute_ppr`.
    edge_index : torch.Tensor
        Original edge indices with shape ``[2, num_edges]``.
    x : torch.Tensor
        Node feature matrix.
    k : int, optional
        Number of PPR neighbors to keep. Default: ``32``.

    Returns
    -------
    tuple
        ``(x_sub, edge_index_sub, ppr_weights, center_idx)`` where
        ``ppr_weights`` are edge weights derived from normalized PPR scores.
    """
    if torch.is_tensor(v):
        v = int(v.item())
    k = min(k, Pi.shape[0])
    scores, nodes = torch.topk(Pi[v], k=k)

    if not torch.any(nodes == v):
        nodes[-1] = v
        scores[-1] = Pi[v, v]

    order = torch.argsort(nodes)
    nodes = nodes[order]
    scores = scores[order]
    score_sum = scores.sum().clamp(min=1e-12)
    scores = scores / score_sum

    edge_index_sub, _ = subgraph(nodes, edge_index, relabel_nodes=True)
    node_to_pos = torch.empty(Pi.shape[0], dtype=torch.long,
                              device=nodes.device)
    node_to_pos[nodes] = torch.arange(nodes.numel(), device=nodes.device)
    center_idx = int(node_to_pos[torch.tensor(v, device=nodes.device)].item())

    if edge_index_sub.numel() == 0:
        ppr_weights = torch.empty(0, dtype=x.dtype, device=x.device)
    else:
        dst = edge_index_sub[1]
        ppr_weights = scores.to(x.device, x.dtype)[dst]
        ppr_weights = ppr_weights / ppr_weights.sum().clamp(min=1e-12)

    return x[nodes.to(x.device)], edge_index_sub.to(x.device), \
        ppr_weights, center_idx
