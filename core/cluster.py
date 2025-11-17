from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import faiss

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def threshold_clusters(vectors: Dict[int, np.ndarray], threshold: float = 0.78) -> Dict[int, List[int]]:
    # Build FAISS index
    ids = list(vectors.keys())
    if not ids:
        return {}
    dim = len(next(iter(vectors.values())))
    xb = np.stack([vectors[i] for i in ids]).astype('float32')
    index = faiss.IndexFlatIP(dim)
    # vectors already L2-normalized by embedder; use inner product as cosine
    faiss.normalize_L2(xb)
    index.add(xb)
    sims, neigh = index.search(xb, 10)  # top-10 neighbors

    # Union-Find
    parent = {i: i for i in range(len(ids))}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(ids)):
        for j in range(1, sims.shape[1]):
            if sims[i, j] >= threshold:
                union(i, neigh[i, j])

    clusters: Dict[int, List[int]] = {}
    for i, pid in enumerate(ids):
        root = find(i)
        clusters.setdefault(root, []).append(pid)

    # Remap roots to dense cluster IDs
    remap = {root: cid for cid, root in enumerate(sorted(clusters.keys()))}
    dense = {remap[root]: members for root, members in clusters.items()}
    return dense
