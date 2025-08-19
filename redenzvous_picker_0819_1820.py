# rendezvous_picker.py
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

from parameter import (
    COMMS_RANGE, INTENT_HORIZON, SENSOR_RANGE, FRONTIER_CELL_SIZE, FREE, UNKNOWN,
)
from utils import get_frontier_in_map, get_cell_position_from_coords, MapInfo

@dataclass
class RDVScore:
    Stime: float
    Sintent: float
    Sgain: float
    Soverlap: float
    total: float

def _nms_points(points: np.ndarray, radius: float, topk: int) -> np.ndarray:
    """简易NMS：按点密度排序后做半径抑制。"""
    if points.size == 0:
        return points
    pts = points.copy()
    picked = []
    used = np.zeros(len(pts), dtype=bool)
    # 以 kNN 密度近似“峰值”
    from sklearn.neighbors import NearestNeighbors  # 若无sklearn，可换成栅格聚类
    k = min(8, len(pts))
    nbrs = NearestNeighbors(n_neighbors=k).fit(pts)
    dists, _ = nbrs.kneighbors(pts)
    density = 1.0 / (np.mean(dists[:, 1:], axis=1) + 1e-6)
    order = np.argsort(-density)
    r2 = radius * radius
    for idx in order:
        if used[idx]:
            continue
        p = pts[idx]
        picked.append(p)
        if len(picked) >= topk:
            break
        # 抑制邻近
        diff = pts - p
        mask = (diff[:, 0] ** 2 + diff[:, 1] ** 2) <= r2
        used = used | mask
    return np.array(picked)

def _graph_index(node_manager):
    """把 nodes_dict 做成：坐标->index 与 index->坐标 的双向索引；邻接表。"""
    coords = []
    for n in node_manager.nodes_dict.__iter__():
        coords.append(np.array(n.data.coords, dtype=float))
    coords = np.array(coords).reshape(-1, 2)
    key_all = coords[:, 0] + 1j * coords[:, 1]
    key_to_idx = {k: i for i, k in enumerate(key_all)}
    adj = [[] for _ in range(len(coords))]
    for i, n in enumerate(node_manager.nodes_dict.__iter__()):
        nb = n.data.neighbor_set
        for c in nb:
            k = complex(c[0], c[1])
            j = key_to_idx.get(k, None)
            if j is not None:
                adj[i].append(j)
    return coords, key_to_idx, adj

def _nearest_node_idx(coords_all: np.ndarray, key_to_idx: dict, node_manager, p: np.ndarray) -> int:
    """找离 p 最近的图节点 index。优先精确匹配，否则 nearest_neighbors。"""
    k = complex(round(float(p[0]), 1), round(float(p[1]), 1))
    j = key_to_idx.get(k, None)
    if j is not None:
        return j
    # 退化：用 nodes_dict 的近邻
    nn = node_manager.nodes_dict.nearest_neighbors(p.tolist(), 1)[0].data.coords
    return key_to_idx.get(complex(nn[0], nn[1]))

def _multi_source_dists(coords_all: np.ndarray, adj: List[List[int]], src_indices: List[int]) -> List[np.ndarray]:
    """对每个源做一次无权图BFS距离（以 hop 数近似）；返回 hop 场（可转成米）。"""
    n = len(coords_all)
    dfields = []
    for s in src_indices:
        dist = np.full(n, np.inf)
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == np.inf:
                    dist[v] = dist[u] + 1
                    q.append(v)
        dfields.append(dist)
    return dfields

def _hop_to_meter(hops: np.ndarray, coords_all: np.ndarray, adj: List[List[int]]) -> np.ndarray:
    """把 hop 粗略映射为米（用平均邻边长度）。"""
    lens = []
    for i, neigh in enumerate(adj):
        p = coords_all[i]
        for j in neigh:
            lens.append(np.linalg.norm(p - coords_all[j]))
    edge_len = np.median(lens) if len(lens) > 0 else 1.0
    out = hops.copy()
    out[np.isfinite(out)] = out[np.isfinite(out)] * edge_len
    return out

def _intent_heat(worker, coords_all: np.ndarray, key_to_idx: dict, radius: float = 2.0) -> np.ndarray:
    """把所有机器人的短 intent 投成节点热力（简单圆核叠加）。"""
    heat = np.zeros(len(coords_all), dtype=float)
    r2 = radius * radius
    # 收集所有 intent 末端（也可把中间点一起投影）
    pts = []
    for r in worker.robots:
        if getattr(r, "intent_seq", None):
            for p in r.intent_seq:
                pts.append(np.array(p, dtype=float))
    if len(pts) == 0:
        return heat
    pts = np.array(pts).reshape(-1, 2)
    for i, c in enumerate(coords_all):
        diff = pts - c
        mask = (diff[:, 0] ** 2 + diff[:, 1] ** 2) <= r2
        heat[i] = float(mask.sum())
    if heat.max() > 0:
        heat = heat / (heat.max() + 1e-6)
    return heat

def _uncertainty_map(agent) -> MapInfo:
    """
    用 agent 的 belief + pred_mean 构造不确定度 U(x):
      U = 1_unknown + p*(1-p)
    返回与 agent.map_info 同原点/尺寸运动学一致的 MapInfo(U)
    """
    belief = agent.map_info.map
    if agent.pred_mean_map_info is None:
        U = (belief == UNKNOWN).astype(np.float32)
    else:
        p = agent.pred_mean_map_info.map.astype(np.float32) / float(FREE)
        U = (belief == UNKNOWN).astype(np.float32) + p * (1 - p)
        U = np.clip(U, 0.0, 2.0)
    return MapInfo(U, agent.map_info.map_origin_x, agent.map_info.map_origin_y, agent.cell_size)

def _gain_around(candidate_idx: int, coords_all: np.ndarray, adj: List[List[int]],
                 Umap: MapInfo, H_post_meter: float = 20.0) -> float:
    """从候选出发一个小规划窗（按米），估计能触达的结点集合的不确定度积分。"""
    # 把“米预算”换成 hop 预算
    # 用平均边长粗估
    lens = []
    for i, neigh in enumerate(adj):
        p = coords_all[i]
        for j in neigh:
            lens.append(np.linalg.norm(p - coords_all[j]))
    edge_len = np.median(lens) if len(lens) > 0 else 1.0
    hop_budget = max(1, int(np.floor(H_post_meter / edge_len)))

    # BFS 限深
    n = len(coords_all)
    seen = np.zeros(n, dtype=bool)
    depth = np.full(n, -1, dtype=int)
    q = deque([candidate_idx])
    seen[candidate_idx] = True
    depth[candidate_idx] = 0
    reach = [candidate_idx]
    while q:
        u = q.popleft()
        if depth[u] >= hop_budget:
            continue
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                depth[v] = depth[u] + 1
                reach.append(v)
                q.append(v)

    # 积分 U
    s = 0.0
    for idx in reach:
        cell = get_cell_position_from_coords(coords_all[idx], Umap)
        if (0 <= cell[1] < Umap.map.shape[0]) and (0 <= cell[0] < Umap.map.shape[1]):
            s += float(Umap.map[cell[1], cell[0]])
    return s

def _frontier_candidates(agent, topk=20) -> np.ndarray:
    """在 agent 的 belief 上取前沿中心并做NMS。"""
    frs = get_frontier_in_map(agent.map_info)  # set of world coords
    if len(frs) == 0:
        return np.zeros((0, 2), dtype=float)
    pts = np.array(list(frs), dtype=float).reshape(-1, 2)
    return _nms_points(pts, radius=agent.cell_size * 6, topk=topk)

def _intent_candidates(worker, topk=20) -> np.ndarray:
    """把多机器人短 intent 的点汇总，做密度NMS，作为“交叠热点”候选。"""
    pts = []
    for r in worker.robots:
        if getattr(r, "intent_seq", None):
            for p in r.intent_seq:
                pts.append(np.array(p, dtype=float))
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=float)
    pts = np.array(pts).reshape(-1, 2)
    return _nms_points(pts, radius=2.0, topk=topk)

def _center_candidates(worker) -> np.ndarray:
    """几何中心类：均值点 + 中位近似（一次Weiszfeld迭代）。"""
    P = np.array([r.location for r in worker.robots], dtype=float)
    mean = P.mean(axis=0, keepdims=True)
    # 简化Weiszfeld一步
    w = 1.0 / (np.linalg.norm(P - mean, axis=1) + 1e-6)
    w = w / w.sum()
    geo = (w[:, None] * P).sum(axis=0, keepdims=True)
    return np.concatenate([mean, geo], axis=0)

def generate_candidates(worker, k_frontier=20, k_intent=20) -> np.ndarray:
    """汇总并去重候选。"""
    # 用 agent0 的视角做近似（你也可以改成融合/全局）
    a0 = worker.robots[0]
    A = _frontier_candidates(a0, topk=k_frontier)
    B = _intent_candidates(worker, topk=k_intent)
    C = _center_candidates(worker)
    cand = np.vstack([A, B, C]) if A.size + B.size + C.size > 0 else np.zeros((0, 2))
    # 去重（0.1m 精度）
    if cand.size == 0:
        return cand
    keys = np.round(cand, 1)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return cand[idx]

def pick_rendezvous_point(worker,
                          H_max_meter: float = 30.0,
                          r_meet_frac: float = 0.4,
                          H_post_meter: float = 20.0):
    """
    只选/打分/返回绘图数据：
      返回 (best_center_xy, r_meet, meta)；若无候选，返回 (None, 0.0, {}).
    """
    cand = generate_candidates(worker)
    if cand.size == 0:
        return None, 0.0, {}

    coords_all, key_to_idx, adj = _graph_index(worker.node_manager)
    if len(coords_all) == 0:
        return None, 0.0, {}

    # 源：当前各机器人所处节点 index
    src_idx = []
    for r in worker.robots:
        idx = _nearest_node_idx(coords_all, key_to_idx, worker.node_manager, r.location)
        if idx is None:
            return None, 0.0, {}
        src_idx.append(idx)

    # 为每个机器人做一次 BFS（hop）
    dfields_hop = _multi_source_dists(coords_all, adj, src_idx)
    dfields_m = [_hop_to_meter(df, coords_all, adj) for df in dfields_hop]

    # 热力与不确定度
    intent_heat = _intent_heat(worker, coords_all, key_to_idx, radius=2.0)
    Umap = _uncertainty_map(worker.robots[0])

    # 开始评估候选
    Stime_list, Sintent_list, Sgain_list, Soverlap_dummy = [], [], [], []
    idx_list = []

    # 把候选落到最近节点
    for p in cand:
        j = _nearest_node_idx(coords_all, key_to_idx, worker.node_manager, p)
        if j is None:
            continue

        # 时间项（max距离）—— 允许等待，只看到齐的最慢者
        lat = max([dfm[j] for dfm in dfields_m])
        if not np.isfinite(lat) or lat > H_max_meter:
            continue  # 简易可行性：超窗直接丢弃

        # 归一化时间分数
        Stime = np.exp(-np.log(5.0) * (lat / max(H_max_meter, 1e-3)))  # H_max时≈1/5

        # 意图项：节点热力
        Sintent = float(intent_heat[j])

        # 信息增益：小窗可达 U 积分
        Sgain = _gain_around(j, coords_all, adj, Umap, H_post_meter=H_post_meter)

        # 占位（不计算路径重叠，先返回1.0）
        Soverlap = 1.0

        Stime_list.append(Stime)
        Sintent_list.append(Sintent)
        Sgain_list.append(Sgain)
        Soverlap_dummy.append(Soverlap)
        idx_list.append(j)

    if len(idx_list) == 0:
        return None, 0.0, {}

    # 归一化（避免全零）
    def norm1(x):
        x = np.array(x, dtype=float)
        return (x - x.min()) / (x.max() - x.min() + 1e-9) if len(x) else x

    St = np.array(Stime_list)
    Si = norm1(Sintent_list)
    Sg = norm1(Sgain_list)
    So = np.array(Soverlap_dummy)

    # 总分
    total = 0.4 * St + 0.25 * Si + 0.25 * Sg + 0.10 * So
    best_k = int(np.argmax(total))
    best_idx = idx_list[best_k]
    best_xy = coords_all[best_idx].copy()
    r_meet = float(r_meet_frac * COMMS_RANGE)

    meta = dict(
        Stime=float(St[best_k]),
        Sintent=float(Si[best_k]),
        Sgain=float(Sg[best_k]),
        Soverlap=float(So[best_k]),
        total=float(total[best_k]),
        lat_est=float(max([dfm[best_idx] for dfm in dfields_m])),
    )
    return best_xy, r_meet, meta
