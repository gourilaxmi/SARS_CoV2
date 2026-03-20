"""
transmission_network.py
========================
Constructs the directed network G of plausible transmission pairs using:
  1. A mechanistic Poisson process model of viral evolution
     (substitution rate mu = 0.091 sub/day, generation time Gamma(4.87,1.98))
  2. Threshold: Hamming distance <= 2, serial interval <= 11 days
  3. Weights = P(D_AB | A->B) x P(H_AB | T_B - T_A)
  4. Prioritised-settings tree sampling (household > school > workplace > family)
  5. Random tree sampling (for sensitivity)

Performance: weight lookup table precomputed for all (dt, h) combos.
"""

import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import gamma, poisson, lognorm
from datetime import date
from collections import defaultdict

RNG = np.random.default_rng(42)

# Model parameters (from paper)
MU               = 0.091
MAX_HAM          = 2
MAX_SERIAL       = 11
GENERATION_MEAN  = 4.87
GENERATION_VAR   = 1.98
INCUBATION_LOG_MU    = np.log(4)
INCUBATION_LOG_SIGMA = np.log(4.7) - np.log(4)

# Gamma generation-time distribution
_GT_SHAPE = GENERATION_MEAN**2 / GENERATION_VAR
_GT_SCALE = GENERATION_VAR / GENERATION_MEAN
_gt_dist  = gamma(_GT_SHAPE, scale=_GT_SCALE)

# Incubation PMF
def _build_inc_pmf():
    rv = lognorm(s=INCUBATION_LOG_SIGMA, scale=np.exp(INCUBATION_LOG_MU))
    return {x: float(rv.cdf(x + 0.5) - rv.cdf(max(x - 0.5, 0))) for x in range(1, 50)}

_INC_PMF = _build_inc_pmf()

def _p_hamming(h, dt):
    total = 0.0
    for i in range(1, min(dt + 15, 40) + 1):
        p_inc = _INC_PMF.get(i, 0.0)
        if p_inc < 1e-8:
            continue
        lam = MU * (2*i - dt) if i <= dt else MU * float(dt)
        total += p_inc * float(poisson.pmf(h, max(lam, 1e-7)))
    return max(total, 1e-12)

# Precompute weight table
_WEIGHT_TABLE = {}
for _dt in range(1, MAX_SERIAL + 1):
    _p_gen = float(_gt_dist.pdf(_dt))
    for _h in range(MAX_HAM + 1):
        _WEIGHT_TABLE[(_dt, _h)] = _p_gen * _p_hamming(_h, _dt)

_MIN_WEIGHT = min(_WEIGHT_TABLE.values()) * 0.01


def edge_weight(dt, h):
    return _WEIGHT_TABLE.get((int(dt), int(h)), 0.0)


def hamming_distance(a, b):
    return int(np.sum(a != b))


def build_transmission_network(pop, genomes):
    G = nx.DiGraph()
    G.add_nodes_from(pop["individual_id"])
    for _, row in pop.iterrows():
        G.nodes[row["individual_id"]].update(row.to_dict())

    ids   = pop["individual_id"].values
    dates = np.array([(d - date(2020, 9, 1)).days for d in pop["test_date"]], dtype=np.int32)
    N     = len(ids)
    print(f"  Scanning {N} individuals for plausible pairs...", flush=True)

    # Group by test day
    day_to_idx = defaultdict(list)
    for idx, d in enumerate(dates):
        day_to_idx[int(d)].append(idx)
    all_days = sorted(day_to_idx.keys())

    edge_count = 0
    for dt in range(1, MAX_SERIAL + 1):
        for d_i in all_days:
            d_j = d_i + dt
            if d_j not in day_to_idx:
                continue
            inf_idxs  = day_to_idx[d_i]
            infe_idxs = day_to_idx[d_j]
            seqs_i = genomes[inf_idxs]
            seqs_j = genomes[infe_idxs]
            # Hamming matrix
            diff = np.sum(seqs_i[:, np.newaxis, :] != seqs_j[np.newaxis, :, :], axis=2)
            rows, cols = np.where(diff <= MAX_HAM)
            for r, c in zip(rows, cols):
                i_idx = inf_idxs[r]
                j_idx = infe_idxs[c]
                if i_idx == j_idx:
                    continue
                h = int(diff[r, c])
                w = edge_weight(dt, h)
                if w > _MIN_WEIGHT:
                    G.add_edge(ids[i_idx], ids[j_idx], weight=w, hamming=h, serial_interval=dt)
                    edge_count += 1

    print(f"  Plausible transmission pairs: {edge_count}")
    return G


def annotate_with_settings(G, social_network):
    for u, v, data in G.edges(data=True):
        s_u = social_network.get(u, {})
        s_v = social_network.get(v, {})
        for s in ["household", "school", "workplace", "family"]:
            data[f"shared_{s}"] = int(s in s_u and s in s_v and s_u[s] == s_v[s])
        data["shared_any"] = int(any(data.get(f"shared_{s}", 0)
                                      for s in ["household","school","workplace","family"]))
    return G


def _weighted_choice(candidates):
    ids, weights = zip(*candidates)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return int(RNG.choice(ids, p=weights))


def sample_prioritised_settings_tree(G, n_trees=100):
    trees = []
    priority = ["household","school","workplace","family"]
    for _ in range(n_trees):
        tree = {}
        for v in G.nodes():
            in_edges = list(G.in_edges(v, data=True))
            if not in_edges:
                tree[v] = None; continue
            chosen = None
            for s in priority:
                cands = [(u, d["weight"]) for u,_,d in in_edges if d.get(f"shared_{s}",0)]
                if cands:
                    chosen = _weighted_choice(cands); break
            if chosen is None:
                chosen = _weighted_choice([(u, d["weight"]) for u,_,d in in_edges])
            tree[v] = chosen
        trees.append(tree)
    return trees


def sample_random_trees(G, n_trees=100):
    trees = []
    for _ in range(n_trees):
        tree = {}
        for v in G.nodes():
            in_edges = list(G.in_edges(v, data=True))
            if not in_edges:
                tree[v] = None; continue
            tree[v] = _weighted_choice([(u, d["weight"]) for u,_,d in in_edges])
        trees.append(tree)
    return trees


def classify_tree_settings(tree, social_network):
    records = []
    for infectee, infector in tree.items():
        if infector is None: continue
        s_inf  = social_network.get(infector, {})
        s_infe = social_network.get(infectee, {})
        setting = "community"
        for s in ["household","school","workplace","family"]:
            if s in s_inf and s in s_infe and s_inf[s] == s_infe[s]:
                setting = s; break
        records.append({"infector": infector, "infectee": infectee, "setting": setting})
    return pd.DataFrame(records)
