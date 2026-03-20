"""
transmission_clusters.py
========================

This module identifies and analyzes transmission clusters by combining
the transmission network (G) with the social network (N).

Main idea:
  - Keep only edges where two individuals share at least one setting
  - Treat the remaining graph as undirected
  - Connected components with size > 5 are considered clusters

We also create randomized networks for comparison to check whether
genetic data actually improves cluster detection.
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

RNG = np.random.default_rng(42)

CLUSTER_MIN_SIZE = 5   # clusters must be larger than this

# ── Build graph based on shared settings ───────────────────────────────────────
def build_settings_subgraph(G: nx.DiGraph, social_network: dict) -> nx.Graph:
    """
    Creates a subgraph by keeping only those edges where both individuals
    share at least one setting (household, school, etc.).

    The result is converted into an undirected graph for cluster detection.
    """
    VN = nx.Graph()
    VN.add_nodes_from(G.nodes(data=True))
    
    for u, v, data in G.edges(data=True):
        if data.get("shared_any", 0) == 1:
            VN.add_edge(u, v, **data)
    
    # Remove nodes that are completely disconnected
    isolated = [n for n in VN.nodes() if VN.degree(n) == 0]
    VN.remove_nodes_from(isolated)
    
    return VN

# ── Extract clusters from the graph ────────────────────────────────────────────
def extract_clusters(VN: nx.Graph,
                     pop: pd.DataFrame,
                     social_network: dict) -> list[dict]:
    """
    Finds connected components (clusters) and computes basic statistics
    for each one.
    """
    components = [c for c in nx.connected_components(VN) if len(c) > CLUSTER_MIN_SIZE]
    
    pop_idx = pop.set_index("individual_id")
    clusters = []
    
    for cid, members in enumerate(components):
        members_list = list(members)
        size = len(members_list)
        
        # Determine dominant variant in the cluster
        variants = [pop_idx.loc[m, "variant"] for m in members_list
                    if m in pop_idx.index]
        dominant_variant = Counter(variants).most_common(1)[0][0] if variants else "unknown"
        
        # Age distribution of cluster members
        ages = [pop_idx.loc[m, "age_group"] for m in members_list
                if m in pop_idx.index]
        
        # Count how many edges belong to each setting
        setting_counts = Counter()
        for e_u, e_v, d in VN.edges(members_list, data=True):
            if e_v in members:
                for s in ["household","school","workplace","family"]:
                    if d.get(f"shared_{s}", 0):
                        setting_counts[s] += 1
        
        # Identify dominant setting
        dominant_setting = "other"
        if setting_counts:
            top_setting, top_count = setting_counts.most_common(1)[0]
            if top_count >= 5:
                dominant_setting = top_setting
        
        # Number of unique regions in the cluster
        regions = [pop_idx.loc[m, "region"] for m in members_list
                   if m in pop_idx.index]
        n_regions = len(set(regions))
        
        clusters.append({
            "cluster_id":        cid,
            "size":              size,
            "dominant_variant":  dominant_variant,
            "dominant_setting":  dominant_setting,
            "n_regions":         n_regions,
            "setting_counts":    dict(setting_counts),
            "age_distribution":  Counter(ages),
        })
    
    return clusters

# ── Summarize clusters by variant ──────────────────────────────────────────────
def cluster_summary(clusters: list[dict]) -> pd.DataFrame:
    """
    Generates summary statistics for each variant:
      - number of clusters
      - largest cluster size
      - average number of regions
      - total individuals in clusters
    """
    records = []
    
    for v in ["B.1.177","Alpha","Eta","Delta","Omicron"]:
        v_clusters = [c for c in clusters if c["dominant_variant"] == v]
        
        if not v_clusters:
            records.append({
                "variant": v,
                "n_clusters": 0,
                "largest_size": 0,
                "mean_regions": 0,
                "prop_individuals": 0
            })
            continue
        
        records.append({
            "variant":           v,
            "n_clusters":        len(v_clusters),
            "largest_size":      max(c["size"] for c in v_clusters),
            "mean_regions":      np.mean([c["n_regions"] for c in v_clusters]),
            "prop_individuals":  sum(c["size"] for c in v_clusters),
        })
    
    return pd.DataFrame(records)

# ── Generate randomized networks ───────────────────────────────────────────────
def generate_randomised_network(G: nx.DiGraph,
                                pop: pd.DataFrame,
                                n_realisations: int = 20) -> list[list[dict]]:
    """
    Creates randomized versions of the network.

    The idea is to:
      - keep the same number of outgoing edges per node
      - ensure time consistency (infection within 11 days)
      - but ignore genetic similarity

    This helps check if real clusters are meaningful or just random.
    """
    pop_idx    = pop.set_index("individual_id")
    date_map   = pop_idx["test_date"].to_dict()
    ids        = list(G.nodes())
    
    all_rand_clusters = []
    
    for r in range(n_realisations):
        G_rand = nx.DiGraph()
        G_rand.add_nodes_from(G.nodes(data=True))
        
        for u in G.nodes():
            out_edges = list(G.out_edges(u))
            if not out_edges:
                continue
            
            k = len(out_edges)  # number of outgoing edges
            
            t_u = date_map.get(u)
            if t_u is None:
                continue
            
            # Possible infectees within 11 days
            candidates = [
                v for v in ids
                if v != u and 0 < (date_map.get(v, t_u) - t_u).days <= 11
            ]
            
            if not candidates:
                continue
            
            chosen = RNG.choice(candidates, size=min(k, len(candidates)), replace=False)
            
            for v in chosen:
                G_rand.add_edge(
                    u, v,
                    weight=1.0,
                    shared_any=0,
                    shared_household=0,
                    shared_school=0,
                    shared_workplace=0,
                    shared_family=0
                )
        
        all_rand_clusters.append(G_rand)
    
    return all_rand_clusters

# ── Extract clusters directly from a network ───────────────────────────────────
def clusters_from_network(G: nx.DiGraph,
                           social_network: dict,
                           pop: pd.DataFrame) -> list[dict]:
    """
    Convenience function: builds subgraph and extracts clusters in one step.
    """
    VN = build_settings_subgraph(G, social_network)
    return extract_clusters(VN, pop, social_network)

if __name__ == "__main__":
    from data_generation import generate_all
    from transmission_network import (build_transmission_network,
                                       annotate_with_settings)
    
    pop, genomes, social, npi = generate_all()
    
    print("\nBuilding transmission network...")
    G = build_transmission_network(pop, genomes)
    G = annotate_with_settings(G, social)
    
    print("Extracting transmission clusters...")
    VN       = build_settings_subgraph(G, social)
    clusters = extract_clusters(VN, pop, social)
    summary  = cluster_summary(clusters)
    
    print(f"\nFound {len(clusters)} clusters (size > {CLUSTER_MIN_SIZE})")
    
    print("\nCluster summary by variant:")
    print(summary.to_string(index=False))
    
    print("\nDominant settings distribution:")
    settings_dist = Counter(c["dominant_setting"] for c in clusters)
    for s, cnt in settings_dist.most_common():
        print(f"  {s}: {cnt}")
