
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_generation       import generate_all
from transmission_network  import (build_transmission_network,
                                    annotate_with_settings,
                                    sample_prioritised_settings_tree,
                                    sample_random_trees,
                                    classify_tree_settings)
from transmission_clusters import (build_settings_subgraph,
                                    extract_clusters,
                                    cluster_summary,
                                    generate_randomised_network)
from reproduction_numbers  import (compute_individual_Rc,
                                    weekly_Rc,
                                    aggregate_Rc_renewal,
                                    estimate_overdispersion_by_setting,
                                    overdispersion_over_time,
                                    weekly_setting_proportions)
from npi_analysis          import (build_regression_data,
                                    run_all_settings,
                                    vaccination_effect_summary,
                                    age_transmission_irr,
                                    NPI_COLS)
from visualizations        import (plot_weekly_proportions,
                                    plot_age_matrices,
                                    plot_cluster_statistics,
                                    plot_reproduction_numbers,
                                    plot_npi_effects,
                                    plot_summary_dashboard)

# ── Output folders ─────────────────────────────────────────────────────────────
FIGURES_DIR = os.path.join("outputs", "figures")
CSV_DIR     = os.path.join("outputs", "csv")
RESULTS_DIR = os.path.join("outputs", "results")

for d in [FIGURES_DIR, CSV_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

N_TREES = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="GYAN Replication — SARS-CoV-2 Transmission Network Analysis"
    )
    parser.add_argument(
        "--fasta", type=str, default=None,
        help="Path to NCBI FASTA file (e.g. data/sequences.fasta). "
             "If not provided, simulated genomes are used."
    )
    parser.add_argument(
        "--trees", type=int, default=N_TREES,
        help=f"Number of transmission trees to sample (default: {N_TREES})"
    )
    return parser.parse_args()


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════════════
# NETWORK ANALYSIS — All required coursework metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_network_metrics(G: nx.DiGraph) -> dict:
    """
    Computes all required network metrics:
      1. Nodes and edges
      2. Degree distribution (in + out)
      3. Network density
      4. Centrality: degree, betweenness, closeness, eigenvector
      5. Clustering coefficient
    Returns a dict of results.
    """
    print("\n  [Network Metrics] Computing basic stats...")
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    print(f"    Nodes   : {n_nodes:,}")
    print(f"    Edges   : {n_edges:,}")
    print(f"    Density : {density:.6f}")

    # ── Degree distribution ──────────────────────────────────────────────────
    print("  [Network Metrics] Computing degree distribution...")
    in_degrees  = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    in_deg_vals  = list(in_degrees.values())
    out_deg_vals = list(out_degrees.values())
    print(f"    In-degree  — mean: {np.mean(in_deg_vals):.3f}  "
          f"max: {max(in_deg_vals)}  min: {min(in_deg_vals)}")
    print(f"    Out-degree — mean: {np.mean(out_deg_vals):.3f}  "
          f"max: {max(out_deg_vals)}  min: {min(out_deg_vals)}")

    # ── Clustering coefficient ───────────────────────────────────────────────
    # Use undirected version for clustering (standard for contact networks)
    print("  [Network Metrics] Computing clustering coefficient...")
    G_und = G.to_undirected()
    avg_clustering = nx.average_clustering(G_und)
    print(f"    Average clustering coefficient: {avg_clustering:.4f}")

    # ── Centrality measures ──────────────────────────────────────────────────
    # For large graphs, betweenness/closeness are computed on a sample
    print("  [Network Metrics] Computing centrality measures...")

    # Degree centrality (fast, exact)
    deg_centrality = nx.degree_centrality(G)
    top_deg = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"    Degree centrality   — mean: {np.mean(list(deg_centrality.values())):.5f}")
    print(f"    Top-5 nodes (degree centrality): {[f'Node {n}' for n,_ in top_deg]}")

    # Betweenness centrality — sampled for speed on large graphs
    sample_k = min(500, n_nodes)
    print(f"    Betweenness centrality (sampled k={sample_k})...")
    bet_centrality = nx.betweenness_centrality(G, k=sample_k, normalized=True, seed=42)
    top_bet = sorted(bet_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"    Betweenness centrality — mean: {np.mean(list(bet_centrality.values())):.6f}")
    print(f"    Top-5 nodes (betweenness): {[f'Node {n}' for n,_ in top_bet]}")

    # Closeness centrality (computed on largest weakly connected component)
    print("    Closeness centrality (on largest WCC)...")
    wcc = max(nx.weakly_connected_components(G), key=len)
    G_wcc = G.subgraph(wcc).copy()
    close_centrality = nx.closeness_centrality(G_wcc)
    top_close = sorted(close_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"    Closeness centrality — mean: {np.mean(list(close_centrality.values())):.5f}")
    print(f"    Top-5 nodes (closeness): {[f'Node {n}' for n,_ in top_close]}")

    # Eigenvector centrality — may not converge for all digraphs; fallback to undirected
    print("    Eigenvector centrality...")
    try:
        eig_centrality = nx.eigenvector_centrality(G, max_iter=500, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        print("    (Digraph did not converge — using undirected graph)")
        eig_centrality = nx.eigenvector_centrality(G_und, max_iter=500, tol=1e-6)
    top_eig = sorted(eig_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"    Eigenvector centrality — mean: {np.mean(list(eig_centrality.values())):.6f}")
    print(f"    Top-5 nodes (eigenvector): {[f'Node {n}' for n,_ in top_eig]}")

    return {
        "n_nodes":          n_nodes,
        "n_edges":          n_edges,
        "density":          density,
        "in_degrees":       in_deg_vals,
        "out_degrees":      out_deg_vals,
        "avg_clustering":   avg_clustering,
        "deg_centrality":   deg_centrality,
        "bet_centrality":   bet_centrality,
        "close_centrality": close_centrality,
        "eig_centrality":   eig_centrality,
        "top_deg":          top_deg,
        "top_bet":          top_bet,
        "top_close":        top_close,
        "top_eig":          top_eig,
    }


def plot_network_metrics(metrics: dict,
                          output_path: str = "outputs/figures/fig6_network_metrics.png"):
    """
    Plots a 2x2 panel of:
      A — In-degree distribution
      B — Out-degree distribution
      C — Top-20 degree centrality
      D — Centrality comparison (deg / betweenness / closeness / eigenvector)
    This constitutes the network analysis figure required for coursework.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Transmission Network Analysis\n"
        "SARS-CoV-2",
        fontsize=13, fontweight="bold"
    )

    # ── A: In-degree distribution ───────────────────────────────────────────
    ax = axes[0, 0]
    in_deg = metrics["in_degrees"]
    cnt = Counter(in_deg)
    xs, ys = zip(*sorted(cnt.items()))
    ax.bar(xs, ys, color="#2196F3", alpha=0.8, width=0.8)
    ax.set_xlabel("In-degree (number of plausible infectors)", fontsize=10)
    ax.set_ylabel("Number of individuals", fontsize=10)
    ax.set_title("A  In-Degree Distribution", fontsize=11, fontweight="bold")
    ax.set_xlim(left=-0.5)
    mean_in = np.mean(in_deg)
    ax.axvline(mean_in, color="red", linestyle="--", lw=1.5,
               label=f"Mean = {mean_in:.2f}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── B: Out-degree distribution ──────────────────────────────────────────
    ax = axes[0, 1]
    out_deg = metrics["out_degrees"]
    cnt = Counter(out_deg)
    xs, ys = zip(*sorted(cnt.items()))
    ax.bar(xs, ys, color="#FF9800", alpha=0.8, width=0.8)
    ax.set_xlabel("Out-degree (number of plausible infectees)", fontsize=10)
    ax.set_ylabel("Number of individuals", fontsize=10)
    ax.set_title("B  Out-Degree Distribution", fontsize=11, fontweight="bold")
    ax.set_xlim(left=-0.5)
    mean_out = np.mean(out_deg)
    ax.axvline(mean_out, color="red", linestyle="--", lw=1.5,
               label=f"Mean = {mean_out:.2f}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── C: Top-20 nodes by degree centrality ────────────────────────────────
    ax = axes[1, 0]
    deg_c = metrics["deg_centrality"]
    top20 = sorted(deg_c.items(), key=lambda x: x[1], reverse=True)[:20]
    nodes_20, vals_20 = zip(*top20)
    y_pos = np.arange(len(nodes_20))
    ax.barh(y_pos, vals_20, color="#4CAF50", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Node {n}" for n in nodes_20], fontsize=7)
    ax.set_xlabel("Degree Centrality", fontsize=10)
    ax.set_title("C  Top-20 Nodes — Degree Centrality", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # ── D: Mean centrality comparison ───────────────────────────────────────
    ax = axes[1, 1]
    centrality_types = ["Degree", "Betweenness", "Closeness", "Eigenvector"]
    mean_vals = [
        np.mean(list(metrics["deg_centrality"].values())),
        np.mean(list(metrics["bet_centrality"].values())),
        np.mean(list(metrics["close_centrality"].values())),
        np.mean(list(metrics["eig_centrality"].values())),
    ]
    colors = ["#2196F3", "#F44336", "#FF9800", "#9C27B0"]
    bars = ax.bar(centrality_types, mean_vals, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, mean_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mean_vals) * 0.01,
                f"{val:.5f}", ha="center", fontsize=8)
    ax.set_ylabel("Mean Centrality Value", fontsize=10)
    ax.set_title("D  Mean Centrality by Type", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Annotate clustering coefficient
    ax.text(0.98, 0.95,
            f"Avg. Clustering Coeff.\n= {metrics['avg_clustering']:.4f}\n\n"
            f"Density = {metrics['density']:.6f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="grey"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args  = parse_args()
    start = time.time()

    # ── STEP 1: Data Generation ───────────────────────────────────────────────
    section("STEP 1 — Data Generation")
    pop, genomes, social, npi = generate_all(fasta_path=args.fasta)

    # ── STEP 2: Transmission Network ─────────────────────────────────────────
    section("STEP 2 — Building Transmission Network G")
    t0 = time.time()
    G  = build_transmission_network(pop, genomes)
    G  = annotate_with_settings(G, social)
    shared = sum(1 for _, _, d in G.edges(data=True) if d.get("shared_any", 0))
    print(f"  Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,} "
          f"| Setting-linked: {shared:,}  [{time.time()-t0:.1f}s]")

    # ── STEP 3: Network Analysis (Coursework Requirement) ────────────────────
    section("STEP 3 — Network Analysis (Degree, Density, Centrality, Clustering)")
    t0      = time.time()
    metrics = compute_network_metrics(G)

    print(f"\n  Network properties:")
    print(f"    Nodes                        : {metrics['n_nodes']:,}")
    print(f"    Edges                        : {metrics['n_edges']:,}")
    print(f"    Density                      : {metrics['density']:.6f}")
    print(f"    Avg. Clustering Coefficient  : {metrics['avg_clustering']:.4f}")
    print(f"    Mean Degree Centrality       : {np.mean(list(metrics['deg_centrality'].values())):.5f}")
    print(f"    Mean Betweenness Centrality  : {np.mean(list(metrics['bet_centrality'].values())):.6f}")
    print(f"    Mean Closeness Centrality    : {np.mean(list(metrics['close_centrality'].values())):.5f}")
    print(f"    Mean Eigenvector Centrality  : {np.mean(list(metrics['eig_centrality'].values())):.6f}")
    print(f"  Done [{time.time()-t0:.1f}s]")

    # Save network metrics figure
    plot_network_metrics(
        metrics,
        output_path=os.path.join(FIGURES_DIR, "fig6_network_metrics.png")
    )

    # Save centrality CSV
    centrality_df = pd.DataFrame({
        "node":              list(metrics["deg_centrality"].keys()),
        "degree_centrality": list(metrics["deg_centrality"].values()),
        "betweenness_centrality": [
            metrics["bet_centrality"].get(n, 0)
            for n in metrics["deg_centrality"].keys()
        ],
        "closeness_centrality": [
            metrics["close_centrality"].get(n, 0)
            for n in metrics["deg_centrality"].keys()
        ],
        "eigenvector_centrality": [
            metrics["eig_centrality"].get(n, 0)
            for n in metrics["deg_centrality"].keys()
        ],
    })
    centrality_df.to_csv(os.path.join(CSV_DIR, "centrality_measures.csv"), index=False)
    print(f"  Saved: outputs/csv/centrality_measures.csv")

    # ── STEP 4: Tree Sampling ─────────────────────────────────────────────────
    section(f"STEP 4 — Sampling {args.trees} Transmission Trees")
    t0       = time.time()
    ps_trees = sample_prioritised_settings_tree(G, n_trees=args.trees)
    r_trees  = sample_random_trees(G, n_trees=args.trees)
    print(f"  Done [{time.time()-t0:.1f}s]")

    setting_counts = Counter()
    for tree in ps_trees:
        df = classify_tree_settings(tree, social)
        setting_counts.update(df["setting"].value_counts().to_dict())
    total_ev = sum(setting_counts.values())
    print("\n  Setting distribution (prioritised trees):")
    for s, c in setting_counts.most_common():
        print(f"    {s:15s}: {c:7,}  ({100*c/total_ev:.1f}%)")

    # ── STEP 5: Community Detection / Clusters ────────────────────────────────
    section("STEP 5 — Transmission Cluster Analysis (Community Detection)")
    t0       = time.time()
    VN       = build_settings_subgraph(G, social)
    clusters = extract_clusters(VN, pop, social)
    cl_sum   = cluster_summary(clusters)
    print(f"  Clusters found (size > 5): {len(clusters):,}  [{time.time()-t0:.1f}s]")
    print(f"\n  By variant:\n{cl_sum.to_string(index=False)}")

    print("\n  Generating randomised networks for comparison...")
    rand_nets     = generate_randomised_network(G, pop, n_realisations=5)
    rand_clusters = []
    for rn in rand_nets:
        rn = annotate_with_settings(rn, social)
        rand_clusters.append(extract_clusters(
            build_settings_subgraph(rn, social), pop, social))

    # ── STEP 6: Reproduction Numbers ─────────────────────────────────────────
    t0      = time.time()
    rc_df   = compute_individual_Rc(ps_trees, pop, social)
    w_rc    = weekly_Rc(rc_df)
    agg_rc  = aggregate_Rc_renewal(pop)
    od_set  = estimate_overdispersion_by_setting(rc_df)
    od_time = overdispersion_over_time(rc_df)
    w_prop  = weekly_setting_proportions(ps_trees, pop, social)
    print(f"  Done [{time.time()-t0:.1f}s]")
    print(f"\n  Overdispersion by setting:")
    print(od_set[["setting", "mean", "k"]].to_string(index=False))

    # ── STEP 7: NPI Regression ────────────────────────────────────────────────
    t0       = time.time()
    reg_data = build_regression_data(rc_df, npi)
    all_res  = run_all_settings(reg_data)
    print(f"  Done [{time.time()-t0:.1f}s]")

    vacc    = pd.DataFrame()
    npi_eff = pd.DataFrame()

    if not all_res.empty:
        NPI_LABELS = {
            "facial_coverings": "Facial coverings",
            "gathering_restr":  "Gathering restrictions",
            "school_restr":     "School restrictions",
            "workplace_restr":  "Workplace restrictions",
            "travel_restr":     "Travel restrictions",
            "stay_home":        "Stay at home",
        }
        npi_eff = all_res[
            (all_res["setting"] == "total") &
            (all_res["covariate"].isin(NPI_COLS))
        ].copy()
        npi_eff["NPI"] = npi_eff["covariate"].map(NPI_LABELS)
        print(f"\n  NPI effect sizes (beta):")
        print(npi_eff[["NPI", "beta", "ci_low", "ci_high", "pvalue"]].to_string(index=False))

        vacc = vaccination_effect_summary(all_res[all_res["setting"] == "total"])
        print(f"\n  Vaccination effects:")
        print(vacc.to_string(index=False))

    # ── STEP 8: Save CSV Outputs ──────────────────────────────────────────────
    csv_files = {
        "cluster_summary.csv":            cl_sum,
        "overdispersion_by_setting.csv":  od_set,
        "weekly_setting_proportions.csv": w_prop,
        "aggregate_rc.csv":               agg_rc,
        "population.csv":                 pop,
        "centrality_measures.csv":        centrality_df,
    }
    if not all_res.empty:
        csv_files["npi_regression_results.csv"] = all_res

    for fname, df in csv_files.items():
        path = os.path.join(CSV_DIR, fname)
        df.to_csv(path, index=False)
        print(f"  Saved: outputs/csv/{fname}")

    # ── STEP 9: Generate Figures ──────────────────────────────────────────────

    # Fig 1: Weekly proportions (REPLICATES paper Figure 2A)
    plot_weekly_proportions(
        w_prop,
        output_path=os.path.join(FIGURES_DIR, "fig1_weekly_proportions.png"))

    # Fig 2: Age-structured matrices (REPLICATES paper Figure 2B-G)
    plot_age_matrices(
        ps_trees, pop, social,
        output_path=os.path.join(FIGURES_DIR, "fig2_age_matrices.png"))

    # Fig 3: Cluster statistics (REPLICATES paper Figure 3)
    plot_cluster_statistics(
        clusters, rand_clusters,
        output_path=os.path.join(FIGURES_DIR, "fig3_clusters.png"))

    # Fig 4: Reproduction numbers (REPLICATES paper Figure 4)
    plot_reproduction_numbers(
        w_rc, agg_rc, od_time, od_set,
        output_path=os.path.join(FIGURES_DIR, "fig4_reproduction_numbers.png"))

    # Fig 5: NPI effects (REPLICATES paper Figure 5)
    if not all_res.empty:
        plot_npi_effects(
            all_res,
            output_path=os.path.join(FIGURES_DIR, "fig5_npi_effects.png"))

    # Fig 6: Network metrics (NEW — coursework requirement)
    plot_network_metrics(
        metrics,
        output_path=os.path.join(FIGURES_DIR, "fig6_network_metrics.png"))

    # Fig 0: Summary dashboard
    plot_summary_dashboard(
        w_prop, od_set,
        all_res if not all_res.empty else pd.DataFrame(),
        cl_sum,
        output_path=os.path.join(FIGURES_DIR, "fig0_summary_dashboard.png"))

    # ── STEP 10: Summary Report ───────────────────────────────────────────────
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SARS-CoV-2 TRANSMISSION NETWORK ANALYSIS — DENMARK\n")

        f.write("DATA SOURCES\n")
        f.write(f"  Genomes    : {'Real NCBI FASTA' if args.fasta else 'Simulated (Poisson model)'}\n")
        f.write(f"  Network    : POLYMOD (Mossong et al. 2008)\n")
        f.write(f"  NPI data   : OxCGRT Denmark (Oxford)\n")
        f.write(f"  Vaccination: SSI Denmark rollout schedule\n\n")

        f.write("NETWORK PROPERTIES\n")
        f.write(f"  Nodes (individuals)          : {metrics['n_nodes']:,}\n")
        f.write(f"  Edges (transmission pairs)   : {metrics['n_edges']:,}\n")
        f.write(f"  Setting-linked pairs         : {shared:,} ({100*shared/max(metrics['n_edges'],1):.1f}%)\n")
        f.write(f"  Network density              : {metrics['density']:.6f}\n")
        f.write(f"  Avg. clustering coefficient  : {metrics['avg_clustering']:.4f}\n\n")

        f.write("DEGREE DISTRIBUTION\n")
        f.write(f"  In-degree  — mean: {np.mean(metrics['in_degrees']):.3f}  "
                f"max: {max(metrics['in_degrees'])}  min: {min(metrics['in_degrees'])}\n")
        f.write(f"  Out-degree — mean: {np.mean(metrics['out_degrees']):.3f}  "
                f"max: {max(metrics['out_degrees'])}  min: {min(metrics['out_degrees'])}\n\n")

        f.write("CENTRALITY MEASURES\n")
        f.write(f"  Degree centrality     (mean): {np.mean(list(metrics['deg_centrality'].values())):.5f}\n")
        f.write(f"  Betweenness centrality(mean): {np.mean(list(metrics['bet_centrality'].values())):.6f}\n")
        f.write(f"  Closeness centrality  (mean): {np.mean(list(metrics['close_centrality'].values())):.5f}\n")
        f.write(f"  Eigenvector centrality(mean): {np.mean(list(metrics['eig_centrality'].values())):.6f}\n")
        f.write(f"  Top-5 by degree     : {[f'Node {n}' for n,_ in metrics['top_deg']]}\n")
        f.write(f"  Top-5 by betweenness: {[f'Node {n}' for n,_ in metrics['top_bet']]}\n\n")

        f.write("COMMUNITY DETECTION (Transmission Clusters)\n")
        f.write(f"  Clusters found (size > 5)    : {len(clusters):,}\n")
        for _, row in cl_sum.iterrows():
            f.write(f"  {row['variant']:10s}: {row['n_clusters']} clusters  "
                    f"(largest={row['largest_size']})\n")

        f.write("\nOVERDISPERSION (k parameter)\n")
        for _, row in od_set.iterrows():
            f.write(f"  {row['setting']:15s}: k = {row['k']:.3f}  (mean Rc = {row['mean']:.3f})\n")

        f.write("\nNPI EFFECTS\n")
        if not npi_eff.empty:
            for _, row in npi_eff.iterrows():
                direction = "REDUCES" if row["beta"] < 0 else "increases"
                sig = "*" if row["pvalue"] < 0.05 else "(not sig.)"
                f.write(f"  {row['NPI']:28s}: {direction}  beta={row['beta']:+.3f}  {sig}\n")

        f.write("\nVACCINATION EFFECTS\n")
        if not vacc.empty:
            for _, row in vacc.iterrows():
                dose  = "1-dose" if "one" in row["covariate"] else "2-dose"
                paper = "~15.5%" if "one" in row["covariate"] else "~23.5%"
                f.write(f"  {dose}: {row['reduction_pct']:.1f}% reduction  (paper: {paper})\n")

        f.write(f"\nFIGURES GENERATED\n")
        f.write(f"  fig0_summary_dashboard.png  — overview dashboard\n")
        f.write(f"  fig1_weekly_proportions.png — replicates paper Fig 2A\n")
        f.write(f"  fig2_age_matrices.png        — replicates paper Fig 2B-G\n")
        f.write(f"  fig3_clusters.png            — replicates paper Fig 3\n")
        f.write(f"  fig4_reproduction_numbers.png— replicates paper Fig 4\n")
        f.write(f"  fig5_npi_effects.png         — replicates paper Fig 5\n")
        f.write(f"  fig6_network_metrics.png     — degree dist, centrality, clustering\n")

        f.write(f"\nTotal runtime: {time.time()-start:.0f} seconds\n")

    print(f"  Saved: outputs/results/summary.txt")

    # ── Final Print ───────────────────────────────────────────────────────────
    print(f"  Runtime : {time.time()-start:.0f} seconds\n")
    print(f"  outputs/figures/  -> {len(os.listdir(FIGURES_DIR))} figures")
    print(f"  outputs/csv/      -> {len(os.listdir(CSV_DIR))} CSV files")
    print(f"  outputs/results/  -> summary.txt\n")
    print(f"  Nodes              : {metrics['n_nodes']:,}")
    print(f"  Edges              : {metrics['n_edges']:,}")
    print(f"  Density            : {metrics['density']:.6f}")
    print(f"  Avg. Clustering    : {metrics['avg_clustering']:.4f}")
    print(f"  Clusters found     : {len(clusters):,}")
    if not all_res.empty:
        v2 = all_res.loc[(all_res["covariate"] == "vacc_two_doses") &
                          (all_res["setting"] == "total"), "beta"]
        if len(v2):
            print(f"  2-dose reduction   : {(1-np.exp(v2.values[0]))*100:.1f}%  (paper: ~23.5%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
