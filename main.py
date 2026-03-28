"""
Transmission Networks and Intervention Effects from
SARS-CoV-2 Genomic and Social Network Data in Denmark
Based on: Curran-Sebastian et al. (2026, medRxiv)

Team GYAN — includes:
  - Network construction (nodes, edges)
  - Degree distribution, density, centrality, clustering
  - Community detection (transmission clusters)
  - Figure replication (Fig 2A, 2B-G, 3, 4, 5)
  - EXTENSION: Robustness & Targeted Attack Analysis
    (Betweenness vs Degree removal strategies)
"""

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
from robustness_attack_analysis import (run_robustness_analysis,
                                         plot_robustness_analysis,
                                         plot_attack_interpretation,
                                         save_attack_csv)

# ── Output folders ─────────────────────────────────────────────────────────────
FIGURES_DIR = os.path.join("outputs", "figures")
CSV_DIR     = os.path.join("outputs", "csv")
RESULTS_DIR = os.path.join("outputs", "results")

for d in [FIGURES_DIR, CSV_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

N_TREES = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="GYAN — SARS-CoV-2 Transmission Network Analysis + Attack Extension"
    )
    parser.add_argument(
        "--fasta", type=str, default=None,
        help="Path to NCBI FASTA file (e.g. data/sequences.fasta)."
    )
    parser.add_argument(
        "--trees", type=int, default=N_TREES,
        help=f"Number of transmission trees to sample (default: {N_TREES})"
    )
    parser.add_argument(
        "--skip-attack", action="store_true",
        help="Skip the robustness/attack analysis extension (saves time)"
    )
    return parser.parse_args()


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════════════
# NETWORK ANALYSIS — Coursework required metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_network_metrics(G: nx.DiGraph) -> dict:
    """
    Computes all required network metrics:
      1. Nodes and edges
      2. Degree distribution (in + out)
      3. Network density
      4. Centrality: degree, betweenness, closeness, eigenvector
      5. Clustering coefficient
    """
    print("\n  [Network Metrics] Computing basic stats...")
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    print(f"    Nodes   : {n_nodes:,}")
    print(f"    Edges   : {n_edges:,}")
    print(f"    Density : {density:.6f}")

    print("  [Network Metrics] Computing degree distribution...")
    in_degrees  = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    in_deg_vals  = list(in_degrees.values())
    out_deg_vals = list(out_degrees.values())
    print(f"    In-degree  — mean: {np.mean(in_deg_vals):.3f}  "
          f"max: {max(in_deg_vals)}  min: {min(in_deg_vals)}")
    print(f"    Out-degree — mean: {np.mean(out_deg_vals):.3f}  "
          f"max: {max(out_deg_vals)}  min: {min(out_deg_vals)}")

    print("  [Network Metrics] Computing clustering coefficient...")
    G_und = G.to_undirected()
    avg_clustering = nx.average_clustering(G_und)
    print(f"    Average clustering coefficient: {avg_clustering:.4f}")

    print("  [Network Metrics] Computing centrality measures...")

    deg_centrality = nx.degree_centrality(G)
    top_deg = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"    Degree centrality   — mean: {np.mean(list(deg_centrality.values())):.5f}")
    print(f"    Top-5 (degree): {[f'Node {n}' for n,_ in top_deg]}")

    sample_k = min(500, n_nodes)
    print(f"    Betweenness centrality (sampled k={sample_k})...")
    bet_centrality = nx.betweenness_centrality(G, k=sample_k, normalized=True, seed=42)
    top_bet = sorted(bet_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"    Betweenness centrality — mean: {np.mean(list(bet_centrality.values())):.6f}")
    print(f"    Top-5 (betweenness): {[f'Node {n}' for n,_ in top_bet]}")

    print("    Closeness centrality (on largest WCC)...")
    wcc = max(nx.weakly_connected_components(G), key=len)
    G_wcc = G.subgraph(wcc).copy()
    close_centrality = nx.closeness_centrality(G_wcc)
    top_close = sorted(close_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"    Closeness centrality — mean: {np.mean(list(close_centrality.values())):.5f}")

    print("    Eigenvector centrality...")
    try:
        eig_centrality = nx.eigenvector_centrality(G, max_iter=500, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        print("    (Digraph did not converge — using undirected graph)")
        eig_centrality = nx.eigenvector_centrality(G_und, max_iter=500, tol=1e-6)
    top_eig = sorted(eig_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"    Eigenvector centrality — mean: {np.mean(list(eig_centrality.values())):.6f}")

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
    """2x2 panel: in-degree dist, out-degree dist, top-20 degree centrality, mean centrality."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Transmission Network Analysis\n"
        "SARS-CoV-2 Denmark (Replication: Curran-Sebastian et al. 2026)",
        fontsize=13, fontweight="bold"
    )

    ax = axes[0, 0]
    in_deg = metrics["in_degrees"]
    cnt = Counter(in_deg)
    xs, ys = zip(*sorted(cnt.items()))
    ax.bar(xs, ys, color="#2196F3", alpha=0.8, width=0.8)
    ax.set_xlabel("In-degree (number of plausible infectors)", fontsize=10)
    ax.set_ylabel("Number of individuals", fontsize=10)
    ax.set_title("A  In-Degree Distribution", fontsize=11, fontweight="bold")
    mean_in = np.mean(in_deg)
    ax.axvline(mean_in, color="red", linestyle="--", lw=1.5, label=f"Mean = {mean_in:.2f}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    out_deg = metrics["out_degrees"]
    cnt = Counter(out_deg)
    xs, ys = zip(*sorted(cnt.items()))
    ax.bar(xs, ys, color="#FF9800", alpha=0.8, width=0.8)
    ax.set_xlabel("Out-degree (number of plausible infectees)", fontsize=10)
    ax.set_ylabel("Number of individuals", fontsize=10)
    ax.set_title("B  Out-Degree Distribution", fontsize=11, fontweight="bold")
    mean_out = np.mean(out_deg)
    ax.axvline(mean_out, color="red", linestyle="--", lw=1.5, label=f"Mean = {mean_out:.2f}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

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
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(mean_vals) * 0.01,
                f"{val:.5f}", ha="center", fontsize=8)
    ax.set_ylabel("Mean Centrality Value", fontsize=10)
    ax.set_title("D  Mean Centrality by Type", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
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

    # ── STEP 3: Network Analysis ──────────────────────────────────────────────
    section("STEP 3 — Network Analysis (Degree, Density, Centrality, Clustering)")
    t0      = time.time()
    metrics = compute_network_metrics(G)
    print(f"  Done [{time.time()-t0:.1f}s]")

    plot_network_metrics(
        metrics,
        output_path=os.path.join(FIGURES_DIR, "fig6_network_metrics.png"))

    centrality_df = pd.DataFrame({
        "node":                   list(metrics["deg_centrality"].keys()),
        "degree_centrality":      list(metrics["deg_centrality"].values()),
        "betweenness_centrality": [metrics["bet_centrality"].get(n, 0)
                                   for n in metrics["deg_centrality"].keys()],
        "closeness_centrality":   [metrics["close_centrality"].get(n, 0)
                                   for n in metrics["deg_centrality"].keys()],
        "eigenvector_centrality": [metrics["eig_centrality"].get(n, 0)
                                   for n in metrics["deg_centrality"].keys()],
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
        df_t = classify_tree_settings(tree, social)
        setting_counts.update(df_t["setting"].value_counts().to_dict())
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
    section("STEP 6 — Reproduction Numbers & Overdispersion")
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
    section("STEP 7 — NPI Effectiveness Regression")
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
    section("STEP 8 — Save CSV Outputs")
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

    # ── STEP 9: Generate Replication Figures ─────────────────────────────────
    section("STEP 9 — Generate Figures")

    plot_weekly_proportions(
        w_prop,
        output_path=os.path.join(FIGURES_DIR, "fig1_weekly_proportions.png"))

    plot_age_matrices(
        ps_trees, pop, social,
        output_path=os.path.join(FIGURES_DIR, "fig2_age_matrices.png"))

    plot_cluster_statistics(
        clusters, rand_clusters,
        output_path=os.path.join(FIGURES_DIR, "fig3_clusters.png"))

    plot_reproduction_numbers(
        w_rc, agg_rc, od_time, od_set,
        output_path=os.path.join(FIGURES_DIR, "fig4_reproduction_numbers.png"))

    if not all_res.empty:
        plot_npi_effects(
            all_res,
            output_path=os.path.join(FIGURES_DIR, "fig5_npi_effects.png"))

    plot_network_metrics(
        metrics,
        output_path=os.path.join(FIGURES_DIR, "fig6_network_metrics.png"))

    plot_summary_dashboard(
        w_prop, od_set,
        all_res if not all_res.empty else pd.DataFrame(),
        cl_sum,
        output_path=os.path.join(FIGURES_DIR, "fig0_summary_dashboard.png"))

    # ── STEP 10: EXTENSION — Robustness & Targeted Attack Analysis ───────────
    attack_results = None
    if not args.skip_attack:
        section("STEP 10 — EXTENSION: Robustness & Targeted Attack Analysis")
        print("  Research question: Is targeting Bridgers (betweenness) or")
        print("  Hubs (degree) more effective for epidemic disruption?\n")
        t0 = time.time()

        # Reuse already-computed centrality from Step 3
        attack_results = run_robustness_analysis(
            G,
            bet_centrality=metrics["bet_centrality"],
            deg_centrality=metrics["deg_centrality"],
        )
        print(f"  Done [{time.time()-t0:.1f}s]")

        # Fig 7: Full 4-panel robustness analysis
        plot_robustness_analysis(
            attack_results,
            output_path=os.path.join(FIGURES_DIR, "fig7_attack_analysis.png"))

        # Fig 8: Key interpretation figure (suitable for report/presentation)
        plot_attack_interpretation(
            attack_results,
            output_path=os.path.join(FIGURES_DIR, "fig8_attack_interpretation.png"))

        # Save CSVs
        save_attack_csv(attack_results, CSV_DIR)

        # Print summary
        print("\n  Attack Analysis Summary:")
        print(attack_results["summary"].to_string(index=False))
    else:
        section("STEP 10 — EXTENSION: Skipped (--skip-attack flag set)")

    # ── STEP 11: Summary Report ───────────────────────────────────────────────
    section("STEP 11 — Summary Report")
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("SARS-CoV-2 TRANSMISSION NETWORK ANALYSIS — DENMARK\n")
        f.write("Team GYAN | Replication: Curran-Sebastian et al. (2026)\n")
        f.write("=" * 60 + "\n\n")

        f.write("DATA SOURCES\n")
        f.write(f"  Genomes    : {'Real NCBI FASTA' if args.fasta else 'Simulated'}\n")
        f.write(f"  Network    : POLYMOD (Mossong et al. 2008)\n")
        f.write(f"  NPI data   : OxCGRT Denmark (Oxford)\n")
        f.write(f"  Vaccination: SSI Denmark rollout schedule\n\n")

        f.write("NETWORK PROPERTIES\n")
        f.write(f"  Nodes                        : {metrics['n_nodes']:,}\n")
        f.write(f"  Edges                        : {metrics['n_edges']:,}\n")
        f.write(f"  Setting-linked pairs         : {shared:,} "
                f"({100*shared/max(metrics['n_edges'],1):.1f}%)\n")
        f.write(f"  Network density              : {metrics['density']:.6f}\n")
        f.write(f"  Avg. clustering coefficient  : {metrics['avg_clustering']:.4f}\n")
        f.write(f"  Mean degree centrality       : "
                f"{np.mean(list(metrics['deg_centrality'].values())):.5f}\n")
        f.write(f"  Mean betweenness centrality  : "
                f"{np.mean(list(metrics['bet_centrality'].values())):.6f}\n\n")

        f.write("DEGREE DISTRIBUTION\n")
        f.write(f"  In-degree  — mean: {np.mean(metrics['in_degrees']):.3f}  "
                f"max: {max(metrics['in_degrees'])}\n")
        f.write(f"  Out-degree — mean: {np.mean(metrics['out_degrees']):.3f}  "
                f"max: {max(metrics['out_degrees'])}\n\n")

        f.write("COMMUNITY DETECTION (Transmission Clusters)\n")
        f.write(f"  Clusters found (size > 5): {len(clusters):,}\n")
        for _, row in cl_sum.iterrows():
            f.write(f"  {row['variant']:10s}: {row['n_clusters']} clusters "
                    f"(largest={row['largest_size']})\n")

        f.write("\nOVERDISPERSION (k parameter)\n")
        for _, row in od_set.iterrows():
            k_str = f"{row['k']:.3f}" if not pd.isna(row['k']) else "NaN (sparse)"
            f.write(f"  {row['setting']:15s}: k = {k_str}  "
                    f"(mean Rc = {row['mean']:.3f})\n")

        f.write("\nNPI EFFECTS\n")
        if not npi_eff.empty:
            for _, row in npi_eff.iterrows():
                direction = "REDUCES" if row["beta"] < 0 else "increases"
                sig = "*" if row["pvalue"] < 0.05 else "(not sig.)"
                f.write(f"  {row['NPI']:28s}: {direction}  "
                        f"beta={row['beta']:+.3f}  {sig}\n")

        f.write("\nVACCINATION EFFECTS\n")
        if not vacc.empty:
            for _, row in vacc.iterrows():
                dose  = "1-dose" if "one" in row["covariate"] else "2-dose"
                paper = "~15.5%" if "one" in row["covariate"] else "~23.5%"
                f.write(f"  {dose}: {row['reduction_pct']:.1f}% reduction "
                        f"(paper: {paper})\n")

        if attack_results is not None:
            f.write("\nEXTENSION: ROBUSTNESS & TARGETED ATTACK ANALYSIS\n")
            f.write("  Research question: Bridger vs Hub removal for epidemic control\n\n")
            s = attack_results["summary"]
            for _, row in s.iterrows():
                f.write(f"  {row['strategy']}\n")
                f.write(f"    LCC drops to 50% at : {row['collapse_at_50pct']*100:.1f}% removal\n")
                f.write(f"    LCC after 10% removal: {row['lcc_at_10pct_removal']*100:.1f}%\n")
                f.write(f"    LCC after 20% removal: {row['lcc_at_20pct_removal']*100:.1f}%\n")
                f.write(f"    LCC after 30% removal: {row['lcc_at_30pct_removal']*100:.1f}%\n\n")
            winner = s.sort_values("collapse_at_50pct").iloc[0]["strategy"]
            f.write(f"  CONCLUSION: {winner} is the more effective\n")
            f.write(f"  disruption strategy for this epidemic network.\n")

        f.write(f"\nFIGURES GENERATED\n")
        f.write(f"  fig0_summary_dashboard.png   — overview dashboard\n")
        f.write(f"  fig1_weekly_proportions.png  — replicates paper Fig 2A\n")
        f.write(f"  fig2_age_matrices.png        — replicates paper Fig 2B-G\n")
        f.write(f"  fig3_clusters.png            — replicates paper Fig 3\n")
        f.write(f"  fig4_reproduction_numbers.png— replicates paper Fig 4\n")
        f.write(f"  fig5_npi_effects.png         — replicates paper Fig 5\n")
        f.write(f"  fig6_network_metrics.png     — degree dist, centrality\n")
        if attack_results is not None:
            f.write(f"  fig7_attack_analysis.png     — EXTENSION: 4-panel attack\n")
            f.write(f"  fig8_attack_interpretation.png — EXTENSION: key finding\n")

        f.write(f"\nTotal runtime: {time.time()-start:.0f} seconds\n")

    print(f"  Saved: outputs/results/summary.txt")

    # ── Final summary ─────────────────────────────────────────────────────────
    section("COMPLETE")
    print(f"  Runtime : {time.time()-start:.0f} seconds\n")
    print(f"  Figures : {len(os.listdir(FIGURES_DIR))}")
    print(f"  CSVs    : {len(os.listdir(CSV_DIR))}")
    print(f"\n  Nodes              : {metrics['n_nodes']:,}")
    print(f"  Edges              : {metrics['n_edges']:,}")
    print(f"  Density            : {metrics['density']:.6f}")
    print(f"  Avg. Clustering    : {metrics['avg_clustering']:.4f}")
    print(f"  Clusters found     : {len(clusters):,}")
    if not all_res.empty:
        v2 = all_res.loc[(all_res["covariate"] == "vacc_two_doses") &
                          (all_res["setting"] == "total"), "beta"]
        if len(v2):
            print(f"  2-dose reduction   : {(1-np.exp(v2.values[0]))*100:.1f}%  (paper: ~23.5%)")
    if attack_results is not None:
        s   = attack_results["summary"].sort_values("collapse_at_50pct")
        win = s.iloc[0]
        print(f"\n  EXTENSION RESULT:")
        print(f"  Most effective strategy : {win['strategy']}")
        print(f"  LCC collapse at 50%     : {win['collapse_at_50pct']*100:.1f}% removal")
    print("=" * 60)


if __name__ == "__main__":
    main()