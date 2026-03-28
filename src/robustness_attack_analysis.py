"""
robustness_attack_analysis.py
==============================
Extension Analysis — Team GYAN
Robustness and Targeted Attack Analysis on the SARS-CoV-2 Transmission Network

Research Question:
  Is it more effective to target high-Betweenness (bridger) nodes or
  high-Degree (hub) nodes to disrupt epidemic spread?

Methods:
  1. Targeted attack by Betweenness Centrality  — removes nodes that act as
     bridges between communities (disrupts long-range spread)
  2. Targeted attack by Degree Centrality       — removes nodes with most
     connections (disrupts local hubs / superspreaders)
  3. Random removal baseline                    — removes nodes at random
     (simulates natural attrition / random vaccination)

Metrics tracked after each removal step:
  - Largest weakly connected component (LCC) size  → proxy for epidemic reach
  - Number of connected components                 → proxy for fragmentation
  - Network density
  - Average shortest path length (on LCC sample)   → proxy for transmission speed

Each strategy removes nodes in increments of STEP_SIZE % of the original
network, up to MAX_REMOVAL % total removal.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ── Parameters ─────────────────────────────────────────────────────────────────
STEP_SIZE   = 0.02   # remove 2% of nodes per step
MAX_REMOVAL = 0.30   # stop after removing 30% of nodes
N_RANDOM_RUNS = 10   # number of random removal replicates for CI bands
PATH_SAMPLE   = 300  # nodes to sample for avg path length (speed vs accuracy)

STRATEGY_COLORS = {
    "Betweenness (Bridger)"  : "#E53935",   # red
    "Degree (Hub)"           : "#1E88E5",   # blue
    "Random Baseline"        : "#43A047",   # green
}


# ══════════════════════════════════════════════════════════════════════════════
# CORE ATTACK SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def _lcc_size(G: nx.DiGraph) -> int:
    """Size of largest weakly connected component."""
    if G.number_of_nodes() == 0:
        return 0
    return len(max(nx.weakly_connected_components(G), key=len))


def _n_components(G: nx.DiGraph) -> int:
    """Number of weakly connected components."""
    return nx.number_weakly_connected_components(G)


def _avg_path_length(G: nx.DiGraph, sample: int = PATH_SAMPLE) -> float:
    """
    Approximate average shortest path length on the LCC.
    Sampled for speed — exact computation is O(N^2).
    Returns np.nan if LCC is too small.
    """
    if G.number_of_nodes() < 5:
        return np.nan
    lcc_nodes = max(nx.weakly_connected_components(G), key=len)
    if len(lcc_nodes) < 5:
        return np.nan
    G_lcc  = G.subgraph(lcc_nodes).copy()
    G_und  = G_lcc.to_undirected()
    sample_nodes = list(G_und.nodes())
    if len(sample_nodes) > sample:
        rng = np.random.default_rng(42)
        sample_nodes = list(rng.choice(sample_nodes, size=sample, replace=False))
    try:
        lengths = []
        for src in sample_nodes:
            sp = nx.single_source_shortest_path_length(G_und, src)
            lengths.extend(sp.values())
        return float(np.mean(lengths)) if lengths else np.nan
    except Exception:
        return np.nan


def _snapshot_metrics(G: nx.DiGraph, n_original: int) -> dict:
    """Collect all metrics for one snapshot of the network."""
    n = G.number_of_nodes()
    lcc = _lcc_size(G)
    return {
        "nodes_remaining":    n,
        "frac_remaining":     n / max(n_original, 1),
        "lcc_frac":           lcc / max(n_original, 1),
        "n_components":       _n_components(G),
        "density":            nx.density(G),
        "avg_path_length":    _avg_path_length(G),
    }


def run_attack(G_orig: nx.DiGraph,
               strategy: str,
               centrality_scores: dict = None,
               step_size: float = STEP_SIZE,
               max_removal: float = MAX_REMOVAL) -> pd.DataFrame:
    """
    Simulate sequential node removal under a given strategy.

    Parameters
    ----------
    G_orig           : original directed transmission network
    strategy         : "betweenness" | "degree" | "random"
    centrality_scores: precomputed dict {node: score} for targeted strategies
    step_size        : fraction of ORIGINAL nodes removed per step
    max_removal      : total fraction of original nodes to remove

    Returns
    -------
    pd.DataFrame with columns:
      step, frac_removed, nodes_remaining, frac_remaining,
      lcc_frac, n_components, density, avg_path_length
    """
    G         = G_orig.copy()
    n_orig    = G_orig.number_of_nodes()
    step_n    = max(1, int(step_size * n_orig))
    max_steps = int(max_removal / step_size)

    # Pre-sort removal order for targeted strategies
    if strategy in ("betweenness", "degree") and centrality_scores:
        sorted_nodes = sorted(
            centrality_scores.keys(),
            key=lambda x: centrality_scores[x],
            reverse=True
        )
        removal_queue = list(sorted_nodes)
    else:
        # Random: shuffle once per run
        rng = np.random.default_rng(None)   # different seed each call
        removal_queue = list(G.nodes())
        rng.shuffle(removal_queue)

    records = []
    # Baseline (0% removed)
    m = _snapshot_metrics(G, n_orig)
    records.append({"step": 0, "frac_removed": 0.0, **m})

    removed_so_far = 0
    queue_ptr      = 0

    for step in range(1, max_steps + 1):
        # Remove next batch
        to_remove = []
        while len(to_remove) < step_n and queue_ptr < len(removal_queue):
            node = removal_queue[queue_ptr]
            queue_ptr += 1
            if G.has_node(node):
                to_remove.append(node)

        if not to_remove:
            break

        G.remove_nodes_from(to_remove)
        removed_so_far += len(to_remove)
        frac_removed    = removed_so_far / n_orig

        m = _snapshot_metrics(G, n_orig)
        records.append({"step": step, "frac_removed": frac_removed, **m})

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_robustness_analysis(G: nx.DiGraph,
                             bet_centrality: dict,
                             deg_centrality: dict) -> dict:
    """
    Run all three attack strategies and return results dict.

    Parameters
    ----------
    G               : full transmission network (DiGraph)
    bet_centrality  : precomputed betweenness centrality {node: score}
    deg_centrality  : precomputed degree centrality {node: score}

    Returns
    -------
    dict with keys:
      "betweenness"  : DataFrame of betweenness attack results
      "degree"       : DataFrame of degree attack results
      "random_runs"  : list of DataFrames (N_RANDOM_RUNS replicates)
      "random_mean"  : mean across random runs
      "summary"      : summary statistics DataFrame
    """
    print("\n  [Attack Analysis] Running betweenness attack...")
    df_bet = run_attack(G, "betweenness", centrality_scores=bet_centrality)

    print("  [Attack Analysis] Running degree attack...")
    df_deg = run_attack(G, "degree", centrality_scores=deg_centrality)

    print(f"  [Attack Analysis] Running {N_RANDOM_RUNS} random baseline runs...")
    random_runs = []
    for i in range(N_RANDOM_RUNS):
        df_r = run_attack(G, "random")
        random_runs.append(df_r)

    # Align random runs on same frac_removed grid as betweenness
    frac_grid = df_bet["frac_removed"].values
    aligned   = []
    for df_r in random_runs:
        interp_lcc = np.interp(frac_grid, df_r["frac_removed"], df_r["lcc_frac"])
        interp_comp = np.interp(frac_grid, df_r["frac_removed"], df_r["n_components"])
        aligned.append(pd.DataFrame({
            "frac_removed": frac_grid,
            "lcc_frac":     interp_lcc,
            "n_components": interp_comp,
        }))

    rand_mean = pd.DataFrame({
        "frac_removed": frac_grid,
        "lcc_frac":     np.mean([d["lcc_frac"].values for d in aligned], axis=0),
        "lcc_std":      np.std( [d["lcc_frac"].values for d in aligned], axis=0),
        "n_components": np.mean([d["n_components"].values for d in aligned], axis=0),
    })

    # Summary: at what % removal does LCC drop below 50% of original?
    def collapse_point(df, threshold=0.5):
        below = df[df["lcc_frac"] < threshold]
        if below.empty:
            return df["frac_removed"].iloc[-1]
        return float(below["frac_removed"].iloc[0])

    summary = pd.DataFrame([
        {
            "strategy":          "Betweenness (Bridger)",
            "collapse_at_50pct": collapse_point(df_bet),
            "lcc_at_10pct_removal": float(df_bet[df_bet["frac_removed"] <= 0.10]["lcc_frac"].iloc[-1]),
            "lcc_at_20pct_removal": float(df_bet[df_bet["frac_removed"] <= 0.20]["lcc_frac"].iloc[-1]),
            "lcc_at_30pct_removal": float(df_bet[df_bet["frac_removed"] <= 0.30]["lcc_frac"].iloc[-1]),
        },
        {
            "strategy":          "Degree (Hub)",
            "collapse_at_50pct": collapse_point(df_deg),
            "lcc_at_10pct_removal": float(df_deg[df_deg["frac_removed"] <= 0.10]["lcc_frac"].iloc[-1]),
            "lcc_at_20pct_removal": float(df_deg[df_deg["frac_removed"] <= 0.20]["lcc_frac"].iloc[-1]),
            "lcc_at_30pct_removal": float(df_deg[df_deg["frac_removed"] <= 0.30]["lcc_frac"].iloc[-1]),
        },
        {
            "strategy":          "Random Baseline",
            "collapse_at_50pct": collapse_point(rand_mean),
            "lcc_at_10pct_removal": float(rand_mean[rand_mean["frac_removed"] <= 0.10]["lcc_frac"].iloc[-1]),
            "lcc_at_20pct_removal": float(rand_mean[rand_mean["frac_removed"] <= 0.20]["lcc_frac"].iloc[-1]),
            "lcc_at_30pct_removal": float(rand_mean[rand_mean["frac_removed"] <= 0.30]["lcc_frac"].iloc[-1]),
        },
    ])

    print("\n  Attack Analysis Summary:")
    print(summary.to_string(index=False))

    return {
        "betweenness":  df_bet,
        "degree":       df_deg,
        "random_runs":  random_runs,
        "random_mean":  rand_mean,
        "summary":      summary,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_robustness_analysis(results: dict,
                              output_path: str = "outputs/figures/fig7_attack_analysis.png"):
    """
    Produces a 4-panel figure:
      A — LCC size vs % nodes removed (main comparison)
      B — Number of components vs % removed (fragmentation)
      C — Network density vs % removed
      D — Summary bar chart: LCC remaining at 10 / 20 / 30% removal
    """
    df_bet  = results["betweenness"]
    df_deg  = results["degree"]
    r_mean  = results["random_mean"]
    summary = results["summary"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Robustness & Targeted Attack Analysis\n"
        "SARS-CoV-2 Transmission Network — Team GYAN Extension",
        fontsize=14, fontweight="bold"
    )

    pct_bet = df_bet["frac_removed"] * 100
    pct_deg = df_deg["frac_removed"] * 100
    pct_rnd = r_mean["frac_removed"] * 100

    # ── Panel A: LCC fraction vs % removed ──────────────────────────────────
    ax = axes[0, 0]
    ax.plot(pct_bet, df_bet["lcc_frac"] * 100,
            color=STRATEGY_COLORS["Betweenness (Bridger)"],
            lw=2.5, label="Betweenness (Bridger)", marker="o", ms=4)
    ax.plot(pct_deg, df_deg["lcc_frac"] * 100,
            color=STRATEGY_COLORS["Degree (Hub)"],
            lw=2.5, label="Degree (Hub)", marker="s", ms=4)
    ax.plot(pct_rnd, r_mean["lcc_frac"] * 100,
            color=STRATEGY_COLORS["Random Baseline"],
            lw=2.5, label="Random Baseline", linestyle="--")
    ax.fill_between(
        pct_rnd,
        (r_mean["lcc_frac"] - r_mean["lcc_std"]) * 100,
        (r_mean["lcc_frac"] + r_mean["lcc_std"]) * 100,
        alpha=0.15, color=STRATEGY_COLORS["Random Baseline"]
    )
    ax.axhline(50, color="grey", linestyle=":", lw=1.2, label="50% threshold")
    ax.set_xlabel("% Nodes Removed", fontsize=10)
    ax.set_ylabel("Largest Component (% of original)", fontsize=10)
    ax.set_title("A  Network Collapse: LCC Size", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, MAX_REMOVAL * 100)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)

    # Add annotation: which collapses first
    s = summary.sort_values("collapse_at_50pct")
    fastest = s.iloc[0]
    ax.annotate(
        f"{fastest['strategy']}\ncollapses first\n@ {fastest['collapse_at_50pct']*100:.1f}%",
        xy=(fastest["collapse_at_50pct"] * 100, 50),
        xytext=(fastest["collapse_at_50pct"] * 100 + 2, 60),
        fontsize=7, color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1)
    )

    # ── Panel B: Number of components (fragmentation) ────────────────────────
    ax = axes[0, 1]
    ax.plot(pct_bet, df_bet["n_components"],
            color=STRATEGY_COLORS["Betweenness (Bridger)"],
            lw=2.5, label="Betweenness (Bridger)", marker="o", ms=4)
    ax.plot(pct_deg, df_deg["n_components"],
            color=STRATEGY_COLORS["Degree (Hub)"],
            lw=2.5, label="Degree (Hub)", marker="s", ms=4)
    ax.plot(pct_rnd, r_mean["n_components"],
            color=STRATEGY_COLORS["Random Baseline"],
            lw=2.5, label="Random Baseline", linestyle="--")
    ax.set_xlabel("% Nodes Removed", fontsize=10)
    ax.set_ylabel("Number of Components", fontsize=10)
    ax.set_title("B  Network Fragmentation", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, MAX_REMOVAL * 100)
    ax.grid(alpha=0.3)
    ax.text(0.97, 0.05,
            "More components = more\nisolated sub-epidemics",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="grey",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey"))

    # ── Panel C: Density vs % removed ───────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(pct_bet, df_bet["density"] * 1000,
            color=STRATEGY_COLORS["Betweenness (Bridger)"],
            lw=2.5, label="Betweenness (Bridger)", marker="o", ms=4)
    ax.plot(pct_deg, df_deg["density"] * 1000,
            color=STRATEGY_COLORS["Degree (Hub)"],
            lw=2.5, label="Degree (Hub)", marker="s", ms=4)
    ax.set_xlabel("% Nodes Removed", fontsize=10)
    ax.set_ylabel("Network Density (×10⁻³)", fontsize=10)
    ax.set_title("C  Network Density Decay", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, MAX_REMOVAL * 100)
    ax.grid(alpha=0.3)

    # ── Panel D: Summary bar chart ───────────────────────────────────────────
    ax = axes[1, 1]
    removal_levels = ["10%", "20%", "30%"]
    cols_map = {
        "10%": "lcc_at_10pct_removal",
        "20%": "lcc_at_20pct_removal",
        "30%": "lcc_at_30pct_removal",
    }
    strategies   = summary["strategy"].tolist()
    x            = np.arange(len(removal_levels))
    bar_width    = 0.25
    offsets      = [-bar_width, 0, bar_width]

    for i, strat in enumerate(strategies):
        row    = summary[summary["strategy"] == strat].iloc[0]
        vals   = [row[cols_map[lv]] * 100 for lv in removal_levels]
        color  = STRATEGY_COLORS.get(strat, "grey")
        bars   = ax.bar(x + offsets[i], vals, bar_width * 0.9,
                        label=strat, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"After {lv}\nremoval" for lv in removal_levels])
    ax.set_ylabel("LCC Remaining (% of original)", fontsize=10)
    ax.set_title("D  LCC Remaining at Key Removal Levels", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(0, 115)
    ax.axhline(50, color="grey", linestyle=":", lw=1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_attack_interpretation(results: dict,
                                output_path: str = "outputs/figures/fig8_attack_interpretation.png"):
    """
    Produces a focused 2-panel interpretation figure:
      Left  — Epidemic reach collapse curve with annotations
      Right — Side-by-side collapse speed comparison with interpretation text
    Designed to be the key figure for the report/presentation.
    """
    df_bet  = results["betweenness"]
    df_deg  = results["degree"]
    r_mean  = results["random_mean"]
    summary = results["summary"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Bridgers vs Hubs: Which Removal Strategy Disrupts Epidemic Spread More?\n"
        "Team GYAN — Extension Analysis",
        fontsize=13, fontweight="bold"
    )

    # ── Left: Main collapse curve ────────────────────────────────────────────
    ax = axes[0]
    pct_bet = df_bet["frac_removed"] * 100
    pct_deg = df_deg["frac_removed"] * 100
    pct_rnd = r_mean["frac_removed"] * 100

    ax.fill_between(
        pct_rnd,
        (r_mean["lcc_frac"] - r_mean["lcc_std"]) * 100,
        (r_mean["lcc_frac"] + r_mean["lcc_std"]) * 100,
        alpha=0.12, color=STRATEGY_COLORS["Random Baseline"], label="_nolegend_"
    )
    ax.plot(pct_rnd, r_mean["lcc_frac"] * 100,
            color=STRATEGY_COLORS["Random Baseline"], lw=2,
            linestyle="--", label="Random Baseline (±1 SD)")
    ax.plot(pct_bet, df_bet["lcc_frac"] * 100,
            color=STRATEGY_COLORS["Betweenness (Bridger)"],
            lw=3, label="Betweenness — Remove Bridgers")
    ax.plot(pct_deg, df_deg["lcc_frac"] * 100,
            color=STRATEGY_COLORS["Degree (Hub)"],
            lw=3, label="Degree — Remove Hubs")

    ax.axhline(50, color="black", linestyle=":", lw=1.2, alpha=0.5)
    ax.text(MAX_REMOVAL * 100 * 0.98, 51.5, "50% collapse threshold",
            ha="right", fontsize=8, color="black", alpha=0.7)

    # Shade the "advantage zone" between the two targeted strategies
    lcc_bet_interp = np.interp(pct_rnd, pct_bet, df_bet["lcc_frac"] * 100)
    lcc_deg_interp = np.interp(pct_rnd, pct_deg, df_deg["lcc_frac"] * 100)
    ax.fill_between(pct_rnd, lcc_bet_interp, lcc_deg_interp,
                    where=(lcc_bet_interp != lcc_deg_interp),
                    alpha=0.12, color="purple",
                    label="Difference between strategies")

    ax.set_xlabel("% of Network Nodes Removed", fontsize=11)
    ax.set_ylabel("Epidemic Reach\n(% of original network reachable)", fontsize=11)
    ax.set_title("Epidemic Reach Under Node Removal Strategies", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, MAX_REMOVAL * 100)
    ax.set_ylim(0, 108)
    ax.grid(alpha=0.25)

    # ── Right: Interpretation panel ──────────────────────────────────────────
    ax = axes[1]
    ax.axis("off")

    # Determine winner
    s = summary.sort_values("collapse_at_50pct")
    winner     = s.iloc[0]["strategy"]
    winner_pct = s.iloc[0]["collapse_at_50pct"] * 100
    runner_pct = s.iloc[1]["collapse_at_50pct"] * 100
    diff_pct   = abs(runner_pct - winner_pct)

    bet_row = summary[summary["strategy"] == "Betweenness (Bridger)"].iloc[0]
    deg_row = summary[summary["strategy"] == "Degree (Hub)"].iloc[0]

    interpretation = (
        f"KEY FINDINGS\n"
        f"{'─'*42}\n\n"
        f"Most effective strategy:\n"
        f"  → {winner}\n\n"
        f"LCC drops to 50% after removing:\n"
        f"  Betweenness : {bet_row['collapse_at_50pct']*100:.1f}% of nodes\n"
        f"  Degree      : {deg_row['collapse_at_50pct']*100:.1f}% of nodes\n"
        f"  Advantage   : {diff_pct:.1f} percentage points\n\n"
        f"LCC remaining after 20% removal:\n"
        f"  Betweenness : {bet_row['lcc_at_20pct_removal']*100:.1f}%\n"
        f"  Degree      : {deg_row['lcc_at_20pct_removal']*100:.1f}%\n\n"
        f"{'─'*42}\n\n"
        f"EPIDEMIOLOGICAL INTERPRETATION\n\n"
        f"Betweenness (Bridger) nodes are\n"
        f"individuals who connect otherwise\n"
        f"separate clusters — removing them\n"
        f"breaks inter-community transmission.\n\n"
        f"Degree (Hub) nodes are super-\n"
        f"spreaders with many direct contacts\n"
        f"— removing them reduces local spread\n"
        f"but may leave bridges intact.\n\n"
        f"{'─'*42}\n\n"
        f"POLICY IMPLICATION\n\n"
        f"Targeted isolation of individuals\n"
        f"who bridge communities (e.g. event\n"
        f"organisers, inter-school contacts)\n"
        f"may be more effective than isolating\n"
        f"those with the most contacts alone.\n\n"
        f"Random intervention (e.g. mass\n"
        f"vaccination without targeting) is\n"
        f"least efficient per node removed."
    )

    ax.text(0.05, 0.97, interpretation,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=9.5,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6",
                      fc="#F8F9FA", ec="#CCCCCC", lw=1.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def save_attack_csv(results: dict, csv_dir: str):
    """Save attack simulation results to CSV files."""
    results["betweenness"].to_csv(
        f"{csv_dir}/attack_betweenness.csv", index=False)
    results["degree"].to_csv(
        f"{csv_dir}/attack_degree.csv", index=False)
    results["random_mean"].to_csv(
        f"{csv_dir}/attack_random_mean.csv", index=False)
    results["summary"].to_csv(
        f"{csv_dir}/attack_summary.csv", index=False)
    print(f"  Saved: {csv_dir}/attack_betweenness.csv")
    print(f"  Saved: {csv_dir}/attack_degree.csv")
    print(f"  Saved: {csv_dir}/attack_random_mean.csv")
    print(f"  Saved: {csv_dir}/attack_summary.csv")
