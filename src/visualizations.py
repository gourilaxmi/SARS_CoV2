"""
visualizations.py
=================
Produces all main figures from the paper replication:
  Fig 1-equivalent: Weekly proportion with infectors in settings
  Fig 2-equivalent: Age-structured transmission matrices
  Fig 3-equivalent: Cluster summary statistics by variant
  Fig 4-equivalent: Reproduction numbers and overdispersion
  Fig 5-equivalent: NPI effect sizes + vaccination IRR
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")

SETTING_COLORS = {
    "household":  "#2196F3",   # blue
    "school":     "#4CAF50",   # green
    "workplace":  "#FF9800",   # orange
    "family":     "#9C27B0",   # purple
    "community":  "#9E9E9E",   # grey
    "total":      "#000000",   # black
}
VARIANT_COLORS = {
    "B.1.177": "#607D8B",
    "Alpha":   "#F44336",
    "Eta":     "#FF9800",
    "Delta":   "#9C27B0",
    "Omicron": "#2196F3",
}

# ── Figure 1: Weekly setting proportions ───────────────────────────────────────
def plot_weekly_proportions(weekly_props: pd.DataFrame,
                             output_path: str = "figures/fig1_weekly_proportions.png"):
    fig, ax = plt.subplots(figsize=(12, 5))
    
    settings_to_plot = ["household","school","workplace","family"]
    for s in settings_to_plot:
        if s not in weekly_props.columns:
            continue
        weeks = [str(w) for w in weekly_props["week"]]
        ax.plot(weeks, weekly_props[s], label=s.capitalize(),
                color=SETTING_COLORS[s], linewidth=1.5, alpha=0.85)
    
    if "total" in weekly_props.columns:
        ax.plot([str(w) for w in weekly_props["week"]], weekly_props["total"],
                label="Total", color="black", linewidth=2.5, linestyle="--")
    
    ax.set_xlabel("Week", fontsize=11)
    ax.set_ylabel("Proportion of individuals with\ninfectors in setting", fontsize=11)
    ax.set_title("Weekly Proportion of Individuals with Plausible Infectors in Settings", fontsize=13)
    ax.legend(frameon=True, loc="upper left", fontsize=9)
    ax.set_ylim(0, 0.8)
    
    # Thin out x-tick labels
    ticks = ax.get_xticks()
    labels = [str(w) for w in weekly_props["week"]]
    step  = max(1, len(labels) // 12)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)
    
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

# ── Figure 2: Age-structured transmission matrices ────────────────────────────
def plot_age_matrices(trees: list[dict], pop: pd.DataFrame,
                       social_network: dict,
                       output_path: str = "figures/fig2_age_matrices.png"):
    """
    Creates 2x3 grid of age-structured heatmaps (one per setting + all + community).
    """
    pop_idx = pop.set_index("individual_id")
    age_groups = ["0-10","11-18","19-39","40-59","60+"]
    settings_to_plot = ["all","household","school","workplace","family","community"]
    
    matrices = {s: np.zeros((5, 5)) for s in settings_to_plot}
    
    for tree in trees:
        for infectee, infector in tree.items():
            if infector is None:
                continue
            if infectee not in pop_idx.index or infector not in pop_idx.index:
                continue
            
            age_inf  = pop_idx.loc[infector, "age_group"]
            age_infe = pop_idx.loc[infectee, "age_group"]
            i_inf  = age_groups.index(age_inf) if age_inf in age_groups else 2
            i_infe = age_groups.index(age_infe) if age_infe in age_groups else 2
            
            # Determine setting
            s_inf  = social_network.get(infector, {})
            s_infe = social_network.get(infectee, {})
            setting = "community"
            for s in ["household","school","workplace","family"]:
                if s in s_inf and s in s_infe and s_inf[s] == s_infe[s]:
                    setting = s
                    break
            
            matrices["all"][i_inf, i_infe]     += 1
            matrices[setting][i_inf, i_infe]   += 1
    
    # Normalise rows
    for s in settings_to_plot:
        row_sums = matrices[s].sum(axis=1, keepdims=True)
        matrices[s] = np.divide(matrices[s], row_sums,
                                 where=row_sums > 0, out=np.zeros_like(matrices[s]))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    titles = ["All","Household","School","Workplace","Family","Community"]
    
    for ax, s, title in zip(axes, settings_to_plot, titles):
        im = ax.imshow(matrices[s], cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(age_groups, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(age_groups, fontsize=8)
        ax.set_xlabel("Infectee Age Group", fontsize=9)
        ax.set_ylabel("Infector Age Group", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    fig.suptitle("Age-Structured Transmission Matrices by Setting", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

# ── Figure 3: Cluster statistics by variant ────────────────────────────────────
def plot_cluster_statistics(clusters: list[dict], rand_cluster_stats: list[dict],
                              output_path: str = "figures/fig3_clusters.png"):
    variants = ["B.1.177","Alpha","Eta","Delta","Omicron"]
    
    def get_stat(cluster_list, variant, stat):
        vc = [c for c in cluster_list if c["dominant_variant"] == variant]
        if not vc:
            return 0
        if stat == "n_clusters":
            return len(vc)
        elif stat == "prop_in_cluster":
            return sum(c["size"] for c in vc) / max(sum(
                c["size"] for c in cluster_list), 1)
        elif stat == "largest_size":
            return max(c["size"] for c in vc)
        elif stat == "mean_regions":
            return np.mean([c["n_regions"] for c in vc])
        return 0
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    stats_labels = [
        ("prop_in_cluster",  "Proportion in a Cluster"),
        ("n_clusters",       "Number of Clusters"),
        ("largest_size",     "Largest Cluster Size"),
        ("mean_regions",     "Mean Regions Spanned"),
    ]
    
    for ax, (stat, label) in zip(axes, stats_labels):
        # True network values
        true_vals = [get_stat(clusters, v, stat) for v in variants]
        
        # Simulated randomised range
        rand_vals = [[get_stat(rc, v, stat) for rc in rand_cluster_stats] for v in variants]
        rand_means = [np.mean(rv) if rv else 0 for rv in rand_vals]
        rand_stds  = [np.std(rv)  if rv else 0 for rv in rand_vals]
        
        y_pos = np.arange(len(variants))
        ax.barh(y_pos, true_vals, color=[VARIANT_COLORS[v] for v in variants],
                alpha=0.8, height=0.4, label="Inferred network")
        ax.errorbar(rand_means, y_pos + 0.3,
                    xerr=rand_stds, fmt="o", color="grey",
                    markersize=5, label="Randomised (mean ± SD)")
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(variants, fontsize=9)
        ax.set_xlabel(label, fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Transmission Cluster Statistics by Variant\n"
                 "(Squares = inferred network, Circles = randomised mean)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

# ── Figure 4: Reproduction numbers and overdispersion ─────────────────────────
def plot_reproduction_numbers(weekly_Rc: pd.DataFrame,
                               agg_Rc: pd.DataFrame,
                               od_time: pd.DataFrame,
                               od_setting: pd.DataFrame,
                               output_path: str = "figures/fig4_reproduction_numbers.png"):
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    
    # Panel A: Rc comparison
    ax_a = fig.add_subplot(gs[0, 0])
    if not weekly_Rc.empty and "median_Rc" in weekly_Rc.columns:
        weeks_ind = [str(w) for w in weekly_Rc["week"]]
        ax_a.plot(weeks_ind, weekly_Rc["median_Rc"],
                  label="Individual-based Rc (prioritised)", color="#2196F3", lw=2)
    if not agg_Rc.empty and "Rc_aggregate" in agg_Rc.columns:
        weeks_agg = [str(w) for w in agg_Rc.get("week", agg_Rc.index)]
        if "week" in agg_Rc.columns:
            ax_a.plot([str(w) for w in agg_Rc["week"]], agg_Rc["Rc_aggregate"],
                      label="Aggregate Rc (renewal eq.)", color="#F44336",
                      lw=2, linestyle="--")
    ax_a.axhline(1.0, color="grey", linestyle=":", lw=1)
    ax_a.set_title("A  Reproduction Number Estimates", fontsize=11)
    ax_a.set_ylabel("Reproduction Number", fontsize=9)
    ax_a.set_ylim(0, 3)
    ax_a.legend(fontsize=8)
    ax_a.tick_params(axis="x", rotation=45, labelsize=7)
    ax_a.grid(alpha=0.2)
    
    # Panel B: Overdispersion over time
    ax_b = fig.add_subplot(gs[0, 1])
    if not od_time.empty and "k" in od_time.columns:
        weeks_od = [str(w) for w in od_time["week"]]
        ax_b.plot(weeks_od, od_time["k"], color="#9C27B0", lw=2)
        ax_b.fill_between(weeks_od,
                           od_time["k"] * 0.8, od_time["k"] * 1.2,
                           alpha=0.2, color="#9C27B0")
    ax_b.set_title("B  Overdispersion (k) Over Time", fontsize=11)
    ax_b.set_ylabel("Dispersion parameter k\n(lower = more overdispersion)", fontsize=9)
    ax_b.tick_params(axis="x", rotation=45, labelsize=7)
    ax_b.grid(alpha=0.2)
    
    # Panel C: Rc by setting over time (stacked area)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_title("C  Rc by Setting (Prioritised Trees)", fontsize=11)
    ax_c.text(0.5, 0.5, "Setting-stratified Rc\n(see weekly_setting_proportions)",
              ha="center", va="center", transform=ax_c.transAxes, fontsize=10,
              color="grey")
    ax_c.set_xlabel("Week")
    ax_c.grid(alpha=0.2)
    
    # Panel D: Overdispersion by setting (bar chart)
    ax_d = fig.add_subplot(gs[1, 1])
    if not od_setting.empty:
        settings_od = od_setting[od_setting["setting"] != "total"]
        settings_od = settings_od.sort_values("k", ascending=False)
        bars = ax_d.barh(settings_od["setting"], settings_od["k"],
                         color=[SETTING_COLORS.get(s, "grey") for s in settings_od["setting"]],
                         alpha=0.85)
        ax_d.set_xlabel("Dispersion k (∞ = no overdispersion, ↓ = more superspreading)", fontsize=8)
        ax_d.set_title("D  Overdispersion by Setting", fontsize=11)
        ax_d.axvline(x=1, color="grey", linestyle=":", lw=1, label="k=1 threshold")
        ax_d.legend(fontsize=8)
        ax_d.grid(axis="x", alpha=0.3)
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

# ── Figure 5: NPI effects and vaccination ──────────────────────────────────────
def plot_npi_effects(all_results: pd.DataFrame,
                      output_path: str = "figures/fig5_npi_effects.png"):
    if all_results.empty:
        print("  No NPI results to plot.")
        return
    
    NPI_LABELS = {
        "facial_coverings":  "Facial coverings",
        "gathering_restr":   "Restriction on gathering size",
        "travel_restr":      "Restrictions on international travel",
        "school_restr":      "School restrictions",
        "stay_home":         "Stay at home order guidance",
        "workplace_restr":   "Workplace restrictions",
    }
    SETTINGS_ORDER = ["total","community","family","household","school","workplace"]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Panel A: Total transmission NPI effects
    ax_a = axes[0]
    total_res = all_results[(all_results["setting"] == "total") &
                             (all_results["covariate"].isin(NPI_LABELS.keys()))].copy()
    total_res["label"] = total_res["covariate"].map(NPI_LABELS)
    total_res = total_res.sort_values("beta")
    
    colors = ["#4CAF50" if b < 0 else "#F44336" for b in total_res["beta"]]
    ax_a.barh(total_res["label"], total_res["beta"],
              xerr=[total_res["beta"] - total_res["ci_low"],
                    total_res["ci_high"] - total_res["beta"]],
              color=colors, alpha=0.8, capsize=4, height=0.5)
    ax_a.axvline(0, color="black", lw=1.5)
    ax_a.set_xlabel("β_i  (change in secondary infections)", fontsize=10)
    ax_a.set_title("A  NPI Effects on Total Transmission", fontsize=12)
    ax_a.grid(axis="x", alpha=0.3)
    
    # Panel B: Vaccination and age IRR
    ax_b = axes[1]
    vacc_rows = all_results[(all_results["covariate"].isin(["vacc_one_dose","vacc_two_doses"])) &
                             (all_results["setting"] == "total")].copy()
    age_rows  = all_results[(all_results["covariate"].str.startswith("age_")) &
                             (all_results["setting"] == "total")].copy()
    
    vacc_rows["IRR"]     = np.exp(vacc_rows["beta"])
    vacc_rows["IRR_low"] = np.exp(vacc_rows["ci_low"])
    vacc_rows["IRR_hi"]  = np.exp(vacc_rows["ci_high"])
    age_rows["IRR"]      = np.exp(age_rows["beta"])
    age_rows["IRR_low"]  = np.exp(age_rows["ci_low"])
    age_rows["IRR_hi"]   = np.exp(age_rows["ci_high"])
    
    combined = pd.concat([
        vacc_rows.assign(group="Vaccination", label=vacc_rows["covariate"].str.replace("vacc_","").str.replace("_"," ")),
        age_rows.assign(group="Age", label=age_rows["covariate"].str.replace("age_","Age "))
    ], ignore_index=True)
    
    if not combined.empty:
        y_pos = np.arange(len(combined))
        ax_b.errorbar(combined["IRR"], y_pos,
                      xerr=[combined["IRR"] - combined["IRR_low"],
                             combined["IRR_hi"] - combined["IRR"]],
                      fmt="o", color="#2196F3", capsize=5, markersize=7, lw=2)
        ax_b.axvline(1.0, color="grey", linestyle="--", lw=1.5)
        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels(combined["label"], fontsize=9)
        ax_b.set_xlabel("Incidence Rate Ratio (IRR) vs reference", fontsize=10)
        ax_b.set_title("B  Vaccination & Age Effects (IRR)", fontsize=12)
        ax_b.fill_betweenx([-0.5, len(combined) - 0.5], 0.75, 1.0,
                            alpha=0.05, color="green")
        ax_b.grid(axis="x", alpha=0.3)
        
        # Annotate reduction %
        for i, (_, row) in enumerate(combined.iterrows()):
            red_pct = (1 - row["IRR"]) * 100
            ax_b.text(row["IRR"] + 0.01, i,
                      f"{red_pct:+.1f}%", fontsize=7, va="center", color="grey")
    
    plt.suptitle("NPI Effectiveness and Individual Risk Factors\n(Replication: Curran-Sebastian et al. 2026)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

# ── Summary  ──────────────────────────────────────────────────────────
def plot_summary_dashboard(weekly_props, od_setting, all_results, cluster_summary,
                            output_path: str = "figures/fig0_summary.png"):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#FAFAFA")
    
    # Title panel
    ax_title = fig.add_axes([0.0, 0.88, 1.0, 0.12])
    ax_title.axis("off")
    ax_title.text(0.5, 0.7,
                  "SARS-CoV-2 Transmission Networks & NPI Effects — Denmark (Replication)",
                  ha="center", va="center", fontsize=16, fontweight="bold")
    ax_title.text(0.5, 0.2,
                  "Team GYAN | Based on: Curran-Sebastian et al. (2026, medRxiv)",
                  ha="center", va="center", fontsize=11, color="grey")
    
    gs = gridspec.GridSpec(2, 3, figure=fig, top=0.87, bottom=0.07,
                           hspace=0.45, wspace=0.38)
    
    # ① Weekly proportions
    ax1 = fig.add_subplot(gs[0, :2])
    if not weekly_props.empty:
        for s in ["household","school","workplace","family","total"]:
            if s not in weekly_props.columns:
                continue
            style = "--" if s == "total" else "-"
            lw    = 2.5 if s == "total" else 1.5
            ax1.plot(range(len(weekly_props)), weekly_props[s],
                     label=s.capitalize(), color=SETTING_COLORS[s],
                     linestyle=style, linewidth=lw, alpha=0.9)
    ax1.set_title("① Weekly Setting-Linked Transmission Proportions", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Proportion", fontsize=9)
    ax1.set_ylim(0, 0.75)
    ax1.legend(fontsize=8, loc="upper left", ncol=3)
    ax1.grid(alpha=0.2)
    ax1.set_xlabel("Week index", fontsize=9)
    
    # ② Cluster summary
    ax2 = fig.add_subplot(gs[0, 2])
    if not cluster_summary.empty and "n_clusters" in cluster_summary.columns:
        variants = cluster_summary["variant"].tolist()
        n_cl     = cluster_summary["n_clusters"].tolist()
        bars     = ax2.barh(variants, n_cl,
                            color=[VARIANT_COLORS.get(v, "grey") for v in variants],
                            alpha=0.85)
        ax2.set_xlabel("Number of clusters", fontsize=9)
        ax2.set_title("② Clusters by Variant", fontsize=10, fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, n_cl):
            ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=9)
    
    # ③ Overdispersion by setting
    ax3 = fig.add_subplot(gs[1, 0])
    if not od_setting.empty and "k" in od_setting.columns:
        od_plot = od_setting[od_setting["setting"] != "total"].copy()
        od_plot = od_plot.sort_values("k")
        ax3.barh(od_plot["setting"], od_plot["k"],
                 color=[SETTING_COLORS.get(s, "grey") for s in od_plot["setting"]],
                 alpha=0.85)
        ax3.axvline(1, color="grey", linestyle=":", lw=1)
        ax3.set_xlabel("k  (↓ more superspreading)", fontsize=9)
        ax3.set_title("③ Overdispersion by Setting", fontsize=10, fontweight="bold")
        ax3.grid(axis="x", alpha=0.2)
    
    # ④ NPI effects
    ax4 = fig.add_subplot(gs[1, 1])
    if not all_results.empty:
        NPI_SHORT = {
            "facial_coverings": "Facial coverings",
            "gathering_restr":  "Gathering restr.",
            "travel_restr":     "Travel restr.",
            "school_restr":     "School restr.",
            "stay_home":        "Stay at home",
            "workplace_restr":  "Workplace restr.",
        }
        npi_res = all_results[(all_results["setting"] == "total") &
                               (all_results["covariate"].isin(NPI_SHORT))].copy()
        npi_res["label"] = npi_res["covariate"].map(NPI_SHORT)
        npi_res = npi_res.sort_values("beta")
        colors_npi = ["#4CAF50" if b < 0 else "#F44336" for b in npi_res["beta"]]
        ax4.barh(npi_res["label"], npi_res["beta"],
                 color=colors_npi, alpha=0.8, height=0.5)
        ax4.axvline(0, color="black", lw=1.2)
        ax4.set_xlabel("β (change in 2° infections)", fontsize=9)
        ax4.set_title("④ NPI Effect Sizes (Total)", fontsize=10, fontweight="bold")
        ax4.grid(axis="x", alpha=0.2)
        green_patch = mpatches.Patch(color="#4CAF50", label="Reduces transmission")
        red_patch   = mpatches.Patch(color="#F44336", label="Increases transmission")
        ax4.legend(handles=[green_patch, red_patch], fontsize=7, loc="lower right")
    
    # ⑤ Vaccination IRR
    ax5 = fig.add_subplot(gs[1, 2])
    if not all_results.empty:
        vacc_res = all_results[(all_results["covariate"].isin(["vacc_one_dose","vacc_two_doses"])) &
                                (all_results["setting"] == "total")].copy()
        if not vacc_res.empty:
            vacc_res["IRR"] = np.exp(vacc_res["beta"])
            vacc_res["reduction"] = (1 - vacc_res["IRR"]) * 100
            labels = ["1 Dose\n(ref: unvacc.)", "2 Doses\n(ref: unvacc.)"]
            bars = ax5.bar(labels[:len(vacc_res)], vacc_res["IRR"].values,
                           color=["#64B5F6","#1565C0"], alpha=0.9, width=0.4)
            ax5.axhline(1.0, color="grey", linestyle="--", lw=1.5)
            ax5.set_ylabel("IRR vs unvaccinated", fontsize=9)
            ax5.set_title("⑤ Vaccination Effect on Transmission", fontsize=10, fontweight="bold")
            ax5.set_ylim(0, 1.3)
            for bar, red in zip(bars, vacc_res["reduction"].values):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f"↓{abs(red):.1f}%", ha="center", fontsize=10, fontweight="bold",
                         color="#1565C0")
            ax5.grid(axis="y", alpha=0.2)
    
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

if __name__ == "__main__":
    print("Run main.py to generate all figures.")
