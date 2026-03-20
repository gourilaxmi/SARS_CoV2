"""
Transmission Networks and Intervention Effects from
SARS-CoV-2 Genomic and Social Network Data in Denmark
Based on: Curran-Sebastian et al. (2026, medRxiv)

"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

# Fix Unicode encoding on Windows
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

# Output folders 
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
    print(f"  {title}")


def main():
    args   = parse_args()
    start  = time.time()

    # STEP 1: Data Generation 
    section("STEP 1 — Data Generation")
    pop, genomes, social, npi = generate_all(fasta_path=args.fasta)

    # STEP 2: Transmission Network 
    section("STEP 2 — Building Transmission Network G")
    t0 = time.time()
    G  = build_transmission_network(pop, genomes)
    G  = annotate_with_settings(G, social)
    shared = sum(1 for _,_,d in G.edges(data=True) if d.get("shared_any", 0))
    print(f"  Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,} "
          f"| Setting-linked: {shared:,}  [{time.time()-t0:.1f}s]")

    # STEP 3: Tree Sampling
    section(f"STEP 3 — Sampling {args.trees} Transmission Trees")
    t0       = time.time()
    ps_trees = sample_prioritised_settings_tree(G, n_trees=args.trees)
    r_trees  = sample_random_trees(G, n_trees=args.trees)
    print(f"  Done [{time.time()-t0:.1f}s]")

    from collections import Counter
    setting_counts = Counter()
    for tree in ps_trees:
        df = classify_tree_settings(tree, social)
        setting_counts.update(df["setting"].value_counts().to_dict())
    total_ev = sum(setting_counts.values())
    print("\n  Setting distribution (prioritised trees):")
    for s, c in setting_counts.most_common():
        print(f"    {s:15s}: {c:7,}  ({100*c/total_ev:.1f}%)")

    # STEP 4: Clusters 
    section("STEP 4 — Transmission Cluster Analysis")
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

    # STEP 5: Reproduction Numbers 
    section("STEP 5 — Reproduction Numbers & Overdispersion")
    t0      = time.time()
    rc_df   = compute_individual_Rc(ps_trees, pop, social)
    w_rc    = weekly_Rc(rc_df)
    agg_rc  = aggregate_Rc_renewal(pop)
    od_set  = estimate_overdispersion_by_setting(rc_df)
    od_time = overdispersion_over_time(rc_df)
    w_prop  = weekly_setting_proportions(ps_trees, pop, social)
    print(f"  Done [{time.time()-t0:.1f}s]")
    print(f"\n  Overdispersion by setting:")
    print(od_set[["setting","mean","k"]].to_string(index=False))

    # STEP 6: NPI Regression 
    section("STEP 6 — NPI Effectiveness Regression")
    t0         = time.time()
    reg_data   = build_regression_data(rc_df, npi)
    all_res    = run_all_settings(reg_data)
    print(f"  Done [{time.time()-t0:.1f}s]")

    vacc = pd.DataFrame()
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
        print(npi_eff[["NPI","beta","ci_low","ci_high","pvalue"]].to_string(index=False))

        vacc = vaccination_effect_summary(all_res[all_res["setting"]=="total"])
        print(f"\n  Vaccination effects:")
        print(vacc.to_string(index=False))

    section(" Save CSV Outputs")
    csv_files = {
        "cluster_summary.csv":            cl_sum,
        "overdispersion_by_setting.csv":  od_set,
        "weekly_setting_proportions.csv": w_prop,
        "aggregate_rc.csv":               agg_rc,
        "population.csv":                 pop,
    }
    if not all_res.empty:
        csv_files["npi_regression_results.csv"] = all_res

    for fname, df in csv_files.items():
        path = os.path.join(CSV_DIR, fname)
        df.to_csv(path, index=False)
        print(f"  Saved: outputs/csv/{fname}")

    # ── STEP 8: Figures ───────────────────────────────────────────────────────
    section(" Generate Figures")

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

    plot_summary_dashboard(
        w_prop, od_set,
        all_res if not all_res.empty else pd.DataFrame(),
        cl_sum,
        output_path=os.path.join(FIGURES_DIR, "fig0_summary_dashboard.png"))

  
    section("Summary")
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"DATA SOURCES\n")
        f.write(f"  Genomes    : {'Real NCBI FASTA' if args.fasta else 'Simulated (Poisson model)'}\n")
        f.write(f"  Network    : POLYMOD (Mossong et al. 2008)\n")
        f.write(f"  NPI data   : OxCGRT Denmark (Oxford)\n")
        f.write(f"  Vaccination: SSI Denmark rollout schedule\n\n")
        f.write(f"NETWORK\n")
        f.write(f"  Individuals           : {len(pop):,}\n")
        f.write(f"  Transmission pairs    : {G.number_of_edges():,}\n")
        f.write(f"  Setting-linked pairs  : {shared:,} ({100*shared/max(G.number_of_edges(),1):.1f}%)\n")
        f.write(f"  Clusters (size > 5)   : {len(clusters):,}\n\n")
        f.write(f"OVERDISPERSION (k parameter)\n")
        for _, row in od_set.iterrows():
            f.write(f"  {row['setting']:15s}: k = {row['k']:.3f}  (mean Rc = {row['mean']:.3f})\n")
        f.write(f"\nNPI EFFECTS\n")
        if not npi_eff.empty:
            for _, row in npi_eff.iterrows():
                direction = "REDUCES" if row["beta"] < 0 else "increases"
                sig = "*" if row["pvalue"] < 0.05 else "(not sig.)"
                f.write(f"  {row['NPI']:28s}: {direction}  beta={row['beta']:+.3f}  {sig}\n")
        f.write(f"\nVACCINATION EFFECTS\n")
        if not vacc.empty:
            for _, row in vacc.iterrows():
                dose = "1-dose" if "one" in row["covariate"] else "2-dose"
                paper = "~15.5%" if "one" in row["covariate"] else "~23.5%"
                f.write(f"  {dose}: {row['reduction_pct']:.1f}% reduction  (paper: {paper})\n")
        f.write(f"\nTotal runtime: {time.time()-start:.0f} seconds\n")

    print(f"  Saved: outputs/results/summary.txt")

    # ── Final Print ───────────────────────────────────────────────────────────
    section("COMPLETE")
    print(f"  Runtime : {time.time()-start:.0f} seconds")
    print(f"\n  outputs/figures/  -> {len(os.listdir(FIGURES_DIR))} figures")
    print(f"  outputs/csv/      -> {len(os.listdir(CSV_DIR))} CSV files")
    print(f"  outputs/results/  -> summary.txt")
    print(f"\n  Transmission pairs : {G.number_of_edges():,}")
    print(f"  Clusters found     : {len(clusters):,}")
    if not all_res.empty:
        v2 = all_res.loc[(all_res["covariate"]=="vacc_two_doses")&
                          (all_res["setting"]=="total"), "beta"]
        if len(v2):
            print(f"  2-dose reduction   : {(1-np.exp(v2.values[0]))*100:.1f}%  (paper: ~23.5%)")
    print("="*60)


if __name__ == "__main__":
    main()