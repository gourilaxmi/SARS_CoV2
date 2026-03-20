"""
reproduction_numbers.py
=======================
Estimates individual-level case reproduction numbers (Rc) and overdispersion
from sampled transmission trees.

Methodology (from paper):
  - Rc_t = number of secondary infections caused by each infector
  - Fit Negative Binomial (community/school/workplace) and
    Beta-Binomial (household) to offspring distributions
  - Overdispersion parameter k: smaller k = more overdispersion / superspreading
  - Compare individual-based Rc with aggregate national estimates
"""

import numpy as np
import pandas as pd
from scipy.stats import nbinom, gamma
from scipy.optimize import minimize
from collections import defaultdict
from datetime import date

RNG = np.random.default_rng(42)

# ── Compute individual reproduction numbers from trees ─────────────────────────
def compute_individual_Rc(trees: list[dict],
                           pop: pd.DataFrame,
                           social_network: dict,
                           freq: str = "W") -> pd.DataFrame:
    """
    For each infector in each tree, count secondary infections by setting.
    Returns a DataFrame aggregated (median across trees) by week/infector.
    """
    pop_idx = pop.set_index("individual_id")
    settings = ["household","school","workplace","family","community"]
    
    # Build vaccination multiplier lookup: vaccinated infectors cause fewer secondaries
    vacc_mult_map = {}
    if "vacc_mult" in pop_idx.columns:
        vacc_mult_map = pop_idx["vacc_mult"].to_dict()

    # Aggregate across trees
    infector_counts = defaultdict(lambda: defaultdict(list))
    
    for tree in trees:
        # Count secondary infections per infector (weighted by vacc multiplier)
        secondary = defaultdict(lambda: {s: 0.0 for s in settings + ["total"]})
        
        for infectee, infector in tree.items():
            if infector is None:
                continue
            # Vaccination effect: vaccinated infectors have reduced onward transmission
            vacc_w = vacc_mult_map.get(infector, 1.0)
            # Determine setting
            s_inf  = social_network.get(infector, {})
            s_infe = social_network.get(infectee, {})
            setting = "community"
            for s in ["household","school","workplace","family"]:
                if (s in s_inf and s in s_infe and s_inf[s] == s_infe[s]):
                    setting = s
                    break
            secondary[infector][setting] += vacc_w
            secondary[infector]["total"] += vacc_w
        
        # All individuals in population (including those with 0 secondary infections)
        for ind_id in pop_idx.index:
            for s in settings + ["total"]:
                infector_counts[ind_id][s].append(secondary[ind_id][s])
    
    # Build summary DataFrame (median Rc per individual)
    rows = []
    for ind_id, counts in infector_counts.items():
        row = {"individual_id": ind_id}
        if ind_id in pop_idx.index:
            row["test_date"] = pop_idx.loc[ind_id, "test_date"]
            row["variant"]   = pop_idx.loc[ind_id, "variant"]
            row["age_group"] = pop_idx.loc[ind_id, "age_group"]
            row["vacc_status"] = pop_idx.loc[ind_id, "vacc_status"]
            row["region"]    = pop_idx.loc[ind_id, "region"]
        for s in settings + ["total"]:
            row[f"rc_{s}"] = np.median(counts[s])
        rows.append(row)
    
    return pd.DataFrame(rows)

# ── Weekly Rc (case reproduction number) ───────────────────────────────────────
def weekly_Rc(rc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates individual Rc by week (median across individuals testing that week).
    """
    if "test_date" not in rc_df.columns:
        return pd.DataFrame()
    
    rc_df = rc_df.copy()
    rc_df["week"] = pd.to_datetime(rc_df["test_date"]).dt.to_period("W")
    weekly = rc_df.groupby("week")["rc_total"].median().reset_index()
    weekly.columns = ["week","median_Rc"]
    return weekly

# ── Aggregate Rc from renewal equation ─────────────────────────────────────────
def aggregate_Rc_renewal(pop: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """
    Approximates the aggregate Rc from weekly case counts using a renewal equation.
    Rc_t = sum_m { R_t+m * g_m }   (Fraser 2007, cited in paper)
    
    For this synthetic data we use a simple Wallinga-Teunis-like estimate.
    """
    pop_copy = pop.copy()
    pop_copy["week"] = pd.to_datetime(pop_copy["test_date"]).dt.to_period("W")
    weekly_cases = pop_copy.groupby("week").size().reset_index(name="cases")
    weekly_cases = weekly_cases.sort_values("week")
    
    # Generation time distribution (Gamma(4.87, 1.98))
    g_shape = 4.87**2 / 1.98
    g_scale = 1.98 / 4.87
    g_dist  = gamma(g_shape, scale=g_scale)
    g_weights = np.array([g_dist.pdf(d) for d in range(1, 8)])
    g_weights /= g_weights.sum()
    
    cases = weekly_cases["cases"].values.astype(float)
    Rc    = np.ones(len(cases))
    
    for t in range(1, len(cases)):
        denom = sum(cases[t - j] * g_weights[j - 1]
                    for j in range(1, min(t + 1, len(g_weights) + 1)))
        Rc[t] = cases[t] / max(denom, 1e-6)
    
    weekly_cases["Rc_aggregate"] = np.clip(Rc, 0.1, 5.0)
    return weekly_cases

# ── Fit Negative Binomial to offspring distribution ────────────────────────────
def fit_negative_binomial(counts: np.ndarray) -> dict:
    """
    Fits Negative Binomial NB(mean=Rc, dispersion=k) to offspring counts.
    Returns dict: {mean, k, log_likelihood}
    
    Overdispersion: k → 0 means high overdispersion (superspreading).
    """
    counts = np.array(counts, dtype=float)
    counts = counts[counts >= 0]
    if len(counts) < 10:
        return {"mean": np.mean(counts), "k": np.inf, "log_likelihood": -np.inf}
    
    mu0 = np.mean(counts)
    k0  = 1.0
    
    def neg_loglik(params):
        mu, k = params
        if mu <= 0 or k <= 0:
            return 1e10
        r = k
        p = k / (k + mu)
        ll = np.sum(nbinom.logpmf(counts.astype(int), r, p))
        return -ll
    
    result = minimize(neg_loglik, [mu0, k0], method="Nelder-Mead",
                      options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000})
    mu_fit, k_fit = result.x
    
    return {
        "mean":           max(mu_fit, 0.0),
        "k":              max(k_fit, 0.0),
        "log_likelihood": -result.fun,
        "converged":      result.success,
    }

# ── Setting-stratified overdispersion ──────────────────────────────────────────
def estimate_overdispersion_by_setting(rc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits NB to each setting's offspring distribution.
    """
    settings = ["total","household","school","workplace","family","community"]
    rows = []
    for s in settings:
        col = f"rc_{s}"
        if col not in rc_df.columns:
            continue
        counts = rc_df[col].dropna().values
        fit = fit_negative_binomial(counts)
        rows.append({"setting": s, **fit})
    return pd.DataFrame(rows)

# ── Overdispersion over time (weekly) ──────────────────────────────────────────
def overdispersion_over_time(rc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits NB per week to capture temporal evolution of overdispersion.
    """
    if "test_date" not in rc_df.columns:
        return pd.DataFrame()
    
    rc_df = rc_df.copy()
    rc_df["week"] = pd.to_datetime(rc_df["test_date"]).dt.to_period("W")
    rows = []
    
    for week, grp in rc_df.groupby("week"):
        counts = grp["rc_total"].values
        if len(counts) < 10:
            continue
        fit = fit_negative_binomial(counts)
        rows.append({"week": week, "k": fit["k"], "mean_Rc": fit["mean"]})
    
    return pd.DataFrame(rows)

# ── Weekly setting proportions (Figure 2A equivalent) ──────────────────────────
def weekly_setting_proportions(trees: list, pop: pd.DataFrame,
                                social_network: dict) -> pd.DataFrame:
    """
    Fast O(N_trees * N_infectees) implementation.
    Computes for each week the proportion of individuals whose sampled
    infector belongs to each setting (reproduces Figure 2A of the paper).
    """
    pop_idx  = pop.set_index("individual_id")
    settings = ["household","school","workplace","family","community"]

    # week label for each individual
    id_to_week = {
        i: pd.Timestamp(pop_idx.loc[i, "test_date"]).to_period("W")
        for i in pop_idx.index
    }
    # weekly population denominators
    week_totals = defaultdict(int)
    for i in pop_idx.index:
        week_totals[id_to_week[i]] += 1

    # Tally setting per (week) across all trees
    week_setting_counts = defaultdict(lambda: {s: 0 for s in settings + ["any"]})

    n_trees = len(trees)
    for tree in trees:
        for infectee, infector in tree.items():
            if infector is None or infectee not in pop_idx.index:
                continue
            wk = id_to_week[infectee]
            s_inf  = social_network.get(infector, {})
            s_infe = social_network.get(infectee, {})
            setting = "community"
            for s in ["household","school","workplace","family"]:
                if s in s_inf and s in s_infe and s_inf[s] == s_infe[s]:
                    setting = s; break
            week_setting_counts[wk][setting] += 1
            week_setting_counts[wk]["any"]    += 1

    rows = []
    for wk in sorted(week_totals.keys()):
        denom = week_totals[wk] * n_trees
        row = {"week": wk}
        for s in settings:
            row[s] = week_setting_counts[wk].get(s, 0) / max(denom, 1)
        row["total"] = week_setting_counts[wk].get("any", 0) / max(denom, 1)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_generation import generate_all
    from transmission_network import (build_transmission_network,
                                       annotate_with_settings,
                                       sample_prioritised_settings_tree)
    
    pop, genomes, social, npi = generate_all()
    G = build_transmission_network(pop, genomes)
    G = annotate_with_settings(G, social)
    
    print("Sampling trees...")
    trees = sample_prioritised_settings_tree(G, n_trees=10)
    
    print("Computing individual Rc...")
    rc_df = compute_individual_Rc(trees, pop, social)
    
    print("\nOverdispersion by setting:")
    od_df = estimate_overdispersion_by_setting(rc_df)
    print(od_df[["setting","mean","k"]].to_string(index=False))
    
    print("\nWeekly setting proportions (first 5 weeks):")
    weekly = weekly_setting_proportions(trees, pop, social)
    print(weekly.head().to_string(index=False))