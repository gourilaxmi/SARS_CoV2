"""
reproduction_numbers.py
=======================

This module estimates individual-level reproduction numbers (Rc) and
overdispersion using sampled transmission trees.

Idea :
  - Rc for each individual = number of people they infect
  - Fit distributions to these counts:
      • Negative Binomial (for community, school, workplace, etc.)
      • Beta-Binomial (for households in the paper)
  - Overdispersion parameter k tells us how uneven transmission is:
      • Small k → more superspreading
  - We also compare individual-based Rc with aggregate estimates
"""

import numpy as np
import pandas as pd
from scipy.stats import nbinom, gamma
from scipy.optimize import minimize
from collections import defaultdict
from datetime import date

RNG = np.random.default_rng(42)

# ── Compute individual reproduction numbers ────────────────────────────────────
def compute_individual_Rc(trees: list[dict],
                           pop: pd.DataFrame,
                           social_network: dict,
                           freq: str = "W") -> pd.DataFrame:
    """
    For each transmission tree, count how many people each individual infects.
    We also split counts by setting (household, school, etc.).

    Final output gives median Rc (across trees) for each individual.
    """
    pop_idx = pop.set_index("individual_id")
    settings = ["household","school","workplace","family","community"]
    
    # If vaccination multipliers exist, use them to reduce transmission
    vacc_mult_map = {}
    if "vacc_mult" in pop_idx.columns:
        vacc_mult_map = pop_idx["vacc_mult"].to_dict()

    # Store counts across all trees
    infector_counts = defaultdict(lambda: defaultdict(list))
    
    for tree in trees:
        # Track secondary infections for each infector
        secondary = defaultdict(lambda: {s: 0.0 for s in settings + ["total"]})
        
        for infectee, infector in tree.items():
            if infector is None:
                continue

            # Apply vaccination effect to reduce transmission weight
            vacc_w = vacc_mult_map.get(infector, 1.0)

            # Identify setting of transmission
            s_inf  = social_network.get(infector, {})
            s_infe = social_network.get(infectee, {})
            setting = "community"

            for s in ["household","school","workplace","family"]:
                if (s in s_inf and s in s_infe and s_inf[s] == s_infe[s]):
                    setting = s
                    break

            secondary[infector][setting] += vacc_w
            secondary[infector]["total"] += vacc_w
        
        # Include all individuals 
        for ind_id in pop_idx.index:
            for s in settings + ["total"]:
                infector_counts[ind_id][s].append(secondary[ind_id][s])
    
    # Build final dataframe
    rows = []
    for ind_id, counts in infector_counts.items():
        row = {"individual_id": ind_id}

        if ind_id in pop_idx.index:
            row["test_date"]   = pop_idx.loc[ind_id, "test_date"]
            row["variant"]     = pop_idx.loc[ind_id, "variant"]
            row["age_group"]   = pop_idx.loc[ind_id, "age_group"]
            row["vacc_status"] = pop_idx.loc[ind_id, "vacc_status"]
            row["region"]      = pop_idx.loc[ind_id, "region"]

        for s in settings + ["total"]:
            row[f"rc_{s}"] = np.median(counts[s])

        rows.append(row)
    
    return pd.DataFrame(rows)

# ── Weekly Rc calculation ──────────────────────────────────────────────────────
def weekly_Rc(rc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups individuals by week and computes median Rc for each week.
    """
    if "test_date" not in rc_df.columns:
        return pd.DataFrame()
    
    rc_df = rc_df.copy()
    rc_df["week"] = pd.to_datetime(rc_df["test_date"]).dt.to_period("W")

    weekly = rc_df.groupby("week")["rc_total"].median().reset_index()
    weekly.columns = ["week","median_Rc"]

    return weekly

# ── Aggregate Rc using renewal idea ────────────────────────────────────────────
def aggregate_Rc_renewal(pop: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """
    Estimates population-level Rc using weekly case counts.

    Uses a simplified renewal approach (similar to Wallinga-Teunis).
    """
    pop_copy = pop.copy()
    pop_copy["week"] = pd.to_datetime(pop_copy["test_date"]).dt.to_period("W")

    weekly_cases = pop_copy.groupby("week").size().reset_index(name="cases")
    weekly_cases = weekly_cases.sort_values("week")
    
    # Generation time distribution (Gamma)
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

# ── Fit Negative Binomial distribution ─────────────────────────────────────────
def fit_negative_binomial(counts: np.ndarray) -> dict:
    """
    Fits a Negative Binomial distribution to offspring counts.

    Returns:
        mean (Rc), dispersion k, and log-likelihood

    Interpretation:
        Smaller k → higher variability → more superspreading
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

# ── Overdispersion by setting ──────────────────────────────────────────────────
def estimate_overdispersion_by_setting(rc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits a Negative Binomial model separately for each setting.
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

# ── Overdispersion over time ───────────────────────────────────────────────────
def overdispersion_over_time(rc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tracks how overdispersion changes week by week.
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

# ── Weekly transmission setting proportions ────────────────────────────────────
def weekly_setting_proportions(trees: list, pop: pd.DataFrame,
                                social_network: dict) -> pd.DataFrame:
    """
    Computes, for each week, what fraction of infections happened in each setting.

    This is similar to Figure 2A in the paper.
    """
    pop_idx  = pop.set_index("individual_id")
    settings = ["household","school","workplace","family","community"]

    # Map each individual to their week
    id_to_week = {
        i: pd.Timestamp(pop_idx.loc[i, "test_date"]).to_period("W")
        for i in pop_idx.index
    }

    # Count individuals per week
    week_totals = defaultdict(int)
    for i in pop_idx.index:
        week_totals[id_to_week[i]] += 1

    # Count infections per setting
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
                    setting = s
                    break

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
