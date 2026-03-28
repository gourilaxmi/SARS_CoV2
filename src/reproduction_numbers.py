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

FIX (v2):
  - k is now capped to [1e-3, 10.0] to prevent numerical blow-up when
    settings have sparse transmission events (workplace/school/community).
  - Minimum sample size raised to 20 for reliable NB fitting.
  - overdispersion_over_time() and estimate_overdispersion_by_setting()
    both use the capped fitter.
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
    settings = ["household", "school", "workplace", "family", "community"]

    # Build vaccination multiplier lookup
    vacc_mult_map = {}
    if "vacc_mult" in pop_idx.columns:
        vacc_mult_map = pop_idx["vacc_mult"].to_dict()

    infector_counts = defaultdict(lambda: defaultdict(list))

    for tree in trees:
        secondary = defaultdict(lambda: {s: 0.0 for s in settings + ["total"]})

        for infectee, infector in tree.items():
            if infector is None:
                continue
            vacc_w = vacc_mult_map.get(infector, 1.0)
            s_inf  = social_network.get(infector, {})
            s_infe = social_network.get(infectee, {})
            setting = "community"
            for s in ["household", "school", "workplace", "family"]:
                if (s in s_inf and s in s_infe and s_inf[s] == s_infe[s]):
                    setting = s
                    break
            secondary[infector][setting] += vacc_w
            secondary[infector]["total"]  += vacc_w

        for ind_id in pop_idx.index:
            for s in settings + ["total"]:
                infector_counts[ind_id][s].append(secondary[ind_id][s])

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


# ── Weekly Rc ──────────────────────────────────────────────────────────────────
def weekly_Rc(rc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates individual Rc by week (median across individuals testing that week).
    """
    if "test_date" not in rc_df.columns:
        return pd.DataFrame()

    rc_df = rc_df.copy()
    rc_df["week"] = pd.to_datetime(rc_df["test_date"]).dt.to_period("W")
    weekly = rc_df.groupby("week")["rc_total"].median().reset_index()
    weekly.columns = ["week", "median_Rc"]
    return weekly


# ── Aggregate Rc from renewal equation ────────────────────────────────────────
def aggregate_Rc_renewal(pop: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """
    Approximates the aggregate Rc from weekly case counts using a renewal equation.
    """
    pop_copy = pop.copy()
    pop_copy["week"] = pd.to_datetime(pop_copy["test_date"]).dt.to_period("W")
    weekly_cases = pop_copy.groupby("week").size().reset_index(name="cases")
    weekly_cases = weekly_cases.sort_values("week")

    g_shape   = 4.87**2 / 1.98
    g_scale   = 1.98 / 4.87
    g_dist    = gamma(g_shape, scale=g_scale)
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


# ── Fit Negative Binomial to offspring distribution ───────────────────────────
def fit_negative_binomial(counts: np.ndarray) -> dict:
    """
    Fits Negative Binomial NB(mean=Rc, dispersion=k) to offspring counts.
    Returns dict: {mean, k, log_likelihood}

    FIX: k is capped to [1e-3, 10.0] to avoid:
      - k → 0   : when too few events exist (workplace/school/community)
      - k → ∞   : when distribution is very tight (household with many zeros)
    Minimum sample raised to 20 for reliable fitting.
    """
    counts = np.array(counts, dtype=float)
    counts = counts[counts >= 0]

    # Require at least 20 observations AND at least some non-zero counts
    if len(counts) < 20 or counts.sum() == 0:
        return {
            "mean":           float(np.mean(counts)) if len(counts) > 0 else 0.0,
            "k":              np.nan,
            "log_likelihood": -np.inf,
            "converged":      False,
        }

    mu0 = max(np.mean(counts), 1e-4)

    # Method-of-moments starting estimate for k
    var0 = np.var(counts)
    k0   = mu0**2 / max(var0 - mu0, 1e-4)
    k0   = np.clip(k0, 0.01, 5.0)

    def neg_loglik(params):
        mu, k = params
        if mu <= 0 or k <= 0:
            return 1e10
        r  = k
        p  = k / (k + mu)
        ll = np.sum(nbinom.logpmf(counts.astype(int), r, p))
        return -ll

    result = minimize(
        neg_loglik, [mu0, k0],
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 3000}
    )
    mu_fit, k_fit = result.x

    # ── KEY FIX: cap k to epidemiologically sensible range ────────────────────
    # k < 0.001 : essentially no information (sparse setting)
    # k > 10    : essentially Poisson — no meaningful overdispersion signal
    k_fit = float(np.clip(k_fit, 1e-3, 10.0))
    mu_fit = float(max(mu_fit, 0.0))

    return {
        "mean":           mu_fit,
        "k":              k_fit,
        "log_likelihood": -result.fun,
        "converged":      result.success,
    }


# ── Setting-stratified overdispersion ─────────────────────────────────────────
def estimate_overdispersion_by_setting(rc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits NB to each setting's offspring distribution.
    Settings with too few events will show k=NaN (plotted as 0 with annotation).
    """
    settings = ["total", "household", "school", "workplace", "family", "community"]
    rows = []
    for s in settings:
        col = f"rc_{s}"
        if col not in rc_df.columns:
            continue
        counts = rc_df[col].dropna().values

        # Only fit if there are enough non-zero transmission events
        nonzero = counts[counts > 0]
        if len(nonzero) < 20:
            rows.append({
                "setting":        s,
                "mean":           float(np.mean(counts)),
                "k":              np.nan,
                "log_likelihood": -np.inf,
                "converged":      False,
                "n_events":       int(len(nonzero)),
                "note":           "insufficient events for NB fit",
            })
            continue

        fit = fit_negative_binomial(counts)
        fit["n_events"] = int(len(nonzero))
        fit["note"]     = "ok"
        rows.append({"setting": s, **fit})

    return pd.DataFrame(rows)


# ── Overdispersion over time ───────────────────────────────────────────────────
def overdispersion_over_time(rc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits NB per week to capture temporal evolution of overdispersion.
    Weeks with < 20 non-zero events are skipped.
    """
    if "test_date" not in rc_df.columns:
        return pd.DataFrame()

    rc_df = rc_df.copy()
    rc_df["week"] = pd.to_datetime(rc_df["test_date"]).dt.to_period("W")
    rows = []

    for week, grp in rc_df.groupby("week"):
        counts  = grp["rc_total"].values
        nonzero = counts[counts > 0]
        if len(nonzero) < 20:
            continue
        fit = fit_negative_binomial(counts)
        if not np.isnan(fit["k"]):
            rows.append({"week": week, "k": fit["k"], "mean_Rc": fit["mean"]})

    return pd.DataFrame(rows)


# ── Weekly setting proportions ────────────────────────────────────────────────
def weekly_setting_proportions(trees: list, pop: pd.DataFrame,
                                social_network: dict) -> pd.DataFrame:
    """
    Computes for each week the proportion of individuals whose sampled
    infector belongs to each setting (reproduces Figure 2A of the paper).
    """
    pop_idx  = pop.set_index("individual_id")
    settings = ["household", "school", "workplace", "family", "community"]

    id_to_week = {
        i: pd.Timestamp(pop_idx.loc[i, "test_date"]).to_period("W")
        for i in pop_idx.index
    }
    week_totals = defaultdict(int)
    for i in pop_idx.index:
        week_totals[id_to_week[i]] += 1

    week_setting_counts = defaultdict(lambda: {s: 0 for s in settings + ["any"]})

    n_trees = len(trees)
    for tree in trees:
        for infectee, infector in tree.items():
            if infector is None or infectee not in pop_idx.index:
                continue
            wk     = id_to_week[infectee]
            s_inf  = social_network.get(infector, {})
            s_infe = social_network.get(infectee, {})
            setting = "community"
            for s in ["household", "school", "workplace", "family"]:
                if s in s_inf and s in s_infe and s_inf[s] == s_infe[s]:
                    setting = s
                    break
            week_setting_counts[wk][setting] += 1
            week_setting_counts[wk]["any"]    += 1

    rows = []
    for wk in sorted(week_totals.keys()):
        denom = week_totals[wk] * n_trees
        row   = {"week": wk}
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
    print(od_df[["setting", "mean", "k", "n_events", "note"]].to_string(index=False))

    print("\nWeekly setting proportions (first 5 weeks):")
    weekly = weekly_setting_proportions(trees, pop, social)
    print(weekly.head().to_string(index=False))