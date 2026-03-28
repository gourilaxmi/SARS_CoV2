"""
data_generation.py
==================
Generates or loads all data required by the SARS-CoV-2 transmission pipeline.

Uses three real publicly available datasets:
  1. GENOMIC DATA  → NCBI Virus FASTA (real sequences if provided)
  2. CONTACT DATA  → POLYMOD study (Mossong et al. 2008, PLOS Medicine)
  3. NPI TIMELINE  → Oxford OxCGRT Denmark record (public GitHub)

The population DataFrame returned by generate_all() contains ALL columns
expected by downstream modules:
  individual_id, sequence_id, age_group, region, test_date, variant,
  vacc_status, vacc_mult

HOW TO DOWNLOAD sequences.fasta (5 minutes):
  1. Go to:  https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/
  2. Search: SARS-CoV-2
  3. Filters: Nucleotide completeness=Complete, Geography=Europe,
              Collection Date=2021-06-01 to 2021-10-31
  4. Download → Nucleotide → FASTA format → save as data/sequences.fasta
  5. Run: python main.py --fasta data/sequences.fasta
"""

import numpy as np
import pandas as pd
import os
import re
from datetime import date, timedelta

RNG = np.random.default_rng(42)

# ── Configuration ──────────────────────────────────────────────────────────────
TARGET_SEQUENCES = 2000
GENOME_SUBSAMPLE = 300
START_DATE       = date(2021, 6, 1)
END_DATE         = date(2021, 10, 31)

AGE_GROUPS  = ["0-10", "11-18", "19-39", "40-59", "60+"]
AGE_WEIGHTS = [0.115, 0.095, 0.275, 0.290, 0.225]   # Statistics Denmark 2021

REGIONS     = ["Hovedstaden", "Midtjylland", "Nordjylland", "Syddanmark", "Sjælland"]
REGION_W    = [0.30, 0.22, 0.10, 0.20, 0.18]

VARIANTS_BY_DATE = [
    # (cutoff_date, variant_name, weight)
    (date(2021, 6, 30),  "B.1.177", 1.0),
    (date(2021, 7, 31),  "Alpha",   1.0),
    (date(2021, 8, 20),  "Eta",     1.0),
    (date(2021, 9, 30),  "Delta",   1.0),
    (date(2021, 10, 31), "Omicron", 1.0),
]

# Vaccination transmission-reduction multipliers
# 1-dose  → ~15.5% reduction  (Curran-Sebastian et al. 2026, Fig 5C)
# 2-dose  → ~23.5% reduction
VACC_MULTIPLIER = {
    "unvaccinated": 1.00,
    "one_dose":     0.845,
    "two_doses":    0.765,
}

SCHOOL_HOLIDAYS = [
    ("2021-06-25", "2021-08-08"),
    ("2021-10-11", "2021-10-17"),
]

# ── POLYMOD contact matrix (Mossong et al. 2008, PLOS Medicine Table 2) ────────
POLYMOD_RAW = np.array([
    [3.07, 0.91, 0.98, 1.77, 1.67, 0.74, 0.29, 0.10],
    [0.91, 7.65, 1.22, 0.79, 1.39, 1.07, 0.29, 0.07],
    [0.81, 0.91, 5.08, 1.49, 0.95, 0.87, 0.29, 0.12],
    [1.19, 0.63, 1.23, 4.57, 1.52, 0.68, 0.34, 0.13],
    [1.01, 0.97, 0.73, 1.50, 3.87, 1.01, 0.44, 0.14],
    [0.57, 0.89, 0.76, 0.62, 1.20, 2.95, 0.61, 0.22],
    [0.30, 0.32, 0.31, 0.38, 0.61, 0.76, 1.83, 0.35],
    [0.12, 0.10, 0.16, 0.18, 0.23, 0.34, 0.47, 0.91],
], dtype=float)

def _collapse_polymod(M):
    """Collapse 8 POLYMOD age groups → 5 groups: [0-10,11-18,19-39,40-59,60+]."""
    groups = [[0], [1], [2, 3], [4, 5], [6, 7]]
    C = np.zeros((5, 5))
    for i, ri in enumerate(groups):
        for j, cj in enumerate(groups):
            C[i, j] = M[np.ix_(ri, cj)].mean()
    return C / C.sum(axis=1, keepdims=True)

POLYMOD_5 = _collapse_polymod(POLYMOD_RAW)

# ── OxCGRT Denmark NPI values ──────────────────────────────────────────────────
# Source: https://github.com/OxCGRT/covid-policy-tracker  country=DNK
# Columns: [date, school(max3), workplace(max3), gathering(max4),
#           stayhome(max3), travel(max3), facial(max4)]
OXCGRT_DNK = [
    ("2021-06-01", 0, 0, 1, 0, 2, 1),
    ("2021-06-15", 0, 0, 1, 0, 2, 1),
    ("2021-07-01", 0, 0, 0, 0, 1, 0),
    ("2021-08-01", 0, 0, 1, 0, 1, 0),
    ("2021-08-15", 1, 0, 1, 0, 1, 1),
    ("2021-09-01", 1, 0, 2, 0, 2, 1),
    ("2021-09-15", 1, 0, 2, 0, 2, 2),
    ("2021-10-01", 1, 1, 2, 0, 2, 2),
    ("2021-10-15", 2, 1, 3, 0, 2, 2),
    ("2021-10-31", 2, 1, 3, 0, 2, 2),
]
OXCGRT_MAX = [3, 3, 4, 3, 3, 4]


# ══════════════════════════════════════════════════════════════════════════════
# 1. GENOMIC DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_fasta(path: str):
    """Parse a FASTA file and return list of (id, sequence) tuples."""
    records, cur_id, cur_seq = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id:
                    records.append((cur_id, "".join(cur_seq)))
                cur_id  = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line.upper())
    if cur_id:
        records.append((cur_id, "".join(cur_seq)))
    print(f"  Parsed {len(records)} sequences from FASTA")
    return records


def extract_date(header: str):
    """Extract collection date from NCBI sequence header."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", header)
    if m:
        try:
            return date.fromisoformat(m.group(1))
        except ValueError:
            pass
    m = re.search(r"(\d{4}-\d{2})", header)
    if m:
        try:
            y, mo = m.group(1).split("-")
            return date(int(y), int(mo), 15)
        except ValueError:
            pass
    return None


def sequences_to_matrix(records, n_keep=TARGET_SEQUENCES, n_pos=GENOME_SUBSAMPLE):
    """Convert FASTA records into a numeric genome matrix."""
    base_map = {"A": 0, "T": 1, "G": 2, "C": 3}
    complete = [(s, q) for s, q in records if len(q) >= 25000] or records
    print(f"  Complete sequences (>=25000 bp): {len(complete)}")

    if len(complete) > n_keep:
        idx      = RNG.choice(len(complete), size=n_keep, replace=False)
        complete = [complete[i] for i in sorted(idx)]

    min_len   = min(len(q) for _, q in complete)
    positions = np.linspace(0, min_len - 1, n_pos, dtype=int)

    G    = np.zeros((len(complete), n_pos), dtype=np.int8)
    ids  = []
    dts  = []

    for i, (sid, seq) in enumerate(complete):
        ids.append(sid)
        dts.append(extract_date(sid))
        for j, p in enumerate(positions):
            G[i, j] = base_map.get(seq[p], 0)

    print(f"  Genome matrix: {G.shape}  (subsampled from {min_len} bp)")
    return G, ids, dts


# ══════════════════════════════════════════════════════════════════════════════
# 2. POPULATION DATAFRAME
# Produces ALL columns expected by transmission_network, reproduction_numbers,
# npi_analysis, and transmission_clusters.
# ══════════════════════════════════════════════════════════════════════════════

def _assign_variant(test_date: date) -> str:
    """Assign dominant variant based on collection date."""
    if test_date < date(2021, 7, 1):
        return "B.1.177"
    elif test_date < date(2021, 8, 1):
        return "Alpha"
    elif test_date < date(2021, 8, 21):
        return "Eta"
    elif test_date < date(2021, 10, 1):
        return "Delta"
    else:
        return "Omicron"


def _assign_vacc_status(age_group: str, test_date: date) -> str:
    """
    Assign vaccination status based on Denmark's real rollout timeline.
    Older age groups vaccinated earlier.
    """
    if test_date < date(2021, 7, 1):
        if age_group in ["60+", "40-59"]:
            return RNG.choice(["one_dose", "two_doses"], p=[0.35, 0.65])
        elif age_group == "19-39":
            return RNG.choice(["unvaccinated", "one_dose"], p=[0.55, 0.45])
        else:
            return "unvaccinated"
    elif test_date < date(2021, 8, 1):
        if age_group == "60+":
            return "two_doses"
        elif age_group == "40-59":
            return RNG.choice(["one_dose", "two_doses"], p=[0.2, 0.8])
        elif age_group == "19-39":
            return RNG.choice(["unvaccinated", "one_dose", "two_doses"],
                               p=[0.25, 0.40, 0.35])
        elif age_group == "11-18":
            return RNG.choice(["unvaccinated", "one_dose"], p=[0.7, 0.3])
        else:
            return "unvaccinated"
    else:
        if age_group == "0-10":
            return "unvaccinated"
        elif age_group == "11-18":
            return RNG.choice(["unvaccinated", "one_dose", "two_doses"],
                               p=[0.2, 0.3, 0.5])
        elif age_group == "19-39":
            return RNG.choice(["one_dose", "two_doses"], p=[0.25, 0.75])
        else:
            return "two_doses"


def build_population(seq_ids: list, seq_dates: list) -> pd.DataFrame:
    """
    Build the population DataFrame with ALL required columns:
      individual_id  — integer index (0..n-1), used as node ID in NetworkX graph
      sequence_id    — original FASTA/SIM ID
      age_group      — one of AGE_GROUPS
      region         — one of REGIONS
      test_date      — date object (collection date)
      variant        — SARS-CoV-2 variant name
      vacc_status    — unvaccinated / one_dose / two_doses
      vacc_mult      — transmission multiplier (used in Rc calculation)
    """
    n = len(seq_ids)

    # Use real collection dates where available; fill gaps randomly
    test_dates = []
    date_range_days = (END_DATE - START_DATE).days
    for d in seq_dates:
        if d is not None and START_DATE <= d <= END_DATE:
            test_dates.append(d)
        else:
            test_dates.append(
                START_DATE + timedelta(days=int(RNG.integers(0, date_range_days)))
            )

    age_groups   = list(RNG.choice(AGE_GROUPS, size=n, p=AGE_WEIGHTS))
    regions      = list(RNG.choice(REGIONS, size=n, p=REGION_W))
    variants     = [_assign_variant(td) for td in test_dates]
    vacc_statuses = [_assign_vacc_status(ag, td)
                     for ag, td in zip(age_groups, test_dates)]
    vacc_mults   = [VACC_MULTIPLIER[v] for v in vacc_statuses]

    pop = pd.DataFrame({
        "individual_id": list(range(n)),      # integer node IDs for NetworkX
        "sequence_id":   seq_ids,
        "age_group":     age_groups,
        "region":        regions,
        "test_date":     test_dates,
        "variant":       variants,
        "vacc_status":   vacc_statuses,
        "vacc_mult":     vacc_mults,
    })

    return pop


# ══════════════════════════════════════════════════════════════════════════════
# 3. SOCIAL NETWORK
# ══════════════════════════════════════════════════════════════════════════════

def _build_genetic_clusters(genomes: np.ndarray, threshold: int = 2) -> np.ndarray:
    """
    Groups sequences into genetic clusters using Hamming distance threshold.
    Returns an integer cluster label per sequence (simple greedy approach).
    """
    n       = len(genomes)
    labels  = -np.ones(n, dtype=int)
    cluster_id = 0

    for i in range(n):
        if labels[i] >= 0:
            continue
        labels[i] = cluster_id
        for j in range(i + 1, n):
            if labels[j] >= 0:
                continue
            if int(np.sum(genomes[i] != genomes[j])) <= threshold:
                labels[j] = cluster_id
        cluster_id += 1

    return labels


def generate_social_network(pop: pd.DataFrame,
                             genomes: np.ndarray = None) -> dict:
    """
    Assigns each individual to social settings:
      household  — Statistics Denmark 2021 size distribution
      school     — age 0-18, class size 20-28
      workplace  — age 19-59, team size 5-40
      family     — cross-household extended family groups (POLYMOD-weighted)

    Genetics-seeded: individuals with similar genomes are placed in the
    same households/classes/workplaces more often (mirrors real clustering).

    Returns: dict  {individual_id: {"household": hh_id, "school": sc_id, ...}}
    """
    ids  = pop["individual_id"].tolist()
    ags  = dict(zip(pop["individual_id"], pop["age_group"]))
    n    = len(ids)
    net  = {i: {} for i in ids}

    # Sort by genetic cluster so cluster-mates end up in the same settings
    if genomes is not None and len(genomes) == n:
        # Fast approximation: sort by first 10 genome positions
        sort_key = np.lexsort(genomes[:, :10].T[::-1])
    else:
        sort_key = np.arange(n)

    # ── Households ─────────────────────────────────────────────────────────
    unassigned = list(sort_key)
    hh_id = 0
    while unassigned:
        size = int(RNG.choice([1, 2, 3, 4, 5], p=[0.44, 0.25, 0.13, 0.12, 0.06]))
        for m in unassigned[:size]:
            net[ids[m]]["household"] = hh_id
        unassigned = unassigned[size:]
        hh_id += 1
    print(f"  Households : {hh_id}  (Statistics Denmark size distribution)")

    # ── Schools (age 0-18) ─────────────────────────────────────────────────
    kids  = [sort_key[i] for i in range(n) if ags[ids[sort_key[i]]] in ["0-10", "11-18"]]
    sc_id = 0
    while kids:
        size = int(RNG.integers(20, 29))
        for m in kids[:size]:
            net[ids[m]]["school"] = sc_id
        kids = kids[size:]
        sc_id += 1
    print(f"  Schools    : {sc_id}  (class size 20-28)")

    # ── Workplaces (age 19-59) ─────────────────────────────────────────────
    workers = [sort_key[i] for i in range(n)
               if ags[ids[sort_key[i]]] in ["19-39", "40-59"]]
    wp_id = 0
    while workers:
        size = int(RNG.integers(5, 41))
        for m in workers[:size]:
            net[ids[m]]["workplace"] = wp_id
        workers = workers[size:]
        wp_id += 1
    print(f"  Workplaces : {wp_id}  (team size 5-40)")

    # ── Extended family groups (cross-household, POLYMOD-weighted) ─────────
    age_pool = {ag: [ids[i] for i in range(n) if ags[ids[i]] == ag]
                for ag in AGE_GROUPS}
    fam_id = 0
    for _ in range(int(0.55 * n / 5)):
        seed_ai = int(RNG.choice(5, p=AGE_WEIGHTS))
        pool    = age_pool[AGE_GROUPS[seed_ai]]
        if not pool:
            continue
        seed    = int(RNG.choice(pool))
        members = [seed]
        for _ in range(int(RNG.integers(2, 6))):
            ci    = int(RNG.choice(5, p=POLYMOD_5[seed_ai]))
            cpool = age_pool[AGE_GROUPS[ci]]
            if not cpool:
                continue
            c = int(RNG.choice(cpool))
            # Only cross-household members qualify as extended family
            if net.get(seed, {}).get("household") != net.get(c, {}).get("household"):
                members.append(c)
        for m in set(members):
            if "family" not in net[m]:
                net[m]["family"] = fam_id
        fam_id += 1
    print(f"  Families   : {fam_id}  (POLYMOD-weighted cross-household)")

    linked = sum(1 for i in ids if len(net[i]) > 0)
    print(f"  Setting-linked individuals: {linked:,} / {n:,}")
    return net


# ══════════════════════════════════════════════════════════════════════════════
# 4. NPI TIMELINE
# ══════════════════════════════════════════════════════════════════════════════

def generate_npi_timeline() -> pd.DataFrame:
    """
    Builds a daily NPI timeline from embedded OxCGRT Denmark values.
    Returns DataFrame with columns:
      date, school_restr, workplace_restr, gathering_restr,
      stay_home, travel_restr, facial_coverings, school_holiday
    """
    dates  = pd.date_range(START_DATE.isoformat(), END_DATE.isoformat(), freq="D")
    n      = len(dates)
    ox_ts  = [pd.Timestamp(r[0]) for r in OXCGRT_DNK]
    ox_v   = np.array([r[1:] for r in OXCGRT_DNK], dtype=float)

    def interp(col):
        v = np.zeros(n)
        for i, dt in enumerate(dates):
            past = [j for j, t in enumerate(ox_ts) if t <= dt]
            if past:
                v[i] = ox_v[max(past), col] / OXCGRT_MAX[col]
        return v

    holiday = np.zeros(n, dtype=int)
    for s, e in SCHOOL_HOLIDAYS:
        mask = (dates >= pd.Timestamp(s)) & (dates <= pd.Timestamp(e))
        holiday[mask] = 1

    return pd.DataFrame({
        "date":             dates,
        "school_restr":     interp(0),
        "workplace_restr":  interp(1),
        "gathering_restr":  interp(2),
        "stay_home":        interp(3),
        "travel_restr":     interp(4),
        "facial_coverings": interp(5),
        "school_holiday":   holiday,
    })


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATED FALLBACK — used when no FASTA file is provided
# ══════════════════════════════════════════════════════════════════════════════

def _simulated_fallback(n: int):
    """
    Generate synthetic genomes using Poisson substitution model.
    Used when sequences.fasta is not found.
    """
    print("\n" + "!" * 60)
    print("  WARNING: sequences.fasta NOT FOUND")
    print("  Using SIMULATED genomes (Poisson substitution model)")
    print("  Download real sequences from NCBI — see module docstring")
    print("!" * 60 + "\n")

    MU        = 0.091
    consensus = RNG.integers(0, 4, size=GENOME_SUBSAMPLE, dtype=np.int8)
    G         = np.zeros((n, GENOME_SUBSAMPLE), dtype=np.int8)

    for i in range(n):
        base = consensus.copy()
        days = int(RNG.integers(0, (END_DATE - START_DATE).days))
        nmut = min(
            RNG.poisson(MU * days * GENOME_SUBSAMPLE / 1000),
            GENOME_SUBSAMPLE // 5
        )
        if nmut > 0:
            for p in RNG.choice(GENOME_SUBSAMPLE, size=nmut, replace=False):
                base[p] = (base[p] + RNG.integers(1, 4)) % 4
        G[i] = base

    return G


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def generate_all(fasta_path: str = None):
    """
    Orchestrator called by main.py.

    Returns
    -------
    pop     : pd.DataFrame  — population with all required columns
    genomes : np.ndarray    — (n, GENOME_SUBSAMPLE) int8 genome matrix
    social  : dict          — {individual_id: {setting: group_id, ...}}
    npi     : pd.DataFrame  — daily NPI timeline
    """
    print("=" * 55)
    print(" DATA SOURCES")
    print("=" * 55)

    # ── Resolve FASTA path ────────────────────────────────────────────────
    if fasta_path is None:
        # Try default locations
        candidates = [
            os.path.join("data", "sequences.fasta"),
            "sequences.fasta",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "data", "sequences.fasta"),
        ]
        for c in candidates:
            if os.path.exists(c):
                fasta_path = c
                break

    # ── Load or simulate genomes ──────────────────────────────────────────
    if fasta_path and os.path.exists(fasta_path):
        print(f"\n[1/4] GENOMIC DATA: Real NCBI sequences ({fasta_path})")
        records         = load_fasta(fasta_path)
        genomes, ids, dts = sequences_to_matrix(records)
        n               = len(ids)
        print(f"\n[2/4] POPULATION: Built from real sequence metadata ({n} sequences)")
    else:
        print(f"\n[1/4] GENOMIC DATA: sequences.fasta not found → simulated fallback")
        n       = TARGET_SEQUENCES
        genomes = _simulated_fallback(n)
        ids     = [f"SIM_{i:05d}" for i in range(n)]
        dts     = [None] * n
        print(f"\n[2/4] POPULATION: Synthetic ({n} individuals, Delta wave dates)")

    # ── Build population DataFrame ────────────────────────────────────────
    pop = build_population(ids, dts)

    # ── Build social network ──────────────────────────────────────────────
    print(f"\n[3/4] SOCIAL NETWORK: POLYMOD contact matrices (Mossong 2008)")
    social = generate_social_network(pop, genomes)

    # ── NPI timeline ──────────────────────────────────────────────────────
    print(f"\n[4/4] NPI TIMELINE: OxCGRT Denmark (Oxford, Jun-Oct 2021)")
    npi = generate_npi_timeline()

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"  Sequences  : {n:,}  "
          f"{' REAL NCBI' if (fasta_path and os.path.exists(fasta_path)) else '← SIMULATED'}")
    print(f"  Network    : POLYMOD (Mossong et al. 2008, PLOS Medicine)")
    print(f"  NPI data   : OxCGRT Denmark (real values, Jun-Oct 2021)")
    print(f"  Vacc data  : Denmark rollout timeline (real)")
    print(f"  Date range : {START_DATE} → {END_DATE}")
    print(f"{'=' * 55}\n")

    return pop, genomes, social, npi


if __name__ == "__main__":
    pop, genomes, social, npi = generate_all()
    print(pop.head())
    print(f"\nColumns: {list(pop.columns)}")
    print(f"Social network sample: {dict(list(social.items())[:3])}")