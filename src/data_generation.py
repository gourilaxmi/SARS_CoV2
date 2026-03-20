"""
data_generation.py — version using real public datasets
======================================================

This script combines three real-world data sources:

1. Genomic data  
   - From the NCBI Virus Database (free access)  
   - SARS-CoV-2 Delta variant sequences  
   - Requires a FASTA file that you download manually (steps below)

2. Contact patterns  
   - Based on the POLYMOD study (Mossong et al. 2008)  
   - Provides realistic age-based interaction patterns

3. NPI timeline  
   - From Oxford’s OxCGRT dataset  
   - Denmark policy data between June–October 2021

How to get sequences.fasta (quick steps):
  1. Go to: https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/
  2. Search for: SARS-CoV-2
  3. Apply filters:
       - Completeness: Complete
       - Region: Europe (or Denmark)
       - Date: 2021-06-01 to 2021-10-31
  4. Download → FASTA format
  5. Save as sequences.fasta in your project folder
  6. Run: python main.py
"""

import numpy as np
import pandas as pd
import os, re
from datetime import date, timedelta

RNG = np.random.default_rng(42)

FASTA_FILE       = "sequences.fasta"
TARGET_SEQUENCES = 2000
GENOME_SUBSAMPLE = 300
START_DATE       = date(2021, 6, 1)
END_DATE         = date(2021, 10, 31)

AGE_GROUPS  = ["0-10", "11-18", "19-39", "40-59", "60+"]
AGE_WEIGHTS = [0.115, 0.095, 0.275, 0.290, 0.225]   # Approximate Denmark population distribution (2021)


# ── POLYMOD contact matrix ─────────────────────────────────────────────────────
# Taken from Mossong et al. (2008)
# Represents average daily contacts between age groups in Europe
# Original matrix has 8 age groups → we later reduce it to 5 groups
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

# Reduce 8 age groups into 5 broader groups used in this project
def _collapse_polymod(M):
    groups = [[0], [1], [2,3], [4,5], [6,7]]
    C = np.zeros((5,5))
    for i,ri in enumerate(groups):
        for j,cj in enumerate(groups):
            C[i,j] = M[np.ix_(ri,cj)].mean()
    return C / C.sum(axis=1, keepdims=True)

POLYMOD_5 = _collapse_polymod(POLYMOD_RAW)


# ── OxCGRT Denmark policy data ────────────────────────────────────────────────
# Extracted from the public Oxford COVID-19 Government Response Tracker
# Values are scaled later to [0,1]
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

# School holiday periods (used as an extra feature)
SCHOOL_HOLIDAYS = [("2021-06-25","2021-08-08"), ("2021-10-11","2021-10-17")]


# ════════════════════════════════════════════════════════════
# 1. LOAD GENOMIC DATA
# ════════════════════════════════════════════════════════════

def load_fasta(path):
    """Simple FASTA parser."""
    records, cur_id, cur_seq = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if cur_id: records.append((cur_id, "".join(cur_seq)))
                cur_id, cur_seq = line[1:].split()[0], []
            else:
                cur_seq.append(line.upper())
    if cur_id: records.append((cur_id, "".join(cur_seq)))
    print(f"  Parsed {len(records)} sequences from FASTA")
    return records

def extract_date(header):
    """Tries to extract a date from sequence metadata."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", header)
    if m:
        try: return date.fromisoformat(m.group(1))
        except: pass
    m = re.search(r"(\d{4}-\d{2})", header)
    if m:
        try:
            y,mo = m.group(1).split("-")
            return date(int(y),int(mo),15)
        except: pass
    return None

def sequences_to_matrix(records, n_keep=TARGET_SEQUENCES, n_pos=GENOME_SUBSAMPLE):
    """Converts sequences into a numeric matrix by sampling genome positions."""
    base_map = {"A":0,"T":1,"G":2,"C":3}
    complete = [(s,q) for s,q in records if len(q)>=25000] or records
    print(f"  Complete sequences (>=25000 bp): {len(complete)}")

    if len(complete) > n_keep:
        idx = RNG.choice(len(complete), size=n_keep, replace=False)
        complete = [complete[i] for i in sorted(idx)]

    min_len   = min(len(q) for _,q in complete)
    positions = np.linspace(0, min_len-1, n_pos, dtype=int)

    G = np.zeros((len(complete), n_pos), dtype=np.int8)
    ids, dates = [], []

    for i,(sid,seq) in enumerate(complete):
        ids.append(sid); dates.append(extract_date(sid))
        for j,p in enumerate(positions):
            G[i,j] = base_map.get(seq[p], 0)

    print(f"  Genome matrix: {G.shape}")
    return G, ids, dates
