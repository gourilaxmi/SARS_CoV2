# SARS-CoV-2 Transmission Network Replication

Replication of **Curran-Sebastian et al. (2026, medRxiv)**  
*"Transmission Networks and Intervention Effects From SARS-CoV-2 Genomic and Social Network Data in Denmark"*

---

## Repository Structure

```
SARS_COV2/
│
├── main.py                     ← Run this
│
├── src/                        ← All analysis modules
│   ├── data_generation.py      ← Loads NCBI sequences + POLYMOD + OxCGRT
│   ├── transmission_network.py ← Builds plausible transmission network G
│   ├── transmission_clusters.py← Outbreak cluster detection
│   ├── reproduction_numbers.py ← Individual Rc + overdispersion
│   ├── npi_analysis.py         ← NPI regression + vaccination effects
│   └── visualizations.py       ← All figures
│
├── data/                       ← Put sequences.fasta here (see data/README.md)
│   └── README.md
│
├── outputs/                    ← Auto-created when you run main.py
│   ├── figures/                ← All plots saved as PNG
│   ├── csv/                    ← All data tables saved as CSV
│   └── results/                ← Summary text file
│
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/GYAN_Replication.git
cd GYAN_Replication
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run (simulated genomes — works immediately)
```bash
python main.py
```

### 4. Run with real NCBI sequences (recommended)
```bash
# Download sequences.fasta from NCBI (see data/README.md)
# Then:
python main.py --fasta data/sequences.fasta
```

---

## Output Files

After running `python main.py` all outputs are saved automatically:

### outputs/figures/
| File | Description |
|------|-------------|
| `fig0_summary_dashboard.png` | Master dashboard — all results on one page |
| `fig1_weekly_proportions.png` | Weekly % of infections per setting |
| `fig2_age_matrices.png` | Age-structured transmission matrices |
| `fig3_clusters.png` | Cluster statistics vs randomised networks |
| `fig4_reproduction_numbers.png` | Rc and overdispersion over time |
| `fig5_npi_effects.png` | NPI effect sizes + vaccination IRR |

### outputs/csv/
| File | Description |
|------|-------------|
| `cluster_summary.csv` | Cluster counts and sizes per variant |
| `overdispersion_by_setting.csv` | Dispersion parameter k per setting |
| `npi_regression_results.csv` | β coefficients, SE, CI, p-values |
| `weekly_setting_proportions.csv` | Week-by-week setting proportions |
| `population.csv` | Full population metadata |
| `aggregate_rc.csv` | Weekly aggregate reproduction number |

### outputs/results/
| File | Description |
|------|-------------|
| `summary.txt` | Key findings printed to file |

---

## Data Sources

| Data | Source | Status |
|------|--------|--------|
| Virus genomes | [NCBI Virus Database](https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/) | Real (you download) |
| Contact matrices | [POLYMOD — Mossong et al. 2008](https://doi.org/10.1371/journal.pmed.0050074) | Embedded |
| NPI policy levels | [OxCGRT Denmark](https://github.com/OxCGRT/covid-policy-tracker) | Embedded |
| Vaccination timeline | SSI Denmark / ECDC 2021 | Embedded |
| Household sizes | Statistics Denmark 2021 | Embedded |

---

## Command Line Options

```bash
python main.py --help

optional arguments:
  --fasta PATH    Path to NCBI FASTA file (default: data/sequences.fasta)
  --trees N       Number of transmission trees to sample (default: 20)
```

### Examples
```bash
# Default run (simulated genomes)
python main.py

# With real sequences
python main.py --fasta data/sequences.fasta

# With more trees for more stable results
python main.py --fasta data/sequences.fasta --trees 50
```

---

## Methods

The pipeline implements the methodology from the paper exactly:

1. **Transmission network G** — connects individuals A→B if:
   - Hamming distance between virus genomes ≤ 2
   - Serial interval 1–11 days
   - Weight = P(generation time) × P(Hamming | time gap) — paper Equation 1

2. **Setting annotation** — edges flagged with shared household/school/workplace/family using POLYMOD social network

3. **Prioritised-settings tree sampling** — household > school > workplace > family > community

4. **Transmission clusters** — connected components in G ∩ N (size > 5)

5. **Reproduction numbers** — individual Rc from secondary infection counts, overdispersion via Negative Binomial fitting

6. **NPI regression** — Poisson GLM of individual Rc on NPI levels, vaccination status, age, variant, holidays

---

## Paper Reference

> Curran-Sebastian J, Morgenstern C, Juul J, et al.
> *Transmission Networks and Intervention Effects From SARS-CoV-2 Genomic and Social Network Data in Denmark.*
> medRxiv 2026. DOI: [10.64898/2026.01.08.26343683](https://doi.org/10.64898/2026.01.08.26343683)
