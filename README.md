# Maternal Risk Stratification in Low-Resource Settings
## Unsupervised K-Means Clustering of Antenatal Health Records

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

### 1. Executive Summary

In Sub-Saharan Africa, maternal mortality remains unacceptably high (~545 per 100,000 live births). A key operational barrier in primary antenatal facilities is the delay in identifying multi-parameter deterioration in pregnant women due to staffing shortages and fragmented paper records.

This research project demonstrates an end-to-end, reproducible **unsupervised machine learning system (K-Means clustering)** for **maternal risk-tier stratification**:
- Evaluated on $8,817$ real patient encounters from primary maternity clinics across 5 districts in **Tanzania** (Zenodo DOI: `10.5281/zenodo.15309733`).
- Benchmarked against the standard **UCI Maternal Health Risk Dataset** ($N=1,014$).
- Fulfills **Objective 1 (Data Standardisation)**, **Objective 2 (K-Means Optimisation & Stability)**, and **Objective 3 (Independent Adverse Outcome Validation)**.
- Features a **fully responsive clinical decision-support frontend** built with Streamlit and Plotly.

---

### 2. Clinical and Statistical Validation Findings

| Metric / Attribute | Tanzania Primary Clinic Cohort (Sub-Saharan Africa) | UCI Benchmark Cohort (South Asia) |
| :--- | :--- | :--- |
| **Cohort Size ($N$)** | **8,817 patient records** | **1,014 patient records** |
| **Clustering Space ($D$)** | Age, SBP, DBP, Pulse, Weight, Hb, Gestational Age, Gravidity | Age, SBP, DBP, Blood Sugar, Body Temp, Heart Rate |
| **Optimal Cluster Partition ($K$)** | $K = 3$ | $K = 3$ |
| **Cluster Stability (Mean ARI across 5 seeds)** | **0.9771** ($\pm 0.0076$) | **1.0000** ($\pm 0.0000$) |
| **Adverse Incidence - Low Risk Tier** | **45.50%** [95% CI: 43.90%, 47.11%] | **13.40%** [95% CI: 10.69%, 16.67%] |
| **Adverse Incidence - Intermediate Tier**| **72.49%** [95% CI: 70.51%, 74.39%] | **17.86%** [95% CI: 14.13%, 22.31%] |
| **Adverse Incidence - High Risk Tier** | **76.19%** [95% CI: 74.65%, 77.65%] | **81.46%** [95% CI: 75.11%, 86.48%] |
| **Chi-Square Independence Test** | $\chi^2 = 788.793$, $p = 5.20 \times 10^{-172}$ | $\chi^2 = 338.122$, $p = 6.43 \times 10^{-72}$ |
| **Cramér's V (Effect Size)** | **0.2991** (Substantial clinical effect) | **0.4083** (Strong effect size) |
| **Relative Risk (High vs Low)** | **$\text{RR} = 1.67$** [95% CI: 1.61, 1.74] | **$\text{RR} = 6.08$** [95% CI: 4.81, 7.68] |
| **Odds Ratio (High vs Low)** | **$\text{OR} = 3.83$** [95% CI: 3.45, 4.26] | **$\text{OR} = 28.40$** [95% CI: 17.97, 44.86] |

---

### 3. Architecture & Project Layout

```
int396/
├── data/
│   ├── raw/                           <- Raw downloaded clinical CSV files
│   │   ├── tanzania_maternal_mhrs.csv
│   │   └── maternal_health_risk.csv
│   └── processed/                     <- Cleaned & standardized feature matrices
│       ├── tanzania_mhrs_standardized.csv
│       └── uci_benchmark_standardized.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 <- Ingestion, schema audit & BP parser
│   ├── preprocessing.py               <- Physiological bounds, imputation & scaling
│   ├── clustering.py                  <- K-Means, K optimization, stability & PCA
│   ├── evaluation.py                  <- Adverse outcome validation, Chi2, RR, OR
│   └── visualisation.py               <- Responsive Plotly interactive charts
├── reports/
│   ├── figures/                       <- High-resolution plots (PNG)
│   └── results/                       <- CSV tables of metrics & profiles
├── app/
│   └── streamlit_app.py               <- Fully responsive interactive web UI
├── main.py                            <- Headless CLI pipeline runner
└── requirements.txt                   <- Dependencies
```

---

### 4. How to Launch and Use the Application

#### A. Interactive Responsive Web Frontend
Launch the modern Streamlit web dashboard:
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

The frontend includes 6 responsive tabs:
1. **🏥 Clinical Overview:** Low-resource context, WHO guidelines, and project boundaries.
2. **📊 Data Quality & EDA:** Distributions, missing data audit, and collinearity heatmaps.
3. **🔬 K-Means Studio:** Elbow curve, Silhouette coefficients, 2D/3D PCA projection, and centroid radar charts.
4. **⚖️ Adverse Outcome Validation:** Chi-Square test, Cramér's V, Relative Risk, and Odds Ratios with 95% CIs.
5. **🩺 Patient Triage Simulator:** Interactive sliders simulating an antenatal clinic visit, computing Euclidean distance to centroids and recommending clinical triage pathways.
6. **📥 Reports & Export:** Download patient cluster assignments and academic CSV summaries.

#### B. Headless Batch Execution
Generate all figures and result CSVs in batch mode:
```bash
# Run for Tanzania Primary Healthcare Dataset:
python main.py --dataset tanzania --k 3

# Run for UCI Benchmark Dataset:
python main.py --dataset uci --k 3
```

---

### 5. Ethical & Clinical Boundaries
- **Research Prototype:** This software is designed for educational research and triage decision support for community health workers and midwives.
- **Non-Diagnostic:** It does not provide definitive medical diagnoses or prescribe treatments.
- **Non-Causal:** Discovered clusters represent statistical proximity in multidimensional physiological space, not direct biological causation.

