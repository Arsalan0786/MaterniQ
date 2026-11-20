"""
Script to generate consolidated performance matrix CSV for reporting.
"""
import pandas as pd

perf_data = [
    {
        'Domain': 'Internal Clustering Quality',
        'Metric': 'Within-Cluster Sum of Squares (Inertia, K=3)',
        'Tanzania Cohort (N=8817)': '55,769.44',
        'UCI Benchmark (N=1014)': '3,289.23',
        'Academic Benchmark Target': 'Lower is superior (Elbow point)'
    },
    {
        'Domain': 'Internal Clustering Quality',
        'Metric': 'Silhouette Coefficient (s)',
        'Tanzania Cohort (N=8817)': '0.1147',
        'UCI Benchmark (N=1014)': '0.2954',
        'Academic Benchmark Target': '> 0.0 (Valid separation in high-dim clinical data)'
    },
    {
        'Domain': 'Internal Clustering Quality',
        'Metric': 'Davies-Bouldin Index (DBI)',
        'Tanzania Cohort (N=8817)': '2.1936',
        'UCI Benchmark (N=1014)': '1.3112',
        'Academic Benchmark Target': 'Lower is superior (Tighter clusters)'
    },
    {
        'Domain': 'Internal Clustering Quality',
        'Metric': 'Calinski-Harabasz Index (CHI)',
        'Tanzania Cohort (N=8817)': '1,166.88',
        'UCI Benchmark (N=1014)': '429.51',
        'Academic Benchmark Target': 'Higher is superior (Dispersion ratio)'
    },
    {
        'Domain': 'Internal Clustering Quality',
        'Metric': 'Cluster Stability (Pairwise Mean ARI, 5 Seeds)',
        'Tanzania Cohort (N=8817)': '0.9771 (+/-0.0076)',
        'UCI Benchmark (N=1014)': '1.0000 (+/-0.0000)',
        'Academic Benchmark Target': '> 0.90 (Deterministic partition stability)'
    },
    {
        'Domain': 'External Clinical Validation',
        'Metric': 'Pearson Chi-Square Independence (Chi2)',
        'Tanzania Cohort (N=8817)': '788.793',
        'UCI Benchmark (N=1014)': '338.122',
        'Academic Benchmark Target': 'p < 0.05 (Non-random outcome association)'
    },
    {
        'Domain': 'External Clinical Validation',
        'Metric': 'Hypothesis Test p-value',
        'Tanzania Cohort (N=8817)': '5.20e-172',
        'UCI Benchmark (N=1014)': '6.43e-72',
        'Academic Benchmark Target': 'Statistically Significant (p <<< 0.001)'
    },
    {
        'Domain': 'External Clinical Validation',
        'Metric': "Cramer's V (Effect Size)",
        'Tanzania Cohort (N=8817)': '0.2991',
        'UCI Benchmark (N=1014)': '0.4083',
        'Academic Benchmark Target': '> 0.25 (Substantial / Strong clinical effect)'
    },
    {
        'Domain': 'External Clinical Validation',
        'Metric': 'Relative Risk: High vs Baseline (RR [95% CI])',
        'Tanzania Cohort (N=8817)': '1.67 [1.61, 1.74]',
        'UCI Benchmark (N=1014)': '6.08 [4.81, 7.68]',
        'Academic Benchmark Target': 'CI strictly excludes 1.0'
    },
    {
        'Domain': 'External Clinical Validation',
        'Metric': 'Odds Ratio: High vs Baseline (OR [95% CI])',
        'Tanzania Cohort (N=8817)': '3.83 [3.45, 4.26]',
        'UCI Benchmark (N=1014)': '28.40 [17.97, 44.86]',
        'Academic Benchmark Target': 'CI strictly excludes 1.0'
    },
    {
        'Domain': 'External Clinical Validation',
        'Metric': 'Adverse Outcome Incidence: Lower Risk Tier',
        'Tanzania Cohort (N=8817)': '45.50% [43.90, 47.11]',
        'UCI Benchmark (N=1014)': '13.40% [10.69, 16.67]',
        'Academic Benchmark Target': 'Statistically separable from intermediate/high'
    },
    {
        'Domain': 'External Clinical Validation',
        'Metric': 'Adverse Outcome Incidence: Intermediate Tier',
        'Tanzania Cohort (N=8817)': '72.49% [70.51, 74.39]',
        'UCI Benchmark (N=1014)': '17.86% [14.13, 22.31]',
        'Academic Benchmark Target': 'Clear gradient progression'
    },
    {
        'Domain': 'External Clinical Validation',
        'Metric': 'Adverse Outcome Incidence: Higher Risk Tier',
        'Tanzania Cohort (N=8817)': '76.19% [74.65, 77.65]',
        'UCI Benchmark (N=1014)': '81.46% [75.11, 86.48]',
        'Academic Benchmark Target': 'Peak observed risk concentration'
    },
    {
        'Domain': 'External Clustering Agreement',
        'Metric': 'Cluster Purity',
        'Tanzania Cohort (N=8817)': '0.6626 (66.3%)',
        'UCI Benchmark (N=1014)': '0.6726 (67.3%)',
        'Academic Benchmark Target': 'Dominant class concentration'
    },
    {
        'Domain': 'External Clustering Agreement',
        'Metric': 'Normalized Mutual Information (NMI)',
        'Tanzania Cohort (N=8817)': '0.0519',
        'UCI Benchmark (N=1014)': '0.2184',
        'Academic Benchmark Target': '> 0.0 (Information shared with ground truth)'
    }
]

df_perf = pd.DataFrame(perf_data)
df_perf.to_csv('reports/results/performance_matrix_consolidated.csv', index=False)
print('Consolidated performance matrix generated at reports/results/performance_matrix_consolidated.csv')

