"""
Maternal Risk Stratification in Low-Resource Settings
Module: src/evaluation.py
Description: Clinical validation against adverse outcomes, contingency analysis,
Chi-square tests, Relative Risk, Odds Ratios, and Cramér's V effect size.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


class ClinicalValidator:
    """
    Validates unsupervised clusters against held-out adverse maternal and pregnancy outcomes.
    Fulfills Objective 3 with rigorous epidemiological and statistical tests.
    """

    def __init__(self, high_risk_tokens: Optional[List[str]] = None):
        if high_risk_tokens is None:
            self.high_risk_tokens = ["high", "high risk", "severe", "complication", "1", "yes"]
        else:
            self.high_risk_tokens = [t.lower() for t in high_risk_tokens]

    def cross_tabulate(
        self, labels: np.ndarray, target: pd.Series
    ) -> pd.DataFrame:
        """
        Builds a contingency matrix comparing cluster IDs to recorded outcome labels.
        """
        cluster_labels = [f"Cluster {l}" for l in labels]
        crosstab = pd.crosstab(
            pd.Series(cluster_labels, name="Cluster"),
            target.rename("Recorded_Outcome"),
            margins=True,
            margins_name="Total"
        )
        return crosstab

    def evaluate_adverse_incidence(
        self, labels: np.ndarray, target: pd.Series
    ) -> pd.DataFrame:
        """
        Calculates adverse outcome incidence (prevalence %) and 95% Wilson score confidence intervals
        for each discovered cluster.
        """
        cluster_names = [f"Cluster {l}" for l in labels]
        df = pd.DataFrame({"Cluster": cluster_names, "Outcome": target.values})

        # Binarize outcome to identify adverse/high-risk state
        is_adverse = df["Outcome"].astype(str).str.lower().str.strip().apply(
            lambda x: 1 if any(tok in x for tok in self.high_risk_tokens) else 0
        )
        df["Is_Adverse"] = is_adverse

        summary = []
        for cluster, group in df.groupby("Cluster"):
            n = len(group)
            events = int(group["Is_Adverse"].sum())
            p = events / n if n > 0 else 0.0

            # Wilson score 95% confidence interval for proportion
            z = 1.96
            denominator = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denominator
            spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
            ci_lower = max(0.0, center - spread)
            ci_upper = min(1.0, center + spread)

            summary.append({
                "Cluster": cluster,
                "Total_Patients": n,
                "Adverse_Events": events,
                "Incidence_Rate (%)": round(p * 100, 2),
                "95%_CI_Lower (%)": round(ci_lower * 100, 2),
                "95%_CI_Upper (%)": round(ci_upper * 100, 2)
            })

        summary_df = pd.DataFrame(summary).sort_values("Incidence_Rate (%)")
        return summary_df

    def perform_statistical_tests(
        self, labels: np.ndarray, target: pd.Series
    ) -> Dict[str, any]:
        """
        Executes Pearson's Chi-Square Test of Independence and computes Cramér's V.
        """
        cluster_names = [f"Cluster {l}" for l in labels]
        contingency = pd.crosstab(cluster_names, target)

        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

        # Cramér's V effect size calculation
        n = contingency.values.sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0

        return {
            "chi2_statistic": round(float(chi2), 3),
            "p_value": float(p_val),
            "degrees_of_freedom": int(dof),
            "cramers_v": round(float(cramers_v), 4),
            "statistically_significant": bool(p_val < 0.05),
            "contingency_matrix": contingency
        }

    def compute_relative_risk_and_odds_ratios(
        self, incidence_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates Relative Risk (RR) and Odds Ratio (OR) comparing each cluster
        against the cluster with the lowest observed incidence (baseline).
        """
        df = incidence_df.copy().reset_index(drop=True)
        baseline = df.iloc[0]  # Lowest incidence cluster
        baseline_cluster = baseline["Cluster"]

        a0 = baseline["Adverse_Events"]
        n0 = baseline["Total_Patients"]
        b0 = n0 - a0
        p0 = a0 / n0 if n0 > 0 else 0.0001

        risk_metrics = []
        for _, row in df.iterrows():
            cluster = row["Cluster"]
            a = row["Adverse_Events"]
            n = row["Total_Patients"]
            b = n - a
            p = a / n if n > 0 else 0.0001

            if cluster == baseline_cluster:
                risk_metrics.append({
                    "Cluster": cluster,
                    "Relative_Risk": 1.0,
                    "RR_95%_CI": "(Reference Baseline)",
                    "Odds_Ratio": 1.0,
                    "OR_95%_CI": "(Reference Baseline)"
                })
                continue

            # Relative Risk (RR)
            rr = p / p0 if p0 > 0 else np.nan
            se_ln_rr = np.sqrt((1/a if a>0 else 0) - (1/n) + (1/a0 if a0>0 else 0) - (1/n0))
            rr_low = np.exp(np.log(rr) - 1.96 * se_ln_rr) if not np.isnan(rr) else np.nan
            rr_high = np.exp(np.log(rr) + 1.96 * se_ln_rr) if not np.isnan(rr) else np.nan

            # Odds Ratio (OR)
            odds = (a / b) if b > 0 else np.nan
            odds0 = (a0 / b0) if b0 > 0 else np.nan
            or_val = odds / odds0 if odds0 and not np.isnan(odds0) and odds0 > 0 else np.nan
            se_ln_or = np.sqrt(1/a + 1/b + 1/a0 + 1/b0) if min(a, b, a0, b0) > 0 else 0.0
            or_low = np.exp(np.log(or_val) - 1.96 * se_ln_or) if not np.isnan(or_val) and se_ln_or > 0 else np.nan
            or_high = np.exp(np.log(or_val) + 1.96 * se_ln_or) if not np.isnan(or_val) and se_ln_or > 0 else np.nan

            risk_metrics.append({
                "Cluster": cluster,
                "Relative_Risk": round(rr, 2) if not np.isnan(rr) else "N/A",
                "RR_95%_CI": f"[{rr_low:.2f}, {rr_high:.2f}]" if not np.isnan(rr_low) else "N/A",
                "Odds_Ratio": round(or_val, 2) if not np.isnan(or_val) else "N/A",
                "OR_95%_CI": f"[{or_low:.2f}, {or_high:.2f}]" if not np.isnan(or_low) else "N/A"
            })

        return pd.DataFrame(risk_metrics)

    def assign_risk_tiers(self, incidence_df: pd.DataFrame) -> Dict[str, str]:
        """
        Maps cluster identifiers to clinically descriptive risk tiers based on empirical
        adverse-outcome prevalence (Lowest -> Intermediate -> Highest).
        """
        sorted_clusters = incidence_df.sort_values("Incidence_Rate (%)")["Cluster"].tolist()
        k = len(sorted_clusters)

        mapping = {}
        if k == 2:
            mapping[sorted_clusters[0]] = "Lower Observed Risk Profile"
            mapping[sorted_clusters[1]] = "Higher Observed Risk Profile"
        elif k == 3:
            mapping[sorted_clusters[0]] = "Lower Observed Risk Profile"
            mapping[sorted_clusters[1]] = "Intermediate Observed Risk Profile"
            mapping[sorted_clusters[2]] = "Higher Observed Risk Profile"
        else:
            for rank, cluster in enumerate(sorted_clusters):
                pct = rank / (k - 1)
                if pct <= 0.33:
                    mapping[cluster] = f"Lower Risk Tier (Rank {rank+1})"
                elif pct <= 0.67:
                    mapping[cluster] = f"Intermediate Risk Tier (Rank {rank+1})"
                else:
                    mapping[cluster] = f"Higher Risk Tier (Rank {rank+1})"

        return mapping

