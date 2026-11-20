"""
Maternal Risk Stratification in Low-Resource Settings
Orchestration Script: main.py
Description: End-to-end command-line execution of the complete ML pipeline,
saving all academic figures and tabular reports.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import DataLoader
from src.preprocessing import ClinicalPreprocessor
from src.clustering import AntenatalKMeans
from src.evaluation import ClinicalValidator


def run_pipeline(dataset_name: str = "tanzania", k: int = 3, random_state: int = 42):
    print("\n" + "=" * 70)
    print("  K-MEANS MATERNAL RISK-TIER STRATIFICATION PIPELINE (HEADLESS)")
    print("=" * 70)

    # 1. Directories
    reports_fig_dir = "reports/figures"
    reports_res_dir = "reports/results"
    os.makedirs(reports_fig_dir, exist_ok=True)
    os.makedirs(reports_res_dir, exist_ok=True)

    # 2. Data Loading
    loader = DataLoader()
    if dataset_name == "tanzania":
        csv_path = os.path.join(loader.processed_data_dir, "tanzania_mhrs_standardized.csv")
        if not os.path.exists(csv_path):
            raw_df = loader.parse_tanzania_mhrs()
        else:
            raw_df = loader.load_dataset(csv_path)
    else:
        csv_path = os.path.join(loader.processed_data_dir, "uci_benchmark_standardized.csv")
        if not os.path.exists(csv_path):
            raw_df = loader.parse_uci_benchmark()
        else:
            raw_df = loader.load_dataset(csv_path)

    print(f"Loaded cohort: {len(raw_df):,} patient records")

    # 3. Preprocessing
    preprocessor = ClinicalPreprocessor(scaler_type="standard", clip_outliers=True)
    X, scaled_df, target_series = preprocessor.fit_transform_pipeline(raw_df)
    features_only, _ = preprocessor.separate_features_and_target(raw_df)

    # 4. Cluster Optimization (K = 2 to 6)
    print("\n[Phase 6] Optimizing cluster count K (Elbow, Silhouette, DBI, CHI)...")
    opt_df = AntenatalKMeans.evaluate_k_range(X, k_range=range(2, 7), random_state=random_state)
    opt_csv_path = os.path.join(reports_res_dir, f"cluster_optimization_{dataset_name}.csv")
    opt_df.to_csv(opt_csv_path, index=False)
    print(f"Saved cluster count optimization metrics to: {opt_csv_path}")
    print(opt_df.to_string(index=False))

    # Matplotlib Optimization plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(opt_df["k"], opt_df["inertia"], marker="o", color="#E76F51", lw=2)
    axes[0].set_title("Elbow Curve (Within-Cluster Sum of Squares)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(opt_df["k"], opt_df["silhouette_score"], marker="s", color="#2A9D8F", lw=2)
    axes[1].set_title("Silhouette Coefficient vs K", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    opt_plot_path = os.path.join(reports_fig_dir, f"elbow_silhouette_{dataset_name}.png")
    plt.savefig(opt_plot_path, dpi=300)
    plt.close()
    print(f"Saved Elbow/Silhouette plot to: {opt_plot_path}")

    # 5. Fit Final Model (K)
    print(f"\n[Phase 5 & 7] Fitting K-Means with K = {k}...")
    kmeans = AntenatalKMeans(n_clusters=k, random_state=random_state)
    labels = kmeans.fit_predict(X)

    # Cluster stability
    stability = AntenatalKMeans.evaluate_cluster_stability(X, k=k)
    print(f"Cluster stability across 5 random seeds (Mean ARI): {stability['mean_ari']:.4f} (std: {stability['std_ari']:.4f})")

    # 6. Clinical Validation (Objective 3)
    print("\n[Phase 8] Validating clusters against recorded adverse outcomes...")
    validator = ClinicalValidator()
    inc_df = validator.evaluate_adverse_incidence(labels, target_series)
    tier_mapping = validator.assign_risk_tiers(inc_df)
    stats_dict = validator.perform_statistical_tests(labels, target_series)
    rr_df = validator.compute_relative_risk_and_odds_ratios(inc_df)

    inc_df["Stratified_Risk_Tier"] = inc_df["Cluster"].map(lambda c: tier_mapping.get(c, c))
    inc_csv_path = os.path.join(reports_res_dir, f"adverse_outcome_validation_{dataset_name}.csv")
    inc_df.to_csv(inc_csv_path, index=False)

    print("\nAdverse Outcome Incidence per Discovered Cluster:")
    print(inc_df[["Cluster", "Stratified_Risk_Tier", "Total_Patients", "Incidence_Rate (%)", "95%_CI_Lower (%)", "95%_CI_Upper (%)"]].to_string(index=False))

    print(f"\nChi-Square Test of Independence:")
    print(f"  chi2 = {stats_dict['chi2_statistic']:.3f}, p-value = {stats_dict['p_value']:.4e}, df = {stats_dict['degrees_of_freedom']}, Cramér's V = {stats_dict['cramers_v']:.4f}")

    print("\nRelative Risk (RR) and Odds Ratios (OR) vs Baseline:")
    print(rr_df.to_string(index=False))

    # Matplotlib Outcome Bar Plot
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sorted_inc = inc_df.sort_values("Incidence_Rate (%)")
    bars = ax.bar(
        sorted_inc["Stratified_Risk_Tier"],
        sorted_inc["Incidence_Rate (%)"],
        color=["#2A9D8F", "#F4A261", "#E76F51"][:k],
        edgecolor="#333333"
    )
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    ax.set_title("Recorded Adverse Outcome Incidence by Stratified Risk Profile", fontsize=11, fontweight="bold")
    ax.set_ylabel("Adverse Outcome Incidence (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.tight_layout()
    bar_plot_path = os.path.join(reports_fig_dir, f"adverse_outcome_incidence_{dataset_name}.png")
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()
    print(f"Saved adverse outcome incidence chart to: {bar_plot_path}")

    # 7. Cluster Profiles
    profiles = kmeans.compute_cluster_profiles(features_only, labels)
    profiles_csv_path = os.path.join(reports_res_dir, f"cluster_profiles_{dataset_name}.csv")
    profiles.to_csv(profiles_csv_path)
    print(f"\nSaved cluster physiological profiles to: {profiles_csv_path}")

    # 8. PCA Projection Plot
    pca_coords, explained_var = kmeans.compute_pca_projection(X, n_components=2)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    scatter = ax.scatter(
        pca_coords[:, 0], pca_coords[:, 1],
        c=labels, cmap="viridis", alpha=0.6, s=20, edgecolors="none"
    )
    ax.set_title(f"PCA 2D Cluster Space (Explained Variance: {sum(explained_var)*100:.1f}%)", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    pca_plot_path = os.path.join(reports_fig_dir, f"pca_clusters_{dataset_name}.png")
    plt.savefig(pca_plot_path, dpi=300)
    plt.close()
    print(f"Saved 2D PCA cluster scatter plot to: {pca_plot_path}")

    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION COMPLETE: ALL ARTIFACTS AND REPORTS GENERATED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maternal Risk Stratification Pipeline")
    parser.add_argument("--dataset", type=str, default="tanzania", choices=["tanzania", "uci"], help="Dataset to process")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    args = parser.parse_args()

    run_pipeline(dataset_name=args.dataset, k=args.k, random_state=args.seed)

