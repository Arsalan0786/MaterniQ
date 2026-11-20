"""
Maternal Risk Stratification in Low-Resource Settings
Module: src/visualisation.py
Description: Interactive, publication-quality Plotly and Matplotlib visualisations
tailored for responsive dashboards and academic reports.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Visualizer:
    """
    Creates responsive, publication-grade interactive visualisations.
    Configured for medical clarity and multi-device responsiveness.
    """

    COLOR_PALETTE = ["#2A9D8F", "#E76F51", "#264653", "#F4A261", "#E63946", "#457B9D"]

    @staticmethod
    def plot_feature_distributions(df: pd.DataFrame) -> go.Figure:
        """Generates distribution histograms for all numeric clinical features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
        n_cols = 3
        n_rows = int(np.ceil(len(numeric_cols) / n_cols))

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[col for col in numeric_cols]
        )

        for idx, col in enumerate(numeric_cols):
            r = (idx // n_cols) + 1
            c = (idx % n_cols) + 1
            fig.add_trace(
                go.Histogram(
                    x=df[col],
                    name=col,
                    marker=dict(color="#00A389", line=dict(color="#ffffff", width=1)),
                    opacity=0.85
                ),
                row=r, col=c
            )

        fig.update_layout(
            title_text="Clinical Feature Distributions",
            height=300 * n_rows,
            showlegend=False,
            template="plotly_white",
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
        """Generates an annotated correlation heatmap for numeric antenatal variables."""
        corr = df.select_dtypes(include=[np.number]).corr().round(2)
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="Viridis",
                zmin=-1, zmax=1,
                text=corr.values,
                texttemplate="%{text}",
                textfont={"size": 12}
            )
        )
        fig.update_layout(
            title="Feature Collinearity Heatmap (Pearson r)",
            template="plotly_white",
            height=500,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig

    @staticmethod
    def plot_elbow_and_silhouette(metrics_df: pd.DataFrame) -> go.Figure:
        """
        Plots the Elbow Curve (WCSS) and Silhouette Score side-by-side
        to enable data-driven selection of K.
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Elbow Method: Within-Cluster Sum of Squares (Inertia)",
                "Silhouette Coefficient (Cluster Cohesion & Separation)"
            ]
        )

        # Elbow trace
        fig.add_trace(
            go.Scatter(
                x=metrics_df["k"],
                y=metrics_df["inertia"],
                mode="lines+markers",
                name="Inertia",
                line=dict(color="#E76F51", width=3),
                marker=dict(size=9, color="#E76F51")
            ),
            row=1, col=1
        )

        # Silhouette trace
        fig.add_trace(
            go.Scatter(
                x=metrics_df["k"],
                y=metrics_df["silhouette_score"],
                mode="lines+markers",
                name="Silhouette Score",
                line=dict(color="#2A9D8F", width=3),
                marker=dict(size=9, color="#2A9D8F")
            ),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Cluster Count (K)", row=1, col=1)
        fig.update_xaxes(title_text="Cluster Count (K)", row=1, col=2)
        fig.update_yaxes(title_text="Inertia (WCSS)", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)

        fig.update_layout(
            title_text="Cluster Count Optimization Metrics",
            template="plotly_white",
            height=420,
            showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig

    @staticmethod
    def plot_pca_2d(
        pca_coords: np.ndarray,
        labels: np.ndarray,
        tier_mapping: Dict[str, str],
        features_df: pd.DataFrame
    ) -> go.Figure:
        """
        Visualizes the 2D Principal Component projection of patients
        color-coded by empirical risk tier with rich hover tooltips.
        """
        cluster_names = [f"Cluster {l}" for l in labels]
        risk_tiers = [tier_mapping.get(c, c) for c in cluster_names]

        plot_df = pd.DataFrame({
            "PC1": pca_coords[:, 0],
            "PC2": pca_coords[:, 1],
            "Cluster": cluster_names,
            "Risk_Tier": risk_tiers
        })

        # Add key features for tooltip
        for col in features_df.columns[:4]:
            plot_df[col] = features_df[col].values

        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color="Risk_Tier",
            hover_data=list(features_df.columns[:4]),
            title="PCA 2D Cluster Projection & Patient Risk Stratification",
            color_discrete_sequence=["#2A9D8F", "#F4A261", "#E76F51", "#264653"],
            opacity=0.75
        )
        fig.update_traces(marker=dict(size=7))
        fig.update_layout(
            template="plotly_white",
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        return fig

    @staticmethod
    def plot_adverse_incidence(
        incidence_df: pd.DataFrame,
        tier_mapping: Dict[str, str]
    ) -> go.Figure:
        """
        Bar chart showing adverse outcome prevalence by cluster with 95% confidence intervals.
        """
        df = incidence_df.copy()
        df["Risk_Tier"] = df["Cluster"].map(lambda c: tier_mapping.get(c, c))
        df["Error_Minus"] = df["Incidence_Rate (%)"] - df["95%_CI_Lower (%)"]
        df["Error_Plus"] = df["95%_CI_Upper (%)"] - df["Incidence_Rate (%)"]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=df["Risk_Tier"],
                    y=df["Incidence_Rate (%)"],
                    text=df["Incidence_Rate (%)"].map(lambda v: f"{v:.1f}%"),
                    textposition="auto",
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=df["Error_Plus"],
                        arrayminus=df["Error_Minus"]
                    ),
                    marker=dict(
                        color=["#2A9D8F", "#F4A261", "#E76F51"][:len(df)],
                        line=dict(color="#333333", width=1)
                    )
                )
            ]
        )
        fig.update_layout(
            title="Adverse Outcome Incidence by Cluster (with 95% Wilson CI)",
            yaxis_title="Observed Adverse Event Rate (%)",
            xaxis_title="Stratified Risk Profile",
            template="plotly_white",
            height=420,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig

    @staticmethod
    def plot_cluster_centroids_radar(
        centroids: np.ndarray,
        feature_names: List[str],
        tier_mapping: Dict[str, str]
    ) -> go.Figure:
        """
        Radar chart comparing standardized physiological centroids across clusters.
        """
        fig = go.Figure()
        colors = ["#2A9D8F", "#F4A261", "#E76F51", "#264653"]

        for idx, center in enumerate(centroids):
            cluster_name = f"Cluster {idx}"
            tier_name = tier_mapping.get(cluster_name, cluster_name)

            fig.add_trace(
                go.Scatterpolar(
                    r=list(center) + [center[0]],
                    theta=list(feature_names) + [feature_names[0]],
                    fill="toself",
                    name=f"{tier_name} ({cluster_name})",
                    line=dict(color=colors[idx % len(colors)]),
                    opacity=0.65
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-2, 3])),
            title="Standardized Physiological Centroid Profiles (Z-Scores)",
            template="plotly_white",
            height=480,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        return fig

