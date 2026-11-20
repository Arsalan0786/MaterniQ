"""
Maternal Risk Stratification in Low-Resource Settings
Module: src/clustering.py
Description: Unsupervised K-Means clustering, cluster-count optimization,
stability auditing across seeds, and PCA dimensionality reduction.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)


class AntenatalKMeans:
    """
    Unsupervised K-Means engine for maternal risk-tier stratification.
    Implements cluster-count optimization, stability assessment, and dimensional projections.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        init: str = "k-means++",
        n_init: int = 20,
        max_iter: int = 300,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

        self.model: KMeans = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.pca_model_: Optional[PCA] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fits K-Means to the standardized feature matrix and returns assigned cluster labels.
        """
        self.labels_ = self.model.fit_predict(X)
        self.cluster_centers_ = self.model.cluster_centers_
        return self.labels_

    @staticmethod
    def evaluate_k_range(
        X: np.ndarray,
        k_range: range = range(2, 9),
        n_init: int = 20,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Calculates cluster evaluation metrics across candidate K values:
        - Inertia (Within-Cluster Sum of Squares)
        - Silhouette Score (Separation & Cohesion)
        - Davies-Bouldin Index (Lower is superior)
        - Calinski-Harabasz Index (Higher is superior)
        """
        metrics = []

        for k in k_range:
            km = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=n_init,
                max_iter=300,
                random_state=random_state
            )
            labels = km.fit_predict(X)

            inertia = float(km.inertia_)
            sil = float(silhouette_score(X, labels))
            dbi = float(davies_bouldin_score(X, labels))
            chi = float(calinski_harabasz_score(X, labels))

            metrics.append({
                "k": k,
                "inertia": round(inertia, 2),
                "silhouette_score": round(sil, 4),
                "davies_bouldin_index": round(dbi, 4),
                "calinski_harabasz_index": round(chi, 2)
            })

        return pd.DataFrame(metrics)

    @staticmethod
    def evaluate_cluster_stability(
        X: np.ndarray,
        k: int = 3,
        seeds: List[int] = [42, 10, 2024, 99, 123]
    ) -> Dict[str, float]:
        """
        Measures clustering stability across independent random initializations
        using the pairwise Adjusted Rand Index (ARI).
        ARI = 1.0 indicates deterministic cluster partition stability.
        """
        run_labels = []
        for seed in seeds:
            km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=seed)
            run_labels.append(km.fit_predict(X))

        pairwise_aris = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                ari = adjusted_rand_score(run_labels[i], run_labels[j])
                pairwise_aris.append(ari)

        return {
            "mean_ari": float(np.mean(pairwise_aris)),
            "std_ari": float(np.std(pairwise_aris)),
            "min_ari": float(np.min(pairwise_aris)),
            "max_ari": float(np.max(pairwise_aris)),
            "n_runs": len(seeds)
        }

    def compute_pca_projection(
        self, X: np.ndarray, n_components: int = 3
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Reduces multidimensional feature matrix to 2D and 3D principal components
        for geometric visualization.
        """
        self.pca_model_ = PCA(n_components=n_components, random_state=self.random_state)
        pca_coords = self.pca_model_.fit_transform(X)
        explained_variance = [
            float(v) for v in self.pca_model_.explained_variance_ratio_
        ]
        return pca_coords, explained_variance

    def compute_cluster_profiles(
        self, original_features_df: pd.DataFrame, labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Calculates cluster-level mean, median, and prevalence for all features in original clinical units.
        """
        df_copy = original_features_df.copy()
        df_copy["Cluster"] = [f"Cluster {l}" for l in labels]

        # Profile: Mean of each continuous feature
        profile_mean = df_copy.groupby("Cluster").mean(numeric_only=True)
        counts = df_copy["Cluster"].value_counts()
        profile_mean.insert(0, "PatientCount", counts)
        profile_mean.insert(1, "Percentage (%)", (counts / len(df_copy) * 100).round(1))

        return profile_mean.sort_index()

