"""
Maternal Risk Stratification in Low-Resource Settings
Module: src/preprocessing.py
Description: Clinical data cleaning, outlier auditing, imputation, and feature standardization.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler


class ClinicalPreprocessor:
    """
    Standardizes antenatal vital signs, obstetric history, and risk factors
    for unsupervised K-Means clustering, preventing data leakage and outlier distortion.
    """

    # Clinically established physiological bounds (WHO / FIGO / ACOG guidelines)
    PHYSIOLOGICAL_LIMITS = {
        "age": (12.0, 55.0),
        "systolicbp": (60.0, 240.0),
        "systolic_bp": (60.0, 240.0),
        "diastolicbp": (30.0, 150.0),
        "diastolic_bp": (30.0, 150.0),
        "pulserate": (40.0, 180.0),
        "pulse_rate": (40.0, 180.0),
        "heartrate": (40.0, 180.0),
        "heart_rate": (40.0, 180.0),
        "bodytemp": (95.0, 106.0),    # Fahrenheit
        "body_temp": (95.0, 106.0),
        "weight": (30.0, 200.0),      # Patient weight in kg (flags errors like 5618 kg)
        "hemoglobin": (4.0, 20.0),    # Hemoglobin in g/dL (flags errors like 137 g/dL)
        "blood_sugar": (2.0, 30.0),   # Blood glucose in mmol/L
        "bloodsugar": (2.0, 30.0),
        "bs": (2.0, 30.0),
        "gestationalage": (4.0, 43.0),
        "gestational_age": (4.0, 43.0),
        "gravidity": (1.0, 16.0),
        "parity": (0.0, 16.0),
        "bmi": (12.0, 60.0)
    }

    OUTCOME_CANDIDATES = [
        "risk", "risklevel", "risk_level", "adverse_outcome",
        "maternal_outcome", "outcome", "target"
    ]

    def __init__(self, scaler_type: str = "standard", clip_outliers: bool = True):
        """
        scaler_type: 'standard' (Z-score) or 'robust' (median & IQR)
        clip_outliers: whether to cap values outside physiological bounds
        """
        self.scaler_type = scaler_type
        self.clip_outliers = clip_outliers
        self.scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        self.feature_names_: List[str] = []
        self.target_name_: Optional[str] = None
        self.imputed_medians_: Dict[str, float] = {}

    def separate_features_and_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Strictly isolates adverse-outcome and risk-level labels from clustering features
        to guarantee zero data leakage during unsupervised segmentation.
        """
        target_col = None
        lower_cols = {col: col.lower().replace(" ", "").replace("_", "") for col in df.columns}

        for orig_col, norm_col in lower_cols.items():
            if norm_col in [cand.replace("_", "") for cand in self.OUTCOME_CANDIDATES]:
                target_col = orig_col
                break

        if target_col is not None:
            self.target_name_ = target_col
            features_df = df.drop(columns=[target_col]).copy()
            target_series = df[target_col].copy()
            print(f"[Preprocessor] Isolated holdout target: '{target_col}' ({target_series.nunique()} classes)")
        else:
            self.target_name_ = None
            features_df = df.copy()
            target_series = None
            print("[Preprocessor] No target column identified; using full dataset as features.")

        return features_df, target_series

    def clean_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing data via median imputation for numerical features
        and mode for categorical variables. Clips implausible typographical extremes.
        """
        cleaned_df = df.copy()

        # Handle numeric features
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            norm_col = col.lower().replace(" ", "").replace("_", "")

            # Physiological plausibility bounding
            if self.clip_outliers:
                for bound_col, (lower, upper) in self.PHYSIOLOGICAL_LIMITS.items():
                    if bound_col.replace("_", "") == norm_col:
                        implausible_mask = (cleaned_df[col] < lower) | (cleaned_df[col] > upper)
                        n_implausible = implausible_mask.sum()
                        if n_implausible > 0:
                            print(f"[Preprocessor] Flagged {n_implausible} implausible entries in '{col}' (<{lower} or >{upper})")
                            cleaned_df.loc[implausible_mask, col] = np.nan
                        break

            # Median imputation (robust against physiological skewness)
            median_val = cleaned_df[col].median()
            self.imputed_medians_[col] = median_val
            cleaned_df[col] = cleaned_df[col].fillna(median_val)

        # Handle categorical features
        cat_cols = cleaned_df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].empty else "Unknown"
            cleaned_df[col] = cleaned_df[col].fillna(mode_val)

        return cleaned_df

    def encode_and_scale(
        self, df: pd.DataFrame, is_training: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        One-hot encodes categorical variables (if present) and scales numerical
        features to prevent Euclidean distance distortion.
        Returns the scaled NumPy matrix X and the processed DataFrame.
        """
        encoded_df = pd.get_dummies(df, drop_first=True, dtype=float)

        if is_training:
            self.feature_names_ = list(encoded_df.columns)
            scaled_matrix = self.scaler.fit_transform(encoded_df)
        else:
            encoded_df = encoded_df.reindex(columns=self.feature_names_, fill_value=0.0)
            scaled_matrix = self.scaler.transform(encoded_df)

        scaled_df = pd.DataFrame(
            scaled_matrix, columns=self.feature_names_, index=df.index
        )

        return scaled_matrix, scaled_df

    def fit_transform_pipeline(
        self, raw_df: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.DataFrame, Optional[pd.Series]]:
        """
        Complete end-to-end preprocessing pipeline execution:
        1. Separate holdout targets.
        2. Clean and impute missing/implausible records.
        3. Encode and scale feature matrix.
        """
        features_raw, target_series = self.separate_features_and_target(raw_df)
        cleaned_df = self.clean_and_impute(features_raw)
        scaled_matrix, scaled_df = self.encode_and_scale(cleaned_df, is_training=True)

        print(f"[Preprocessor] Pipeline completed: Matrix shape = {scaled_matrix.shape}")
        return scaled_matrix, scaled_df, target_series
