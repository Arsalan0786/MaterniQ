"""
Maternal Risk Stratification in Low-Resource Settings
Module: src/data_loader.py
Description: Data loading, candidate schema verification, metadata auditing,
and dedicated parsers for Tanzania MHRS and benchmark maternal datasets.
"""

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class DataLoader:
    """
    Robust data loader and schema auditor for antenatal health records.
    Designed for clinical reproducibility and data integrity verification.
    """

    EXPECTED_DOMAINS = {
        "demographic": ["age", "gestational_age", "district", "residence"],
        "vitals": [
            "systolic_bp", "diastolic_bp", "heart_rate", "pulse_rate",
            "body_temp", "bmi", "weight"
        ],
        "obstetric": ["parity", "gravidity", "previous_complications", "no_pregnancy"],
        "clinical_tests": ["blood_glucose", "blood_sugar", "proteinuria", "hemoglobin"],
        "outcome": ["risk_level", "risk", "adverse_outcome", "outcome_label"]
    }

    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Loads dataset from CSV or Excel file with strict file-existence checking.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at: {file_path}")

        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, low_memory=False)
        elif file_path.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        print(f"[DataLoader] Successfully loaded {len(df)} records from {file_path}")
        return df

    def parse_tanzania_mhrs(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Extracts and parses standardized clinical antenatal features from the
        Tanzania MHRS dataset (Zenodo DOI: 10.5281/zenodo.15309733, N=8,817).
        """
        if file_path is None:
            file_path = os.path.join(self.raw_data_dir, "tanzania_maternal_mhrs.csv")

        raw_df = self.load_dataset(file_path)

        def parse_bp(val):
            if pd.isna(val) or not isinstance(val, str) or "/" not in val:
                return np.nan, np.nan
            try:
                parts = val.split("/")
                s = float(parts[0].strip())
                d = float(parts[1].strip())
                if 50 <= s <= 260 and 30 <= d <= 160:
                    return s, d
            except Exception:
                pass
            return np.nan, np.nan

        parsed_df = pd.DataFrame()
        sbp, dbp = zip(*raw_df["blood_pressure_v1"].apply(parse_bp))
        parsed_df["Age"] = pd.to_numeric(raw_df["age"], errors="coerce")
        parsed_df["SystolicBP"] = sbp
        parsed_df["DiastolicBP"] = dbp
        parsed_df["PulseRate"] = pd.to_numeric(raw_df["pulse_rate_v1"], errors="coerce")
        parsed_df["Weight"] = pd.to_numeric(raw_df["weight_kg_v1"], errors="coerce")
        parsed_df["Hemoglobin"] = pd.to_numeric(raw_df["hemoglobin_check_result_v1"], errors="coerce")
        parsed_df["GestationalAge"] = pd.to_numeric(raw_df["duration_of_pregnancy_weeks_"], errors="coerce")
        parsed_df["Gravidity"] = pd.to_numeric(raw_df["no_pregnancy"], errors="coerce")

        # Map Risk outcome
        if "Risk" in raw_df.columns:
            parsed_df["RiskLevel"] = raw_df["Risk"].astype(str).str.lower().str.strip()
        else:
            parsed_df["RiskLevel"] = "unknown"

        output_path = os.path.join(self.processed_data_dir, "tanzania_mhrs_standardized.csv")
        parsed_df.to_csv(output_path, index=False)
        print(f"[DataLoader] Processed Tanzania MHRS dataset saved to {output_path} ({parsed_df.shape})")
        return parsed_df

    def parse_uci_benchmark(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Loads and standardizes the UCI Maternal Health Risk dataset (Ahmed et al., N=1,014).
        """
        if file_path is None:
            file_path = os.path.join(self.raw_data_dir, "maternal_health_risk.csv")

        raw_df = self.load_dataset(file_path)
        col_rename = {
            "Age": "Age",
            "SystolicBP": "SystolicBP",
            "DiastolicBP": "DiastolicBP",
            "BS": "BloodSugar",
            "BodyTemp": "BodyTemp",
            "HeartRate": "PulseRate",
            "Risk": "RiskLevel",
            "RiskLevel": "RiskLevel"
        }
        df = raw_df.rename(columns=col_rename)
        df["RiskLevel"] = df["RiskLevel"].astype(str).str.lower().str.strip()

        output_path = os.path.join(self.processed_data_dir, "uci_benchmark_standardized.csv")
        df.to_csv(output_path, index=False)
        print(f"[DataLoader] Processed UCI benchmark dataset saved to {output_path} ({df.shape})")
        return df

    def audit_dataset_schema(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Audits dataset dimensions, column data types, missing value percentages,
        and identifies candidate features corresponding to clinical domains.
        """
        audit_report = {
            "n_records": int(df.shape[0]),
            "n_features": int(df.shape[1]),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_counts": df.isnull().sum().to_dict(),
            "missing_percentages": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "duplicate_records": int(df.duplicated().sum()),
            "domain_mapping": self._map_columns_to_domains(df.columns)
        }
        return audit_report

    def _map_columns_to_domains(self, columns: List[str]) -> Dict[str, List[str]]:
        """
        Maps existing dataset columns to standardized clinical domains using fuzzy matching.
        """
        mapping = {domain: [] for domain in self.EXPECTED_DOMAINS}
        normalized_cols = {col: col.lower().replace(" ", "_").replace("-", "_") for col in columns}

        for orig_col, norm_col in normalized_cols.items():
            assigned = False
            for domain, keywords in self.EXPECTED_DOMAINS.items():
                for kw in keywords:
                    if kw in norm_col or norm_col in kw:
                        mapping[domain].append(orig_col)
                        assigned = True
                        break
                if assigned:
                    break
            if not assigned:
                mapping.setdefault("unclassified", []).append(orig_col)

        return mapping
