"""
Data loading utilities for healthcare datasets.
Supports eICU, MIMIC, and synthetic data (Synthea).
"""
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from .cohort import CohortConfig, extract_eicu_cohort, extract_mimic_cohort, create_cohort, split_by_hospital, simulate_non_iid_partition


class DataLoader:
    """Load and prepare healthcare datasets for federated learning."""
    
    def __init__(self, config: CohortConfig, data_dir: str = 'data/raw'):
        self.config = config
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_eicu(self, conn) -> Tuple[pd.DataFrame, pd.Series]:
        """Load eICU cohort from database connection."""
        df = extract_eicu_cohort(conn, self.config)
        return create_cohort(df, self.config)
    
    def load_mimic(self, conn) -> Tuple[pd.DataFrame, pd.Series]:
        """Load MIMIC cohort from database connection."""
        df = extract_mimic_cohort(conn, self.config)
        return create_cohort(df, self.config)
    
    def load_synthetic(self, n_samples: int = 10000, n_hospitals: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic EHR data for testing."""
        np.random.seed(42)
        
        # Generate patient demographics
        data = {
            'patient_id': np.arange(n_samples),
            'age': np.random.normal(65, 15, n_samples).clip(18, 100).astype(int),
            'gender': np.random.choice(['M', 'F'], n_samples, p=[0.55, 0.45]),
            'ethnicity': np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], n_samples),
            'hospital_id': np.random.choice(range(n_hospitals), n_samples),
            'admission_type': np.random.choice(['Emergency', 'Elective', 'Urgent'], n_samples, p=[0.6, 0.2, 0.2]),
            'icu_type': np.random.choice(['MICU', 'SICU', 'CCU', 'CSRU'], n_samples),
        }
        
        # Generate comorbidities
        comorbidities = [
            'chf', 'afib', 'ckd', 'copd', 'diabetes', 'hypertension',
            'liver_disease', 'cancer', 'stroke', 'dementia'
        ]
        for com in comorbidities:
            data[com] = np.random.binomial(1, 0.15, n_samples)
        
        # Generate first 24h vitals (mean values)
        vital_means = {
            'heart_rate': 85,
            'systolic_bp': 120,
            'diastolic_bp': 70,
            'mean_bp': 85,
            'respiratory_rate': 18,
            'temperature': 37.0,
            'spo2': 97,
        }
        for vital, mean in vital_means.items():
            data[vital] = np.random.normal(mean, mean * 0.15, n_samples)
        
        # Generate first 24h labs
        lab_means = {
            'glucose': 120,
            'potassium': 4.0,
            'sodium': 138,
            'creatinine': 1.2,
            'bun': 20,
            'wbc': 8,
            'hemoglobin': 11,
            'platelets': 250,
            'lactate': 2.0,
            'ph': 7.4,
            'pco2': 40,
            'po2': 90,
            'bicarbonate': 24,
        }
        for lab, mean in lab_means.items():
            data[lab] = np.random.normal(mean, mean * 0.2, n_samples)
        
        # Generate severity scores
        data['sofa_score'] = np.random.poisson(4, n_samples)
        data['apache_score'] = np.random.normal(20, 8, n_samples).clip(0, 60)
        
        # Generate target (mortality ~10%)
        # Higher risk for older, sicker patients
        risk_score = (
            (data['age'] - 18) / 82 * 0.3 +
            data['sofa_score'] / 20 * 0.4 +
            data['apache_score'] / 60 * 0.3
        )
        mortality_prob = 1 / (1 + np.exp(-5 * (risk_score - 0.5)))
        data['mortality'] = np.random.binomial(1, mortality_prob, n_samples)
        
        df = pd.DataFrame(data)
        y = df['mortality']
        X = df.drop(columns=['mortality', 'patient_id'])
        
        return X, y
    
    def prepare_federated_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        partition_method: str = 'hospital',
        **kwargs
    ) -> Dict[str, Dict]:
        """
        Prepare federated data splits.
        
        Args:
            X: Feature matrix
            y: Target vector
            partition_method: 'hospital', 'dirichlet', 'label_skew', 'feature_skew', 'quantity_skew'
            **kwargs: Additional arguments for partition method
        
        Returns:
            Dict with client splits and global test set
        """
        if partition_method == 'hospital':
            if 'hospital_id' not in X.columns and 'hospitalid' not in X.columns:
                raise ValueError("Hospital ID column not found for hospital-based partitioning")
            hospital_col = 'hospital_id' if 'hospital_id' in X.columns else 'hospitalid'
            return split_by_hospital(X, y, hospital_id_col=hospital_col)
        else:
            return simulate_non_iid_partition(X, y, partition_type=partition_method, **kwargs)
    
    def save_processed_data(
        self,
        splits: Dict,
        output_dir: str = 'data/processed',
    ) -> None:
        """Save processed data splits to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, split in splits.items():
            split_dir = output_path / name
            split_dir.mkdir(exist_ok=True)
            
            for key, df in split.items():
                if isinstance(df, pd.DataFrame):
                    df.to_parquet(split_dir / f'{key}.parquet', index=False)
                elif isinstance(df, pd.Series):
                    df.to_frame().to_parquet(split_dir / f'{key}.parquet', index=False)
                else:
                    pd.DataFrame(df).to_parquet(split_dir / f'{key}.parquet', index=False)
    
    def load_processed_data(
        self,
        input_dir: str = 'data/processed',
    ) -> Dict[str, Dict]:
        """Load processed data splits from disk."""
        input_path = Path(input_dir)
        splits = {}
        
        for split_dir in input_path.iterdir():
            if split_dir.is_dir():
                split = {}
                for file in split_dir.glob('*.parquet'):
                    key = file.stem
                    split[key] = pd.read_parquet(file)
                splits[split_dir.name] = split
        
        return splits


def load_demo_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load a small demo dataset for quick testing."""
    loader = DataLoader(CohortConfig())
    return loader.load_synthetic(n_samples=1000, n_hospitals=3)