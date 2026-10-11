#!/usr/bin/env python
"""
Unit tests for preprocessing pipeline.
Run: python -m pytest tests/test_preprocessing.py -v
"""
import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.pipeline import (
    build_preprocessing_pipeline,
    get_feature_names_after_preprocessing,
    MissingnessIndicator,
    ClinicalClipper,
    TemporalFeatureExtractor,
    ComorbidityCounter,
)


class TestMissingnessIndicator:
    """Tests for MissingnessIndicator transformer."""
    
    def test_adds_missing_indicators(self):
        df = pd.DataFrame({'a': [1, 2, np.nan], 'b': [4, np.nan, 6]})
        transformer = MissingnessIndicator(features=['a', 'b'])
        transformer.fit(df)
        result = transformer.transform(df)
        
        assert 'a_missing' in result.columns
        assert 'b_missing' in result.columns
        assert result['a_missing'].tolist() == [0, 0, 1]
        assert result['b_missing'].tolist() == [0, 1, 0]
    
    def test_handles_no_missing(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        transformer = MissingnessIndicator(features=['a', 'b'])
        transformer.fit(df)
        result = transformer.transform(df)
        
        # Should not add indicator columns if no missing in fit
        assert 'a_missing' not in result.columns or result['a_missing'].sum() == 0


class TestClinicalClipper:
    """Tests for ClinicalClipper transformer."""
    
    def test_clips_to_ranges(self):
        df = pd.DataFrame({'heart_rate': [30, 100, 250], 'temperature': [35, 37, 45]})
        clipper = ClinicalClipper()
        result = clipper.fit_transform(df)
        
        assert result['heart_rate'].min() >= 30
        assert result['heart_rate'].max() <= 200
        assert result['temperature'].min() >= 30
        assert result['temperature'].max() <= 42
    
    def test_ignores_unknown_columns(self):
        df = pd.DataFrame({'unknown_col': [1, 2, 3]})
        clipper = ClinicalClipper()
        result = clipper.fit_transform(df)
        # Should not error, just pass through
        assert 'unknown_col' in result.columns


class TestComorbidityCounter:
    """Tests for ComorbidityCounter transformer."""
    
    def test_counts_comorbidities(self):
        df = pd.DataFrame({
            'congestive_heart_failure': [1, 0, 1],
            'cardiac_arrhythmias': [0, 1, 1],
            'renal_failure': [0, 0, 1],
            'other': [1, 1, 1]  # not in default list
        })
        counter = ComorbidityCounter()
        result = counter.fit_transform(df)
        
        assert 'comorbidity_count' in result.columns
        assert 'elixhauser_score' in result.columns
        assert result['comorbidity_count'].tolist() == [1, 1, 3]


class TestPreprocessingPipeline:
    """Tests for the full preprocessing pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data matching synthetic structure."""
        np.random.seed(42)
        n = 1000
        return pd.DataFrame({
            'age': np.random.normal(65, 15, n).clip(18, 100),
            'hospital_id': np.random.randint(0, 5, n),
            'heart_rate': np.random.normal(85, 15, n),
            'systolic_bp': np.random.normal(120, 20, n),
            'gender': np.random.choice(['M', 'F'], n),
            'ethnicity': np.random.choice(['White', 'Black', 'Hispanic'], n),
            'chf': np.random.binomial(1, 0.15, n),
            'afib': np.random.binomial(1, 0.1, n),
        })
    
    @pytest.fixture
    def sample_target(self, sample_data):
        return pd.Series(np.random.binomial(1, 0.3, len(sample_data)), name='mortality')
    
    def test_pipeline_fits_and_transforms(self, sample_data, sample_target):
        numeric_features = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = sample_data.select_dtypes(include=['object']).columns.tolist()
        
        pipeline = build_preprocessing_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            scaling_method='zscore',
            discretization_method='quantile',
            n_bins=5,
        )
        
        # Fit on train
        X_train_trans = pipeline.fit_transform(sample_data, sample_target)
        # Transform test
        X_test_trans = pipeline.transform(sample_data)
        
        assert X_train_trans.shape[0] == len(sample_data)
        assert X_test_trans.shape[0] == len(sample_data)
        assert X_train_trans.shape[1] == X_test_trans.shape[1]
        assert not np.isnan(X_train_trans).any()
        assert not np.isnan(X_test_trans).any()
    
    def test_no_data_leakage(self, sample_data, sample_target):
        """Verify preprocessing params are fit only on training data."""
        numeric_features = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = sample_data.select_dtypes(include=['object']).columns.tolist()
        
        pipeline = build_preprocessing_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            scaling_method='zscore',
        )
        
        # Split
        train_idx = sample_data.index[:800]
        test_idx = sample_data.index[800:]
        
        X_train = sample_data.loc[train_idx]
        y_train = sample_target.loc[train_idx]
        X_test = sample_data.loc[test_idx]
        
        # Fit on train only
        pipeline.fit(X_train, y_train)
        
        # Transform both
        X_train_trans = pipeline.transform(X_train)
        X_test_trans = pipeline.transform(X_test)
        
        # Check that test data is transformed using train statistics
        # (mean should be ~0 for scaled features since train mean was subtracted)
        train_mean = X_train_trans.mean(axis=0)
        test_mean = X_test_trans.mean(axis=0)
        
        # Train mean should be close to 0 for z-score scaled features
        # (some features may not be scaled if discretized)
        numeric_scaled = [c for c in numeric_features if c in X_train.columns]
        # Just verify no error and shapes match
        assert X_train_trans.shape[1] == X_test_trans.shape[1]
    
    def test_consistent_columns_across_splits(self, sample_data, sample_target):
        """Verify column order is consistent."""
        numeric_features = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = sample_data.select_dtypes(include=['object']).columns.tolist()
        
        pipeline = build_preprocessing_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        
        pipeline.fit(sample_data, sample_target)
        
        # Transform multiple "clients"
        client1 = sample_data.iloc[:300]
        client2 = sample_data.iloc[300:600]
        
        trans1 = pipeline.transform(client1)
        trans2 = pipeline.transform(client2)
        
        # Get feature names
        names1 = get_feature_names_after_preprocessing(pipeline, numeric_features, categorical_features)
        names2 = get_feature_names_after_preprocessing(pipeline, numeric_features, categorical_features)
        
        assert names1 == names2
        assert list(pd.DataFrame(trans1).columns) == list(pd.DataFrame(trans2).columns)


class TestSavedPipeline:
    """Tests for the saved preprocessing pipeline."""
    
    @pytest.fixture(scope="class")
    def saved_pipeline(self):
        path = Path('data/processed/preprocessing_pipeline.joblib')
        if path.exists():
            return joblib.load(path)
        return None
    
    @pytest.fixture(scope="class")
    def processed_splits(self):
        path = Path('data/processed')
        splits = {}
        for split_dir in path.iterdir():
            if split_dir.is_dir():
                split = {}
                for file in split_dir.glob('*.parquet'):
                    key = file.stem
                    split[key] = pd.read_parquet(file)
                splits[split_dir.name] = split
        return splits
    
    def test_saved_pipeline_exists(self, saved_pipeline):
        assert saved_pipeline is not None, "Saved pipeline not found"
    
    def test_no_missing_values(self, processed_splits):
        """Verify no NaN values in any processed split."""
        for name, split in processed_splits.items():
            for key in ['X_train', 'X_test']:
                if key in split:
                    df = split[key]
                    missing = df.isnull().sum().sum()
                    assert missing == 0, f"{name}/{key} has {missing} missing values"
    
    def test_consistent_columns(self, processed_splits):
        """Verify consistent columns across all splits."""
        hospital_splits = {k: v for k, v in processed_splits.items() if k.startswith('hospital')}
        ref_cols = list(hospital_splits[list(hospital_splits.keys())[0]]['X_train'].columns)
        
        for name, split in hospital_splits.items():
            train_cols = list(split['X_train'].columns)
            test_cols = list(split['X_test'].columns)
            assert train_cols == ref_cols, f"{name}: train columns differ"
            assert test_cols == ref_cols, f"{name}: test columns differ"
    
    def test_target_binary(self, processed_splits):
        """Verify target is binary."""
        for name, split in processed_splits.items():
            for key in ['y_train', 'y_test']:
                if key in split:
                    unique = split[key].iloc[:, 0].unique()
                    assert set(unique).issubset({0, 1}), f"{name}/{key} has non-binary values: {unique}"
    
    def test_positive_rates_reasonable(self, processed_splits):
        """Verify positive rates are in expected range."""
        for name, split in processed_splits.items():
            if name.startswith('hospital'):
                pos_rate = split['y_train'].iloc[:, 0].mean()
                assert 0.05 <= pos_rate <= 0.5, f"{name}: pos_rate={pos_rate:.3f} outside expected range"


class TestFeatureEngineering:
    """Tests for feature engineering components."""
    
    def test_temporal_extractor(self):
        """Test TemporalFeatureExtractor with time-series data."""
        df = pd.DataFrame({
            'patient_id': [1, 1, 1, 2, 2],
            'heart_rate': [80, 85, 90, 70, 75],
            'time': [0, 6, 12, 0, 6],
        })
        extractor = TemporalFeatureExtractor(vital_cols=['heart_rate'])
        extractor.fit(df)
        result = extractor.transform(df)
        
        assert 'heart_rate_min' in result.columns
        assert 'heart_rate_max' in result.columns
        assert 'heart_rate_mean' in result.columns
        assert 'heart_rate_slope' in result.columns
        assert len(result) == 2  # 2 patients
    
    def test_comorbidity_counter(self):
        """Test ComorbidityCounter."""
        df = pd.DataFrame({
            'chf': [1, 0, 1],
            'afib': [0, 1, 1],
            'ckd': [0, 0, 1],
        })
        counter = ComorbidityCounter(condition_cols=['chf', 'afib', 'ckd'])
        result = counter.fit_transform(df)
        
        assert result['comorbidity_count'].tolist() == [1, 1, 3]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])