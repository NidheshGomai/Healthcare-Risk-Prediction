"""
Preprocessing pipeline for healthcare risk prediction.
Implements imputation, scaling, discretization, and feature engineering.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold


class MissingnessIndicator(BaseEstimator, TransformerMixin):
    """Add binary indicators for missing values."""
    
    def __init__(self, features: Optional[List[str]] = None):
        self.features = features
        self.missing_features_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        if self.features is None:
            self.features = X.columns.tolist()
        self.missing_features_ = [f for f in self.features if X[f].isnull().any()]
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feat in self.missing_features_:
            X[f'{feat}_missing'] = X[feat].isnull().astype(int)
        return X
    
    def get_feature_names_out(self, input_features=None):
        if self.missing_features_ is None:
            return np.array([])
        return np.array([f'{f}_missing' for f in self.missing_features_])


class ClinicalClipper(BaseEstimator, TransformerMixin):
    """Clip values to physiologically plausible ranges."""
    
    CLINICAL_RANGES = {
        'age': (18, 100),
        'heart_rate': (30, 200),
        'systolic_bp': (50, 250),
        'diastolic_bp': (20, 150),
        'mean_bp': (30, 200),
        'respiratory_rate': (5, 50),
        'temperature': (30, 42),
        'spo2': (50, 100),
        'glucose': (20, 500),
        'potassium': (1.5, 8.0),
        'sodium': (110, 170),
        'creatinine': (0.1, 15.0),
        'bun': (1, 150),
        'wbc': (0.5, 100),
        'hemoglobin': (3, 20),
        'platelets': (10, 1000),
        'lactate': (0.1, 20),
        'ph': (6.8, 7.6),
        'pco2': (10, 100),
        'po2': (20, 600),
        'bicarbonate': (5, 40),
    }
    
    def __init__(self, ranges: Optional[Dict] = None, method: str = 'clip'):
        self.ranges = ranges or self.CLINICAL_RANGES
        self.method = method
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            if col in self.ranges:
                low, high = self.ranges[col]
                if self.method == 'clip':
                    X[col] = X[col].clip(low, high)
                elif self.method == 'iqr':
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    X[col] = X[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return X


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features (min, max, mean, slope) from time-series vitals."""
    
    def __init__(self, vital_cols: Optional[List[str]] = None):
        self.vital_cols = vital_cols or [
            'heart_rate', 'systolic_bp', 'diastolic_bp', 'mean_bp',
            'respiratory_rate', 'temperature', 'spo2'
        ]
        self.feature_names_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_ = []
        for col in self.vital_cols:
            if col in X.columns:
                self.feature_names_.extend([
                    f'{col}_min', f'{col}_max', f'{col}_mean', f'{col}_std', f'{col}_slope'
                ])
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame) and 'patient_id' in X.columns:
            grouped = X.groupby('patient_id')
            new_rows = []
            for pid, group in grouped:
                row = {'patient_id': pid}
                for col in self.vital_cols:
                    if col in group.columns:
                        vals = group[col].dropna().values
                        if len(vals) > 0:
                            row[f'{col}_min'] = vals.min()
                            row[f'{col}_max'] = vals.max()
                            row[f'{col}_mean'] = vals.mean()
                            row[f'{col}_std'] = vals.std() if len(vals) > 1 else 0
                            if len(vals) > 1:
                                row[f'{col}_slope'] = np.polyfit(np.arange(len(vals)), vals, 1)[0]
                            else:
                                row[f'{col}_slope'] = 0
                        else:
                            row[f'{col}_min'] = np.nan
                            row[f'{col}_max'] = np.nan
                            row[f'{col}_mean'] = np.nan
                            row[f'{col}_std'] = np.nan
                            row[f'{col}_slope'] = np.nan
                new_rows.append(row)
            return pd.DataFrame(new_rows)
        return X
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


class ComorbidityCounter(BaseEstimator, TransformerMixin):
    """Count comorbidities from ICD codes or comorbidity flags."""
    
    ELIXHAUSER_CONDITIONS = [
        'congestive_heart_failure', 'cardiac_arrhythmias', 'valvular_disease',
        'pulmonary_circulation', 'peripheral_vascular', 'hypertension',
        'paralysis', 'other_neurological', 'chronic_pulmonary',
        'diabetes_uncomplicated', 'diabetes_complicated', 'hypothyroidism',
        'renal_failure', 'liver_disease', 'peptic_ulcer', 'aids',
        'lymphoma', 'metastatic_cancer', 'solid_tumor', 'rheumatoid_arthritis',
        'coagulopathy', 'obesity', 'weight_loss', 'fluid_electrolyte',
        'blood_loss_anemia', 'deficiency_anemia', 'alcohol_abuse',
        'drug_abuse', 'psychoses', 'depression'
    ]
    
    def __init__(self, condition_cols: Optional[List[str]] = None):
        self.condition_cols = condition_cols or self.ELIXHAUSER_CONDITIONS
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        present_conditions = [c for c in self.condition_cols if c in X.columns]
        if present_conditions:
            X['comorbidity_count'] = X[present_conditions].sum(axis=1)
            X['elixhauser_score'] = X[present_conditions].sum(axis=1)
        return X


def build_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    scaling_method: str = 'zscore',
    discretization_method: str = 'quantile',
    n_bins: int = 10,
    add_missingness_indicators: bool = True,
    clip_outliers: bool = True,
    add_temporal_features: bool = False,
    add_comorbidity_count: bool = True,
) -> Pipeline:
    """
    Build the complete preprocessing pipeline.
    
    Args:
        numeric_features: List of numeric feature column names
        categorical_features: List of categorical feature column names
        scaling_method: 'zscore', 'minmax', or 'none'
        discretization_method: 'quantile', 'uniform', 'kmeans', or 'none'
        n_bins: Number of bins for discretization
        add_missingness_indicators: Whether to add missingness indicator columns
        clip_outliers: Whether to clip outliers to clinical ranges
        add_temporal_features: Whether to extract temporal features (requires patient_id and time)
        add_comorbidity_count: Whether to add comorbidity count feature
    
    Returns:
        sklearn Pipeline object
    """
    
    numeric_transformers = []
    
    if add_missingness_indicators:
        numeric_transformers.append(('missing_indicator', MissingnessIndicator(numeric_features)))
    
    if clip_outliers:
        numeric_transformers.append(('clipper', ClinicalClipper()))
    
    # Imputation
    numeric_transformers.append(('imputer', SimpleImputer(strategy='median')))
    
    # Scaling
    if scaling_method == 'zscore':
        numeric_transformers.append(('scaler', StandardScaler()))
    elif scaling_method == 'minmax':
        numeric_transformers.append(('scaler', MinMaxScaler()))
    # 'none' = no scaling
    
    # Discretization for Naive Bayes
    if discretization_method != 'none':
        numeric_transformers.append((
            'discretizer', 
            KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=discretization_method)
        ))
    
    numeric_pipeline = Pipeline(numeric_transformers)
    
    categorical_transformers = []
    if categorical_features:
        categorical_transformers.append(('imputer', SimpleImputer(strategy='most_frequent')))
        categorical_transformers.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
    
    categorical_pipeline = Pipeline(categorical_transformers) if categorical_transformers else 'drop'
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_pipeline, numeric_features),
            ('categorical', categorical_pipeline, categorical_features),
        ],
        remainder='drop',
        sparse_threshold=0,
    )
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('variance_filter', VarianceThreshold(threshold=0.0)),
    ])


def get_feature_names_after_preprocessing(
    pipeline: Pipeline,
    numeric_features: List[str],
    categorical_features: List[str],
) -> List[str]:
    """Get feature names after preprocessing pipeline."""
    preprocessor = pipeline.named_steps['preprocessor']
    feature_names = []
    
    # Get feature names from column transformer
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        # Fallback: manually construct
        for name, trans, cols in preprocessor.transformers_:
            if name == 'numeric' and trans != 'drop':
                if hasattr(trans, 'get_feature_names_out'):
                    try:
                        feature_names.extend(trans.get_feature_names_out(cols))
                    except AttributeError:
                        feature_names.extend(cols)
                else:
                    feature_names.extend(cols)
            elif name == 'categorical' and trans != 'drop':
                if hasattr(trans, 'get_feature_names_out'):
                    try:
                        feature_names.extend(trans.get_feature_names_out(cols))
                    except AttributeError:
                        feature_names.extend(cols)
                else:
                    feature_names.extend(cols)
    
    return feature_names