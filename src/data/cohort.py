"""
Cohort extraction for healthcare risk prediction.
Defines the patient cohort, observation window, and target labels.
"""
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class CohortConfig:
    """Configuration for cohort extraction."""
    dataset: str = 'eicu'  # 'eicu', 'mimic', 'synthea'
    target: str = 'mortality'  # 'mortality', 'readmission_30d'
    observation_window_hours: int = 24
    min_age: int = 18
    max_age: int = 100
    first_stay_only: bool = True
    min_los_hours: float = 1.0
    exclude_doa: bool = True  # Exclude dead on arrival
    
    # MIMIC-specific
    mimic_version: str = 'iv'
    
    # eICU-specific
    eicu_hospital_ids: Optional[List[int]] = None  # None = all hospitals


def extract_eicu_cohort(
    conn,  # database connection
    config: CohortConfig,
) -> pd.DataFrame:
    """
    Extract cohort from eICU database.
    
    Requires tables: patient, admissiondrug, diagnosis, lab, vitalperiodic, 
    nursecharting, respiratorycharting, infusiondrug, treatment.
    """
    # This is a template - actual implementation depends on database schema
    query = f"""
    WITH first_icu_stay AS (
        SELECT 
            p.patienthealthsystemstayid,
            p.patientid,
            p.age,
            p.gender,
            p.ethnicity,
            p.hospitalid,
            p.unitdischargestatus,
            p.hospitaldischargestatus,
            p.unitdischargeoffset,
            p.hospitaldischargeoffset,
            ap.admissionheight,
            ap.admissionweight,
            ap.apachescore,
            ap.predictedhospitalmortality
        FROM patient p
        JOIN admissiondrug ap ON p.patienthealthsystemstayid = ap.patienthealthsystemstayid
        WHERE p.age >= {config.min_age}
          AND p.age <= {config.max_age}
          AND p.unitdischargeoffset >= {config.min_los_hours * 60}
    ),
    first_24h_vitals AS (
        SELECT 
            vp.patienthealthsystemstayid,
            vp.observationoffset,
            vp.heartrate,
            vp.systolic,
            vp.diastolic,
            vp.meanbp,
            vp.respiration,
            vp.temperature,
            vp.spo2
        FROM vitalperiodic vp
        JOIN first_icu_stay fis ON vp.patienthealthsystemstayid = fis.patienthealthsystemstayid
        WHERE vp.observationoffset BETWEEN 0 AND {config.observation_window_hours * 60}
    ),
    first_24h_labs AS (
        SELECT 
            l.patienthealthsystemstayid,
            l.labname,
            l.labresult,
            l.labresultoffset
        FROM lab l
        JOIN first_icu_stay fis ON l.patienthealthsystemstayid = fis.patienthealthsystemstayid
        WHERE l.labresultoffset BETWEEN 0 AND {config.observation_window_hours * 60}
    )
    SELECT * FROM first_icu_stay
    """
    return pd.read_sql(query, conn)


def extract_mimic_cohort(
    conn,
    config: CohortConfig,
) -> pd.DataFrame:
    """Extract cohort from MIMIC-III/IV database."""
    # Template for MIMIC - requires chartevents, labevents, icustays, patients, admissions
    pass


def define_target(
    df: pd.DataFrame,
    config: CohortConfig,
) -> pd.Series:
    """Define the target variable based on configuration."""
    if config.target == 'mortality':
        if 'hospitaldischargestatus' in df.columns:
            return (df['hospitaldischargestatus'] == 'Expired').astype(int)
        elif 'unitdischargestatus' in df.columns:
            return (df['unitdischargestatus'] == 'Expired').astype(int)
        elif 'hospital_expire_flag' in df.columns:
            return df['hospital_expire_flag'].astype(int)
    elif config.target == 'readmission_30d':
        # Requires readmission data
        pass
    raise ValueError(f"Unknown target: {config.target}")


def create_cohort(
    df: pd.DataFrame,
    config: CohortConfig,
    hospital_id_col: str = 'hospitalid',
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create the final cohort dataframe with features and target.
    
    Returns:
        (features_df, target_series)
    """
    # Apply inclusion/exclusion criteria
    df = df.copy()
    
    # Age filter
    if 'age' in df.columns:
        df = df[(df['age'] >= config.min_age) & (df['age'] <= config.max_age)]
    
    # LOS filter
    if 'unitdischargeoffset' in df.columns:
        df = df[df['unitdischargeoffset'] >= config.min_los_hours * 60]
    
    # First stay only
    if config.first_stay_only and 'patientid' in df.columns:
        df = df.sort_values('hospitaldischargeoffset').drop_duplicates('patientid', keep='first')
    
    # Define target
    y = define_target(df, config)
    
    # Drop target-related columns from features
    drop_cols = [
        'hospitaldischargestatus', 'unitdischargestatus',
        'hospitaldischargeoffset', 'unitdischargeoffset',
        'hospital_expire_flag',
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    return X, y


def split_by_hospital(
    X: pd.DataFrame,
    y: pd.Series,
    hospital_id_col: str = 'hospitalid',
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Split data by hospital for federated learning.
    Returns dict with train/test splits per hospital + global test set.
    """
    from sklearn.model_selection import train_test_split
    
    hospitals = X[hospital_id_col].unique()
    splits = {}
    
    # Global test set (stratified)
    X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    splits['global'] = {
        'X_train': X_train_global, 'y_train': y_train_global,
        'X_test': X_test_global, 'y_test': y_test_global,
    }
    
    # Per-hospital splits
    for hid in hospitals:
        mask = X[hospital_id_col] == hid
        X_h = X[mask]
        y_h = y[mask]
        
        if len(X_h) < 10:  # Skip very small hospitals
            continue
            
        X_train, X_test, y_train, y_test = train_test_split(
            X_h, y_h, test_size=test_size, stratify=y_h, random_state=random_state
        )
        splits[f'hospital_{hid}'] = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
        }
    
    return splits


def simulate_non_iid_partition(
    X: pd.DataFrame,
    y: pd.Series,
    n_clients: int = 5,
    alpha: float = 0.5,
    partition_type: str = 'dirichlet',
    random_state: int = 42,
) -> dict:
    """
    Simulate non-IID partitions for federated learning.
    
    Args:
        X: Features
        y: Labels
        n_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        partition_type: 'dirichlet', 'label_skew', 'feature_skew', 'quantity_skew'
        random_state: Random seed
    
    Returns:
        Dict mapping client_id to (X_train, y_train, X_test, y_test)
    """
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if partition_type == 'dirichlet':
        # Dirichlet partition for label skew
        n_classes = len(np.unique(y))
        client_props = np.random.dirichlet([alpha] * n_clients, n_classes)
        client_indices = [[] for _ in range(n_clients)]
        
        for c in range(n_classes):
            class_indices = indices[y == c]
            np.random.shuffle(class_indices)
            props = client_props[c]
            split_points = (np.cumsum(props) * len(class_indices)).astype(int)[:-1]
            class_splits = np.split(class_indices, split_points)
            for i, split in enumerate(class_splits):
                client_indices[i].extend(split)
    
    elif partition_type == 'label_skew':
        # Extreme label skew: each client gets mostly one class
        n_classes = len(np.unique(y))
        client_indices = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            class_indices = indices[y == c]
            np.random.shuffle(class_indices)
            # Assign each class predominantly to one client
            primary_client = c % n_clients
            n_primary = int(0.8 * len(class_indices))
            client_indices[primary_client].extend(class_indices[:n_primary])
            # Distribute rest
            remaining = class_indices[n_primary:]
            for idx in remaining:
                client_indices[np.random.randint(n_clients)].append(idx)
    
    elif partition_type == 'feature_skew':
        # Partition by feature (e.g., age groups)
        if 'age' in X.columns:
            age_col = X['age'].values
        else:
            age_col = X.iloc[:, 0].values
        
        # Create age groups
        bins = np.percentile(age_col, np.linspace(0, 100, n_clients + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        client_indices = [[] for _ in range(n_clients)]
        for i in range(n_samples):
            for j in range(n_clients):
                if bins[j] <= age_col[i] < bins[j + 1]:
                    client_indices[j].append(i)
                    break
    
    elif partition_type == 'quantity_skew':
        # Unequal client sizes (power law)
        sizes = np.random.pareto(1.5, n_clients)
        sizes = (sizes / sizes.sum() * n_samples).astype(int)
        sizes[-1] = n_samples - sizes[:-1].sum()
        
        np.random.shuffle(indices)
        client_indices = []
        start = 0
        for size in sizes:
            client_indices.append(indices[start:start + size])
            start += size
    
    else:
        raise ValueError(f"Unknown partition_type: {partition_type}")
    
    # Create splits
    splits = {}
    for i, idx in enumerate(client_indices):
        if len(idx) < 10:
            continue
        X_client = X.iloc[idx]
        y_client = y.iloc[idx]
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_client, y_client, test_size=0.2, stratify=y_client, random_state=random_state
        )
        splits[f'client_{i}'] = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
        }
    
    return splits