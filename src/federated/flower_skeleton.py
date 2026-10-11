"""
Federated Learning skeleton using Flower.
Implements FedAvg for linear models as the foundation.
"""
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import pandas as pd
from collections import OrderedDict
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Context,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FlowerClient(fl.client.NumPyClient):
    """Flower client for federated learning."""
    
    def __init__(
        self,
        client_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_type: str = 'logistic',
        scaler: Optional[StandardScaler] = None,
    ):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_type = model_type
        self.scaler = scaler
        self.model = self._create_model()
        self._fit_scaler()
    
    def _create_model(self):
        """Create local model."""
        if self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                solver='lbfgs',
            )
        elif self.model_type == 'linear':
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def _fit_scaler(self):
        """Fit scaler on local training data."""
        if self.scaler is not None:
            self.scaler.fit(self.X_train)
    
    def get_parameters(self, config):
        """Return model parameters as list of numpy arrays."""
        # Check if model is fitted
        if not hasattr(self.model, 'coef_') or self.model.coef_ is None:
            # Return initial parameters (zeros)
            n_features = self.X_train.shape[1]
            if self.model_type == 'logistic':
                return [np.zeros((1, n_features)), np.zeros(1)]
            else:
                return [np.zeros(n_features), np.array([0.0])]
        else:
            if self.model_type == 'logistic':
                return [self.model.coef_, self.model.intercept_]
            else:
                return [self.model.coef_, np.array([self.model.intercept_])]
    
    def set_parameters(self, parameters):
        """Set model parameters from list of numpy arrays."""
        if self.model_type == 'logistic':
            self.model.coef_ = parameters[0]
            self.model.intercept_ = parameters[1]
        else:
            self.model.coef_ = parameters[0]
            self.model.intercept_ = parameters[1][0]
    
    def fit(self, parameters: Parameters, config) -> FitRes:
        """Train model on local data."""
        self.set_parameters(parameters)
        
        # Scale training data if scaler available
        if self.scaler is not None:
            X_train_scaled = self.scaler.transform(self.X_train)
        else:
            X_train_scaled = self.X_train
        
        # Local training
        self.model.fit(X_train_scaled, self.y_train)
        
        # Return updated parameters
        new_params = self.get_parameters(config)
        num_examples = len(self.X_train)
        
        # Local metrics
        if self.scaler is not None:
            X_test_scaled = self.scaler.transform(self.X_test)
        else:
            X_test_scaled = self.X_test
        
        train_score = self.model.score(X_train_scaled, self.y_train)
        test_score = self.model.score(X_test_scaled, self.y_test)
        
        metrics = {
            'train_score': float(train_score),
            'test_score': float(test_score),
            'num_examples': num_examples,
        }
        
        return FitRes(
            parameters=new_params,
            num_examples=num_examples,
            metrics=metrics,
        )
    
    def evaluate(self, parameters: Parameters, config):
        """Evaluate model on local test data."""
        self.set_parameters(parameters)
        
        if self.scaler is not None:
            X_test_scaled = self.scaler.transform(self.X_test)
        else:
            X_test_scaled = self.X_test
        
        test_score = self.model.score(X_test_scaled, self.y_test)
        num_examples = len(self.X_test)
        
        return float(1 - test_score), num_examples, {'accuracy': float(test_score)}


def create_client_fn(
    splits: Dict,
    model_type: str = 'logistic',
    use_federated_scaler: bool = False,
) -> Callable[[Context], fl.client.Client]:
    """Create client factory function for Flower simulation (new API with Context)."""
    
    # Prepare hospital splits list for numeric partition IDs
    hospital_splits = {k: v for k, v in splits.items() if k.startswith('hospital')}
    hospital_ids = sorted(hospital_splits.keys())
    
    # Prepare global scaler if using federated scaling
    if use_federated_scaler:
        all_X_train = np.vstack([v['X_train'].values for v in hospital_splits.values()])
        global_scaler = StandardScaler()
        global_scaler.fit(all_X_train)
    else:
        global_scaler = None
    
    def client_fn(context: Context) -> fl.client.Client:
        # Extract partition ID from context (numeric: 0, 1, 2, ...)
        partition_id = context.node_config.get("partition-id", 0)
        
        # Map partition ID to hospital ID
        if isinstance(partition_id, str):
            try:
                partition_id = int(partition_id)
            except ValueError:
                partition_id = 0
        
        if partition_id >= len(hospital_ids):
            partition_id = 0
        
        cid = hospital_ids[partition_id]
        
        split = hospital_splits[cid]
        X_train = split['X_train'].values
        y_train = split['y_train'].values.ravel()
        X_test = split['X_test'].values
        y_test = split['y_test'].values.ravel()
        
        # Use global scaler for federated scaling, None for local
        scaler = global_scaler if use_federated_scaler else StandardScaler()
        
        client = FlowerClient(
            client_id=cid,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_type=model_type,
            scaler=scaler,
        )
        return client.to_client()
    
    return client_fn


class FedAvgWithLogging(FedAvg):
    """FedAvg strategy with custom logging."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_metrics_ = []
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate and log metrics."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Log round metrics
        if results:
            train_scores = [r.metrics.get('train_score', 0) for _, r in results]
            test_scores = [r.metrics.get('test_score', 0) for _, r in results]
            num_examples = [r.num_examples for _, r in results]
            
            round_metric = {
                'round': server_round,
                'avg_train_score': np.mean(train_scores),
                'avg_test_score': np.mean(test_scores),
                'weighted_train_score': np.average(train_scores, weights=num_examples),
                'weighted_test_score': np.average(test_scores, weights=num_examples),
                'num_clients': len(results),
            }
            self.round_metrics_.append(round_metric)
            
            print(f"Round {server_round}: "
                  f"train={round_metric['weighted_train_score']:.4f}, "
                  f"test={round_metric['weighted_test_score']:.4f}, "
                  f"clients={len(results)}")
        
        return aggregated_parameters, aggregated_metrics
    
    def get_round_metrics(self) -> List[Dict]:
        return self.round_metrics_


def run_flower_simulation(
    splits: Dict,
    model_type: str = 'logistic',
    num_rounds: int = 10,
    num_clients: int = 5,
    client_fraction: float = 1.0,
    use_federated_scaler: bool = False,
    output_dir: str = 'results',
) -> Dict:
    """
    Run Flower simulation for federated learning.
    
    Args:
        splits: Data splits from preprocessing
        model_type: 'logistic' or 'linear'
        num_rounds: Number of federated rounds
        num_clients: Number of clients per round
        client_fraction: Fraction of clients to sample per round
        use_federated_scaler: Whether to use global scaler
        output_dir: Output directory for results
    
    Returns:
        Dict with round metrics and final model
    """
    hospital_splits = {k: v for k, v in splits.items() if k.startswith('hospital')}
    client_ids = list(hospital_splits.keys())[:num_clients]
    
    print(f"\n=== Flower FedAvg Simulation ===")
    print(f"Model: {model_type}")
    print(f"Rounds: {num_rounds}")
    print(f"Clients: {client_ids}")
    print(f"Federated scaler: {use_federated_scaler}")
    
    # Create client function
    client_fn = create_client_fn(splits, model_type, use_federated_scaler)
    
    # Create strategy
    strategy = FedAvgWithLogging(
        fraction_fit=client_fraction,
        fraction_evaluate=client_fraction,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )
    
    # Create initial parameters from a dummy model (without Flower wrapper)
    from sklearn.linear_model import LogisticRegression, LinearRegression
    hospital_splits = {k: v for k, v in splits.items() if k.startswith('hospital')}
    X_sample = hospital_splits[client_ids[0]]['X_train'].values
    y_sample = hospital_splits[client_ids[0]]['y_train'].values.ravel()
    
    if model_type == 'logistic':
        dummy_model = LogisticRegression(max_iter=1, class_weight='balanced', random_state=42, solver='lbfgs')
        dummy_model.fit(X_sample, y_sample)
        initial_params = [dummy_model.coef_, dummy_model.intercept_]
    else:
        dummy_model = LinearRegression()
        dummy_model.fit(X_sample, y_sample)
        initial_params = [dummy_model.coef_, np.array([dummy_model.intercept_])]
    
    initial_parameters = ndarrays_to_parameters(initial_params)
    
    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_ids),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )
    
    # Save round metrics
    round_metrics = strategy.get_round_metrics()
    metrics_df = pd.DataFrame(round_metrics)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    scaler_suffix = "_fed_scaler" if use_federated_scaler else "_local_scaler"
    metrics_df.to_csv(output_path / f'fl_{model_type}{scaler_suffix}_metrics.csv', index=False)
    
    print(f"\nSimulation complete!")
    print(f"Final test score: {metrics_df['weighted_test_score'].iloc[-1]:.4f}" if len(metrics_df) > 0 else "No metrics")
    
    return {
        'history': history,
        'round_metrics': round_metrics,
        'metrics_df': metrics_df,
        'strategy': strategy,
    }


def run_centralized_baseline_flower(
    splits: Dict,
    model_type: str = 'logistic',
    output_dir: str = 'results',
) -> Dict:
    """Run centralized baseline using same model for comparison."""
    X_train = splits['global']['X_train'].values
    y_train = splits['global']['y_train'].values.ravel()
    X_test = splits['global']['X_test'].values
    y_test = splits['global']['y_test'].values.ravel()
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    else:
        model = LinearRegression()
    
    model.fit(X_train_scaled, y_train)
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"\n=== Centralized Baseline ({model_type}) ===")
    print(f"Train score: {train_score:.4f}")
    print(f"Test score: {test_score:.4f}")
    
    return {
        'train_score': train_score,
        'test_score': test_score,
        'model': model,
        'scaler': scaler,
    }


def run_local_only_flower(
    splits: Dict,
    model_type: str = 'logistic',
    output_dir: str = 'results',
) -> Dict:
    """Run local-only training for each client."""
    hospital_splits = {k: v for k, v in splits.items() if k.startswith('hospital')}
    
    results = {}
    for cid, split in hospital_splits.items():
        X_train = split['X_train'].values
        y_train = split['y_train'].values.ravel()
        X_test = split['X_test'].values
        y_test = split['y_test'].values.ravel()
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == 'logistic':
            model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        else:
            model = LinearRegression()
        
        model.fit(X_train_scaled, y_train)
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        results[cid] = {
            'train_score': train_score,
            'test_score': test_score,
            'model': model,
            'scaler': scaler,
        }
        print(f"  {cid}: train={train_score:.4f}, test={test_score:.4f}")
    
    return results