"""
Manual Federated Learning implementation (FedAvg) for healthcare risk prediction.
This replaces the complex Flower simulation with a clean, controllable implementation.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')


class FederatedClient:
    """Local client for federated learning."""
    
    def __init__(
        self,
        client_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_type: str = 'logistic',
        use_global_scaler: bool = False,
        global_scaler: Optional[StandardScaler] = None,
    ):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_type = model_type
        self.use_global_scaler = use_global_scaler
        self.global_scaler = global_scaler
        
        # Initialize local scaler and model
        self.scaler = global_scaler if use_global_scaler else StandardScaler()
        self.model = self._create_model()
        
        # Fit scaler
        self.scaler.fit(X_train)
    
    def _create_model(self):
        if self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                solver='lbfgs',
            )
        else:
            return LinearRegression()
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as list of numpy arrays."""
        n_features = self.X_train.shape[1]
        if not hasattr(self.model, 'coef_') or self.model.coef_ is None:
            if self.model_type == 'logistic':
                return [np.zeros((1, n_features)), np.zeros(1)]
            else:
                return [np.zeros(n_features), np.array([0.0])]
        else:
            if self.model_type == 'logistic':
                return [self.model.coef_.copy(), self.model.intercept_.copy()]
            else:
                return [self.model.coef_.copy(), np.array([self.model.intercept_])]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from list of numpy arrays."""
        if self.model_type == 'logistic':
            self.model.coef_ = parameters[0]
            self.model.intercept_ = parameters[1]
        else:
            self.model.coef_ = parameters[0]
            self.model.intercept_ = parameters[1][0]
    
    def train(self) -> Dict[str, float]:
        """Train model locally and return metrics."""
        X_train_scaled = self.scaler.transform(self.X_train)
        self.model.fit(X_train_scaled, self.y_train)
        
        # Compute metrics
        X_test_scaled = self.scaler.transform(self.X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'num_examples': len(self.X_train),
        }
        
        # For logistic regression, compute probabilities
        if self.model_type == 'logistic':
            y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
            metrics['auroc'] = roc_auc_score(self.y_test, y_prob)
            metrics['auprc'] = average_precision_score(self.y_test, y_prob)
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set."""
        X_test_scaled = self.scaler.transform(self.X_test)
        
        if self.model_type == 'logistic':
            if not hasattr(self.model, 'classes_'):
                # Model not fitted yet
                return {'accuracy': 0.0, 'num_examples': len(self.X_test)}
            y_pred = self.model.predict(X_test_scaled)
            y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
            return {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'auroc': roc_auc_score(self.y_test, y_prob),
                'auprc': average_precision_score(self.y_test, y_prob),
                'num_examples': len(self.X_test),
            }
        else:
            y_pred = self.model.predict(X_test_scaled)
            return {
                'accuracy': accuracy_score(self.y_test, (y_pred > 0.5).astype(int)),
                'num_examples': len(self.X_test),
            }


class FederatedServer:
    """Server for federated learning (FedAvg)."""
    
    def __init__(
        self,
        model_type: str = 'logistic',
        n_features: int = 38,
    ):
        self.model_type = model_type
        self.n_features = n_features
        self.global_model = None
        self.round_metrics = []
    
    def initialize_global_model(self):
        """Initialize global model with zeros."""
        if self.model_type == 'logistic':
            self.global_model = {
                'coef': np.zeros((1, self.n_features)),
                'intercept': np.zeros(1),
            }
        else:
            self.global_model = {
                'coef': np.zeros(self.n_features),
                'intercept': np.array([0.0]),
            }
    
    def get_global_parameters(self) -> List[np.ndarray]:
        """Get global model parameters."""
        return [self.global_model['coef'].copy(), self.global_model['intercept'].copy()]
    
    def aggregate(self, client_updates: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """
        Aggregate client updates using FedAvg (weighted by number of examples).
        
        Args:
            client_updates: List of (parameters, num_examples) from each client
        
        Returns:
            Aggregated global parameters
        """
        total_examples = sum(n for _, n in client_updates)
        
        # Weighted average of parameters
        weighted_coef = np.zeros_like(client_updates[0][0][0])
        weighted_intercept = np.zeros_like(client_updates[0][0][1])
        
        for params, n_examples in client_updates:
            weight = n_examples / total_examples
            weighted_coef += weight * params[0]
            weighted_intercept += weight * params[1]
        
        self.global_model['coef'] = weighted_coef
        self.global_model['intercept'] = weighted_intercept
        
        return self.get_global_parameters()


def run_federated_learning(
    splits: Dict,
    model_type: str = 'logistic',
    num_rounds: int = 10,
    num_clients: int = 5,
    use_global_scaler: bool = False,
    output_dir: str = 'results',
) -> Dict:
    """
    Run federated learning simulation with FedAvg.
    
    Args:
        splits: Data splits from preprocessing
        model_type: 'logistic' or 'linear'
        num_rounds: Number of federated rounds
        num_clients: Number of clients per round
        use_global_scaler: Whether to use global scaler (federated) or local scalers
        output_dir: Output directory for results
    
    Returns:
        Dict with round metrics and final results
    """
    from pathlib import Path
    
    print(f"\n=== Manual FedAvg Simulation ===")
    print(f"Model: {model_type}")
    print(f"Rounds: {num_rounds}")
    print(f"Clients: {num_clients}")
    print(f"Global scaler: {use_global_scaler}")
    
    # Prepare hospital splits
    hospital_splits = {k: v for k, v in splits.items() if k.startswith('hospital')}
    client_ids = list(hospital_splits.keys())[:num_clients]
    
    # Prepare global scaler if needed
    if use_global_scaler:
        all_X_train = np.vstack([v['X_train'].values for v in hospital_splits.values()])
        global_scaler = StandardScaler()
        global_scaler.fit(all_X_train)
    else:
        global_scaler = None
    
    # Create clients
    clients = {}
    for cid in client_ids:
        split = hospital_splits[cid]
        clients[cid] = FederatedClient(
            client_id=cid,
            X_train=split['X_train'].values,
            y_train=split['y_train'].values.ravel(),
            X_test=split['X_test'].values,
            y_test=split['y_test'].values.ravel(),
            model_type=model_type,
            use_global_scaler=use_global_scaler,
            global_scaler=global_scaler,
        )
    
    # Initialize server
    n_features = splits['global']['X_train'].shape[1]
    server = FederatedServer(model_type=model_type, n_features=n_features)
    server.initialize_global_model()
    
    # Federated rounds
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num}/{num_rounds} ---")
        
        # Distribute global parameters to clients
        global_params = server.get_global_parameters()
        
        client_updates = []
        round_client_metrics = {}
        
        # Each client trains locally
        for cid in client_ids:
            client = clients[cid]
            client.set_parameters(global_params)
            metrics = client.train()
            client_params = client.get_parameters()
            
            client_updates.append((client_params, metrics['num_examples']))
            round_client_metrics[cid] = metrics
            
            print(f"  {cid}: acc={metrics['accuracy']:.4f}", end='')
            if 'auroc' in metrics:
                print(f", auroc={metrics['auroc']:.4f}", end='')
            print()
        
        # Server aggregates
        server.aggregate(client_updates)
        
        # Evaluate global model on each client's test set
        global_params = server.get_global_parameters()
        eval_metrics = {}
        
        for cid in client_ids:
            client = clients[cid]
            client.set_parameters(global_params)
            eval_metrics[cid] = client.evaluate()
        
        # Compute weighted averages
        total_examples = sum(m['num_examples'] for m in eval_metrics.values())
        weighted_acc = sum(m['accuracy'] * m['num_examples'] for m in eval_metrics.values()) / total_examples
        
        round_summary = {
            'round': round_num,
            'weighted_accuracy': weighted_acc,
            'client_metrics': eval_metrics,
        }
        
        if model_type == 'logistic':
            weighted_auroc = sum(m.get('auroc', 0) * m['num_examples'] for m in eval_metrics.values()) / total_examples
            weighted_auprc = sum(m.get('auprc', 0) * m['num_examples'] for m in eval_metrics.values()) / total_examples
            round_summary['weighted_auroc'] = weighted_auroc
            round_summary['weighted_auprc'] = weighted_auprc
            print(f"  Global: acc={weighted_acc:.4f}, auroc={weighted_auroc:.4f}, auprc={weighted_auprc:.4f}")
        else:
            print(f"  Global: acc={weighted_acc:.4f}")
        
        server.round_metrics.append(round_summary)
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_global_params = server.get_global_parameters()
    
    for cid in client_ids:
        client = clients[cid]
        client.set_parameters(final_global_params)
        final_metrics = client.evaluate()
        print(f"  {cid}: {final_metrics}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    scaler_suffix = "_fed_scaler" if use_global_scaler else "_local_scaler"
    metrics_df = pd.DataFrame(server.round_metrics)
    metrics_df.to_csv(output_path / f'fl_{model_type}{scaler_suffix}_metrics.csv', index=False)
    
    return {
        'server': server,
        'clients': clients,
        'round_metrics': server.round_metrics,
        'final_global_params': final_global_params,
    }


def run_centralized_baseline(
    splits: Dict,
    model_type: str = 'logistic',
) -> Dict:
    """Run centralized baseline for comparison."""
    X_train = splits['global']['X_train'].values
    y_train = splits['global']['y_train'].values.ravel()
    X_test = splits['global']['X_test'].values
    y_test = splits['global']['y_test'].values.ravel()
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, solver='lbfgs')
    else:
        model = LinearRegression()
    
    model.fit(X_train_scaled, y_train)
    
    if model_type == 'logistic':
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'auroc': roc_auc_score(y_test, y_prob),
            'auprc': average_precision_score(y_test, y_prob),
            'model': model,
            'scaler': scaler,
        }
    else:
        y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'model': model,
            'scaler': scaler,
        }


def run_local_only(
    splits: Dict,
    model_type: str = 'logistic',
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
            model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, solver='lbfgs')
        else:
            model = LinearRegression()
        
        model.fit(X_train_scaled, y_train)
        
        if model_type == 'logistic':
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            results[cid] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'auroc': roc_auc_score(y_test, y_prob),
                'auprc': average_precision_score(y_test, y_prob),
                'model': model,
                'scaler': scaler,
            }
        else:
            y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
            results[cid] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'model': model,
                'scaler': scaler,
            }
        
        print(f"  {cid}: acc={results[cid]['accuracy']:.4f}", end='')
        if 'auroc' in results[cid]:
            print(f", auroc={results[cid]['auroc']:.4f}", end='')
        print()
    
    return results