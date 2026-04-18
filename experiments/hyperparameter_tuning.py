"""
Hyperparameter tuning classes for NLP models.

Provides two approaches for hyperparameter optimization:
- RandomizedSearchCV: Uses scikit-learn's randomized search
- OptunaSearchCV: Uses Optuna for Bayesian optimization
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import os
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV as SklearnRandomizedSearchCV
from sklearn.base import clone
from scipy.stats import randint, uniform, loguniform
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import get_scorer


class RandomizedSearchCV:
    """
    Hyperparameter tuning using scikit-learn's RandomizedSearchCV.
    
    This class wraps scikit-learn's RandomizedSearchCV for consistent
    hyperparameter tuning across different models.
    """
    
    def __init__(
        self,
        estimator: Any,
        param_distributions: Dict[str, list],
        n_iter: int = 20,
        cv: int = 5,
        scoring: Optional[str] = None,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 1,
    ):
        """
        Initialize RandomizedSearchCV tuner.
        
        Args:
            estimator: The model to tune (e.g., LogisticRegression, XGBClassifier)
            param_distributions: Dictionary of parameter names and distributions to sample from
            n_iter: Number of parameter settings sampled
            cv: Number of cross-validation folds
            scoring: Scoring metric (e.g., 'accuracy', 'f1', 'roc_auc')
            n_jobs: Number of jobs to run in parallel (-1 = use all processors)
            random_state: Random seed for reproducibility
            verbose: Verbosity level
        """
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        self.search = None
        self.best_params = None
        self.best_score = None
        self.best_model = None
    
    def _normalize_param_distributions(self) -> Dict[str, Any]:
        """
        Accept both sklearn-native distributions/lists and Optuna-style specs.

        Supported schema for each parameter:
        - sklearn-native list/distribution object (passed through)
        - {'type': 'categorical', 'choices': [...]} 
        - {'type': 'int', 'low': a, 'high': b}
        - {'type': 'float', 'low': a, 'high': b, 'log': bool}
        """
        normalized: Dict[str, Any] = {}

        for name, spec in self.param_distributions.items():
            if not isinstance(spec, dict) or "type" not in spec:
                normalized[name] = spec
                continue

            param_type = spec.get("type")
            if param_type == "categorical":
                normalized[name] = spec["choices"]
            elif param_type == "int":
                low = int(spec["low"])
                high = int(spec["high"])
                normalized[name] = randint(low, high + 1)
            elif param_type == "float":
                low = float(spec["low"])
                high = float(spec["high"])
                if spec.get("log", False):
                    normalized[name] = loguniform(low, high)
                else:
                    normalized[name] = uniform(low, high - low)
            else:
                raise ValueError(f"Unsupported param type for '{name}': {param_type}")

        return normalized

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomizedSearchCV":
        """
        Run randomized search hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            self
        """
        self.search = SklearnRandomizedSearchCV(
            estimator=self.estimator,
            param_distributions=self._normalize_param_distributions(),
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        
        self.search.fit(X, y)
        self.best_params = self.search.best_params_
        self.best_score = self.search.best_score_
        self.best_model = self.search.best_estimator_
        
        return self
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best hyperparameters found."""
        if self.best_params is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.best_params
    
    def get_best_score(self) -> float:
        """Get best cross-validation score."""
        if self.best_score is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.best_score
    
    def get_best_model(self) -> Any:
        """Get the best model with optimal hyperparameters."""
        if self.best_model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.best_model

    def get_search(self) -> SklearnRandomizedSearchCV:
        """Get the fitted sklearn RandomizedSearchCV object."""
        if self.search is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.search
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best model."""
        if self.best_model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.best_model.predict(X)
    
    def get_visualisations(self, exp_dir: str) -> None:
        """Get visualizations for hyperparameter importance and optimization history."""
        if self.search is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        plots_dir = os.path.join(exp_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Note: sklearn's RandomizedSearchCV does not have built-in visualizations like Optuna.
        # We can create some basic plots using the cv_results_ attribute.

        results = self.search.cv_results_
        params = results['params']
        mean_test_scores = results['mean_test_score']

        # Example: Plot mean test score vs. one of the hyperparameters (if it's numeric)
        for param_name in self.param_distributions.keys():
            if isinstance(params[0][param_name], (int, float)):
                param_values = [p[param_name] for p in params]
                plt.figure(figsize=(8, 6))
                plt.scatter(param_values, mean_test_scores)
                plt.xlabel(param_name)
                plt.ylabel('Mean CV Test Score')
                plt.title(f'Hyperparameter Tuning: {param_name} vs. Mean CV Score')
                plt.grid()
                plt.savefig(f"{plots_dir}/cv_score_vs_{param_name}.png")
                plt.close()


class OptunaSearchCV:
    """
    Hyperparameter tuning using Optuna for Bayesian optimization.
    
    Optuna provides efficient hyperparameter optimization with pruning
    of unpromising trials, making it faster than random search.
    """
    
    def __init__(
        self,
        estimator: Any,
        param_distributions: Dict[str, Dict[str, Any]],
        cv: int = 5,
        scoring: Optional[str] = None,
        n_trials: int = 20,
        timeout: Optional[int] = None,
        random_state: int = 42,
        verbose: int = 1,
    ):
        """
        Initialize OptunaSearchCV tuner.
        
        Args:
            estimator: The model to tune (e.g., LogisticRegression, XGBClassifier)
            param_distributions: Dictionary of parameter names and their search space definitions.
                Example: {
                    'C': {'type': 'float', 'low': 0.01, 'high': 100},
                    'max_iter': {'type': 'int', 'low': 100, 'high': 1000}
                }
            cv: Number of cross-validation folds
            scoring: Scoring metric (e.g., 'accuracy', 'f1', 'roc_auc')
            n_trials: Maximum number of trials to run
            timeout: Timeout in seconds (optional)
            random_state: Random seed for reproducibility
            verbose: Verbosity level (0=silent, 1=with progress bar)
        """
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.cv = cv
        self.scoring = scoring
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        self.verbose = verbose
        
        self.study = None
        self.best_params = None
        self.best_score = None
        self.best_model = None
        self.best_trial = None

    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """
        Optimized objective function that enables pruning after each CV fold.
        """
        # 1. Sample parameters (Same logic as your previous version)
        params = {}
        for param_name, param_config in self.param_distributions.items():
            param_type = param_config.get('type', 'float')
            if param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, param_config['low'], param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config['choices']
                )

        # 2. Setup Model and CV
        # clone() ensures we start with a fresh version of your estimator
        model = clone(self.estimator)
        model.set_params(**params)
        
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        scorer = get_scorer(self.scoring) if self.scoring else None #e.g F1
        
        fold_scores = []
        
        # 3. Manual CV Loop for Pruning
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            try:
                model.fit(X_train, y_train)
            except ValueError as exc:
                message = str(exc)
                if "validation dataset" not in message and "early stopping" not in message:
                    raise

                # Some estimators (notably XGBoost/CatBoost when early stopping is enabled)
                # require a validation set during fit(). Use the held-out CV fold as the
                # validation data for this trial.
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
            
            # Calculate score for this fold
            if scorer:
                score = scorer(model, X_val, y_val)
            else:
                score = model.score(X_val, y_val)
                
            fold_scores.append(score)
            
            # --- THE KEY STEP FOR PRUNING ---
            # Report the average score up to this fold
            current_avg = np.mean(fold_scores)
            trial.report(current_avg, fold_idx)
            
            # Check if Optuna thinks we should stop this trial now
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        return np.mean(fold_scores)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "OptunaSearchCV":
        """
        Run Optuna hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            self
        """
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner()
        
        # Suppress Optuna's verbose output unless explicitly requested
        if self.verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction='maximize',
        )
        
        self.study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=self.verbose > 0,
        )
        
        self.best_trial = self.study.best_trial
        self.best_params = self.best_trial.params
        self.best_score = self.best_trial.value

        self.best_model = clone(self.estimator)
        self.best_model.set_params(**self.best_params)

        self.best_model.fit(X, y)
        return self
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best hyperparameters found."""
        if self.best_params is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.best_params
    
    def get_best_score(self) -> float:
        """Get best cross-validation score."""
        if self.best_score is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.best_score
    
    def get_best_model(self) -> Any:
        """Get the best model with optimal hyperparameters."""
        if self.best_model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.best_model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best model."""
        if self.best_model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.best_model.predict(X)
    
    def get_study(self) -> optuna.Study:
        """Get the Optuna study object for advanced analysis."""
        if self.study is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.study
    
    def get_visualisations(self, exp_dir: str) -> None:
        """Get Optuna visualizations for hyperparameter importance and optimization history."""
        if self.study is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        plots_dir = os.path.join(exp_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        optimisation_history = optuna.visualization.plot_optimization_history(self.study)
        param_importances = optuna.visualization.plot_param_importances(self.study)
        parallel_coordinates = optuna.visualization.plot_parallel_coordinate(self.study)

        optimisation_history.write_html(f"{plots_dir}/optimization_history.html")
        param_importances.write_html(f"{plots_dir}/param_importances.html")
        parallel_coordinates.write_html(f"{plots_dir}/parallel_coordinates.html")

        for var in self.param_distributions.keys():
            slice_plot = optuna.visualization.plot_slice(self.study, params=[var])
            slice_plot.write_html(f"{plots_dir}/slice_{var}.html")
