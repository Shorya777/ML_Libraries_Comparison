import pandas as pd
import numpy as pd
import gc
import numpy as np
from itertools import product
import mlflow 
import dagshub
from sklearn.metrics import accuracy_score

def custom_grid_search(model_name, model, param_grid, X, y, cv=3, scoring='accuracy'):
    gc.collect()
    if cv < 2:
        raise ValueError("Cross-validation requires at least 2 folds (cv >= 2).")

    best_score = 0
    best_params = None
    
    param_combinations = list(product(*param_grid.values()))
    param_keys = list(param_grid.keys())

    with mlflow.start_run(run_name = model_name, nested = True):
        for i, combination in enumerate(param_combinations, 1):
            params = dict(zip(param_keys, combination))
            model.set_params(**params)
            print(f"Training model {i}/{len(param_combinations)} with params: {params}")
    
            np.random.seed(42)
            indices = np.arange(len(X))
            np.random.shuffle(indices)
    
            fold_sizes = np.full(cv, len(X) // cv)
            fold_sizes[:len(X) % cv] += 1
            folds = np.split(indices, np.cumsum(fold_sizes)[:-1])
    
            cv_scores = []
    
            with mlflow.start_run(run_name = f'{model_name}_{i}', nested = True):
                mlflow.log_params(params)
    
                for fold_idx, valid_indices in enumerate(folds):
                    train_indices = np.concatenate([folds[i] for i in range(cv) if i != fold_idx])
                    X_train, X_valid = X[train_indices], X[valid_indices]
                    y_train, y_valid = y[train_indices], y[valid_indices]
    
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_valid)
    
                        # Handle scoring
                        if scoring == 'mean_squared_error':
                            score = np.mean((y_valid - y_pred) ** 2)
                        elif scoring == 'accuracy':
                            score = accuracy_score(y_valid, y_pred)
                        else:
                            raise ValueError(f"Scoring method {scoring} is not supported")
    
                        cv_scores.append(score)
    
                    except Exception as e:
                        print(f"Error with params {params} during fold {fold_idx}: {e}")
                        break
    
                if cv_scores:
                    mean_cv_score = np.mean(cv_scores)
                    mlflow.log_metric("mean_cv_score", mean_cv_score)
    
                    if mean_cv_score > best_score:
                        best_score = mean_cv_score
                        best_params = params

    return best_params, best_score

def custom_grid_search_models(models, param_grid, X_train, y_train, cv=3):
    best_models = {}
    dagshub.init(repo_owner='Shorya777', repo_name='ML_Libraries_Comparison', mlflow=True)
    with mlflow.start_run(run_name = "sklearn_run"):
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            best_params, best_score = custom_grid_search(model_name, model, param_grid[model_name], X_train, y_train, cv)
    
            print(f"Best Parameters for {model_name}: {best_params}")
            print(f"Best Score (accuracy): {best_score}")
    
            best_models[model_name] = {
                'best_params': best_params,
                'best_score': best_score,
            }

    return best_models
