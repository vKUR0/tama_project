import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 1. On récupère le chemin absolu du dossier où se trouve le script actuel
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. On remonte d'un niveau (l'équivalent du "..")
parent_dir = os.path.dirname(current_dir)

# 3. On construit le chemin vers le dossier qui contient le dossier "Code"
# (Si "Code" est directement dans le dossier parent)
sys.path.append(parent_dir)

# 4. Maintenant tu peux importer normalement en spécifiant le dossier "Code"
from Code.Tools.data import load_dataset_tsunami


def get_prepared_tsunami_data(data_path="./dataset/dataset_earthquake.csv"):
    """Charge les données, applique le split stratifié et renvoie les versions brutes et scalées."""
    X, y = load_dataset_tsunami(data_path)
    
    # Stratify=y est indispensable ici pour conserver le ratio de tsunamis
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling pour le MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Calcul du poids des classes pour XGBoost
    num_neg = len(y_train) - sum(y_train)
    num_pos = sum(y_train)
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, pos_weight


def tune_tsunami_xgb(X_train, y_train, pos_weight):
    """Optimise le XGBoost Classifier pour maximiser le F1-Score."""
    print("\n>>> Tuning XGBoost Classifier...")
    
    param_dist = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    # StratifiedKFold garantit que chaque fold a ses 1% de tsunamis
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)

    search = RandomizedSearchCV(
        estimator=xgb.XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss", random_state=69),
        param_distributions=param_dist,
        n_iter=15,
        scoring="f1",  # On cherche à maximiser le F1-score !
        cv=cv_strategy,
        random_state=69,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print("Meilleurs paramètres XGBoost:", search.best_params_)
    return search.best_params_


def tune_tsunami_random_forest(X_train, y_train):
    """Optimise le Random Forest Classifier."""
    print("\n>>> Tuning Random Forest Classifier...")
    
    param_dist = {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        # class_weight="balanced" aide le RF à pénaliser les erreurs sur la classe rare
        "class_weight": ["balanced", "balanced_subsample", None] 
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=69),
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1",
        cv=cv_strategy,
        random_state=69,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print("Meilleurs paramètres Random Forest:", search.best_params_)
    return search.best_params_


def tune_tsunami_mlp(X_train_scaled, y_train):
    """Optimise le MLP Classifier (Deep Learning) sur données scalées."""
    print("\n>>> Tuning MLP Classifier...")
    
    param_dist = {
        "hidden_layer_sizes": [(64, 32, 16), (100, 50), (128, 64), (50, 50)],
        "activation": ["tanh", "relu"],
        "solver": ["adam", "sgd"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01, 0.05],
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)

    search = RandomizedSearchCV(
        estimator=MLPClassifier(max_iter=1000, random_state=69, early_stopping=True),
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1",
        cv=cv_strategy,
        random_state=69,
        n_jobs=-1,
    )
    search.fit(X_train_scaled, y_train)
    print("Meilleurs paramètres MLP:", search.best_params_)
    return search.best_params_


if __name__ == "__main__":
    # 1. Préparation des données spécifiques
    X_train, _, y_train, _, X_train_scaled, _, pos_weight = get_prepared_tsunami_data()

    print("--- DÉBUT DU TUNING POUR LA CLASSIFICATION TSUNAMI ---")
    print(f"Poids de compensation calculé pour la classe rare : {pos_weight:.2f}")

    # 2. Exécution des recherches d'hyperparamètres
    best_xgb_config = tune_tsunami_xgb(X_train, y_train, pos_weight)
    best_rf_config = tune_tsunami_random_forest(X_train, y_train)
    best_mlp_config = tune_tsunami_mlp(X_train_scaled, y_train)

    print("\n=======================================================")
    print("Tuning terminé ! Injecte ces paramètres dans models_config_tsunami.py")
    print("=======================================================")