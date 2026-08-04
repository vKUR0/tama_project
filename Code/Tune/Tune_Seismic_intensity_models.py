import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# 1. Dossier où se trouve ce script (work/Code/Tune/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. On remonte d'UN niveau pour atteindre le dossier "Code" (work/Code/)
CODE_DIR = os.path.dirname(CURRENT_DIR)

# 3. On ajoute le dossier "Code" au chemin de recherche Python
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# 4. Maintenant Python trouve directement Tools/data.py
from Tools.data import load_dataset_seismic_intensity, scale_features


def tune_xgboost(X_train, y_train):
    """Recherche les meilleurs hyperparamètres pour XGBoost Regressor."""
    print("\n>>> 🛠️ Tuning XGBoost Regressor en cours...")

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],   
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=69)

    search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(random_state=69),
        param_distributions=param_dist,
        n_iter=15,  # 15 combinaisons testées
        scoring="neg_root_mean_squared_error",  # Maximise l'opposé du RMSE
        cv=cv_strategy,
        random_state=69,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    print(
        f"  ✅ Meilleurs paramètres XGBoost : {search.best_params_}"
    )
    print(f"  📊 Meilleur RMSE (CV) : {-search.best_score_:.4f}")
    return search.best_params_, search.best_estimator_


def tune_random_forest(X_train, y_train):
    """Recherche les meilleurs hyperparamètres pour Random Forest Regressor."""
    print("\n>>> 🛠️ Tuning Random Forest Regressor en cours...")

    param_dist = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [1.0, "sqrt", "log2"],
    }

    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=69)

    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=69),
        param_distributions=param_dist,
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        cv=cv_strategy,
        random_state=69,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    print(
        f"  ✅ Meilleurs paramètres Random Forest : {search.best_params_}"
    )
    print(f"  📊 Meilleur RMSE (CV) : {-search.best_score_:.4f}")
    return search.best_params_, search.best_estimator_


def tune_mlp(X_train_scaled, y_train):
    """Recherche les meilleurs hyperparamètres pour MLP Regressor (données normalisées)."""
    print("\n>>> 🛠️ Tuning MLP Regressor en cours...")

    param_dist = {
        "hidden_layer_sizes": [(100, 50), (64, 32), (128, 64, 32), (50, 50)],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "sgd"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01, 0.1],
    }

    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=69)

    search = RandomizedSearchCV(
        estimator=MLPRegressor(
            max_iter=1000, early_stopping=True, random_state=69
        ),
        param_distributions=param_dist,
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        cv=cv_strategy,
        random_state=69,
        n_jobs=-1,
    )

    search.fit(X_train_scaled, y_train)
    print(f"  ✅ Meilleurs paramètres MLP : {search.best_params_}")
    print(f"  📊 Meilleur RMSE (CV) : {-search.best_score_:.4f}")
    return search.best_params_, search.best_estimator_


def main():
    file_path = "./dataset/dataset_earthquake_full.csv"

    print("1. Chargement et préparation des données...")
    X, y = load_dataset_seismic_intensity(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    print("\n================ DÉBUT DU TUNING ================")

    # 1. Tuning XGBoost
    best_xgb_params, best_xgb_model = tune_xgboost(X_train, y_train)

    # 2. Tuning Random Forest
    best_rf_params, best_rf_model = tune_random_forest(X_train, y_train)

    # 3. Tuning MLP (sur données scalées)
    best_mlp_params, best_mlp_model = tune_mlp(X_train_scaled, y_train)

    print("\n=================================================")
    print("✨ Tuning terminé avec succès !")
    print("Recopie ces dictionnaires dans ton fichier Models_Magnitude.py.")


if __name__ == "__main__":
    main()