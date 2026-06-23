from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

from Code.Tools.data import load_dataset

def get_prepared_data(data_path="./dataset/dataset_updated_renamed.csv"):
    """Loads data, splits it into train/test, and returns raw and scaled versions."""
    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale data specifically for Deep Learning (MLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

def tune_xgb(X_train, y_train):
    """Tunes the XGBoost Regressor using RandomizedSearchCV."""
    print("\n>>> Tuning XGBoost Regressor...")
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(random_state=69),
        param_distributions=param_dist,
        n_iter=15,
        scoring="neg_root_mean_squared_error",
        cv=5,
        random_state=69,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print("Best parameters for XGBoost:", search.best_params_)
    return search.best_params_

def tune_random_forest(X_train, y_train):
    """Tunes the Random Forest Regressor using RandomizedSearchCV."""
    print("\n>>> Tuning Random Forest...")
    param_dist = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [1.0, "sqrt", "log2"],
    }

    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=69),
        param_distributions=param_dist,
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        cv=5,
        random_state=69,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print("Best parameters for Random Forest:", search.best_params_)
    return search.best_params_


def tune_mlp(X_train_scaled, y_train):
    """Tunes the MLP Regressor (Deep Learning) using SCALED data."""
    print("\n>>> Tuning MLP (Deep Learning)...")
    param_dist = {
        "hidden_layer_sizes": [
            (64, 32, 16),
            (100, 50),
            (128, 64, 32),
            (50, 25, 12),
        ],
        "activation": ["tanh", "relu"],
        "solver": ["sgd", "adam"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01, 0.1],
    }

    search = RandomizedSearchCV(
        estimator=MLPRegressor(max_iter=1000, random_state=69),
        param_distributions=param_dist,
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        cv=5,
        random_state=69,
        n_jobs=-1,
    )
    search.fit(X_train_scaled, y_train)
    print("Best parameters for MLP:", search.best_params_)
    return search.best_params_

def main():
    X_train, _, y_train, _, X_train_scaled, _ = get_prepared_data()

    print("--- STARTING TARGETED TUNING PROCESS ---")

    # 2. Call individual tuning functions sequentially
    best_xgb_params = tune_xgb(X_train, y_train)
    best_rf_params = tune_random_forest(X_train, y_train)
    best_mlp_params = tune_mlp(X_train_scaled, y_train)

    print("\n==========================================")
    print("All tuning processes finished successfully!")
    print("==========================================")

if __name__ == "__main__":
    main()