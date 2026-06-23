from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor


def get_models():
    """Retourne un dictionnaire contenant les instances des modèles à comparer."""
    return {
        "Dummy": DummyRegressor(strategy="mean"),
        "Random Forest": RandomForestRegressor(
            n_estimators=50,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=69,
            n_jobs=-1),
        "XGBoost Regressor": xgb.XGBRegressor(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.2,
            subsample=0.8,
            colsample_bytree=1.0,
            random_state=69,
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="sgd",
            learning_rate_init=0.1,
            alpha=0.0001,
            max_iter=1000,
            early_stopping=True,
            random_state=69,
        ),
    }