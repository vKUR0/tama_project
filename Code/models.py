from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor


def get_models():
    """Retourne un dictionnaire contenant les instances des modèles à comparer."""
    return {
        "Dummy": DummyRegressor(strategy="mean"),
        "Random Forest": RandomForestRegressor(random_state=69, n_estimators=100),
        "XGBoost Regressor": xgb.XGBRegressor(random_state=69),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            solver="adam",
            max_iter=1000,  # Nombre d'époques max pour converger
            random_state=42,
        ),
    }