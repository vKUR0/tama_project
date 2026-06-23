# models_config_tsunami.py
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import matplotlib.pyplot as plt


def get_trained_classifiers(scale_pos_weight_value=1.0):
    """Retourne les instances des Classifiers avec leurs hyperparamètres optimisés."""
    return {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        
        # --- RANDOM FOREST OPTIMISÉ ---
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=69,
            n_jobs=-1,
        ),
        
        # --- XGBOOST OPTIMISÉ ---
        "XGBoost Classifier": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=9,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            scale_pos_weight=scale_pos_weight_value,
            eval_metric="logloss",
            random_state=69,
        ),
        
        # --- MLP OPTIMISÉ ---
        "MLP Classifier": MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="tanh",
            solver="adam",
            learning_rate_init=0.05,
            alpha=0.0001,
            max_iter=1000,
            early_stopping=True,
            random_state=69,
        ),
    }
