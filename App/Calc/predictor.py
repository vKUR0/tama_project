# predictor.py
import numpy as np
import joblib

# Simulation de chargement (remplace par tes vrais fichiers .joblib)
try:
    model_magnitude = joblib.load("App/Saved_Models/xgboost_magnitude_model.pkl")
    model_tsunami = joblib.load("App/Saved_Models/xgboost_tsunami_model.pkl")
except:
    # Fallback si tes fichiers ne sont pas encore créés
    model_magnitude = None
    model_tsunami = None

def predict_earthquake_events(lat, lon, depth, year, month, hour, time_since_eq):
    """Prend les entrées de l'utilisateur et renvoie la magnitude et le risque de tsunami."""
    
    # 1. Préparation du vecteur de caractéristiques (doit suivre EXACTEMENT l'ordre de tes colonnes)
    # Exemple pour la magnitude : ["Latitude", "Longitude", "Depth", "Year", "Month", "Hour", "Num_Stations", "Time_Since_Last_EQ"]
    features_reg = [[lat, lon, depth, year, month, hour, 50, time_since_eq]] # 50 stations par défaut
    
    # Exemple pour le tsunami : ["Latitude", "Longitude", "Depth", "Magnitude", "Year", "Month", "Hour", "Time_Since_Last_EQ"]
    
    # 2. Prédiction de la Magnitude
    if model_magnitude:
        pred_magnitude = model_magnitude.predict(features_reg)[0]
    else:
        # Simulation mathématique réaliste si pas de modèle chargé
        pred_magnitude = 3.5 + (depth * 0.01) + np.random.uniform(0, 2)
    
    # 3. Prédiction du Tsunami (en incluant la magnitude prédite !)
    features_clf = [[lat, lon, depth, pred_magnitude, year, month, hour, time_since_eq]]
    if model_tsunami:
        pred_tsunami = model_tsunami.predict(features_clf)[0]
    else:
        pred_tsunami = 1 if pred_magnitude > 6.5 and depth < 30 else 0


    return round(pred_magnitude, 2), pred_tsunami