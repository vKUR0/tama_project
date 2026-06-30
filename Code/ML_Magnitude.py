import os
import time
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

from Tools.data import load_dataset_magnitude, scale_features, encode_250m_mesh
from Models.Models_Magnitude import get_models


def main(data_dir):
    # Construction du chemin vers le fichier CSV
    file_path = os.path.join(data_dir, "dataset_earthquake.csv")

    print("1. Chargement et nettoyage des données...")
    X, y = load_dataset_magnitude(file_path)

    # Découpage Train (80%) / Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    print(
        f"   -> Données d'entraînement : {X_train.shape[0]} lignes | Données de test : {X_test.shape[0]} lignes"
    )

    print("\n2. Entraînement et évaluation des modèles...")
    modeles = get_models()
    resultats = []

    # Boucle d'évaluation
    for nom, model in modeles.items():
        print(f"   - Entraînement en cours pour : {nom}...")
        if nom == "MLP":
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test
        
        # --- MESURE DU TEMPS D'ENTRAÎNEMENT ---
        start_train = time.time()
        model.fit(X_tr, y_train)
        end_train = time.time()
        temps_entrainement = end_train - start_train

        # --- MESURE DU TEMPS DE PRÉDICTION ---
        start_pred = time.time()
        y_pred = model.predict(X_te)
        end_pred = time.time()
        temps_prediction = end_pred - start_pred

        # Calcul des métriques de performance
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Sauvegarde des scores
        resultats.append(
            {
                "Model": nom,
                "RMSE": round(rmse, 3),
                "R² Score": round(r2, 3),
                "Training Time (s)": round(temps_entrainement, 4),
                "Prediction Time (s)": round(temps_prediction, 4),
            }
        )

    # 3. Affichage du tableau comparatif final
    df_performance = pd.DataFrame(resultats)
    print("\n========== PERFORMANCE METRICS ==========")
    print(df_performance.to_string(index=False))
    print("=========================================\n")
    # save xgboost model
    xgb_model = modeles.get("XGBoost Regressor")
    if xgb_model:
        joblib.dump(xgb_model, os.path.join(data_dir, "xgboost_magnitude_model.pkl"))
        print("XGBoost model saved as 'xgboost_magnitude_model.pkl' in the dataset directory.")





if __name__ == "__main__":
    # Point d'entrée du programme avec le chemin vers ton dossier de données
    main("./dataset/")