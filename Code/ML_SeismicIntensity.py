import os
import time
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
from matplotlib import pyplot as plt
import xgboost as xgb

from Tools.data import load_dataset_seismic_intensity, scale_features
from Models.Models_Seismic_Intensity import get_models


def main(file_path):
    # Construction du chemin vers le fichier CSV

    print("1. Chargement et nettoyage des données...")
    X, y = load_dataset_seismic_intensity(file_path)

    # Découpage Train (80%) / Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
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
        #Create a scatter plot of the predictions vs the true values
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
        joblib.dump(xgb_model, os.path.join("./Dataset/", "xgboost_magnitude_model.pkl"))
        print("XGBoost model saved as 'xgboost_magnitude_model.pkl' in the dataset directory.")
    # use the xgboost model to do a scatter plot of the predictions vs the true values
    y_pred_xgb = xgb_model.predict(X_test)
    xgb.plot_importance(xgb_model, max_num_features=10, importance_type='weight')
    plt.title("Top 10 Feature Importances for XGBoost Model")
    plt.show()




if __name__ == "__main__":
    # file_path = "./dataset/dataset_earthquake.csv"  # Remplace par le chemin réel vers ton fichier CSV
    # main(file_path)
    file_path = "./dataset/updated2.csv"  # Remplace par le chemin réel vers ton fichier CSV
    main(file_path)
    