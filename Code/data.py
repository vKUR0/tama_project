import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_dataset(path):
    df = pd.read_csv(path)

    # --- NETTOYAGE DES NOMS DE COLONNES POUR XGBOOST ---
    # On remplace '[', ']' et '<' par rien ou un underscore
    df.columns = (
        df.columns.str.replace("[", "_", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("<", "", regex=False)
    )
    # Résultat : "Max_Acc[gal]" devient "Max_Acc_gal"

    # Gestion de la date
    df["Origin_Time"] = pd.to_datetime(df["Origin_Time"])
    df["Year"] = df["Origin_Time"].dt.year
    df["Month"] = df["Origin_Time"].dt.month
    df["Hour"] = df["Origin_Time"].dt.hour
    df = df.drop("Origin_Time", axis=1)

    # Séparation Features / Target (Le nom de "Magnitude" n'a pas bougé)
    # X contient uniquement les colonnes : Latitude,Longitude,Depth,Year,Month,Num_Stations
    X = df[["Latitude", "Longitude", "Depth", "Year", "Month", "Num_Stations"]]
    y = df["Magnitude"]

    return X, y

def scale_features(X_train, X_test):
    """Normalise les données d'entrée (indispensable pour le Deep Learning)."""
    scaler = StandardScaler()

    # On apprend la moyenne et l'écart-type sur le train et on applique
    X_train_scaled = scaler.fit_transform(X_train)

    # On applique la MEME transformation sur le test (sans ré-apprendre !)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled