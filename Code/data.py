import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_dataset(path):
    """Charge le dataset, nettoie les noms de colonnes et sépare les features de la target."""
    df = pd.read_csv(path)
    # --- NETTOYAGE DES NOMS DE COLONNES POUR XGBOOST ---
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
    X = df[["Latitude", "Longitude", "Depth", "Year", "Month" ,"Num_Stations"]]
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

def encode_250m_mesh(lat, lon):
    """
    Calculates the 10-digit 250m grid mesh code (JIS X 0410) from coordinates.
    """
    # Step 1: 1st-Mesh (approx. 80km x 90km)
    p = int(lat * 1.5)
    q = int(lon - 100)
    
    # Step 2: 2nd-Mesh (approx. 10km x 11km)
    lat_rem_2 = (lat * 1.5) - p
    lon_rem_2 = (lon - 100) - q
    r = int(lat_rem_2 * 8)
    s = int(lon_rem_2 * 8)
    
    # Step 3: 3rd-Mesh / 1km Mesh
    lat_rem_3 = (lat_rem_2 * 8) - r
    lon_rem_3 = (lon_rem_2 * 8) - s
    t = int(lat_rem_3 * 10)
    u = int(lon_rem_3 * 10)
    
    # Step 4: 500m Mesh Division (9th digit)
    lat_rem_4 = (lat_rem_3 * 10) - t
    lon_rem_4 = (lon_rem_3 * 10) - u
    
    if lat_rem_4 < 0.5 and lon_rem_4 < 0.5:
        m9 = 1
    elif lat_rem_4 < 0.5 and lon_rem_4 >= 0.5:
        m9 = 2
    elif lat_rem_4 >= 0.5 and lon_rem_4 < 0.5:
        m9 = 3
    else:
        m9 = 4
        
    # Step 5: 250m Mesh Division (10th digit)
    lat_rem_5 = (lat_rem_4 * 2) - int(lat_rem_4 * 2)
    lon_rem_5 = (lon_rem_4 * 2) - int(lon_rem_4 * 2)
    
    if lat_rem_5 < 0.5 and lon_rem_5 < 0.5:
        m10 = 1
    elif lat_rem_5 < 0.5 and lon_rem_5 >= 0.5:
        m10 = 2
    elif lat_rem_5 >= 0.5 and lon_rem_5 < 0.5:
        m10 = 3
    else:
        m10 = 4
        
    # Construct the final 10-digit code string
    mesh_code = f"{p}{q:02d}{r}{s}{t}{u}{m9}{m10}"
    return mesh_code