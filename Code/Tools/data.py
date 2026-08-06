import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_dataset_seismic_intensity(path):
    """Loads the dataset, cleans column names, engineers historical features,

    includes soil/mesh characteristics (AVS, ARV, Mesh_Code), and splits X and y.
    """
    df = pd.read_csv(path)

    # --- CLEAN COLUMN NAMES FOR XGBOOST ---
    df.columns = (
        df.columns.str.replace("[", "_", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("<", "", regex=False)
    )

    # 1. Parse and preserve the original timestamp for calculations
    df["Timestamp"] = pd.to_datetime(df["Origin_Time"])

    # Extract time components
    df["Year"] = df["Timestamp"].dt.year
    df["Month"] = df["Timestamp"].dt.month
    df["Hour"] = df["Timestamp"].dt.hour

    # Drop the original string column
    if "Origin_Time" in df.columns:
        df = df.drop("Origin_Time", axis=1)

    # 2. Historical Feature Engineering (Spatio-Temporal Grid)
    df["Lat_Grid"] = df["Latitude"].round(1)
    df["Lon_Grid"] = df["Longitude"].round(1)

    df["Time_Since_Last_EQ"] = (
        df.groupby(["Lat_Grid", "Lon_Grid"])["Timestamp"]
        .diff()
        .dt.total_seconds()
    )
    df["Time_Since_Last_EQ"] = df["Time_Since_Last_EQ"].fillna(-1)

    # 3. Clean soil/mesh columns if present
    for col in ["AVS", "ARV", "Mesh_Code"]:
        if col in df.columns:
            df[col] = df[col].fillna(-1.0)

    # --- SEPARATING FEATURES / TARGET ---
    # Included soil characteristics (AVS, ARV, Mesh_Code) for enhanced prediction accuracy
    feature_cols = [
        "Latitude",
        "Longitude",
        "Depth",
        "Magnitude",
        "Num_Stations",
        "Time_Since_Last_EQ",
        "Mesh_Code",
        "AVS",
        "ARV",
        "Is_Offshore",
    ]

    # Filter features that are present in the dataset
    available_features = [col for col in feature_cols if col in df.columns]

    X = df[available_features]
    y = df["Seismic_Intensity"]

    return X, y

def load_dataset_magnitude(path):
    """Loads the dataset, cleans column names, engineers historical features, and splits X and y."""
    df = pd.read_csv(path)
    
    # --- CLEAN COLUMN NAMES FOR XGBOOST ---
    df.columns = (
        df.columns.str.replace("[", "_", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("<", "", regex=False)
    )

    # 1. Parse and preserve the original timestamp for calculations
    df["Timestamp"] = pd.to_datetime(df["Origin_Time"])
    
    # Extract time components
    df["Year"] = df["Timestamp"].dt.year
    df["Month"] = df["Timestamp"].dt.month
    df["Hour"] = df["Timestamp"].dt.hour
    
    # Drop the original string column
    if "Origin_Time" in df.columns:
        df = df.drop("Origin_Time", axis=1)


    df['Lat_Grid'] = df['Latitude'].round(1)
    df['Lon_Grid'] = df['Longitude'].round(1)

    df['Time_Since_Last_EQ'] = df.groupby(['Lat_Grid', 'Lon_Grid'])['Timestamp'].diff().dt.total_seconds()
    df['Time_Since_Last_EQ'] = df['Time_Since_Last_EQ'].fillna(-1)

    # --- SEPARATING FEATURES / TARGET ---
    # Define explicitly which columns go into your Machine Learning model
    feature_cols = ["Latitude", "Longitude", "Depth", "Year", "Month", "Hour", "Num_Stations",  "Time_Since_Last_EQ"]
    
    X = df[feature_cols]
    y = df["Magnitude"]

    return X, y

def load_dataset_tsunami(path):
    df = pd.read_csv(path)

    # Clean column names
    df.columns = (
        df.columns.str.replace("[", "_", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("<", "", regex=False)
    )

    # Rename Tsunami column if it's named Tsunami_ID in the CSV
    if "Tsunami_ID" in df.columns:
        df["Tsunami"] = df["Tsunami_ID"].apply(lambda x: 1 if x > 0 else 0)

    df["Timestamp"] = pd.to_datetime(df["Origin_Time"])
    df["Year"] = df["Timestamp"].dt.year
    df["Month"] = df["Timestamp"].dt.month
    df["Hour"] = df["Timestamp"].dt.hour

    df["Lat_Grid"] = df["Latitude"].round(1)
    df["Lon_Grid"] = df["Longitude"].round(1)

    df = df.sort_values(by="Timestamp").reset_index(drop=True)

    df["Time_Since_Last_EQ"] = (
        df.groupby(["Lat_Grid", "Lon_Grid"])["Timestamp"]
        .diff()
        .dt.total_seconds()
    )
    df["Time_Since_Last_EQ"] = df["Time_Since_Last_EQ"].fillna(-1)

    feature_cols = [
        "Latitude",
        "Longitude",
        "Depth",
        "Magnitude", 
        "Year",
        "Month",
        "Hour",
        "Time_Since_Last_EQ",
    ]

    X = df[feature_cols]
    y = df["Tsunami"]

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