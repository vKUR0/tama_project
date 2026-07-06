import pandas as pd
import streamlit as st


@st.cache_data
def load_cached_seismic_history(path):
    """Charge le dataset historique UNE SEULE FOIS et pré-calcule les index spatio-temporels."""
    df = pd.read_csv(path)
    df["Origin_Time"] = pd.to_datetime(df["Origin_Time"])
    df["Lat_Grid"] = df["Latitude"].round(1)
    df["Lon_Grid"] = df["Longitude"].round(1)
    # Trier par temps pour accélérer les recherches futures
    df = df.sort_values("Origin_Time")
    return df


def get_time_last_EQ(lat, lon, target_datetime, df_history):
    """Calcule instantanément le temps écoulé (en secondes) depuis le dernier séisme

    dans la même grille, par rapport à une date/heure cible complète.
    """
    # 1. Arrondir les coordonnées cibles pour correspondre à la grille
    target_lat_grid = round(lat, 1)
    target_lon_grid = round(lon, 1)

    # 2. Convertir l'entrée en Timestamp Pandas complet (avec jour/minute/seconde)
    target_ts = pd.to_datetime(target_datetime)

    # 3. Filtrer en une seule ligne : même endroit ET dans le passé
    mask = (
        (df_history["Lat_Grid"] == target_lat_grid)
        & (df_history["Lon_Grid"] == target_lon_grid)
        & (df_history["Origin_Time"] < target_ts)
    )

    matched_events = df_history[mask]

    # 4. Si aucun séisme passé n'est trouvé dans cette maille
    if matched_events.empty:
        return -1.0  # Valeur par défaut (comme ton fillna(-1) du tuning)

    # 5. Trouver le plus récent dans le passé
    last_eq_time = matched_events["Origin_Time"].max()

    # 6. Calcul du delta en secondes
    time_since_last_eq = (target_ts - last_eq_time).total_seconds()

    return float(time_since_last_eq)