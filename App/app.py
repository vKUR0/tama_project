# app.py
import streamlit as st
import folium
from streamlit_folium import st_folium
from streamlit_extras.let_it_rain import rain
import datetime
from pathlib import Path

from Calc.predictor import predict_earthquake_events
from Calc.feature import load_cached_seismic_history, get_time_last_EQ



# 1. On part de la racine du projet (/mount/src/tama_project)
ROOT_DIR = Path(__file__).resolve().parents[1]

# 2. On cherche de manière récursive tous les fichiers .csv qui contiennent "earthquake"
csv_files = list(ROOT_DIR.rglob("*earthquake*.csv"))

if "df_history" not in st.session_state:
    if csv_files:
        # On prend le premier fichier trouvé qui correspond
        DETECTED_PATH = csv_files[0]
        st.success(f"🔍 Fichier détecté automatiquement : `{DETECTED_PATH.relative_to(ROOT_DIR)}`")
        st.session_state.df_history = load_cached_seismic_history(
            str(DETECTED_PATH)
        )
    else:
        # Si vraiment aucun fichier ne correspond, on affiche le contenu du projet pour comprendre
        st.error(
            "❌ Aucun fichier contenant 'earthquake' et se terminant par '.csv' n'a été trouvé."
        )

        # Petit outil de debug pour lister ce qu'il y a dans App/Calc/
        calc_dir = ROOT_DIR / "App" / "Calc"
        if calc_dir.exists():
            st.write(
                "Contenu du dossier `App/Calc/` :", list(calc_dir.iterdir())
            )
        else:
            st.write(
                "Le dossier `App/Calc/` n'existe pas à la racine. Dossiers disponibles :",
                list(ROOT_DIR.iterdir()),
            )

        st.stop()

st.title("Tama University - Seismic Event Predictor")
st.write("Enter the characteristics of the event to estimate its power and associated risks.")

st.markdown("---")

st.header("Paramètres du Séisme")

# Entrées Géographiques
col1, col2, col3 = st.columns(3)

lat_input = float(col1.text_input("Latitude", value="35.00"))
lon_input = float(col2.text_input("Longitude", value="140.00"))
depth_input = float(col3.text_input("Depth Km", value="25.0"))

date_input = col1.date_input("Event Date", datetime.date.today())
time_input = col2.time_input("Event Time", datetime.time(12, 0))
nb_station_input = float(col3.text_input("Number of Stations", value="100"))


# Extraction des composants temporels requis par tes modèles
year = date_input.year
month = date_input.month
hour = time_input.hour

st.markdown("---")

lat = float(lat_input)
lon = float(lon_input)
depth = float(depth_input)
nb_station = int(float(nb_station_input))
target_datetime = datetime.datetime.combine(date_input, time_input)
time_since_eq = get_time_last_EQ(
    lat, lon, target_datetime, st.session_state.df_history
)

# Calcul des prédictions via notre fichier predictor.py
magnitude, tsunami = predict_earthquake_events(
    float(lat),
    float(lon),
    float(depth),
    year,
    month,
    hour,
    int(nb_station),
    time_since_eq
)

# Affichage des résultats sous forme de cartes d'indicateurs (Metrics)
col_m1, col_m2 = st.columns(2)
col_m1.header("Predictions")
col_m1.metric(label="Predicted Magnitude", value=f"{magnitude:.2f}")
if tsunami == 1:
    col_m1.error("Tsunami Alert : YES")
else:
    col_m1.success("Tsunami Alert : NO")

col_m2.header("Event Location")
col_m2.metric(label="Localisation", value="Lat: {:.2f}, Lon: {:.2f}".format(lat, lon))

with col_m2:
    # Création d'une carte Folium centrée sur les coordonnées saisies
    m = folium.Map(location=[lat, lon], zoom_start=6)

    # Un marqueur rouge pour l'épicentre
    folium.Marker(
        [lat, lon], 
        popup=f"Épicentre\nMagnitude: {magnitude:.2f}", 
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    # Affichage de la carte dans Streamlit
    st_folium(m, width=800, height=450)

if st.button("Send balloons!"):
    st.balloons()

if st.button("It's snow time!"):
    st.snow()

if st.button("Make it rain!"):
    rain(
        emoji="❤️",  # Emoji de pluie
        font_size=54,
        falling_speed=5,
        animation_length=2,
    )