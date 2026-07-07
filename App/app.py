# app.py
import streamlit as st
import folium
from streamlit_folium import st_folium
from streamlit_extras.let_it_rain import rain
import datetime
from pathlib import Path

from Calc.predictor import predict_earthquake_events
from Calc.feature import load_cached_seismic_history, get_time_last_EQ



# 1. On trouve où est situé le fichier app.py actuel (App/)
CURRENT_DIR = Path(__file__).resolve().parent

# 2. On reconstruit proprement le chemin vers le fichier CSV
# Cela va donner : /mount/src/tama_project/App/Calc/dataset/dataset_earthquake.csv
DATASET_PATH = CURRENT_DIR / "Calc" / "dataset" / "dataset_earthquake.csv"

# 3. Chargement sécurisé avec vérification
if "df_history" not in st.session_state:
    if not DATASET_PATH.exists():
        st.error(f"❌ Fichier historique introuvable au chemin construit : {DATASET_PATH}")
        st.stop() # Arrête proprement l'application avec un message explicite
        
    st.session_state.df_history = load_cached_seismic_history(str(DATASET_PATH))

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