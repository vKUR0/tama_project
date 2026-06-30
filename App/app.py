# app.py
import streamlit as st
import folium
from streamlit_folium import st_folium
from streamlit_extras.let_it_rain import rain
import datetime

# Import de notre fonction de prédiction
from Calc.predictor import predict_earthquake_events

# Configuration de la page
st.set_page_config(page_title="Seismic Predictor Dashboard", layout="wide")

st.title("🌋 Système d'Analyse et de Prédiction Sismique")
st.write("Entrez les caractéristiques de l'événement pour estimer sa puissance et les risques associés.")

st.markdown("---")

# Création de deux colonnes : Gauche pour les entrées, Droite pour les résultats et la carte
col_inputs, col_outputs = st.columns([1, 2])

with col_inputs:
    st.header("Paramètres du Séisme")
    
    # Entrées Géographiques
    lat = st.number_input("Latitude", value=67.0, min_value=-90.0, max_value=90.0, step=0.01)
    lon = st.number_input("Longitude", value=67.0, min_value=-180.0, max_value=180.0, step=0.01)
    depth = st.number_input("Profondeur Km", min_value=0.0, max_value=700.0, value=25.0)
    
    # Entrées Temporelles
    date_input = st.date_input("Date de l'événement", datetime.date.today())
    time_input = st.time_input("Heure de l'événement", datetime.time(12, 0))
    time_since_eq = st.number_input("Temps depuis le dernier séisme régional (s)", value=3600)

    # Extraction des composants temporels requis par tes modèles
    year = date_input.year
    month = date_input.month
    hour = time_input.hour

with col_outputs:
    st.header("Prédictions des Modèles IA")
    
    # Calcul des prédictions via notre fichier predictor.py
    magnitude, tsunami= predict_earthquake_events(lat, lon, depth, year, month, hour, time_since_eq)
    
    # Affichage des résultats sous forme de cartes d'indicateurs (Metrics)
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric(label="Magnitude Estimée (XGBoost)", value=f"{magnitude} Mw")
    
    if tsunami == 1:
        col_m2.error("Alerte Tsunami : OUI")
    else:
        col_m2.success("Alerte Tsunami : NON")
        

    st.write("### Localisation de l'Épicentre et Zone Impactée")
    
    # Création d'une carte Folium centrée sur les coordonnées saisies
    m = folium.Map(location=[lat, lon], zoom_start=6)
    
    # Un marqueur rouge pour l'épicentre
    folium.Marker(
        [lat, lon], 
        popup=f"Épicentre\nMagnitude: {magnitude}", 
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    
            
    # Affichage de la carte dans Streamlit
    st_folium(m, width=800, height=450)

if st.button("Send balloons!"):
    st.balloons()

if st.button("It's snow time!"):
    st.snow()  # Streamlit n'a pas de confetti, mais on peut utiliser snow pour un effet festif

if st.button("Make it rain!"):
    rain(
        emoji="🖕",  # Emoji de pluie
        font_size=54,
        falling_speed=5,
        animation_length=2,
    )
