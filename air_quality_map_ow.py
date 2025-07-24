import geopandas as gpd
import folium
import requests
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import BallTree
import streamlit as st
from streamlit_folium import st_folium
import branca.colormap as cm
from datetime import datetime, timedelta

# CONFIG
st.set_page_config(page_title="Chicago AQI Map", layout="wide")
SHAPEFILE_PATH = "neighborhoods_shapefile.shp"
API_KEY = "48b8cf776845b1b3b76e183c60826568"

# Load shapefile
@st.cache_resource
def load_neighborhoods():
    gdf = gpd.read_file(SHAPEFILE_PATH).rename(columns={'neighborho': 'neighborhood'})
    gdf["neighborhood"] = gdf["neighborhood"].str.strip().str.title()
    gdf["centroid"] = gdf.centroid
    return gdf

# Fetch air pollution current data from OpenWeather
@st.cache_data(ttl=600)
def fetch_current_aqi(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        if "list" in data and len(data["list"]) > 0:
            return data["list"][0]
    return None

# Fetch forecast data from OpenWeather
@st.cache_data(ttl=600)
def fetch_forecast_aqi(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        if "list" in data and len(data["list"]) > 0:
            return data["list"]
    return None

# Fetch historic data from OpenWeather (timestamps in unix UTC)
@st.cache_data(ttl=600)
def fetch_historic_aqi(lat, lon, start_unix, end_unix):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_unix}&end={end_unix}&appid={API_KEY}"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        if "list" in data and len(data["list"]) > 0:
            return data["list"]
    return None

# Convert pollutant name to key in OpenWeather API components
def pollutant_key_map(pollutant):
    mapping = {
        "pm25": "pm2_5",
        "pm10": "pm10",
        "o3": "o3",
        "no2": "no2",
        "so2": "so2",
        "co": "co"
    }
    return mapping.get(pollutant.lower(), pollutant.lower())

# --- UI Header ---
st.markdown("<h1 style='text-align: center; color: #333;'>🌇 Chicago Neighborhood Air Quality Map</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Live air quality and pollutant forecasts by neighborhood, colored by nearest sensor data</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Filters (TOP) ---
pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
col1, col2, col3, col4 = st.columns([1.5, 1.2, 1.3, 2])

with col1:
    selected_pollutants = st.multiselect("Select Pollutants to Show", pollutants, default=['pm25'])
with col2:
    display_mode = st.radio("Data Mode", ["Current", "Forecast", "Historic"], horizontal=True)
with col3:
    color_by = st.selectbox("Color Map By", selected_pollutants)
with col4:
    if display_mode == "Historic":
        start_date = st.date_input("Start Date", datetime.today() - timedelta(days=7), max_value=datetime.today())
        end_date = st.date_input("End Date", datetime.today(), min_value=start_date, max_value=datetime.today())
    else:
        start_date = None
        end_date = None

# Load neighborhoods
neighborhoods = load_neighborhoods()

# Prepare coordinates and BallTree for nearest lookup
coords = np.array([[geom.y, geom.x] for geom in neighborhoods["centroid"]])
tree = BallTree(np.radians(coords), metric='haversine')

# Function to get nearest neighborhood centroid coords (just self here but pattern kept)
def get_nearest_station_coord(point):
    # For this example, use neighborhood centroid directly (since we call API per neighborhood)
    return (point.y, point.x)

# Fetch and assign AQI data per neighborhood
aqi_data_list = []
for idx, row in neighborhoods.iterrows():
    lat, lon = row["centroid"].y, row["centroid"].x
    if display_mode == "Current":
        data = fetch_current_aqi(lat, lon)
    elif display_mode == "Forecast":
        data = fetch_forecast_aqi(lat, lon)
    else:  # Historic
        # Convert dates to unix timestamps
        start_unix = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        end_unix = int(datetime.combine(end_date, datetime.min.time()).timestamp())
        data = fetch_historic_aqi(lat, lon, start_unix, end_unix)
    aqi_data_list.append(data)

neighborhoods["aq_data"] = aqi_data_list

# Extract pollutant value per neighborhood for coloring
def extract_color_val(aq_data):
    if not aq_data:
        return None
    p_key = pollutant_key_map(color_by)
    try:
        if display_mode == "Current":
            return aq_data["components"].get(p_key)
        elif display_mode == "Forecast":
            # Take average of next 24 hours (or all available)
            vals = [entry["components"].get(p_key) for entry in aq_data if p_key in entry["components"]]
            vals = [v for v in vals if v is not None]
            if vals:
                return sum(vals)/len(vals)
        else:  # Historic
            vals = [entry["components"].get(p_key) for entry in aq_data if p_key in entry["components"]]
            vals = [v for v in vals if v is not None]
            if vals:
                return sum(vals)/len(vals)
    except Exception:
        return None
    return None

neighborhoods["color_val"] = neighborhoods["aq_data"].apply(extract_color_val)

# Setup colormap green to red based on available values
valid_vals = neighborhoods["color_val"].dropna()
if not valid_vals.empty:
    min_val, max_val = valid_vals.min(), valid_vals.max()
else:
    min_val, max_val = 0, 1  # fallback

colormap = cm.LinearColormap(
    colors=["green", "yellow", "red"],
    vmin=min_val,
    vmax=max_val,
    caption=f"Air Quality: {color_by.upper()} ({display_mode})"
)

# Create Folium map
m = folium.Map(location=[41.8781, -87.6298], zoom_start=11, tiles="openstreetmap")

for _, row in neighborhoods.iterrows():
    name = row["neighborhood"]
    centroid = row["centroid"]
    geometry = row["geometry"]
    aq_data = row["aq_data"]
    color_val = row["color_val"]

    fill_color = colormap(color_val) if color_val is not None else "gray"

    # Tooltip content
    tooltip_html = f"<b>{name}</b><br>"
    if aq_data:
        if display_mode == "Current":
            tooltip_html += f"AQI (Overall): {aq_data['main']['aqi']}<br>"
            #tooltip_html += f"Updated: {datetime.utcfromtimestamp(aq_data['dt']).strftime('%Y-%m-%d %H:%M UTC')}<br>"
            for p in selected_pollutants:
                p_key = pollutant_key_map(p)
                val = aq_data["components"].get(p_key, "N/A")
                tooltip_html += f"{p.upper()}: {val}<br>"
        else:
            # Forecast or Historic list of entries
            vals_to_show = []
            if isinstance(aq_data, list):
                # Show average of first 24h for forecast or average for historic
                entries = aq_data[:24] if display_mode == "Forecast" else aq_data
                for p in selected_pollutants:
                    p_key = pollutant_key_map(p)
                    vals = [e["components"].get(p_key) for e in entries if p_key in e["components"]]
                    vals = [v for v in vals if v is not None]
                    avg_val = sum(vals)/len(vals) if vals else "N/A"
                    tooltip_html += f"{p.upper()} ({display_mode} avg): {avg_val}<br>"
            else:
                tooltip_html += "No detailed data available.<br>"
    else:
        tooltip_html += "No data available.<br>"

    folium.GeoJson(
        geometry,
        style_function=lambda feature, fill_color=fill_color: {
            'fillColor': fill_color,
            'color': 'black',
            'weight': 1.25,
            'fillOpacity': 0.7,
        },
        tooltip=folium.Tooltip(tooltip_html, sticky=True)
    ).add_to(m)

    folium.map.Marker(
        [centroid.y, centroid.x],
        icon=folium.DivIcon(html=f"<div style='font-size:8pt; color:black'>{name}</div>")
    ).add_to(m)

# Add legend
colormap.add_to(m)

# Show map
st_folium(m, width=1100, height=750)