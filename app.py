import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from streamlit_folium import st_folium
import folium 
import sys
sys.path.append('..')
from utils.funciones3 import distancia_minima_turistica

# Cargar el modelo previamente entrenado
modelDTree = joblib.load('../modelos/regresion/decisionTree_AirbnbNY_V1.pkl')
modelDtreeCcp = joblib.load('../modelos/regresion/decisionTree_Ccp_AirbnbNY_V1.pkl')
modelRF = joblib.load('../modelos/regresion/randomForest_AirbnbNY_V1.pkl')
modelGrdB = joblib.load('../modelos/regresion/gradientBoosting_AirbnbNY_V1.pkl')
modelXGB = joblib.load('../modelos/regresion/ExtremegradientBoosting_AirbnbNY_V1.pkl')

models = [modelDTree, modelDtreeCcp, modelRF, modelGrdB, modelXGB]

#Cargar el scaler si es necesario
scaler = joblib.load('../modelos/scaler/MinMaxFull_Airbnb_V1.pkl')
# Orden scaler
columns_scaler = ['latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms',
       'beds', 'review_scores_value', 'license', 'instant_bookable',
       'amenity_tv', 'amenity_parking', 'amenity_dryer',
       'amenity_hair_dryer', 'amenity_cleaning_products',
       'amenity_washer', 'amenity_luxurious', 'distance_min',
       'is_entire_home']

def predecir_precio(input_data, model, scaler=None):
    if scaler:
        input_data = scaler.transform(input_data)
    # Aquí se puede elegir el modelo que se desea utilizar para la predicción
    prediction = model.predict(input_data)
    return float(prediction)

def promedio_precio(input_data, models, scaler=None):
    predictions = [predecir_precio(input_data, model, scaler) for model in models]
    return np.max(predictions),predictions

# Titulo de la aplicación
st.title('Predicción de precios de alojamientos en Airbnb')
st.write("Esta aplicación predice el precio de un alojamiento en Airbnb basado en ciertas características.")

st.write("Seleccione la ubicación del alojamiento haciendo clic en el mapa:")

# Coordenadas aproximadas del rectángulo que cubre NYC
ny_bounds = [[40.4774, -74.2591], [40.9176, -73.7004]]  # [Suroeste, Noreste]

# Crear mapa centrado en NYC
m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, max_bounds=True)

# Agregar un rectángulo que muestre los límites
folium.Rectangle(bounds=ny_bounds, color="blue", fill=True, fill_opacity=0.05).add_to(m)

# Mostrar el mapa en Streamlit
mapa = st_folium(m, width=700, height=500)

# Validar que el clic esté dentro del área
if mapa and mapa["last_clicked"]:
    lat = mapa["last_clicked"]["lat"]
    lon = mapa["last_clicked"]["lng"]

    if (ny_bounds[0][0] <= lat <= ny_bounds[1][0]) and (ny_bounds[0][1] <= lon <= ny_bounds[1][1]):
        st.success(f"✅ Coordenadas dentro de NYC: Latitud = {lat:.6f}, Longitud = {lon:.6f}")
    else:
        st.error("⚠️ El punto seleccionado está fuera de los límites de la ciudad.")
else:
    st.info("Haz clic en el mapa para obtener las coordenadas.")
    st.warning("Selecciona una ubicación en el mapa antes de predecir.")
    st.stop()


st.subheader('Ingrese las características generales del alojamiento:')


is_entire_home = st.selectbox("Tipo de alojamiento", ["Alojamiento completo", "Cuarto privado"])
if is_entire_home == "Alojamiento completo":
    is_entire_home = True
else:
    is_entire_home = False

review_scores_value = st.number_input("Calificación del alojamiento", min_value=0.0, max_value=5.0, step=0.1, value=4.0)

colA, colB = st.columns(2)

with colA:
    accommodates = st.number_input("Número de huespedes", min_value=1, max_value=16, value=2)
    bathrooms = st.number_input("Baños", min_value=0, max_value=5, step=1, value=1)
    license = st.checkbox("Cuenta con licencia?")

with colB:
    bedrooms = st.number_input("Habitaciones", min_value=0, max_value=7, step=1, value=1)
    beds = st.number_input("Camas", min_value=1, max_value=10, step=1, value=2)
    instant_bookable = st.checkbox("Reserva instantanea?")

st.subheader('Ingrese las comodidades del alojamiento:')
col1, col2 = st.columns(2)

with col1:
    amenity_tv = st.checkbox("TV")
    amenity_hair_dryer = st.checkbox("Secador de cabello")
    amenity_parking = st.checkbox("Estacionamiento")
    amenity_cleaning_products = st.checkbox("Productos de limpieza")

with col2:
    amenity_washer = st.checkbox("Lavadora")
    amenity_dryer = st.checkbox("Secadora")
    amenity_luxurious = st.checkbox("Comodidades lujosas")

input_data = pd.DataFrame([{
    'is_entire_home': is_entire_home,
    'accommodates': accommodates,
    'bathrooms': bathrooms,
    'bedrooms': bedrooms,
    'beds': beds,
    'review_scores_value': review_scores_value,
    'license': license,
    'instant_bookable': instant_bookable,
    'amenity_tv': amenity_tv,
    'amenity_hair_dryer': amenity_hair_dryer,
    'amenity_parking': amenity_parking,
    'amenity_cleaning_products': amenity_cleaning_products,
    'amenity_washer': amenity_washer,
    'amenity_dryer': amenity_dryer,
    'amenity_luxurious': amenity_luxurious,
    'latitude': lat,
    'longitude': lon,
    'distance_min': distancia_minima_turistica(lat, lon)
}])

input_data = input_data[columns_scaler]

# Predict button
if st.button("Predecir precio"):
    prediction, predictions = promedio_precio(input_data, models, scaler )
    prediction  = round(np.exp(prediction))  # Convertir de log-precio a precio real
    
    # Guardamos en sesión
    st.session_state.prediction = prediction
    st.session_state.predictions = predictions
    
    st.write(f"El precio predecido para el alquiler de esta propiedad en Airbnb es ${prediction}.")
    
if "prediction" in st.session_state:
    if st.checkbox('Mostrar resultados por modelo'):
        st.write(f"El precio predecido para el alquiler de esta propiedad en Airbnb es ${st.session_state.prediction}.")
        st.table(pd.DataFrame({
            'Modelo': models,
            'Predicción ($)': [np.exp(pred) for pred in st.session_state.predictions]
        }))
    
