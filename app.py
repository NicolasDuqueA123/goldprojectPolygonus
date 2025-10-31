import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.serialize import model_from_json
# Para este código, asumiremos que se guardó el modelo ARIMA con joblib/pickle como un objeto "fit".
from tensorflow.keras.models import load_model

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---

# Configura la página
st.set_page_config(layout="wide", page_title="Predicción Precio del Oro")

# Cargar los datos (con caché para que solo se carguen una vez)
@st.cache_data
def load_data():
    # Asegúrate de que este nombre de archivo coincida con tu CSV en GitHub
    try:
        df = pd.read_csv('XAU_1d_data_V2.csv', sep=None, engine='python')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.columns = df.columns.str.strip()
        # Limpiar filas con NaN en el precio
        df.dropna(subset=['Close'], inplace=True) 
        df.sort_values('Date', inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: Archivo 'XAU_1d_data_V2.csv' no encontrado. Asegúrate de que esté en la raíz de tu repositorio de GitHub.")
        return pd.DataFrame()

# Cargar los modelos (con caché para que solo se carguen una vez)
@st.cache_resource
def load_all_models():
    models = {}
    
    # Intenta cargar cada modelo; si alguno falta, muestra un error específico.
    try:
        models['linear'] = joblib.load('linear_model.joblib')
        models['arima'] = joblib.load('arima_model.pkl')
        
        with open('prophet_model.json', 'r') as fin:
            models['prophet'] = model_from_json(fin.read())
            
        models['cnn'] = load_model('cnn_model.keras')
        models['cnn_scaler'] = joblib.load('cnn_scaler.joblib')
        
    except FileNotFoundError as e:
        st.error(f"Error al cargar un archivo de modelo: {e}. Asegúrate de que todos los archivos (.joblib, .pkl, .json, .keras) estén en la raíz del repositorio.")
        return None
    except Exception as e:
        st.error(f"Error inesperado al cargar modelos: {e}")
        return None
    
    return models

# Cargar todo al inicio
df = load_data()
if df.empty:
    st.stop()
    
models = load_all_models()
if models is None:
    st.stop()


# --- 2. INTERFAZ DE PESTAÑAS ---

st.title("📈 Proyecto de Pronóstico del Precio del Oro")

# Definición de las 2 pestañas
tab1, tab2 = st.tabs(["Introducción y Análisis", "Hacer Predicción"])


# --- PESTAÑA 1: INTRODUCCIÓN Y ANÁLISIS (Nueva estructura) ---
with tab1:
    st.header("Análisis de Modelos y Datos Históricos")
    col1, col2 = st.columns(2)
    
    # COLUMNA 1: Introducción y Modelos
    with col1:
        st.write("""
        Esta es una aplicación web interactiva construida con Streamlit para presentar 
        un proyecto de pronóstico del precio del oro (XAU/USD) creada por Nicolás Duque Aguirre. 
        
        El objetivo es implementar, comparar y visualizar el rendimiento de 
        diferentes modelos de series temporales y machine learning.
        
        **Modelos Implementados:**
        * Regresión Lineal Simple
        * Red Neuronal Convolucional (CNN 1D)
        * ARIMA (Autoregressive Integrated Moving Average)
        * Prophet (de Meta)
        
        """)
        st.image('gold.png')
    
    # COLUMNA 2: Gráfica y Tabla de Métricas (Contenido movido y CORREGIDO)
    with col2:
        # Gráfica de precios históricos
        st.subheader("Precio Histórico del Oro (XAU)")
        fig_hist = px.line(df, x='Date', y='Close', title='Precio de Cierre (Close) - Serie temporal')
        st.plotly_chart(fig_hist, use_container_width=True)

        # Tabla de Métricas (AÑADIDA AQUÍ Y REVISADA)
        st.subheader("Métricas de Rendimiento (Evaluación del Notebook)")
        metrics_data = {
            # 5 elementos
            'Modelo': ['Regresión Lineal*', 'CNN 1D', 'ARIMA', 'Prophet*', 'Híbrido (P+CNN)*'], 
            # 5 elementos
            'Métrica Principal': ['R²: 0.7316', 'R²: 0.9311', 'R²: -0.6166', 'R²: 0.9942', 'R²: 0.9372'], 
            # 5 elementos
            'Nota': ['Sobreajustado', 'Realista', 'Requiere ajuste', 'Sobreajustado', 'Sobreajustado'] 
        }
        
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
        st.warning("*Nota: Las métricas marcadas con '*' estaban sobreajustadas (evaluadas en datos de entrenamiento) en el notebook original. La métrica R² de la CNN es la más confiable.")


# --- PESTAÑA 2: PREDICCIÓN EN VIVO (Antigua tab3) ---
with tab2:
    st.header("Generador de Pronóstico")
    st.write("Selecciona un modelo y la cantidad de días que deseas proyectar.")

    model_choice = st.selectbox("Elige un modelo:", ["Prophet", "ARIMA", "Regresión Lineal", "CNN"])
    days_to_forecast = st.number_input("Días a proyectar (1-365):", min_value=1, max_value=365, value=30)
    
    if st.button("Generar Predicción"):
        with st.spinner(f"Cargando {model_choice} y generando pronóstico..."):
            
            # Obtener el último precio conocido
            last_price = df['Close'].iloc[-1]
            last_date = df['Date'].iloc[-1]
            pred_value = 0
            
            # Crear un nuevo índice de fechas para el pronóstico
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days_to_forecast + 1)]
            
            # --- Lógica de predicción ---
            
            if model_choice == 'Prophet':
                future_df = models['prophet'].make_future_dataframe(periods=days_to_forecast)
                forecast = models['prophet'].predict(future_df)
                pred_value = forecast.iloc[-1]['yhat']
                
                fig = models['prophet'].plot(forecast)
                fig.gca().axhline(y=last_price, color='red', linestyle='--', lw=2, label=f'Último Precio: ${last_price:.2f}')
                st.pyplot(fig)
            
            elif model_choice == 'ARIMA':
                # Asumiendo que arima_model.pkl es el objeto .fit() de statsmodels
                forecast_series = models['arima'].predict(start=len(df), end=len(df) + days_to_forecast - 1)
                forecast = pd.Series(forecast_series.values, index=future_dates)
                pred_value = forecast.iloc[-1]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df['Date'].iloc[-200:], df['Close'].iloc[-200:], label='Datos Históricos')
                ax.plot(future_dates, forecast.values, label='Pronóstico ARIMA', linestyle='--')
                ax.axhline(y=last_price, color='red', linestyle='--', lw=2, label=f'Último Precio: ${last_price:.2f}')
                ax.legend()
                st.pyplot(fig)
            
            elif model_choice == 'Regresión Lineal':
                last_date_ordinal = df['Date'].map(pd.Timestamp.toordinal).iloc[-1]
                future_dates_ordinal = np.array([last_date_ordinal + i for i in range(1, days_to_forecast + 1)]).reshape(-1, 1)
                
                forecast_values = models['linear'].predict(future_dates_ordinal)
                pred_value = forecast_values[-1]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df['Date'].iloc[-200:], df['Close'].iloc[-200:], label='Datos Históricos')
                ax.plot(future_dates, forecast_values, label='Pronóstico Lineal', linestyle='--')
                ax.axhline(y=last_price, color='red', linestyle='--', lw=2, label=f'Último Precio: ${last_price:.2f}')
                ax.legend()
                st.pyplot(fig)

            elif model_choice == 'CNN':
                # La CNN usa una ventana de tiempo (LOOKBACK), asumimos 30 días
                LOOKBACK = 30 
                scaler = models['cnn_scaler']
                model_cnn = models['cnn']
                
                # Obtener la última secuencia de datos. Manejo de error si los datos no son suficientes
                if len(df) < LOOKBACK:
                    st.error(f"Datos insuficientes. Se necesitan {LOOKBACK} días de datos históricos para la CNN.")
                    st.stop()
                    
                last_sequence_raw = df['Close'].iloc[-LOOKBACK:].values.reshape(-1, 1)
                last_sequence_scaled = scaler.transform(last_sequence_raw)
                
                current_batch = last_sequence_scaled.reshape((1, LOOKBACK, 1))
                future_predictions_scaled = []
                
                # Walk-forward prediction
                for i in range(days_to_forecast):
                    next_pred_scaled = model_cnn.predict(current_batch, verbose=0)[0]
                    future_predictions_scaled.append(next_pred_scaled)
                    # Mover la ventana de tiempo para incluir la nueva predicción
                    current_batch = np.append(current_batch[:, 1:, :], [[next_pred_scaled]], axis=1)
                
                # Des-escalar
                forecast_values = scaler.inverse_transform(future_predictions_scaled)
                pred_value = forecast_values[-1][0]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df['Date'].iloc[-200:], df['Close'].iloc[-200:], label='Datos Históricos')
                ax.plot(future_dates, forecast_values, label='Pronóstico CNN', linestyle='--')
                ax.axhline(y=last_price, color='red', linestyle='--', lw=2, label=f'Último Precio: ${last_price:.2f}')
                ax.legend()
                st.pyplot(fig)

            # --- Mostrar la predicción final ---
            st.subheader(f"Predicción de {model_choice} a {days_to_forecast} días:")
            st.metric(label=f"Precio estimado del día {days_to_forecast} ({future_dates[-1].strftime('%Y-%m-%d')})", 
                      value=f"${pred_value:,.2f}",
                      delta=f"${pred_value - last_price:,.2f} vs último precio")
