import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.serialize import model_from_json
from statsmodels.tsa.arima.model import ARIMAResults
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---

# Configura la página (esto debe ser lo primero)
st.set_page_config(layout="wide", page_title="Predicción Precio del Oro")

# Cargar los datos y modelos (con caché para que solo se carguen una vez)
@st.cache_data
def load_data():
    df = pd.read_csv('XAU_1d_data_V2.csv', sep=None, engine='python')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def load_all_models():
    models = {}
    
    # Modelo Lineal
    models['linear'] = joblib.load('linear_model.joblib')
    
    # Modelo ARIMA
    models['arima'] = joblib.load('arima_model.pkl')
    
    # Modelo Prophet
    with open('prophet_model.json', 'r') as fin:
        models['prophet'] = model_from_json(fin.read())
        
    # Modelo CNN y su scaler
    models['cnn'] = load_model('cnn_model.keras')
    models['cnn_scaler'] = joblib.load('cnn_scaler.joblib')

    # Modelo Logístico
    #models['logistic'] = joblib.load('logistic_model.joblib')
    
    return models

# Cargar todo al inicio
df = load_data()
models = load_all_models()


# --- 2. INTERFAZ DE PESTAÑAS ---

st.title("📈 Proyecto de Pronóstico del Precio del Oro")

tab1, tab2, tab3 = st.tabs(["Introducción", "Análisis de Modelos", "Hacer Predicción"])


# --- PESTAÑA 1: INTRODUCCIÓN ---
with tab1:
    st.header("Bienvenido al Proyecto")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        Esta es una aplicación web interactiva construida con Streamlit para presentar 
        un proyecto de pronóstico del precio del oro (XAU/USD). 
        
        El objetivo es implementar, comparar y visualizar el rendimiento de 
        diferentes modelos de series temporales y machine learning.
        
        **Modelos Implementados:**
        * Regresión Lineal Simple
        * Red Neuronal Convolucional (CNN 1D)
        * Prophet (de Meta)
        * Modelo Híbrido (Prophet + CNN)
        
        Navega a la pestaña **"Análisis de Modelos"** para ver sus gráficas y métricas,
        o ve a **"Hacer Predicción"** para probar los modelos en vivo.
        """)
    
    with col2:
        # Cargar una imagen (asegúrate de subir una imagen a tu repo y cambiar el nombre)
        st.image("gold.png", caption="El oro como activo financiero")
        st.info("El oro es considerado un activo refugio que preserva valor en el tiempo y se adelanta a crisis inflacionarias.")


# --- PESTAÑA 2: GRÁFICAS Y ANÁLISIS ---
with tab2:
    st.header("Análisis Gráfico de los Modelos")
    
    # Gráfica de precios históricos
    st.subheader("Precio Histórico del Oro (XAU)")
    fig_hist = px.line(df, x='Date', y='Close', title='Precio de Cierre (Close) - Serie temporal')
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Métricas (Copiadas de tu notebook)
    # Reemplaza todo el bloque metrics_data con ESTO:

    st.subheader("Métricas de Rendimiento (Evaluación del Notebook)")
    metrics_data = {
        'Modelo': [
            'Regresión Lineal*', 
            'CNN 1D', 
            'ARIMA', 
            'Prophet*', 
            'Híbrido (P+CNN)*'
        ],
        'Métrica Principal': [
            'R²: 0.7316', 
            'R²: 0.9311', 
            'R²: -0.6166', 
            'R²: 0.9942', 
            'R²: 0.9372'
        ],
        'Nota': [
            'Sobreajustado', 
            'Realista', 
            'Requiere ajuste', 
            'Sobreajustado', 
            'Sobreajustado'
        ]
    }

    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    st.warning("*Nota: Las métricas de Reg. Lineal, Prophet e Híbrido en el notebook original estaban sobreajustadas (evaluadas en datos de entrenamiento).")



    # Gráfica del Modelo Logístico (Matriz de Confusión)
    #st.subheader("Análisis del Modelo Logístico (Predicción de Dirección)")
    #st.write("Este modelo predice si el precio 'Subirá' (1) o 'Bajará/Se mantendrá' (0).")
    
    # Recrear los datos para la matriz de confusión (código de tu notebook)
    log_data = df[['Date', 'Close']].copy()
    log_data['price_diff'] = log_data['Close'].diff()
    log_data['y'] = (log_data['price_diff'] > 0).astype(int)
    n_lags = 5
    for i in range(1, n_lags + 1):
        log_data[f'lag_diff_{i}'] = log_data['price_diff'].shift(i)
    log_data = log_data.dropna()
    
    X_log = log_data[[f'lag_diff_{i}' for i in range(1, n_lags + 1)]]
    y_log = log_data['y']
    
    # Escalar
    scaler_log = joblib.load('cnn_scaler.joblib') # Reutilizamos el scaler, aunque lo ideal sería uno propio
    X_log_scaled = scaler_log.transform(X_log) # Asumiendo que el scaler logístico se guardó como cnn_scaler
    
    # Predecir sobre todos los datos (para la gráfica de tu notebook)
    y_pred_log = models['logistic'].predict(X_log_scaled)
    cm = confusion_matrix(y_log, y_pred_log)
    
    fig_cm = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred. Baja/Mantiene', 'Pred. Sube'],
                yticklabels=['Real Baja/Mantiene', 'Real Sube'])
    plt.title('Matriz de Confusión del Modelo Logístico')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    st.pyplot(fig_cm)


# --- PESTAÑA 3: PREDICCIÓN EN VIVO ---
with tab3:
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
            
            if model_choice == 'Prophet':
                # Hacer el dataframe futuro
                future_df = models['prophet'].make_future_dataframe(periods=days_to_forecast)
                # NOTA: Si tu modelo Prophet usa regresores (ma_20, etc.), esta parte fallará.
                # Para un pronóstico real, necesitarías predecir los regresores también.
                # Por simplicidad, asumimos que el modelo guardado no los *requiere* para predecir.
                forecast = models['prophet'].predict(future_df)
                pred_value = forecast.iloc[-1]['yhat']
                
                # Graficar
                fig = models['prophet'].plot(forecast)
                ax = fig.gca()
                ax.axhline(y=last_price, color='red', linestyle='--', lw=2, label=f'Último Precio: ${last_price:.2f}')
                ax.legend()
                st.pyplot(fig)
            
            elif model_choice == 'ARIMA':
                forecast = models['arima'].forecast(steps=days_to_forecast)
                pred_value = forecast.iloc[-1]
                
                # Graficar
                fig = plt.figure(figsize=(10, 6))
                plt.plot(df['Date'].iloc[-200:], df['Close'].iloc[-200:], label='Datos Históricos')
                plt.plot(forecast, label='Pronóstico ARIMA', linestyle='--')
                plt.axhline(y=last_price, color='red', linestyle='--', lw=2, label=f'Último Precio: ${last_price:.2f}')
                plt.legend()
                st.pyplot(fig)
            
            elif model_choice == 'Regresión Lineal':
                # Crear fechas futuras y convertirlas a ordinal
                last_date_ordinal = df['Date'].map(pd.Timestamp.toordinal).iloc[-1]
                future_dates_ordinal = np.array([last_date_ordinal + i for i in range(1, days_to_forecast + 1)]).reshape(-1, 1)
                
                forecast_values = models['linear'].predict(future_dates_ordinal)
                pred_value = forecast_values[-1]
                
                # Graficar
                future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days_to_forecast + 1)]
                fig = plt.figure(figsize=(10, 6))
                plt.plot(df['Date'].iloc[-200:], df['Close'].iloc[-200:], label='Datos Históricos')
                plt.plot(future_dates, forecast_values, label='Pronóstico Lineal', linestyle='--')
                plt.axhline(y=last_price, color='red', linestyle='--', lw=2, label=f'Último Precio: ${last_price:.2f}')
                plt.legend()
                st.pyplot(fig)

            elif model_choice == 'CNN':
                # La CNN necesita la última secuencia de datos para predecir
                LOOKBACK = 30 # (El mismo LOOKBACK que usaste en el notebook)
                scaler = models['cnn_scaler']
                model_cnn = models['cnn']
                
                # Obtener los últimos 30 días y escalarlos
                last_sequence_raw = df['Close'].iloc[-LOOKBACK:].values.reshape(-1, 1)
                last_sequence_scaled = scaler.transform(last_sequence_raw)
                
                # La CNN espera la forma (1, 30, 1)
                current_batch = last_sequence_scaled.reshape((1, LOOKBACK, 1))
                
                future_predictions_scaled = []
                
                # Predecir día por día, alimentando el modelo con su propia predicción
                for i in range(days_to_forecast):
                    next_pred_scaled = model_cnn.predict(current_batch)[0]
                    future_predictions_scaled.append(next_pred_scaled)
                    # Actualizar el 'current_batch' para la siguiente predicción
                    current_batch = np.append(current_batch[:, 1:, :], [[next_pred_scaled]], axis=1)
                
                # Des-escalar los resultados
                forecast_values = scaler.inverse_transform(future_predictions_scaled)
                pred_value = forecast_values[-1][0]
                
                # Graficar
                future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days_to_forecast + 1)]
                fig = plt.figure(figsize=(10, 6))
                plt.plot(df['Date'].iloc[-200:], df['Close'].iloc[-200:], label='Datos Históricos')
                plt.plot(future_dates, forecast_values, label='Pronóstico CNN', linestyle='--')
                plt.axhline(y=last_price, color='red', linestyle='--', lw=2, label=f'Último Precio: ${last_price:.2f}')
                plt.legend()
                st.pyplot(fig)

            # --- Mostrar la predicción final ---
            st.subheader(f"Predicción de {model_choice} a {days_to_forecast} días:")
            st.metric(label=f"Precio estimado del último día", 
                      value=f"${pred_value:,.2f}",
                      delta=f"${pred_value - last_price:,.2f} vs último precio")