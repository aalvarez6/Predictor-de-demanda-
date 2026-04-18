import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import altair as alt
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(page_title="Predicción de Demanda - LSTM", layout="wide")

# Definición del modelo LSTM
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        # Inicializar estado oculto
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Función para normalizar datos
def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    std = std if std != 0 else 1  # Evitar división por cero
    return (data - mean) / std, mean, std

# Función para desnormalizar datos
def denormalize_data(data, mean, std):
    return data * std + mean

# Función para crear secuencias
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:(i + seq_length)]
        sequences.append(seq)
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)

# Función para cargar datos

#def load_data(file):
#    try:
#        df = pd.read_csv(file, index_col=0)
#        df['Datetime'] = pd.to_datetime(df['Datetime'])
#        df = df.dropna()
#        return df.sort_values('Datetime')
#    except Exception as e:
#        st.error(f"Error al cargar los datos: {str(e)}")
#        return None

def load_data(file):
    try:
        df = pd.read_csv(file, index_col=0)
        
        # Verificar el formato de las columnas
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df = df.dropna()
            return df.sort_values('Datetime')
        elif 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            # Renombrar la columna Time a Datetime para mantener consistencia
            df = df.rename(columns={'Time': 'Datetime'})
            df = df.dropna()
            return df.sort_values('Datetime')
        else:
            raise ValueError("El archivo debe contener una columna 'Datetime' o 'Time'")
            
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None



# Función para realizar predicciones
def predict_future(model, last_sequence, n_steps, mean, std):
    model.eval()
    predictions = []
    current_sequence = last_sequence.clone()
    
    with torch.no_grad():
        for _ in range(n_steps):
            # Preparar input
            x = current_sequence.view(1, -1, 1)
            # Predecir siguiente valor
            output = model(x)
            predictions.append(output.item())
            # Actualizar secuencia
            current_sequence = torch.cat((current_sequence[1:], output.view(1, 1)), 0)
    
    # Desnormalizar predicciones
    return denormalize_data(np.array(predictions), mean, std)

# Título de la aplicación
st.title("🔮 Predicción de Consumo Eléctrico con LSTM")

# Configuración en la barra lateral
st.sidebar.header("Configuración")
uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=['csv'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.sidebar.header("Parámetros del Modelo")
        seq_length = st.sidebar.slider("Longitud de secuencia (horas)", 
                                     min_value=1, max_value=48, value=24)
        prediction_hours = st.sidebar.slider("Horas a predecir", 
                                           min_value=1, max_value=72, value=24)
        hidden_size = st.sidebar.slider("Tamaño capa oculta", 
                                      min_value=10, max_value=100, value=50)
        epochs = st.sidebar.slider("Épocas de entrenamiento", 
                                 min_value=10, max_value=200, value=50)
        
        # Botón de entrenamiento
        train_button = st.sidebar.button("Entrenar Modelo")
        
        # Mostrar datos originales
        st.subheader("Vista previa de los datos")
        st.write(df.head())
        
        if train_button:
            with st.spinner('Entrenando el modelo...'):
                try:
                    # Preparar datos
                    data = df['Kwh'].values.reshape(-1, 1)
                    data_normalized, mean, std = normalize_data(data.ravel())
                    
                    # Crear secuencias
                    X, y = create_sequences(data_normalized, seq_length)
                    X = torch.FloatTensor(X).view(-1, seq_length, 1)
                    y = torch.FloatTensor(y)
                    
                    # Crear y entrenar modelo
                    model = LSTMPredictor(hidden_size=hidden_size)
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters())
                    
                    # Entrenamiento
                    progress_bar = st.progress(0)
                    train_losses = []
                    
                    for epoch in range(epochs):
                        model.train()
                        optimizer.zero_grad()
                        
                        # Forward pass
                        outputs = model(X).squeeze()
                        loss = criterion(outputs, y)
                        
                        # Backward pass
                        loss.backward()
                        optimizer.step()
                        
                        train_losses.append(loss.item())
                        progress_bar.progress((epoch + 1) / epochs)
                    
                    # Preparar última secuencia para predicción
                    last_sequence = torch.FloatTensor(data_normalized[-seq_length:])
                    predictions = predict_future(model, last_sequence.view(-1, 1), prediction_hours, mean, std)
                    
                    # Crear fechas para predicciones
                    last_date = df['Datetime'].iloc[-1]
                    future_dates = [last_date + timedelta(hours=i+1) for i in range(len(predictions))]
                    
                    # Visualización
                    st.header("📈 Resultados de la Predicción")
                    
                    # Preparar datos para visualización
                    historical_data = df[['Datetime', 'Kwh']].copy()
                    historical_data['Tipo'] = 'Histórico'
                    
                    predictions_df = pd.DataFrame({
                        'Datetime': future_dates,
                        'Kwh': predictions,
                        'Tipo': 'Predicción'
                    })
                    
                    # Combinar datos históricos y predicciones
                    viz_data = pd.concat([
                        historical_data,
                        predictions_df
                    ])
                    
                    # Crear gráfico
                    chart = alt.Chart(viz_data).mark_line().encode(
                        x=alt.X('Datetime:T', title='Fecha y Hora'),
                        y=alt.Y('Kwh:Q', title='Consumo (kWh)'),
                        color=alt.Color('Tipo:N', 
                                      scale=alt.Scale(domain=['Histórico', 'Predicción'],
                                                    range=['#1f77b4', '#ff7f0e']))
                    ).properties(
                        width=800,
                        height=400
                    ).interactive()
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Mostrar tabla de predicciones
                    st.header("📋 Tabla de Predicciones")
                    prediction_table = pd.DataFrame({
                        'Fecha y Hora': [d.strftime('%Y-%m-%d %H:%M:%S') for d in future_dates],
                        'Predicción (kWh)': predictions.round(4)
                    })
                    st.dataframe(prediction_table, height=400)
                    
                    # Botón de descarga
                    st.download_button(
                        label="Descargar Predicciones CSV",
                        data=prediction_table.to_csv(index=False),
                        file_name="predicciones.csv",
                        mime="text/csv",
                    )
                    
                except Exception as e:
                    st.error(f"Error durante el entrenamiento: {str(e)}")
                    st.info("Intenta ajustar los parámetros del modelo o verificar los datos.")

else:
    st.info("👆 Por favor, carga un archivo CSV para comenzar el análisis.")
    st.markdown("""
    El archivo CSV debe contener las siguientes columnas:
    - `Datetime`: Fecha y hora de la medición
    - `Kwh`: Consumo eléctrico en kilovatios-hora
    """)

# Información de uso
st.sidebar.markdown("""
---
### Información de Uso
- Ajusta la longitud de secuencia según el patrón temporal
- Define cuántas horas hacia el futuro predecir
- Modifica el tamaño de la capa oculta para ajustar la complejidad del modelo
- Aumenta las épocas de entrenamiento para mejor precisión
""")
