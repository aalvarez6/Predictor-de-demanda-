import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import altair as alt
from datetime import datetime, timedelta
import io

try:
    import openpyxl  # noqa: F401
    _OPENPYXL_OK = True
except ImportError:
    _OPENPYXL_OK = False

# ─── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Predicción de Demanda — Series Temporales",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.level-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.badge-1 { background: #0e4f3a; color: #4fffb0; border: 1px solid #4fffb0; }
.badge-2 { background: #1a2f5e; color: #5eb3ff; border: 1px solid #5eb3ff; }
.badge-3 { background: #3d1a5e; color: #c97bff; border: 1px solid #c97bff; }

.metric-card {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 4px 0;
}
.metric-label { color: #9ca3af; font-size: 0.78rem; font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; }
.metric-value { color: #f9fafb; font-size: 1.5rem; font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODELO LSTM
# ══════════════════════════════════════════════════════════════════════════════
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES UTILITARIAS
# ══════════════════════════════════════════════════════════════════════════════
def normalize_data(data):
    mean, std = np.mean(data), np.std(data)
    std = std if std != 0 else 1
    return (data - mean) / std, mean, std

def denormalize_data(data, mean, std):
    return data * std + mean

def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)

def predict_future(model, last_sequence, n_steps, mean, std):
    model.eval()
    predictions = []
    current_sequence = last_sequence.clone()
    with torch.no_grad():
        for _ in range(n_steps):
            x = current_sequence.view(1, -1, 1)
            output = model(x)
            predictions.append(output.item())
            current_sequence = torch.cat((current_sequence[1:], output.view(1, 1)), 0)
    return denormalize_data(np.array(predictions), mean, std)


# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS (CSV y XLSX)
# ──────────────────────────────────────────────────────────────────────────────
def load_data(file, sheet_name=None):
    """Carga CSV o XLSX. Busca columna de tiempo y columna numérica de consumo."""
    try:
        fname = file.name.lower()
        is_excel = fname.endswith(".xlsx") or fname.endswith(".xls")

        if is_excel:
            if not _OPENPYXL_OK:
                st.error("Falta la librería **openpyxl**. Instálala con: `pip install openpyxl`")
                return None
            # Leer el contenido completo en memoria para evitar errores de buffer/stream
            raw_bytes = file.read()
            buf = io.BytesIO(raw_bytes)
            engine = "openpyxl" if fname.endswith(".xlsx") else "xlrd"
            try:
                df = pd.read_excel(buf, sheet_name=sheet_name or 0, engine=engine)
            except Exception as e_excel:
                st.error(f"No se pudo leer el archivo Excel: {e_excel}")
                st.info("Intenta guardar el archivo como **CSV** (UTF-8) desde Excel y cárgalo de nuevo.")
                return None
        else:
            raw_bytes = file.read()
            buf = io.BytesIO(raw_bytes)
            # Detectar separador automáticamente
            sample = raw_bytes[:2048].decode("utf-8", errors="replace")
            sep = ";" if sample.count(";") > sample.count(",") else ","
            df = pd.read_csv(buf, sep=sep)

        # ── Detectar columna de tiempo ──────────────────────────────────────
        time_col = None
        for candidate in ["Datetime", "Time", "datetime", "time", "Fecha", "fecha", "Date", "date", "timestamp"]:
            if candidate in df.columns:
                time_col = candidate
                break
        if time_col is None:
            # intentar detectar automáticamente la primera columna parseable como fecha
            for col in df.columns:
                try:
                    pd.to_datetime(df[col].dropna().iloc[:3])
                    time_col = col
                    break
                except Exception:
                    pass
        if time_col is None:
            st.error("No se encontró columna de fecha/tiempo. Asegúrate de tener una columna 'Datetime', 'Time', 'Fecha' o similar.")
            return None

        df = df.rename(columns={time_col: "Datetime"})
        df["Datetime"] = pd.to_datetime(df["Datetime"])

        # ── Detectar columna de consumo ────────────────────────────────────
        value_col = None
        for candidate in ["Kwh", "kwh", "KWH", "Consumo", "consumo", "Value", "value", "Demand", "demand", "MW", "kW"]:
            if candidate in df.columns:
                value_col = candidate
                break
        if value_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                value_col = numeric_cols[0]
                st.info(f"Se usará la columna numérica: **{value_col}**")
            else:
                st.error(f"No se encontró columna numérica de consumo. Columnas disponibles: {list(df.columns)}")
                return None

        df = df.rename(columns={value_col: "Kwh"})
        df = df[["Datetime", "Kwh"]].dropna().sort_values("Datetime").reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        st.info("💡 **Pistas para resolver el error:**\n"
                "- Asegúrate de que el archivo no esté abierto en Excel al mismo tiempo\n"
                "- Intenta exportar desde Excel como **CSV UTF-8** y carga ese archivo\n"
                "- Verifica que el archivo no esté protegido con contraseña\n"
                "- El archivo debe tener al menos una columna de fecha y una numérica")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# NIVEL 1 — EXPLORADOR: Limpieza y EDA
# ══════════════════════════════════════════════════════════════════════════════
def nivel_1(df):
    st.markdown('<span class="level-badge badge-1">Nivel 1 — Explorador</span>', unsafe_allow_html=True)
    st.markdown("### 🔍 Análisis Exploratorio y Limpieza")

    col1, col2 = st.columns(2)

    # ── 1. NaN y Outliers ──────────────────────────────────────────────────
    n_nan = df["Kwh"].isna().sum()
    q1, q3 = df["Kwh"].quantile(0.25), df["Kwh"].quantile(0.75)
    iqr = q3 - q1
    n_outliers = ((df["Kwh"] < q1 - 1.5 * iqr) | (df["Kwh"] > q3 + 1.5 * iqr)).sum()

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total registros</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">NaN detectados</div>
            <div class="metric-value">{n_nan}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Outliers (IQR)</div>
            <div class="metric-value">{n_outliers}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── 2. Imputación ──────────────────────────────────────────────────────
    impute_method = st.selectbox(
        "Método de imputación",
        ["Forward Fill", "Media simple", "Mediana", "Interpolación lineal"],
        key="impute_method"
    )

    df_clean = df.copy()
    if impute_method == "Forward Fill":
        df_clean["Kwh"] = df_clean["Kwh"].ffill()
    elif impute_method == "Media simple":
        df_clean["Kwh"] = df_clean["Kwh"].fillna(df_clean["Kwh"].mean())
    elif impute_method == "Mediana":
        df_clean["Kwh"] = df_clean["Kwh"].fillna(df_clean["Kwh"].median())
    else:
        df_clean["Kwh"] = df_clean["Kwh"].interpolate()

    with col2:
        mean_val = df_clean["Kwh"].mean()
        std_val = df_clean["Kwh"].std()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Media (limpio)</div>
            <div class="metric-value">{mean_val:.3f} kWh</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Desviación estándar</div>
            <div class="metric-value">{std_val:.3f} kWh</div>
        </div>
        """, unsafe_allow_html=True)

    # ── 3. Gráfica Original vs Limpia ─────────────────────────────────────
    orig = df[["Datetime", "Kwh"]].copy(); orig["Serie"] = "Original"
    clean = df_clean[["Datetime", "Kwh"]].copy(); clean["Serie"] = "Limpia"
    combined = pd.concat([orig, clean])

    chart = alt.Chart(combined).mark_line(opacity=0.85).encode(
        x=alt.X("Datetime:T", title="Fecha"),
        y=alt.Y("Kwh:Q", title="Consumo (kWh)"),
        color=alt.Color("Serie:N", scale=alt.Scale(
            domain=["Original", "Limpia"],
            range=["#4b5563", "#4fffb0"]
        )),
        strokeDash=alt.StrokeDash("Serie:N", scale=alt.Scale(
            domain=["Original", "Limpia"],
            range=[[4, 4], [0]]
        ))
    ).properties(height=300).interactive()

    st.altair_chart(chart, use_container_width=True)
    return df_clean


# ══════════════════════════════════════════════════════════════════════════════
# NIVEL 2 — CONSTRUCTOR: Modelos estadísticos + Forecast
# ══════════════════════════════════════════════════════════════════════════════
def nivel_2(df_clean, forecast_weeks):
    st.markdown('<span class="level-badge badge-2">Nivel 2 — Constructor</span>', unsafe_allow_html=True)
    st.markdown("### 📐 Modelos Estadísticos + Forecast con IC 90%")

    data = df_clean["Kwh"].values
    n = len(data)
    if n < 2:
        st.warning("No hay suficientes datos."); return None, None

    # ── Holt-Winters simplificado (tendencia + estacionalidad) ─────────────
    alpha, beta, gamma = 0.3, 0.1, 0.2
    period = 52  # anual en semanas

    # Inicializar
    level = np.mean(data[:period]) if n >= period else np.mean(data)
    trend = (np.mean(data[period:2*period]) - np.mean(data[:period])) / period if n >= 2*period else 0
    seasonal = np.zeros(period)
    if n >= period:
        for i in range(period):
            seasonal[i] = data[i] - level

    fitted = np.zeros(n)
    for t in range(n):
        s_idx = t % period
        if t == 0:
            fitted[t] = level + trend + seasonal[s_idx]
        else:
            prev_level = level
            fitted[t] = level + trend + seasonal[s_idx]
            if n > period and t >= period:
                level_new = alpha * (data[t] - seasonal[s_idx]) + (1 - alpha) * (prev_level + trend)
                trend = beta * (level_new - prev_level) + (1 - beta) * trend
                seasonal[s_idx] = gamma * (data[t] - level_new) + (1 - gamma) * seasonal[s_idx]
                level = level_new

    residuals = data - fitted
    mae = np.mean(np.abs(residuals))
    mape = np.mean(np.abs(residuals / (data + 1e-9))) * 100

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MAE (Holt-Winters)</div>
            <div class="metric-value">{mae:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MAPE (Holt-Winters)</div>
            <div class="metric-value">{mape:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Forecast con IC 90% ───────────────────────────────────────────────
    n_steps = forecast_weeks
    std_resid = np.std(residuals)
    z90 = 1.645

    future_preds = []
    fut_level, fut_trend = level, trend
    fut_seasonal = seasonal.copy()
    for i in range(n_steps):
        s_idx = (n + i) % period
        val = fut_level + fut_trend + fut_seasonal[s_idx]
        future_preds.append(val)

    future_preds = np.array(future_preds)
    ic_low = future_preds - z90 * std_resid
    ic_high = future_preds + z90 * std_resid

    last_date = df_clean["Datetime"].iloc[-1]
    freq = pd.infer_freq(df_clean["Datetime"]) or "W"
    future_dates = pd.date_range(start=last_date, periods=n_steps + 1, freq=freq)[1:]

    # ── Histograma de residuos ────────────────────────────────────────────
    resid_df = pd.DataFrame({"Residuo": residuals})
    hist_chart = alt.Chart(resid_df).mark_bar(color="#5eb3ff", opacity=0.75).encode(
        alt.X("Residuo:Q", bin=alt.Bin(maxbins=40), title="Residuo"),
        alt.Y("count()", title="Frecuencia")
    ).properties(height=180, title="Histograma de Residuos — Calidad del Ajuste").interactive()
    st.altair_chart(hist_chart, use_container_width=True)

    # ── Gráfico interactivo con IC ─────────────────────────────────────────
    hist_df = pd.DataFrame({
        "Datetime": df_clean["Datetime"],
        "Kwh": data,
        "Tipo": "Histórico",
        "IC_low": np.nan,
        "IC_high": np.nan,
    })
    pred_df = pd.DataFrame({
        "Datetime": future_dates,
        "Kwh": future_preds,
        "Tipo": "Forecast",
        "IC_low": ic_low,
        "IC_high": ic_high,
    })
    viz = pd.concat([hist_df, pred_df])

    base = alt.Chart(viz)
    band = base.mark_area(opacity=0.2, color="#5eb3ff").encode(
        x=alt.X("Datetime:T"),
        y=alt.Y("IC_low:Q"),
        y2=alt.Y2("IC_high:Q"),
        tooltip=[alt.Tooltip("Datetime:T", title="Fecha"),
                 alt.Tooltip("IC_low:Q", title="IC Inferior", format=".3f"),
                 alt.Tooltip("IC_high:Q", title="IC Superior", format=".3f")]
    )
    lines = base.mark_line().encode(
        x="Datetime:T",
        y=alt.Y("Kwh:Q", title="Consumo (kWh)"),
        color=alt.Color("Tipo:N", scale=alt.Scale(
            domain=["Histórico", "Forecast"],
            range=["#9ca3af", "#5eb3ff"]
        )),
        tooltip=[alt.Tooltip("Datetime:T", title="Fecha"),
                 alt.Tooltip("Kwh:Q", title="Valor", format=".3f"),
                 alt.Tooltip("Tipo:N", title="Serie")]
    )
    chart = (band + lines).properties(height=350, title="Forecast con Intervalo de Confianza 90%").interactive()
    st.altair_chart(chart, use_container_width=True)

    # ── Tabla de predicciones ─────────────────────────────────────────────
    st.markdown("**Tabla de predicciones (semana, valor central, IC inferior, IC superior)**")
    table = pd.DataFrame({
        "Semana": [f"S+{i+1}" for i in range(n_steps)],
        "Fecha": [d.strftime("%Y-%m-%d") for d in future_dates],
        "Valor Central": future_preds.round(3),
        "IC Inferior (90%)": ic_low.round(3),
        "IC Superior (90%)": ic_high.round(3),
    })
    st.dataframe(table, use_container_width=True, height=300)

    return pred_df, mae, mape


# ══════════════════════════════════════════════════════════════════════════════
# NIVEL 3 — ARQUITECTO: LSTM + Comparación de modelos + Log de anomalías
# ══════════════════════════════════════════════════════════════════════════════
def nivel_3(df_clean, seq_length, prediction_steps, hidden_size, epochs, nivel2_mape):
    st.markdown('<span class="level-badge badge-3">Nivel 3 — Arquitecto</span>', unsafe_allow_html=True)
    st.markdown("### 🏗️ LSTM + Comparación de Modelos + Anomalías")

    data = df_clean["Kwh"].values
    data_norm, mean, std = normalize_data(data)

    X, y = create_sequences(data_norm, seq_length)
    X_t = torch.FloatTensor(X).view(-1, seq_length, 1)
    y_t = torch.FloatTensor(y)

    model = LSTMPredictor(hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    progress_bar = st.progress(0, text="Entrenando LSTM…")
    train_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_t).squeeze()
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs, text=f"Época {epoch+1}/{epochs} — Loss: {loss.item():.5f}")

    # ── Ajuste sobre histórico ────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        fitted_norm = model(X_t).squeeze().numpy()
    fitted = denormalize_data(fitted_norm, mean, std)
    actual = data[seq_length:]

    residuals_lstm = actual - fitted
    mae_lstm = np.mean(np.abs(residuals_lstm))
    mape_lstm = np.mean(np.abs(residuals_lstm / (actual + 1e-9))) * 100

    # ── Comparación de modelos ────────────────────────────────────────────
    st.markdown("#### 📊 Comparación de Modelos (por MAPE)")
    model_data = pd.DataFrame({
        "Modelo": ["Holt-Winters", "LSTM"],
        "MAPE (%)": [nivel2_mape, mape_lstm],
        "MAE": [None, mae_lstm],  # solo para mostrar en tabla
    })
    best_model = model_data.loc[model_data["MAPE (%)"].idxmin(), "Modelo"]
    st.info(f"✅ Mejor modelo por MAPE: **{best_model}** ({model_data['MAPE (%)'].min():.2f}%)")

    bar = alt.Chart(model_data).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Modelo:N", title="Modelo"),
        y=alt.Y("MAPE (%):Q", title="MAPE (%)"),
        color=alt.condition(
            alt.datum.Modelo == best_model,
            alt.value("#c97bff"),
            alt.value("#4b5563")
        ),
        tooltip=["Modelo:N", alt.Tooltip("MAPE (%):Q", format=".2f")]
    ).properties(height=220)
    st.altair_chart(bar, use_container_width=True)

    # ── Predicción LSTM ───────────────────────────────────────────────────
    last_sequence = torch.FloatTensor(data_norm[-seq_length:]).view(-1, 1)
    predictions = predict_future(model, last_sequence, prediction_steps, mean, std)

    last_date = df_clean["Datetime"].iloc[-1]
    freq = pd.infer_freq(df_clean["Datetime"]) or "W"
    future_dates = pd.date_range(start=last_date, periods=prediction_steps + 1, freq=freq)[1:]

    hist_df = pd.DataFrame({"Datetime": df_clean["Datetime"], "Kwh": data, "Tipo": "Histórico"})
    pred_df = pd.DataFrame({"Datetime": future_dates, "Kwh": predictions, "Tipo": "Predicción LSTM"})
    viz = pd.concat([hist_df, pred_df])

    chart = alt.Chart(viz).mark_line().encode(
        x=alt.X("Datetime:T", title="Fecha"),
        y=alt.Y("Kwh:Q", title="Consumo (kWh)"),
        color=alt.Color("Tipo:N", scale=alt.Scale(
            domain=["Histórico", "Predicción LSTM"],
            range=["#9ca3af", "#c97bff"]
        )),
        tooltip=["Datetime:T", alt.Tooltip("Kwh:Q", format=".3f"), "Tipo:N"]
    ).properties(height=300, title="Predicción LSTM").interactive()
    st.altair_chart(chart, use_container_width=True)

    # ── Tabla de predicciones LSTM ────────────────────────────────────────
    pred_table = pd.DataFrame({
        "Paso": [f"S+{i+1}" for i in range(prediction_steps)],
        "Fecha": [d.strftime("%Y-%m-%d") for d in future_dates],
        "Predicción LSTM (kWh)": predictions.round(4),
    })
    st.dataframe(pred_table, use_container_width=True, height=280)

    # ── Log de anomalías ──────────────────────────────────────────────────
    st.markdown("#### 🚨 Log de Anomalías Detectadas")
    q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    anom_mask = (data < lower) | (data > upper)
    anom_df = df_clean[anom_mask].copy()
    anom_df["Tipo"] = np.where(data[anom_mask] < lower, "Valor bajo", "Valor alto")
    anom_df = anom_df.rename(columns={"Kwh": "Valor Original"})
    anom_df["Límite inferior"] = round(lower, 3)
    anom_df["Límite superior"] = round(upper, 3)

    if len(anom_df) == 0:
        st.success("No se detectaron anomalías en el dataset.")
    else:
        st.warning(f"Se detectaron **{len(anom_df)}** anomalías.")
        st.dataframe(anom_df[["Datetime", "Valor Original", "Tipo", "Límite inferior", "Límite superior"]].reset_index(drop=True),
                     use_container_width=True, height=260)

    # ── Descarga de predicciones ──────────────────────────────────────────
    csv_bytes = pred_table.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar predicciones LSTM (.csv)", data=csv_bytes,
                       file_name="predicciones_lstm.csv", mime="text/csv")

    return pred_table


# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
st.title("📊 Predicción de Demanda — Series Temporales")
st.caption("Explorador  ·  Constructor  ·  Arquitecto")

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("📁 Datos")
uploaded_file = st.sidebar.file_uploader("Cargar archivo (CSV o XLSX)", type=["csv", "xlsx", "xls"])

# Selector de hoja para XLSX
sheet_name = None
if uploaded_file is not None and uploaded_file.name.lower().endswith((".xlsx", ".xls")):
    try:
        raw = uploaded_file.read()
        uploaded_file.seek(0)  # resetear el cursor
        xf = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
        sheets = xf.sheet_names
        if len(sheets) > 1:
            sheet_name = st.sidebar.selectbox("Hoja del archivo Excel", sheets)
        else:
            sheet_name = sheets[0]
    except Exception:
        sheet_name = 0

# Nivel activo
nivel = st.sidebar.radio("Nivel de análisis", ["Nivel 1 — Explorador", "Nivel 2 — Constructor", "Nivel 3 — Arquitecto"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parámetros")

# Parámetros Nivel 2
forecast_weeks = st.sidebar.slider("Semanas a predecir (N2)", min_value=4, max_value=52, value=12)

# Parámetros Nivel 3
seq_length  = st.sidebar.slider("Longitud de secuencia LSTM", min_value=4, max_value=52, value=12)
hidden_size = st.sidebar.slider("Neuronas capa oculta", min_value=10, max_value=150, value=50)
epochs      = st.sidebar.slider("Épocas de entrenamiento", min_value=10, max_value=300, value=80)

train_btn = st.sidebar.button("🚀 Ejecutar Análisis", type="primary")

# ── Instrucciones cuando no hay archivo ──────────────────────────────────
if uploaded_file is None:
    st.info("👆 Carga un archivo CSV o XLSX para comenzar.")
    st.markdown("""
    **Formato esperado:**
    | Datetime | Kwh |
    |---|---|
    | 2023-01-01 00:00 | 245.3 |
    | 2023-01-01 01:00 | 238.7 |

    *Se detectan automáticamente columnas con nombres alternativos (Time, Fecha, Value, Consumo, etc.)*
    """)
    st.stop()

# ── Carga ──────────────────────────────────────────────────────────────────
df = load_data(uploaded_file, sheet_name=sheet_name)
if df is None:
    # Último recurso: selector manual de columnas
    st.warning("Selección manual de columnas:")
    try:
        uploaded_file.seek(0)
        raw2 = uploaded_file.read()
        fname2 = uploaded_file.name.lower()
        if fname2.endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(io.BytesIO(raw2), sheet_name=sheet_name or 0, engine="openpyxl")
        else:
            df_raw = pd.read_csv(io.BytesIO(raw2))
        st.dataframe(df_raw.head(5), use_container_width=True)
        cols = list(df_raw.columns)
        col_time = st.selectbox("Columna de fecha/tiempo", cols, key="manual_time")
        col_val  = st.selectbox("Columna de consumo (numérica)", cols, key="manual_val")
        if st.button("Aplicar selección manual"):
            df_raw = df_raw.rename(columns={col_time: "Datetime", col_val: "Kwh"})
            df_raw["Datetime"] = pd.to_datetime(df_raw["Datetime"])
            df = df_raw[["Datetime", "Kwh"]].dropna().sort_values("Datetime").reset_index(drop=True)
            st.success(f"Cargados {len(df):,} registros correctamente.")
        else:
            st.stop()
    except Exception as e2:
        st.error(f"No se pudo leer el archivo: {e2}")
        st.stop()

st.success(f"✅ Dataset cargado: **{len(df):,} registros** | {df['Datetime'].min().date()} → {df['Datetime'].max().date()}")
with st.expander("Vista previa de los datos"):
    st.dataframe(df.head(20), use_container_width=True)

if not train_btn:
    st.info("Ajusta los parámetros en la barra lateral y presiona **Ejecutar Análisis**.")
    st.stop()

# ── Ejecución por nivel ────────────────────────────────────────────────────
df_clean = nivel_1(df)
st.markdown("---")

nivel2_mape = None
if "Constructor" in nivel or "Arquitecto" in nivel:
    result = nivel_2(df_clean, forecast_weeks)
    if result[0] is not None:
        _, nivel2_mae, nivel2_mape = result
    st.markdown("---")

if "Arquitecto" in nivel:
    if nivel2_mape is None:
        # Calcular mape básico si se saltó nivel 2
        result = nivel_2(df_clean, forecast_weeks)
        if result[0] is not None:
            _, nivel2_mae, nivel2_mape = result
        nivel2_mape = nivel2_mape or 0.0

    nivel_3(df_clean, seq_length, forecast_weeks, hidden_size, epochs, nivel2_mape)

# ── Sidebar info ───────────────────────────────────────────────────────────
st.sidebar.markdown("""
---
**Niveles:**
- 🟢 **Nivel 1** — EDA, NaN, outliers, imputación
- 🔵 **Nivel 2** — Holt-Winters, MAE/MAPE, IC 90%, tabla forecast
- 🟣 **Nivel 3** — LSTM, comparación modelos, log anomalías
""")
