import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import altair as alt
import io

try:
    import openpyxl  # noqa: F401
    _OPENPYXL_OK = True
except ImportError:
    _OPENPYXL_OK = False

# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Predicción de Demanda", page_icon="📦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif; }
h1,h2,h3                    { font-family: 'IBM Plex Mono', monospace; }
.badge { display:inline-block; padding:3px 12px; border-radius:4px;
         font-family:'IBM Plex Mono',monospace; font-size:.72rem;
         font-weight:600; letter-spacing:.07em; text-transform:uppercase; margin-bottom:6px; }
.b1 { background:#0e4f3a; color:#4fffb0; border:1px solid #4fffb0; }
.b2 { background:#1a2f5e; color:#5eb3ff; border:1px solid #5eb3ff; }
.b3 { background:#3d1a5e; color:#c97bff; border:1px solid #c97bff; }
.kpi { background:#111827; border:1px solid #374151; border-radius:8px;
       padding:14px 18px; margin:3px 0; }
.kpi-label { color:#9ca3af; font-size:.72rem; font-family:'IBM Plex Mono',monospace; text-transform:uppercase; }
.kpi-value { color:#f9fafb; font-size:1.4rem; font-family:'IBM Plex Mono',monospace; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODELO LSTM
# ══════════════════════════════════════════════════════════════════════════════
class LSTMPredictor(nn.Module):
    def __init__(self, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
        self.h    = hidden_size
        self.nl   = num_layers
    def forward(self, x):
        h0 = torch.zeros(self.nl, x.size(0), self.h)
        c0 = torch.zeros(self.nl, x.size(0), self.h)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════
def norm(arr):
    m, s = arr.mean(), arr.std()
    s = s if s else 1
    return (arr - m) / s, m, s

def denorm(arr, m, s):
    return arr * s + m

def make_seq(arr, L):
    X, y = [], []
    for i in range(len(arr) - L):
        X.append(arr[i:i+L])
        y.append(arr[i+L])
    return np.array(X), np.array(y)

def forecast_lstm(model, seq, n, m, s):
    model.eval()
    preds = []
    cur = seq.clone()
    with torch.no_grad():
        for _ in range(n):
            out = model(cur.view(1, -1, 1))
            preds.append(out.item())
            cur = torch.cat((cur[1:], out.view(1, 1)), 0)
    return denorm(np.array(preds), m, s)

def holt_winters(arr, alpha=.3, beta=.1, gamma=.2, period=52):
    n = len(arr)
    lv  = arr[:period].mean() if n >= period else arr.mean()
    tr  = ((arr[period:2*period].mean() - arr[:period].mean()) / period
           if n >= 2*period else 0.0)
    sea = (arr[:period] - lv).copy() if n >= period else np.zeros(period)
    fitted = np.zeros(n)
    for t in range(n):
        s = t % period
        fitted[t] = lv + tr + sea[s]
        if n > period and t >= period:
            lv_prev = lv
            lv  = alpha * (arr[t] - sea[s]) + (1 - alpha) * (lv + tr)
            tr  = beta  * (lv - lv_prev)    + (1 - beta)  * tr
            sea[s] = gamma * (arr[t] - lv)  + (1 - gamma) * sea[s]
    return fitted, lv, tr, sea

def future_hw(lv, tr, sea, n, period=52):
    return np.array([lv + tr + sea[i % period] for i in range(n)])

def detect_anomalies(series, dates):
    q1, q3 = np.percentile(series, 25), np.percentile(series, 75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    mask = (series < lo) | (series > hi)
    df = pd.DataFrame({
        "Semana":          pd.to_datetime(dates[mask]).strftime("%Y-%m-%d"),
        "Tipo de semana":  np.where(series[mask] < lo, "🔴 Valor bajo", "🟡 Valor alto"),
        "Valor original":  series[mask].round(2),
        "Límite inferior": round(lo, 2),
        "Límite superior": round(hi, 2),
    })
    return df, lo, hi

def parse_fecha(col):
    """Corrige fechas duplicadas tipo '2014-01-06 2014-01-06' y parsea."""
    col = col.astype(str).str.strip()
    col = col.str.extract(r'^(\d{4}-\d{2}-\d{2})', expand=False).fillna(col)
    return pd.to_datetime(col, errors="coerce")

@st.cache_data(show_spinner="Leyendo archivo…")
def load_raw(file_bytes, fname, sheet):
    if fname.endswith((".xlsx", ".xls")):
        if not _OPENPYXL_OK:
            return None, "Falta openpyxl en requirements.txt"
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet, engine="openpyxl")
    else:
        sample = file_bytes[:2048].decode("utf-8", errors="replace")
        sep = ";" if sample.count(";") > sample.count(",") else ","
        df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
    return df, None

def build_ts(df, col_fecha, col_valor, col_sku, sku_sel):
    tmp = df.copy()
    tmp["_fecha"] = parse_fecha(tmp[col_fecha])
    tmp["_valor"] = pd.to_numeric(tmp[col_valor], errors="coerce")
    if col_sku and sku_sel != "__TODOS__":
        tmp = tmp[tmp[col_sku] == sku_sel]
    tmp = tmp.dropna(subset=["_fecha", "_valor"])
    tmp = tmp[tmp["_valor"] >= 0]
    weekly = (tmp.groupby(pd.Grouper(key="_fecha", freq="W"))["_valor"]
                 .sum().reset_index()
                 .rename(columns={"_fecha": "Fecha", "_valor": "Demanda"}))
    return weekly[weekly["Demanda"] > 0].reset_index(drop=True)

def kpi(label, value):
    return f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.title("📦 Predicción de Demanda — Series Temporales")
st.caption("Explorador · Constructor · Arquitecto")

with st.sidebar:
    st.header("📁 Datos")
    uploaded = st.file_uploader("CSV o XLSX", type=["csv","xlsx","xls"])

    sheet = 0
    if uploaded and uploaded.name.lower().endswith((".xlsx",".xls")):
        try:
            raw_peek = uploaded.read()
            uploaded.seek(0)
            xf = pd.ExcelFile(io.BytesIO(raw_peek), engine="openpyxl")
            sheet = st.selectbox("Hoja Excel", xf.sheet_names) if len(xf.sheet_names) > 1 else xf.sheet_names[0]
        except Exception:
            sheet = 0

    nivel = st.radio("Nivel de análisis",
                     ["1 — Explorador", "2 — Constructor", "3 — Arquitecto"])
    st.markdown("---")
    st.header("⚙️ Parámetros")
    forecast_n  = st.slider("Semanas a predecir",      4,  52, 12)
    seq_length  = st.slider("Secuencia LSTM",           4,  52, 12)
    hidden_size = st.slider("Neuronas ocultas",        10, 150, 50)
    epochs      = st.slider("Épocas entrenamiento",    10, 200, 60)
    run_btn     = st.button("🚀 Ejecutar análisis", type="primary")
    st.markdown("""---
**Niveles**
- 🟢 **1** EDA, limpieza, imputación
- 🔵 **2** Holt-Winters, IC 90%, residuos
- 🟣 **3** LSTM, comparación, anomalías""")


# ── Sin archivo ───────────────────────────────────────────────────────────────
if not uploaded:
    st.info("👆 Carga un archivo CSV o XLSX para comenzar.")
    st.markdown("""
**Columnas esperadas (nombres flexibles):**

| fecha | sku | producto | unidades_vendid | nota |
|---|---|---|---|---|
| 2014-01-06 | SKU-001 | Arroz_500g | 754.8 | ok |

*Las fechas duplicadas `2014-01-06 2014-01-06` se corrigen automáticamente.*
""")
    st.stop()


# ── Cargar archivo ────────────────────────────────────────────────────────────
raw_bytes = uploaded.read()
uploaded.seek(0)
df_raw, err = load_raw(raw_bytes, uploaded.name.lower(), sheet)
if err:
    st.error(err)
    st.stop()

cols = list(df_raw.columns)

# ── Mapeo de columnas ─────────────────────────────────────────────────────────
def pick(candidates):
    for c in candidates:
        for col in cols:
            if col.lower().strip() == c:
                return col
    return cols[0]

with st.expander("🗂️ Vista previa y mapeo de columnas", expanded=True):
    st.dataframe(df_raw.head(8), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    col_fecha = c1.selectbox("Columna fecha",
        cols, index=cols.index(pick(["fecha","datetime","date","time"])))
    col_valor = c2.selectbox("Columna demanda (numérica)",
        cols, index=cols.index(pick(["unidades_vendid","unidades","kwh","value","demand","consumo","ventas"])))
    col_sku_opt = ["(ninguna)"] + cols
    col_sku   = c3.selectbox("Columna SKU / agrupación", col_sku_opt, index=0)

col_sku_real = None if col_sku == "(ninguna)" else col_sku

# ── Selector de SKU ───────────────────────────────────────────────────────────
sku_sel = "__TODOS__"
if col_sku_real:
    skus = sorted(df_raw[col_sku_real].dropna().unique().tolist())
    sku_sel = st.sidebar.selectbox(
        "SKU a analizar", ["__TODOS__"] + skus,
        format_func=lambda x: "Todos los SKUs" if x == "__TODOS__" else x)


# ── Construir serie temporal ──────────────────────────────────────────────────
try:
    ts = build_ts(df_raw, col_fecha, col_valor, col_sku_real, sku_sel)
except Exception as e:
    st.error(f"Error construyendo la serie: {e}")
    st.stop()

if len(ts) < 8:
    st.warning(f"Solo {len(ts)} semanas disponibles tras el filtrado. Necesitas al menos 8.")
    st.stop()

st.success(f"✅ **{len(ts)} semanas** | {ts['Fecha'].min().date()} → {ts['Fecha'].max().date()} | SKU: {sku_sel}")


# ── Dashboard exploratorio rápido (siempre visible antes de ejecutar) ─────────
st.subheader("📊 Dashboard Exploratorio")
c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi("Total semanas",   len(ts)), unsafe_allow_html=True)
c2.markdown(kpi("Media semanal",   f"{ts['Demanda'].mean():.1f}"), unsafe_allow_html=True)
c3.markdown(kpi("Máximo",          f"{ts['Demanda'].max():.1f}"), unsafe_allow_html=True)
c4.markdown(kpi("Mínimo",          f"{ts['Demanda'].min():.1f}"), unsafe_allow_html=True)

area = alt.Chart(ts).mark_area(
    line={"color":"#5eb3ff"},
    color=alt.Gradient(gradient="linear",
        stops=[alt.GradientStop(color="#5eb3ff", offset=0),
               alt.GradientStop(color="transparent", offset=1)],
        x1=1, x2=1, y1=1, y2=0)
).encode(
    x=alt.X("Fecha:T", title="Semana"),
    y=alt.Y("Demanda:Q", title="Unidades"),
    tooltip=["Fecha:T", alt.Tooltip("Demanda:Q", format=".1f")]
).properties(height=260, title="Serie histórica de demanda").interactive()
st.altair_chart(area, use_container_width=True)

# Top SKUs si se ven todos
if col_sku_real and sku_sel == "__TODOS__":
    top = (df_raw.assign(_v=pd.to_numeric(df_raw[col_valor], errors="coerce"))
                 .groupby(col_sku_real)["_v"].sum()
                 .nlargest(15).reset_index()
                 .rename(columns={col_sku_real:"SKU","_v":"Total"}))
    bar_top = alt.Chart(top).mark_bar(color="#c97bff").encode(
        x=alt.X("Total:Q", title="Unidades totales"),
        y=alt.Y("SKU:N", sort="-x", title=""),
        tooltip=["SKU:N", alt.Tooltip("Total:Q", format=",.0f")]
    ).properties(height=320, title="Top 15 SKUs por demanda total").interactive()
    st.altair_chart(bar_top, use_container_width=True)

if not run_btn:
    st.info("Ajusta los parámetros en la barra lateral y presiona **Ejecutar análisis**.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# NIVEL 1 — EXPLORADOR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<span class="badge b1">Nivel 1 — Explorador</span>', unsafe_allow_html=True)
st.markdown("### 🔍 Limpieza y EDA")

arr_raw = ts["Demanda"].values.copy()
n_nan   = int(np.isnan(arr_raw).sum())
q1r, q3r = np.nanpercentile(arr_raw, 25), np.nanpercentile(arr_raw, 75)
n_out   = int(((arr_raw < q1r-1.5*(q3r-q1r)) | (arr_raw > q3r+1.5*(q3r-q1r))).sum())

c1,c2,c3,c4 = st.columns(4)
for col_w, lbl, val in zip([c1,c2,c3,c4],
    ["Semanas","NaN detectados","Outliers (IQR)","Semanas válidas"],
    [len(ts), n_nan, n_out, len(ts)-n_nan]):
    col_w.markdown(kpi(lbl, val), unsafe_allow_html=True)

impute = st.selectbox("Método de imputación",
    ["Forward Fill","Media simple","Mediana","Interpolación lineal"])

ts_clean = ts.copy()
ts_clean["Demanda"] = ts_clean["Demanda"].replace(0, np.nan)
if   impute == "Forward Fill":       ts_clean["Demanda"] = ts_clean["Demanda"].ffill()
elif impute == "Media simple":       ts_clean["Demanda"] = ts_clean["Demanda"].fillna(ts_clean["Demanda"].mean())
elif impute == "Mediana":            ts_clean["Demanda"] = ts_clean["Demanda"].fillna(ts_clean["Demanda"].median())
else:                                ts_clean["Demanda"] = ts_clean["Demanda"].interpolate()

comp = pd.concat([ts.assign(Serie="Original"), ts_clean.assign(Serie="Limpia")])
chart_comp = alt.Chart(comp).mark_line(opacity=.85).encode(
    x=alt.X("Fecha:T", title="Semana"),
    y=alt.Y("Demanda:Q", title="Unidades"),
    color=alt.Color("Serie:N", scale=alt.Scale(
        domain=["Original","Limpia"], range=["#4b5563","#4fffb0"])),
    strokeDash=alt.StrokeDash("Serie:N", scale=alt.Scale(
        domain=["Original","Limpia"], range=[[4,4],[0]])),
    tooltip=["Fecha:T","Serie:N", alt.Tooltip("Demanda:Q",format=".1f")]
).properties(height=250, title="Original vs. Limpia").interactive()
st.altair_chart(chart_comp, use_container_width=True)

c1,c2 = st.columns(2)
c1.markdown(kpi("Media (limpio)",    f"{ts_clean['Demanda'].mean():.2f}"), unsafe_allow_html=True)
c2.markdown(kpi("Desv. estándar",    f"{ts_clean['Demanda'].std():.2f}"),  unsafe_allow_html=True)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# NIVEL 2 — CONSTRUCTOR
# ══════════════════════════════════════════════════════════════════════════════
nivel2_mape = None
if "2" in nivel or "3" in nivel:
    st.markdown('<span class="badge b2">Nivel 2 — Constructor</span>', unsafe_allow_html=True)
    st.markdown("### 📐 Holt-Winters + Forecast IC 90%")

    arr   = ts_clean["Demanda"].values.astype(float)
    dates = ts_clean["Fecha"].values

    fitted_hw, lv, tr, sea = holt_winters(arr)
    resid_hw  = arr - fitted_hw
    mae_hw    = np.mean(np.abs(resid_hw))
    mape_hw   = np.mean(np.abs(resid_hw / (arr + 1e-9))) * 100
    nivel2_mape = mape_hw

    c1,c2 = st.columns(2)
    c1.markdown(kpi("MAE Holt-Winters",  f"{mae_hw:.2f}"),   unsafe_allow_html=True)
    c2.markdown(kpi("MAPE Holt-Winters", f"{mape_hw:.2f}%"), unsafe_allow_html=True)

    # ── Histograma de residuos ────────────────────────────────────────────
    st.markdown("#### 📉 Histograma de Residuos — Calidad del Ajuste")
    rdf = pd.DataFrame({"Residuo": resid_hw})
    hist_hw = alt.Chart(rdf).mark_bar(color="#5eb3ff", opacity=.75).encode(
        alt.X("Residuo:Q", bin=alt.Bin(maxbins=35), title="Residuo"),
        alt.Y("count()", title="Frecuencia"),
        tooltip=[alt.Tooltip("Residuo:Q", bin=True), "count()"]
    ).properties(height=200, title="Distribución de residuos (centrada en 0 = buen ajuste)").interactive()
    zero_rule = alt.Chart(pd.DataFrame({"x":[0]})).mark_rule(
        color="#ff4b4b", strokeDash=[4,4], strokeWidth=2).encode(x="x:Q")
    st.altair_chart(hist_hw + zero_rule, use_container_width=True)

    # ── Forecast IC 90% ───────────────────────────────────────────────────
    z90     = 1.645
    std_r   = resid_hw.std()
    fut_hw  = future_hw(lv, tr, sea, forecast_n)
    ic_lo   = fut_hw - z90 * std_r
    ic_hi   = fut_hw + z90 * std_r
    last_d  = pd.Timestamp(dates[-1])
    fut_d   = pd.date_range(start=last_d, periods=forecast_n+1, freq="W")[1:]

    h_plot = pd.DataFrame({"Fecha":pd.to_datetime(dates),"Valor":arr,"Tipo":"Histórico","IC_lo":np.nan,"IC_hi":np.nan})
    p_plot = pd.DataFrame({"Fecha":fut_d,"Valor":fut_hw,"Tipo":"Forecast HW","IC_lo":ic_lo,"IC_hi":ic_hi})
    viz    = pd.concat([h_plot, p_plot])

    base  = alt.Chart(viz)
    band  = base.mark_area(opacity=.15, color="#5eb3ff").encode(
        x="Fecha:T", y="IC_lo:Q", y2="IC_hi:Q",
        tooltip=[alt.Tooltip("Fecha:T",title="Semana"),
                 alt.Tooltip("IC_lo:Q",format=".1f",title="IC Inf"),
                 alt.Tooltip("IC_hi:Q",format=".1f",title="IC Sup")])
    lines = base.mark_line(strokeWidth=2).encode(
        x=alt.X("Fecha:T", title="Semana"),
        y=alt.Y("Valor:Q",  title="Unidades"),
        color=alt.Color("Tipo:N", scale=alt.Scale(
            domain=["Histórico","Forecast HW"], range=["#9ca3af","#5eb3ff"])),
        tooltip=["Fecha:T", alt.Tooltip("Valor:Q",format=".1f"), "Tipo:N"])
    st.altair_chart((band+lines).properties(
        height=300, title="Forecast Holt-Winters con IC 90%").interactive(),
        use_container_width=True)

    tabla_hw = pd.DataFrame({
        "Semana":          [f"S+{i+1}" for i in range(forecast_n)],
        "Fecha":           fut_d.strftime("%Y-%m-%d"),
        "Valor central":   fut_hw.round(2),
        "IC inferior 90%": ic_lo.round(2),
        "IC superior 90%": ic_hi.round(2),
    })
    st.dataframe(tabla_hw, use_container_width=True, height=260)
    st.download_button("⬇️ Descargar forecast HW (.csv)",
        data=tabla_hw.to_csv(index=False).encode(), file_name="forecast_hw.csv", mime="text/csv")
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# NIVEL 3 — ARQUITECTO
# ══════════════════════════════════════════════════════════════════════════════
if "3" in nivel:
    st.markdown('<span class="badge b3">Nivel 3 — Arquitecto</span>', unsafe_allow_html=True)
    st.markdown("### 🏗️ LSTM + Comparación de Modelos + Anomalías")

    arr   = ts_clean["Demanda"].values.astype(float)
    dates = ts_clean["Fecha"].values

    if len(arr) <= seq_length:
        st.warning(f"Necesitas más de {seq_length} semanas. Reduce la secuencia LSTM en la barra lateral.")
        st.stop()

    arr_n, m, s = norm(arr)
    X, y = make_seq(arr_n, seq_length)
    Xt = torch.FloatTensor(X).view(-1, seq_length, 1)
    yt = torch.FloatTensor(y)

    model = LSTMPredictor(hidden_size=hidden_size)
    opt   = torch.optim.Adam(model.parameters())
    crit  = nn.MSELoss()

    prog   = st.progress(0, text="Entrenando LSTM…")
    losses = []
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out  = model(Xt).squeeze()
        loss = crit(out, yt)
        loss.backward(); opt.step()
        losses.append(loss.item())
        prog.progress((ep+1)/epochs, text=f"Época {ep+1}/{epochs}  loss={loss.item():.5f}")

    # Curva de pérdida
    ldf = pd.DataFrame({"Época": range(1, epochs+1), "Loss": losses})
    lc  = alt.Chart(ldf).mark_line(color="#c97bff", strokeWidth=1.5).encode(
        x=alt.X("Época:Q"), y=alt.Y("Loss:Q", title="MSE Loss"),
        tooltip=["Época:Q", alt.Tooltip("Loss:Q", format=".6f")]
    ).properties(height=180, title="Curva de entrenamiento LSTM").interactive()
    st.altair_chart(lc, use_container_width=True)

    # Ajuste sobre histórico
    model.eval()
    with torch.no_grad():
        fit_n = model(Xt).squeeze().numpy()
    fit        = denorm(fit_n, m, s)
    act        = arr[seq_length:]
    resid_lstm = act - fit
    mae_lstm   = np.mean(np.abs(resid_lstm))
    mape_lstm  = np.mean(np.abs(resid_lstm / (act + 1e-9))) * 100

    # Histograma residuos LSTM
    st.markdown("#### 📉 Histograma de Residuos LSTM")
    rdf2 = pd.DataFrame({"Residuo": resid_lstm})
    h2   = alt.Chart(rdf2).mark_bar(color="#c97bff", opacity=.75).encode(
        alt.X("Residuo:Q", bin=alt.Bin(maxbins=35)),
        alt.Y("count()", title="Frecuencia"),
        tooltip=[alt.Tooltip("Residuo:Q", bin=True), "count()"]
    ).properties(height=180, title="Residuos LSTM").interactive()
    z2 = alt.Chart(pd.DataFrame({"x":[0]})).mark_rule(
        color="#ff4b4b", strokeDash=[4,4], strokeWidth=2).encode(x="x:Q")
    st.altair_chart(h2+z2, use_container_width=True)

    # Comparación de modelos
    st.markdown("#### 📊 Comparación de Modelos por MAPE")
    hw_val = nivel2_mape if nivel2_mape is not None else 999
    cmp = pd.DataFrame({"Modelo":["Holt-Winters","LSTM"], "MAPE (%)": [hw_val, mape_lstm]})
    best = cmp.loc[cmp["MAPE (%)"].idxmin(), "Modelo"]
    st.info(f"✅ Mejor modelo: **{best}** — MAPE = {cmp['MAPE (%)'].min():.2f}%")

    cbar = alt.Chart(cmp).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Modelo:N"),
        y=alt.Y("MAPE (%):Q"),
        color=alt.condition(alt.datum.Modelo == best,
                            alt.value("#c97bff"), alt.value("#374151")),
        tooltip=["Modelo:N", alt.Tooltip("MAPE (%):Q", format=".2f")]
    ).properties(height=200)
    st.altair_chart(cbar, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.markdown(kpi("MAE LSTM",  f"{mae_lstm:.2f}"),  unsafe_allow_html=True)
    c2.markdown(kpi("MAPE LSTM", f"{mape_lstm:.2f}%"), unsafe_allow_html=True)

    # Predicción futura
    last_seq  = torch.FloatTensor(arr_n[-seq_length:]).view(-1, 1)
    preds     = forecast_lstm(model, last_seq, forecast_n, m, s)
    last_d    = pd.Timestamp(dates[-1])
    fut_d     = pd.date_range(start=last_d, periods=forecast_n+1, freq="W")[1:]

    h_df = pd.DataFrame({"Fecha":pd.to_datetime(dates), "Demanda":arr,  "Tipo":"Histórico"})
    p_df = pd.DataFrame({"Fecha":fut_d,                 "Demanda":preds, "Tipo":"Predicción LSTM"})
    viz2 = pd.concat([h_df, p_df])

    lc2 = alt.Chart(viz2).mark_line(strokeWidth=2).encode(
        x=alt.X("Fecha:T", title="Semana"),
        y=alt.Y("Demanda:Q", title="Unidades"),
        color=alt.Color("Tipo:N", scale=alt.Scale(
            domain=["Histórico","Predicción LSTM"], range=["#9ca3af","#c97bff"])),
        tooltip=["Fecha:T", alt.Tooltip("Demanda:Q",format=".1f"), "Tipo:N"]
    ).properties(height=300, title="Predicción LSTM vs Histórico").interactive()
    st.altair_chart(lc2, use_container_width=True)

    pred_table = pd.DataFrame({
        "Semana":            [f"S+{i+1}" for i in range(forecast_n)],
        "Fecha":             fut_d.strftime("%Y-%m-%d"),
        "Predicción LSTM":   preds.round(2),
    })
    st.dataframe(pred_table, use_container_width=True, height=260)
    st.download_button("⬇️ Descargar predicciones LSTM (.csv)",
        data=pred_table.to_csv(index=False).encode(), file_name="pred_lstm.csv", mime="text/csv")

    # Log de anomalías
    st.markdown("#### 🚨 Log de Anomalías — Tipo de Semana y Valor Original")
    anom_df, lo, hi = detect_anomalies(arr, dates)

    ca, cb = st.columns(2)
    ca.markdown(kpi("Límite inferior", f"{lo:.2f}"), unsafe_allow_html=True)
    cb.markdown(kpi("Límite superior", f"{hi:.2f}"), unsafe_allow_html=True)

    if anom_df.empty:
        st.success("No se detectaron anomalías en el dataset.")
    else:
        st.warning(f"Se detectaron **{len(anom_df)}** anomalías.")
        st.dataframe(anom_df, use_container_width=True, height=280)

        ts_p    = pd.DataFrame({"Fecha":pd.to_datetime(dates), "Demanda":arr})
        anom_p  = ts_p[(arr < lo)|(arr > hi)]
        base_l  = alt.Chart(ts_p).mark_line(color="#9ca3af").encode(x="Fecha:T", y="Demanda:Q")
        anom_pt = alt.Chart(anom_p).mark_point(color="#ff4b4b", size=90, filled=True).encode(
            x="Fecha:T", y="Demanda:Q",
            tooltip=[alt.Tooltip("Fecha:T",title="Semana"), alt.Tooltip("Demanda:Q",format=".1f")])
        lo_r = alt.Chart(pd.DataFrame({"y":[lo]})).mark_rule(color="#ff4b4b",strokeDash=[4,4]).encode(y="y:Q")
        hi_r = alt.Chart(pd.DataFrame({"y":[hi]})).mark_rule(color="#ff4b4b",strokeDash=[4,4]).encode(y="y:Q")
        st.altair_chart((base_l+anom_pt+lo_r+hi_r).properties(
            height=250, title="Serie con anomalías marcadas (●)").interactive(),
            use_container_width=True)

        st.download_button("⬇️ Descargar log de anomalías (.csv)",
            data=anom_df.to_csv(index=False).encode(), file_name="anomalias.csv", mime="text/csv")
