"""
pages/1_🤖_Analista_IA.py
Asistente de IA para análisis de series de demanda.
Usa la API de Claude con clave ingresada por el usuario.
"""

import streamlit as st
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Analista IA — Demanda",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg0:#06060e; --bg1:#0d0d1a; --bg2:#111120;
    --surface:#161625; --surface2:#1c1c2e; --surface3:#22223a;
    --bd:rgba(255,255,255,0.06); --bd-hi:rgba(255,255,255,0.10);
    --accent:#4fffb0; --accent-dim:rgba(79,255,176,0.08); --accent-glow:rgba(79,255,176,0.15);
    --blue:#5eb3ff; --blue-dim:rgba(94,179,255,0.08);
    --purple:#c97bff; --purple-dim:rgba(201,123,255,0.08);
    --t1:#eeeef4; --t2:#8888a8; --t3:#3a3a58;
    --font:'IBM Plex Sans',system-ui,sans-serif;
    --mono:'IBM Plex Mono',monospace;
    --r:10px; --rl:18px; --pill:100px;
}

html, body, [class*="css"] { font-family: var(--font) !important; }
.stApp { background: var(--bg0) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 960px !important; margin: 0 auto !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: var(--bd-hi); border-radius: 4px; }

/* ── HERO ── */
.page-hero { padding: 1.6rem 0 1.2rem; border-bottom: 1px solid var(--bd); margin-bottom: 1.6rem; }
.page-hero h1 { font-family: var(--mono); font-size: clamp(1.5rem,3.5vw,2.2rem);
    font-weight: 600; color: var(--t1); letter-spacing: -.03em; margin: 0 0 .3rem; }
.page-hero p  { font-size: .88rem; color: var(--t2); margin: 0; }
.page-hero .hl { color: var(--accent); }

/* ── API KEY BOX ── */
.key-box { background: var(--surface); border: 1px solid var(--bd-hi);
    border-radius: var(--rl); padding: 1.2rem 1.5rem; margin-bottom: 1.4rem; }
.key-box-title { font-family: var(--mono); font-size: .75rem; font-weight: 600;
    letter-spacing: .09em; text-transform: uppercase; color: var(--t3); margin-bottom: .7rem; }
.stTextInput input {
    background: var(--surface2) !important; border: 1px solid var(--bd-hi) !important;
    border-radius: var(--r) !important; color: var(--t1) !important;
    font-family: var(--mono) !important; font-size: .82rem !important;
    caret-color: var(--accent) !important; }
.stTextInput input:focus {
    border-color: rgba(79,255,176,.35) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important; }

/* ── CONTEXT BANNER ── */
.ctx { display:flex; align-items:center; gap:8px; flex-wrap:wrap;
    background: var(--accent-dim); border: 1px solid rgba(79,255,176,.15);
    border-radius: var(--r); padding: .6rem 1rem; margin-bottom: 1.2rem;
    font-size: .78rem; color: var(--accent); font-family: var(--mono); font-weight: 500; }
.ctx span { color: var(--t2); font-weight: 400; }

/* ── CHAT SHELL ── */
.chat-shell { background: var(--bg1); border: 1px solid var(--bd);
    border-radius: var(--rl); overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,.6); margin-bottom: 1.5rem; }
.chat-topbar { display:flex; align-items:center; gap:8px; padding: .75rem 1.2rem;
    background: rgba(6,6,14,.9); border-bottom: 1px solid var(--bd); }
.tls { display:flex; gap:5px; margin-right:4px; }
.tl  { width:9px; height:9px; border-radius:50%; }
.tl-r{background:#ff5f57} .tl-y{background:#febc2e} .tl-g{background:#28c840}
.tb-name { font-size:.8rem; font-weight:600; color:var(--t1); }
.tb-model { margin-left:auto; font-size:.62rem; font-family:var(--mono); color:var(--t3);
    background:var(--surface2); border:1px solid var(--bd);
    padding:2px 9px; border-radius:var(--pill); }

/* ── MESSAGES ── */
.chat-msgs { padding: 1.2rem; min-height: 300px; }
.mw { display:flex; align-items:flex-start; gap:8px; margin-bottom:1rem;
    animation: bi .22s cubic-bezier(.34,1.56,.64,1); }
.mw.usr { flex-direction: row-reverse; }
@keyframes bi { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
.av { width:28px; height:28px; border-radius:50%; display:flex; align-items:center;
    justify-content:center; font-size:.7rem; flex-shrink:0; }
.av-ai  { background: linear-gradient(135deg,rgba(79,255,176,.2),rgba(94,179,255,.15));
    border: 1px solid rgba(79,255,176,.25); }
.av-usr { background: var(--surface3); border: 1px solid var(--bd-hi); }
.bbl { max-width:74%; padding:.68rem .9rem; border-radius:var(--r);
    font-size:.86rem; line-height:1.65; word-break:break-word; }
.bbl-ai  { background:var(--surface); border:1px solid var(--bd);
    color:var(--t1); border-bottom-left-radius:3px; }
.bbl-usr { background:linear-gradient(135deg,rgba(79,255,176,.09),rgba(79,255,176,.04));
    border:1px solid rgba(79,255,176,.16); color:var(--t1); border-bottom-right-radius:3px; }
.bbl-ai strong { color:var(--accent); font-weight:600; }
.bbl-ai code { font-family:var(--mono); font-size:.78rem; background:var(--accent-dim);
    color:var(--accent); padding:1px 5px; border-radius:3px; }
.ts { font-size:.6rem; color:var(--t3); margin-top:2px; padding:0 2px; }
.mw.usr .ts { text-align:right; }

/* ── TYPING ── */
.typing-row { display:flex; align-items:center; gap:8px; padding-bottom:.4rem; }
.dots { display:flex; gap:4px; }
.dots span { width:5px; height:5px; background:var(--accent); border-radius:50%;
    animation:dw 1.3s ease-in-out infinite; }
.dots span:nth-child(2){animation-delay:.18s} .dots span:nth-child(3){animation-delay:.36s}
@keyframes dw{0%,80%,100%{transform:scale(.6);opacity:.3}40%{transform:scale(1.1);opacity:1}}
.typing-lbl { font-size:.68rem; color:var(--t3); font-style:italic; font-family:var(--mono); }

/* ── INPUT ── */
.input-zone { border-top:1px solid var(--bd); padding:.9rem 1.2rem;
    background:rgba(4,4,12,.7); }
.stTextArea textarea {
    background: var(--surface2) !important; border:1px solid var(--bd-hi) !important;
    border-radius: var(--r) !important; color:var(--t1) !important;
    font-family:var(--font) !important; font-size:.86rem !important;
    line-height:1.55 !important; caret-color:var(--accent) !important;
    resize:none !important; }
.stTextArea textarea:focus {
    border-color:rgba(79,255,176,.35) !important;
    box-shadow:0 0 0 3px var(--accent-dim) !important; }
.stTextArea textarea::placeholder { color:var(--t3) !important; }

/* ── BUTTONS ── */
.stButton>button[kind="primary"] {
    background: linear-gradient(135deg,rgba(79,255,176,.15),rgba(79,255,176,.06)) !important;
    border:1px solid rgba(79,255,176,.35) !important; color:var(--accent) !important;
    font-family:var(--font) !important; font-weight:600 !important; font-size:.82rem !important;
    border-radius:var(--r) !important; width:100% !important; transition:all .17s !important; }
.stButton>button[kind="primary"]:hover {
    box-shadow:0 0 14px var(--accent-glow) !important; transform:translateY(-1px) !important; }
.stButton>button:not([kind="primary"]) {
    background:var(--surface) !important; border:1px solid var(--bd-hi) !important;
    color:var(--t2) !important; font-family:var(--font) !important; font-size:.76rem !important;
    border-radius:var(--pill) !important; transition:all .17s !important; white-space:nowrap !important; }
.stButton>button:not([kind="primary"]):hover {
    background:var(--accent-dim) !important; border-color:rgba(79,255,176,.3) !important;
    color:var(--accent) !important; }

/* ── QUICK CHIPS LABEL ── */
.chips-lbl { font-size:.63rem; font-weight:600; letter-spacing:.09em; text-transform:uppercase;
    color:var(--t3); margin-bottom:.5rem; font-family:var(--mono); }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] { background:var(--bg2) !important;
    border-right:1px solid var(--bd) !important; }

/* ── FOOTER ── */
.footer { text-align:center; margin-top:1.5rem; font-size:.62rem;
    color:var(--t3); letter-spacing:.07em; }
.stSpinner>div { border-top-color:var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ts() -> str:
    return datetime.now().strftime("%H:%M")


def _render_msg(role: str, content: str):
    is_user = role == "user"
    wrap = "mw usr" if is_user else "mw"
    bbl  = "bbl bbl-usr" if is_user else "bbl bbl-ai"
    av_c = "av av-usr" if is_user else "av av-ai"
    icon = "👤" if is_user else "🤖"
    st.markdown(f"""
    <div class="{wrap}">
      <div class="{av_c}">{icon}</div>
      <div>
        <div class="{bbl}">{content}</div>
        <div class="ts">{_ts()}</div>
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT dinámico según contexto de sesión
# ══════════════════════════════════════════════════════════════════════════════

def _build_system() -> str:
    # Detectar si hay datos de predicción en sesión (del app.py principal)
    has_pred = st.session_state.get("predictions") is not None

    context_block = ""
    if has_pred:
        import numpy as np
        preds = st.session_state.get("predictions", [])
        arr = np.asarray(preds, dtype=float)
        n = len(arr)
        context_block = f"""
════════════════════════════════════════════════════
DATOS DE PREDICCIÓN ACTIVOS
════════════════════════════════════════════════════
Registros predichos : {n}
Promedio            : {arr.mean():.3f}
Máximo              : {arr.max():.3f}
Mínimo              : {arr.min():.3f}
Desv. estándar      : {arr.std():.3f}
Modelo LSTM usado   : {st.session_state.get('modo', 'N/A')}
════════════════════════════════════════════════════
"""

    return f"""Eres un experto en análisis de series temporales de demanda energética y forecasting.
Tienes profundo conocimiento en:
  · Modelos de predicción: LSTM, ARIMA, Holt-Winters, Prophet, XGBoost
  · Evaluación de modelos: MAE, MAPE, RMSE, R², intervalos de confianza
  · Análisis exploratorio: detección de tendencias, estacionalidad, outliers
  · Ingeniería de features para series temporales
  · Imputación de datos faltantes y limpieza de series
  · Interpretación de resultados y recomendaciones de negocio
  · Planificación de demanda energética, forecasting de carga
{context_block}
REGLAS DE RESPUESTA:
  · Responde siempre en español claro y profesional.
  · Da primero el número o resultado concreto, luego explica.
  · Usa negritas para cifras clave y listas cuando hay 3+ ítems.
  · Si te preguntan sobre el modelo o los datos, usa el contexto de arriba.
  · Si no hay datos cargados, aún puedes responder preguntas generales sobre series temporales.
  · Termina cada respuesta técnica con una recomendación práctica breve precedida de 💡.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  LLAMADA A LA API DE CLAUDE
# ══════════════════════════════════════════════════════════════════════════════

def _call_claude(api_key: str, system: str, history: list) -> str:
    try:
        import anthropic
    except ImportError:
        return "❌ Paquete <code>anthropic</code> no instalado. Agrégalo al <code>requirements.txt</code>."

    try:
        client = anthropic.Anthropic(api_key=api_key)
        r = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1500,
            system=system,
            messages=[{"role": m["role"], "content": m["content"]} for m in history],
        )
        return r.content[0].text
    except anthropic.AuthenticationError:
        return "❌ <strong>API key inválida.</strong> Verifica que la clave sea correcta (comienza con <code>sk-ant-</code>)."
    except anthropic.RateLimitError:
        return "⏱️ <strong>Límite de rate alcanzado.</strong> Espera un momento y vuelve a intentarlo."
    except anthropic.APIConnectionError:
        return "🌐 <strong>Error de conexión.</strong> Verifica tu conexión a internet."
    except Exception as e:
        return f"❌ <strong>Error:</strong> <code>{str(e)[:220]}</code>"


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK CHIPS
# ══════════════════════════════════════════════════════════════════════════════

QUICK = [
    ("📊", "Calidad del modelo",
     "¿Qué tan confiables son las predicciones del modelo LSTM? ¿Cuáles son sus limitaciones?"),
    ("🔍", "Interpretar resultados",
     "Explícame cómo interpretar los valores de MAE y MAPE obtenidos en el ajuste histórico."),
    ("📈", "Mejorar precisión",
     "¿Qué técnicas podría aplicar para mejorar la precisión del forecast de demanda?"),
    ("⚠️", "Anomalías detectadas",
     "Hay anomalías en el dataset. ¿Cómo debo tratarlas antes de reentrenar el modelo?"),
    ("🔋", "Horizonte de predicción",
     "¿Cuál es el horizonte de predicción recomendado para series de demanda energética semanal?"),
    ("⚙️", "Comparar modelos",
     "¿En qué casos conviene usar ARIMA vs LSTM vs Holt-Winters para predecir demanda?"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def _sidebar(has_pred: bool):
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:.4rem 0 1rem'>
          <div style='font-size:1.7rem;margin-bottom:4px'>🤖</div>
          <div style='font-size:.82rem;font-weight:600;color:#eeeef4'>Analista IA</div>
          <div style='font-size:.66rem;color:#3a3a58;margin-top:2px'>Series Temporales · Claude</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")

        if has_pred:
            st.success("✅ Predicción cargada desde la app")
        else:
            st.info("ℹ️ Sin predicción activa.\nPuedes igual hacer preguntas generales.")

        st.metric("Mensajes en sesión", len(st.session_state.get("ai_history", [])))
        st.markdown("---")

        if st.button("🗑️ Nueva conversación", use_container_width=True):
            st.session_state["ai_history"] = []
            st.session_state["_ai_typing"] = False
            st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style='font-size:.7rem;color:#3a3a58;line-height:1.7'>
        <strong style='color:#8888a8'>Temas que puedo analizar:</strong><br>
        · Calidad y confianza del modelo<br>
        · Interpretación de MAE / MAPE<br>
        · Detección y tratamiento de outliers<br>
        · Técnicas de imputación<br>
        · Comparación de modelos<br>
        · Horizonte y frecuencia de forecast<br>
        · Estacionalidad y tendencias<br><br>
        <strong style='color:#8888a8'>Modelo activo:</strong><br>
        claude-sonnet-4-5
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER CHAT
# ══════════════════════════════════════════════════════════════════════════════

def _render_chat(api_key: str, system: str):
    # Shell
    st.markdown("""
    <div class="chat-shell">
      <div class="chat-topbar">
        <div class="tls">
          <div class="tl tl-r"></div><div class="tl tl-y"></div><div class="tl tl-g"></div>
        </div>
        <span class="tb-name">🤖 Analista IA — Series de Demanda</span>
        <span class="tb-model">claude-sonnet-4-5</span>
      </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chat-msgs">', unsafe_allow_html=True)

    # Saludo inicial
    if not st.session_state.get("ai_history"):
        _render_msg("assistant",
            "¡Hola! Soy tu <strong>Analista IA</strong> 🤖 especializado en series temporales de demanda. "
            "Puedo ayudarte a interpretar las predicciones del modelo LSTM, evaluar la calidad del ajuste, "
            "detectar anomalías, comparar modelos y mucho más. ¿Por dónde empezamos? 💡"
        )

    for m in st.session_state.get("ai_history", []):
        _render_msg(m["role"], m["content"])

    if st.session_state.get("_ai_typing"):
        st.markdown("""
        <div class="typing-row">
          <div class="av av-ai">🤖</div>
          <div>
            <div class="dots"><span></span><span></span><span></span></div>
            <div class="typing-lbl">Analizando…</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # /chat-msgs

    # Input
    st.markdown('<div class="input-zone">', unsafe_allow_html=True)
    ci, cs = st.columns([5, 1])
    with ci:
        txt = st.text_area("",
            placeholder="Pregunta sobre el modelo, los datos, anomalías, métricas…",
            key="ai_input", height=78, label_visibility="collapsed")
    with cs:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        send = st.button("💬 Enviar", type="primary", use_container_width=True, key="ai_send")
    st.markdown('</div>', unsafe_allow_html=True)  # /input-zone
    st.markdown('</div>', unsafe_allow_html=True)  # /chat-shell

    # Resolver input: chip o texto
    qp    = st.session_state.pop("_ai_qp", None)
    final = qp or (txt.strip() if send and txt.strip() else None)

    if final:
        st.session_state.setdefault("ai_history", [])
        st.session_state["ai_history"].append({"role": "user", "content": final})
        st.session_state["_ai_typing"] = True
        st.rerun()

    if st.session_state.get("_ai_typing"):
        ans = _call_claude(api_key, system, st.session_state["ai_history"])
        st.session_state["ai_history"].append({"role": "assistant", "content": ans})
        st.session_state["_ai_typing"] = False
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    has_pred = st.session_state.get("predictions") is not None
    _sidebar(has_pred)

    # Hero
    st.markdown("""
    <div class="page-hero">
      <h1>🤖 <span class="hl">Analista IA</span> — Series de Demanda</h1>
      <p>Pregúntale a Claude sobre tus predicciones, métricas, anomalías y técnicas de forecasting.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── API KEY ────────────────────────────────────────────────────────────────
    # Intentar leer de secrets primero, sino pedir al usuario
    api_key = ""
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass

    if not api_key:
        st.markdown('<div class="key-box">', unsafe_allow_html=True)
        st.markdown('<div class="key-box-title">🔑 API Key de Anthropic</div>', unsafe_allow_html=True)
        api_key = st.text_input(
            "",
            type="password",
            placeholder="sk-ant-api03-…",
            key="user_api_key",
            label_visibility="collapsed",
            help="Tu clave no se almacena. Se usa solo durante esta sesión.",
        )
        st.markdown(
            "<div style='font-size:.72rem;color:#3a3a58;margin-top:.4rem'>"
            "La clave solo se usa en esta sesión y no se guarda en ningún servidor. "
            "Obténla en <a href='https://console.anthropic.com' target='_blank' "
            "style='color:#4fffb0;text-decoration:none'>console.anthropic.com</a>.</div>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if not api_key:
        st.info("👆 Ingresa tu API key de Anthropic para activar el asistente.")
        st.stop()

    # ── Context banner si hay datos ───────────────────────────────────────────
    if has_pred:
        import numpy as np
        arr = np.asarray(st.session_state["predictions"], dtype=float)
        modo = st.session_state.get("modo", "N/A")
        st.markdown(
            f'<div class="ctx">📊 Predicción cargada &nbsp;·&nbsp;'
            f'<span>{len(arr)} pasos &nbsp;|&nbsp; avg {arr.mean():.2f} &nbsp;|&nbsp; modelo: {modo}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Quick chips ───────────────────────────────────────────────────────────
    st.markdown('<div class="chips-lbl">⚡ Preguntas rápidas</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (ico, label, prompt) in enumerate(QUICK):
        with cols[i % 3]:
            if st.button(f"{ico}  {label}", key=f"qchip_{i}", use_container_width=True):
                st.session_state["_ai_qp"] = prompt

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chat ──────────────────────────────────────────────────────────────────
    system = _build_system()
    _render_chat(api_key, system)

    st.markdown("""
    <div class="footer">
      🤖 Analista IA · Series de Demanda · Claude Sonnet 4.5
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
