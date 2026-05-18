import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SimFolio — Predicción de Acciones",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2236;
    --accent: #00d4aa;
    --accent2: #f59e0b;
    --accent3: #818cf8;
    --danger: #f43f5e;
    --text: #e2e8f0;
    --muted: #64748b;
    --border: #1e2d40;
}

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background-color: var(--bg) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Header hero */
.hero-block {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f2027 50%, #0a0e1a 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 4px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-block::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(0,212,170,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.02em;
    margin: 0 0 0.3rem 0;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.4rem;
    flex: 1;
    min-width: 140px;
}
.metric-label { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.3rem; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 700; color: var(--accent); }
.metric-value.warn { color: var(--accent2); }
.metric-value.danger { color: var(--danger); }
.metric-value.neutral { color: var(--accent3); }

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #fff;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 2rem 0 1rem 0;
    letter-spacing: -0.01em;
}
.section-header span { color: var(--accent); }

/* Winner badge */
.winner-badge {
    display: inline-block;
    background: rgba(0,212,170,0.12);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-size: 0.65rem;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
    margin-left: 0.5rem;
}
.loser-badge {
    display: inline-block;
    background: rgba(100,116,139,0.1);
    border: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.65rem;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Recommendation box */
.rec-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
}
.rec-box.best { border-left-color: var(--accent); }
.rec-box.risk { border-left-color: var(--danger); }
.rec-box.safe { border-left-color: var(--accent3); }
.rec-title { font-family: 'Syne', sans-serif; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); margin-bottom: 0.3rem; }
.rec-content { font-size: 0.9rem; color: var(--text); }

/* Warning box */
.warning-box {
    background: rgba(244,63,94,0.07);
    border: 1px solid rgba(244,63,94,0.3);
    border-radius: 4px;
    padding: 1rem 1.4rem;
    margin-top: 1.5rem;
    font-size: 0.8rem;
    color: #fca5a5;
    line-height: 1.6;
}

/* Streamlit overrides */
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label { color: var(--muted) !important; font-size: 0.75rem !important; }

.stDataFrame { border: 1px solid var(--border) !important; border-radius: 4px; }

div[data-testid="stTabs"] button {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent) !important; }

.stSpinner > div { border-color: var(--accent) transparent transparent transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-block">
    <div class="hero-title">Sim<span>Folio</span></div>
    <div class="hero-sub">Universidad Externado · Valoración de Activos · GBM / Heston / Merton</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")

    TICKERS_DISPONIBLES = {
        "AAPL — Apple (Tech)": "AAPL",
        "MSFT — Microsoft (Tech)": "MSFT",
        "NVDA — Nvidia (Tech)": "NVDA",
        "AMZN — Amazon (Consumo)": "AMZN",
        "TSLA — Tesla (Consumo)": "TSLA",
        "JPM — JPMorgan (Financiero)": "JPM",
        "BAC — Bank of America (Fin.)": "BAC",
        "GS — Goldman Sachs (Fin.)": "GS",
        "JNJ — J&J (Salud)": "JNJ",
        "PFE — Pfizer (Salud)": "PFE",
        "UNH — UnitedHealth (Salud)": "UNH",
        "XOM — ExxonMobil (Energía)": "XOM",
        "CVX — Chevron (Energía)": "CVX",
        "CAT — Caterpillar (Industrial)": "CAT",
        "GE — GE Aerospace (Industrial)": "GE",
        "HD — Home Depot (Consumo)": "HD",
    }

    seleccion = st.multiselect(
        "Seleccionar 4 acciones",
        options=list(TICKERS_DISPONIBLES.keys()),
        default=["AAPL — Apple (Tech)", "MSFT — Microsoft (Tech)",
                 "JPM — JPMorgan (Financiero)", "XOM — ExxonMobil (Energía)"],
        max_selections=4,
        help="Selecciona exactamente 4 acciones de sectores distintos."
    )

    st.markdown("---")
    n_sim = st.slider("N° simulaciones Monte Carlo", 500, 2000, 1000, 100)
    ventana_anios = st.slider("Ventana histórica (años)", 1, 5, 2)
    st.markdown("---")
    ejecutar = st.button("🚀 Ejecutar Análisis", use_container_width=True)

tickers_sel = [TICKERS_DISPONIBLES[k] for k in seleccion]

if len(tickers_sel) != 4:
    st.warning("⚠️ Selecciona exactamente **4 acciones** en el panel lateral para iniciar el análisis.")
    st.stop()

if not ejecutar:
    st.info("👈 Configura los parámetros en el panel lateral y presiona **Ejecutar Análisis**.")
    st.stop()

# ─────────────────────────────────────────────
# HELPERS — MODELOS
# ─────────────────────────────────────────────

def simular_gbm(S0, mu, sigma, T, dt, n_sim):
    n_steps = int(T / dt)
    Z = np.random.standard_normal((n_sim, n_steps))
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    paths = S0 * np.exp(np.cumsum(log_ret, axis=1))
    paths = np.hstack([np.full((n_sim, 1), S0), paths])
    return paths


def simular_heston(S0, mu, v0, kappa, theta, xi, rho, T, dt, n_sim):
    n_steps = int(T / dt)
    paths = np.zeros((n_sim, n_steps + 1))
    paths[:, 0] = S0
    v = np.full(n_sim, v0)

    for t in range(n_steps):
        Z1 = np.random.standard_normal(n_sim)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_sim)
        v_pos = np.maximum(v, 0)
        dv = kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * Z2
        v = np.maximum(v_pos + dv, 0)
        dS = mu * dt + np.sqrt(v_pos * dt) * Z1
        paths[:, t + 1] = paths[:, t] * np.exp(dS - 0.5 * v_pos * dt)

    return paths


def simular_merton(S0, mu, sigma, lam, mu_j, sigma_j, T, dt, n_sim):
    n_steps = int(T / dt)
    paths = np.zeros((n_sim, n_steps + 1))
    paths[:, 0] = S0

    for t in range(n_steps):
        Z = np.random.standard_normal(n_sim)
        N = np.random.poisson(lam * dt, n_sim)
        J = np.random.normal(mu_j, sigma_j, n_sim) * N
        drift = (mu - 0.5 * sigma**2 - lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)) * dt
        paths[:, t + 1] = paths[:, t] * np.exp(drift + sigma * np.sqrt(dt) * Z + J)

    return paths


def calcular_rmse(precios_obs, paths_sim):
    n_obs = len(precios_obs)
    sim_slice = paths_sim[:, 1:n_obs + 1]
    media_sim = sim_slice.mean(axis=0)
    rmse = np.sqrt(np.mean((media_sim - precios_obs) ** 2))
    return rmse, media_sim


def estimar_heston_params(retornos, v0_base):
    kappa = 2.0
    theta = v0_base
    xi = 0.3
    rho = -0.7
    return kappa, theta, xi, rho


# ─────────────────────────────────────────────
# DESCARGA DE DATOS
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📥 Descarga de <span>Datos</span></div>', unsafe_allow_html=True)

import datetime
fecha_fin = datetime.date.today()
fecha_ini = fecha_fin - datetime.timedelta(days=int(365 * ventana_anios))

datos = {}
retornos_dict = {}
validos = []

cols_load = st.columns(4)
for i, ticker in enumerate(tickers_sel):
    with cols_load[i]:
        with st.spinner(f"Descargando {ticker}..."):
            try:
                df = yf.download(ticker, start=str(fecha_ini), end=str(fecha_fin),
                                 auto_adjust=True, progress=False)
                if df.empty or len(df) < 100:
                    st.error(f"{ticker}: sin datos")
                    continue
                # Manejar MultiIndex que genera yfinance >= 0.2.40
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                precio = df["Close"].squeeze().dropna()
                ret = np.log(precio / precio.shift(1)).dropna()
                datos[ticker] = precio
                retornos_dict[ticker] = ret
                validos.append(ticker)
                st.success(f"✅ {ticker} — {len(precio)} días")
            except Exception as e:
                st.error(f"{ticker}: {e}")

if len(validos) < 4:
    st.error("No se pudieron descargar todas las acciones. Revisa la conexión.")
    st.stop()

# ─────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Histórico & Retornos",
    "🔬 Simulaciones",
    "🏆 Backtesting RMSE",
    "🔭 Proyección 1 Mes",
    "💼 Recomendación"
])

# ═══════════════════════════════════════════
# TAB 1 — HISTÓRICO & RETORNOS
# ═══════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📈 Precio <span>Histórico</span></div>', unsafe_allow_html=True)

    # Precio histórico normalizado
    fig_hist = go.Figure()
    colores = ["#00d4aa", "#f59e0b", "#818cf8", "#f43f5e"]
    for idx, ticker in enumerate(validos):
        p = datos[ticker]
        p_norm = p / p.iloc[0] * 100
        fig_hist.add_trace(go.Scatter(
            x=p_norm.index, y=p_norm.values,
            name=ticker, line=dict(color=colores[idx], width=1.8),
            hovertemplate=f"<b>{ticker}</b><br>%{{x|%Y-%m-%d}}<br>Base 100: %{{y:.1f}}<extra></extra>"
        ))

    fig_hist.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.6)",
        font=dict(family="DM Mono", size=11),
        legend=dict(orientation="h", y=1.05, x=0),
        yaxis_title="Índice Base 100",
        xaxis_title="",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified"
    )
    fig_hist.update_xaxes(gridcolor="#1e2d40", showgrid=True)
    fig_hist.update_yaxes(gridcolor="#1e2d40", showgrid=True)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Métricas de precios actuales
    st.markdown('<div class="section-header">📋 Estadísticas de <span>Precios</span></div>', unsafe_allow_html=True)
    cols_m = st.columns(4)
    for idx, ticker in enumerate(validos):
        p = datos[ticker]
        ret = retornos_dict[ticker]
        precio_actual = float(p.iloc[-1])
        ret_total = float((p.iloc[-1] / p.iloc[0] - 1) * 100)
        vol_anual = float(ret.std() * np.sqrt(252) * 100)
        with cols_m[idx]:
            color_ret = "warn" if ret_total >= 0 else "danger"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{ticker}</div>
                <div class="metric-value">${precio_actual:.2f}</div>
                <div style="font-size:0.7rem; color:{'#00d4aa' if ret_total>=0 else '#f43f5e'}; margin-top:0.2rem;">
                    {'▲' if ret_total>=0 else '▼'} {abs(ret_total):.1f}% ({ventana_anios}a)
                </div>
                <div style="font-size:0.65rem; color:#64748b; margin-top:0.2rem;">
                    Vol. anual: {vol_anual:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Retornos logarítmicos
    st.markdown('<div class="section-header">📉 Retornos <span>Logarítmicos</span> Diarios</div>', unsafe_allow_html=True)
    fig_ret = make_subplots(rows=2, cols=2, subplot_titles=validos,
                            vertical_spacing=0.12, horizontal_spacing=0.08)

    posiciones = [(1,1),(1,2),(2,1),(2,2)]
    for idx, ticker in enumerate(validos):
        ret = retornos_dict[ticker]
        r, c = posiciones[idx]
        fig_ret.add_trace(go.Scatter(
            x=ret.index, y=ret.values,
            mode="lines", name=ticker,
            line=dict(color=colores[idx], width=0.8),
            showlegend=False
        ), row=r, col=c)
        fig_ret.add_hline(y=0, line_dash="dot", line_color="#64748b", line_width=0.8, row=r, col=c)

    fig_ret.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.6)",
        font=dict(family="DM Mono", size=10),
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    fig_ret.update_xaxes(gridcolor="#1e2d40")
    fig_ret.update_yaxes(gridcolor="#1e2d40", tickformat=".1%")
    st.plotly_chart(fig_ret, use_container_width=True)

    # Distribución de retornos
    st.markdown('<div class="section-header">🔔 Distribución de <span>Retornos</span></div>', unsafe_allow_html=True)
    fig_dist = go.Figure()
    for idx, ticker in enumerate(validos):
        ret = retornos_dict[ticker]
        fig_dist.add_trace(go.Violin(
            y=ret.values, name=ticker,
            fillcolor=colores[idx],
            line_color=colores[idx],
            opacity=0.7,
            box_visible=True,
            meanline_visible=True
        ))

    fig_dist.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.6)",
        font=dict(family="DM Mono", size=11),
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(tickformat=".1%", gridcolor="#1e2d40"),
        showlegend=False
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 2 — SIMULACIONES
# ═══════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔬 Simulaciones <span>Monte Carlo</span></div>', unsafe_allow_html=True)

    ticker_sim = st.selectbox("Ver simulaciones de:", validos)

    p_full = datos[ticker_sim]
    ret_full = retornos_dict[ticker_sim]
    S0_sim = float(p_full.iloc[-1])
    mu_sim = float(ret_full.mean() * 252)
    sigma_sim = float(ret_full.std() * np.sqrt(252))
    v0_sim = float(ret_full.std() ** 2 * 252)

    T_sim = 21 / 252
    dt_sim = 1 / 252

    n_show = min(n_sim, 200)

    with st.spinner("Simulando trayectorias..."):
        paths_gbm = simular_gbm(S0_sim, mu_sim, sigma_sim, T_sim, dt_sim, n_sim)
        kappa_h, theta_h, xi_h, rho_h = estimar_heston_params(ret_full, v0_sim)
        paths_heston = simular_heston(S0_sim, mu_sim, v0_sim, kappa_h, theta_h, xi_h, rho_h, T_sim, dt_sim, n_sim)
        lam_m = max(5.0, float(np.abs(ret_full[np.abs(ret_full) > 2*ret_full.std()]).count()) / len(ret_full) * 252)
        ret_extremos = ret_full[np.abs(ret_full) > 2*ret_full.std()]
        mu_j = float(ret_extremos.mean()) if len(ret_extremos) > 0 else 0.0
        sigma_j = max(float(ret_extremos.std()) if len(ret_extremos) > 0 else 0.05, 0.01)
        paths_merton = simular_merton(S0_sim, mu_sim, sigma_sim, lam_m, mu_j, sigma_j, T_sim, dt_sim, n_sim)

    dias_proj = np.arange(paths_gbm.shape[1])

    def fig_sim(paths, titulo, color, n_show):
        fig = go.Figure()
        alpha_line = max(0.03, 80 / n_sim)
        for i in range(min(n_show, paths.shape[0])):
            fig.add_trace(go.Scatter(
                x=dias_proj, y=paths[i],
                mode="lines", line=dict(color=color, width=0.5),
                opacity=alpha_line, showlegend=False,
                hoverinfo="skip"
            ))
        # Media
        media = paths.mean(axis=0)
        p5 = np.percentile(paths, 5, axis=0)
        p95 = np.percentile(paths, 95, axis=0)

        # Convertir color hex a rgba para el relleno
        def hex_to_rgba(hex_color, alpha=0.12):
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        fig.add_trace(go.Scatter(x=dias_proj, y=p95, fill=None,
                                 line=dict(color=color, width=1, dash="dot"),
                                 showlegend=False, name="P95"))
        fig.add_trace(go.Scatter(x=dias_proj, y=p5, fill="tonexty",
                                 fillcolor=hex_to_rgba(color, 0.12),
                                 line=dict(color=color, width=1, dash="dot"),
                                 showlegend=False, name="P5"))
        fig.add_trace(go.Scatter(x=dias_proj, y=media,
                                 line=dict(color="#ffffff", width=2.5),
                                 name="Precio Esperado"))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,14,26,0.6)",
            title=dict(text=titulo, font=dict(family="Syne", size=14, color="#fff")),
            font=dict(family="DM Mono", size=10),
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(title="Días hábiles", gridcolor="#1e2d40"),
            yaxis=dict(title="Precio (USD)", gridcolor="#1e2d40"),
            showlegend=False
        )
        return fig

    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        st.plotly_chart(fig_sim(paths_gbm, f"GBM — {ticker_sim}", "#00d4aa", n_show), use_container_width=True)
    with col_sim2:
        st.plotly_chart(fig_sim(paths_heston, f"Heston — {ticker_sim}", "#818cf8", n_show), use_container_width=True)
    with col_sim3:
        st.plotly_chart(fig_sim(paths_merton, f"Merton — {ticker_sim}", "#f59e0b", n_show), use_container_width=True)

    # Parámetros estimados
    st.markdown('<div class="section-header">🔢 Parámetros <span>Estimados</span></div>', unsafe_allow_html=True)
    df_params = pd.DataFrame({
        "Parámetro": ["µ anual", "σ anual", "v₀ (varianza)", "κ (rev. media)", "θ (varianza LP)", "ξ (vol-vol)", "ρ (correlación)", "λ (saltos/año)", "µⱼ (tamaño salto)", "σⱼ (vol saltos)"],
        "Valor": [f"{mu_sim:.4f}", f"{sigma_sim:.4f}", f"{v0_sim:.6f}", f"{kappa_h:.2f}", f"{theta_h:.6f}", f"{xi_h:.2f}", f"{rho_h:.2f}", f"{lam_m:.1f}", f"{mu_j:.4f}", f"{sigma_j:.4f}"],
        "Modelo": ["GBM/Todos", "GBM/Merton", "Heston", "Heston", "Heston", "Heston", "Heston", "Merton", "Merton", "Merton"]
    })
    st.dataframe(df_params, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════
# TAB 3 — BACKTESTING RMSE
# ═══════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🏆 Backtesting <span>RMSE</span> por Modelo</div>', unsafe_allow_html=True)
    st.caption("Ventana de prueba: últimos 21 días hábiles. Ventana de entrenamiento: todo lo anterior.")

    resultados = {}

    progress = st.progress(0, text="Ejecutando backtesting...")

    for idx_t, ticker in enumerate(validos):
        p_full = datos[ticker]
        ret_full = retornos_dict[ticker]

        # Split train / test
        n_test = 21
        p_train = p_full.iloc[:-n_test]
        p_test = p_full.iloc[-n_test:]

        # Parámetros desde train
        ret_train = np.log(p_train / p_train.shift(1)).dropna()
        mu_ = float(ret_train.mean() * 252)
        sigma_ = float(ret_train.std() * np.sqrt(252))
        v0_ = float(ret_train.std() ** 2 * 252)
        S0_ = float(p_train.iloc[-1])
        T_ = n_test / 252
        dt_ = 1 / 252

        # GBM
        pg = simular_gbm(S0_, mu_, sigma_, T_, dt_, n_sim)
        rmse_gbm, _ = calcular_rmse(p_test.values, pg)

        # Heston
        kappa_, theta_, xi_, rho_ = estimar_heston_params(ret_train, v0_)
        ph = simular_heston(S0_, mu_, v0_, kappa_, theta_, xi_, rho_, T_, dt_, n_sim)
        rmse_hes, _ = calcular_rmse(p_test.values, ph)

        # Merton
        lam_ = max(5.0, float(np.abs(ret_train[np.abs(ret_train) > 2*ret_train.std()]).count()) / len(ret_train) * 252)
        ret_ex = ret_train[np.abs(ret_train) > 2*ret_train.std()]
        mu_j_ = float(ret_ex.mean()) if len(ret_ex) > 0 else 0.0
        sigma_j_ = max(float(ret_ex.std()) if len(ret_ex) > 0 else 0.05, 0.01)
        pm = simular_merton(S0_, mu_, sigma_, lam_, mu_j_, sigma_j_, T_, dt_, n_sim)
        rmse_mer, _ = calcular_rmse(p_test.values, pm)

        rmses = {"GBM": rmse_gbm, "Heston": rmse_hes, "Merton": rmse_mer}
        ganador = min(rmses, key=rmses.get)

        # Guardar paths del ganador para proyectar
        resultados[ticker] = {
            "rmse_gbm": rmse_gbm, "rmse_hes": rmse_hes, "rmse_mer": rmse_mer,
            "ganador": ganador,
            "S0_proj": float(p_full.iloc[-1]),
            "mu": mu_, "sigma": sigma_, "v0": v0_,
            "kappa": kappa_, "theta": theta_, "xi": xi_, "rho": rho_,
            "lam": lam_, "mu_j": mu_j_, "sigma_j": sigma_j_,
            "p_test": p_test,
            "paths_test_gbm": pg,
            "paths_test_hes": ph,
            "paths_test_mer": pm,
        }
        progress.progress((idx_t + 1) / len(validos), text=f"Backtesting {ticker}...")

    progress.empty()

    # Gráfico backtesting vs observado
    fig_bt = make_subplots(rows=2, cols=2, subplot_titles=validos,
                           vertical_spacing=0.15, horizontal_spacing=0.08)

    for idx, ticker in enumerate(validos):
        r, c = posiciones[idx]
        d = resultados[ticker]
        p_test_v = d["p_test"]
        dias_bt = np.arange(len(p_test_v))

        for model_name, paths_bt, col_m in [
            ("GBM", d["paths_test_gbm"], "#00d4aa"),
            ("Heston", d["paths_test_hes"], "#818cf8"),
            ("Merton", d["paths_test_mer"], "#f59e0b"),
        ]:
            media_m = paths_bt[:, 1:len(p_test_v)+1].mean(axis=0)
            showleg = idx == 0
            fig_bt.add_trace(go.Scatter(
                x=p_test_v.index, y=media_m,
                name=model_name, line=dict(color=col_m, width=1.5, dash="dot"),
                showlegend=showleg, legendgroup=model_name
            ), row=r, col=c)

        # Observado
        fig_bt.add_trace(go.Scatter(
            x=p_test_v.index, y=p_test_v.values,
            name="Observado", line=dict(color="#ffffff", width=2),
            showlegend=idx == 0, legendgroup="obs"
        ), row=r, col=c)

    fig_bt.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.6)",
        font=dict(family="DM Mono", size=10),
        height=550,
        margin=dict(l=10, r=10, t=40, b=40),
        legend=dict(orientation="h", y=1.05),
        hovermode="x unified"
    )
    fig_bt.update_xaxes(gridcolor="#1e2d40")
    fig_bt.update_yaxes(gridcolor="#1e2d40")
    st.plotly_chart(fig_bt, use_container_width=True)

    # Tabla RMSE
    st.markdown('<div class="section-header">📋 Tabla de <span>RMSE</span></div>', unsafe_allow_html=True)

    rows_rmse = []
    for ticker in validos:
        d = resultados[ticker]
        rows_rmse.append({
            "Acción": ticker,
            "RMSE GBM": round(d["rmse_gbm"], 4),
            "RMSE Heston": round(d["rmse_hes"], 4),
            "RMSE Merton": round(d["rmse_mer"], 4),
            "Modelo Ganador": d["ganador"],
            "Mejor RMSE": round(min(d["rmse_gbm"], d["rmse_hes"], d["rmse_mer"]), 4)
        })

    df_rmse = pd.DataFrame(rows_rmse)

    # Mostrar tabla con colores
    def highlight_winner(row):
        col_map = {"GBM": "RMSE GBM", "Heston": "RMSE Heston", "Merton": "RMSE Merton"}
        styles = [""] * len(row)
        idx_win = row.index.get_loc(col_map[row["Modelo Ganador"]])
        styles[idx_win] = "background-color: rgba(0,212,170,0.15); color: #00d4aa; font-weight: bold"
        return styles

    st.dataframe(
        df_rmse.style.apply(highlight_winner, axis=1),
        use_container_width=True, hide_index=True
    )

    # RMSE bar chart comparativo
    st.markdown('<div class="section-header">📊 Comparación de <span>RMSE</span></div>', unsafe_allow_html=True)
    fig_rmse = go.Figure()

    for modelo, color in [("RMSE GBM", "#00d4aa"), ("RMSE Heston", "#818cf8"), ("RMSE Merton", "#f59e0b")]:
        fig_rmse.add_trace(go.Bar(
            name=modelo.replace("RMSE ", ""),
            x=df_rmse["Acción"], y=df_rmse[modelo],
            marker_color=color, marker_opacity=0.85
        ))

    fig_rmse.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.6)",
        font=dict(family="DM Mono", size=11),
        barmode="group",
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.05),
        xaxis=dict(gridcolor="#1e2d40"),
        yaxis=dict(gridcolor="#1e2d40", title="RMSE (USD)")
    )
    st.plotly_chart(fig_rmse, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 4 — PROYECCIÓN A 1 MES
# ═══════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🔭 Proyección a <span>1 Mes</span> — Modelo Ganador</div>', unsafe_allow_html=True)

    T_proj = 21 / 252
    dt_proj = 1 / 252
    proyecciones = {}

    for ticker in validos:
        d = resultados[ticker]
        S0_p = d["S0_proj"]
        ganador = d["ganador"]

        if ganador == "GBM":
            paths_p = simular_gbm(S0_p, d["mu"], d["sigma"], T_proj, dt_proj, n_sim)
        elif ganador == "Heston":
            paths_p = simular_heston(S0_p, d["mu"], d["v0"], d["kappa"], d["theta"], d["xi"], d["rho"], T_proj, dt_proj, n_sim)
        else:
            paths_p = simular_merton(S0_p, d["mu"], d["sigma"], d["lam"], d["mu_j"], d["sigma_j"], T_proj, dt_proj, n_sim)

        precios_finales = paths_p[:, -1]
        proyecciones[ticker] = {
            "actual": S0_p,
            "esperado": float(np.percentile(precios_finales, 50)),
            "p5": float(np.percentile(precios_finales, 5)),
            "p95": float(np.percentile(precios_finales, 95)),
            "ganador": ganador,
            "paths": paths_p,
            "var_esp": float((np.percentile(precios_finales, 50) / S0_p - 1) * 100),
            "amplitud": float((np.percentile(precios_finales, 95) - np.percentile(precios_finales, 5)) / S0_p * 100)
        }

    # Gráfico de abanico por acción
    fig_proj = make_subplots(rows=2, cols=2, subplot_titles=validos,
                             vertical_spacing=0.28, horizontal_spacing=0.08)

    for idx, ticker in enumerate(validos):
        r, c = posiciones[idx]
        proj = proyecciones[ticker]
        paths_p = proj["paths"]
        dias_p = np.arange(paths_p.shape[1])
        col_p = colores[idx]

        # Banda P5–P95
        p5_line = np.percentile(paths_p, 5, axis=0)
        p95_line = np.percentile(paths_p, 95, axis=0)
        media_p = np.percentile(paths_p, 50, axis=0)

        # Algunas trayectorias
        for i in range(min(80, n_sim)):
            fig_proj.add_trace(go.Scatter(
                x=dias_p, y=paths_p[i],
                mode="lines", line=dict(color=col_p, width=0.4),
                opacity=0.06, showlegend=False, hoverinfo="skip"
            ), row=r, col=c)

        fig_proj.add_trace(go.Scatter(
            x=dias_p, y=p95_line, fill=None,
            line=dict(color=col_p, width=1, dash="dot"),
            name="P95", showlegend=False
        ), row=r, col=c)
        fig_proj.add_trace(go.Scatter(
            x=dias_p, y=p5_line, fill="tonexty",
            fillcolor="rgba({},{},{},0.12)".format(*[int(col_p.lstrip("#")[i:i+2], 16) for i in (0,2,4)]),
            line=dict(color=col_p, width=1, dash="dot"),
            name="P5", showlegend=False
        ), row=r, col=c)
        fig_proj.add_trace(go.Scatter(
            x=dias_p, y=media_p,
            line=dict(color="#ffffff", width=2),
            name="Mediana", showlegend=False
        ), row=r, col=c)

    fig_proj.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.6)",
        font=dict(family="DM Mono", size=10),
        height=550,
        margin=dict(l=10, r=10, t=40, b=40),
    )
    fig_proj.update_xaxes(gridcolor="#1e2d40", title_text="Días hábiles")
    fig_proj.update_yaxes(gridcolor="#1e2d40", title_text="Precio (USD)")
    st.plotly_chart(fig_proj, use_container_width=True)

    # Tabla de proyección
    st.markdown('<div class="section-header">📋 Tabla de <span>Proyección</span> — Próximo Mes</div>', unsafe_allow_html=True)

    rows_proj = []
    for ticker in validos:
        proj = proyecciones[ticker]
        rows_proj.append({
            "Acción": ticker,
            "Precio Actual": f"${proj['actual']:.2f}",
            "Rango Inf. (P5%)": f"${proj['p5']:.2f}",
            "Precio Esperado (P50%)": f"${proj['esperado']:.2f}",
            "Rango Sup. (P95%)": f"${proj['p95']:.2f}",
            "Var. Esperada": f"{'▲' if proj['var_esp']>=0 else '▼'} {abs(proj['var_esp']):.2f}%",
            "Amplitud Rango": f"{proj['amplitud']:.1f}%",
            "Modelo": proj["ganador"]
        })

    df_proj = pd.DataFrame(rows_proj)
    st.dataframe(df_proj, use_container_width=True, hide_index=True)

    # Gráfico de rangos tipo "bullet"
    st.markdown('<div class="section-header">🎯 Rango Proyectado <span>Visual</span></div>', unsafe_allow_html=True)

    fig_range = go.Figure()
    for idx, ticker in enumerate(validos):
        proj = proyecciones[ticker]
        fig_range.add_trace(go.Scatter(
            x=[proj["p5"], proj["p95"]], y=[ticker, ticker],
            mode="lines", line=dict(color=colores[idx], width=10),
            opacity=0.35, showlegend=False, hoverinfo="skip"
        ))
        fig_range.add_trace(go.Scatter(
            x=[proj["esperado"]], y=[ticker],
            mode="markers", marker=dict(color="#ffffff", size=12, symbol="diamond"),
            name=ticker, showlegend=True,
            hovertemplate=f"<b>{ticker}</b><br>P5: ${proj['p5']:.2f}<br>P50: ${proj['esperado']:.2f}<br>P95: ${proj['p95']:.2f}<extra></extra>"
        ))

    fig_range.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.6)",
        font=dict(family="DM Mono", size=11),
        height=250,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title="Precio USD", gridcolor="#1e2d40"),
        yaxis=dict(gridcolor="#1e2d40"),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_range, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 5 — RECOMENDACIÓN EJECUTIVA
# ═══════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">💼 Recomendación <span>Ejecutiva</span> para Don Rigoberto</div>', unsafe_allow_html=True)

    # Calcular rankings
    mejor_valorizacion = max(validos, key=lambda t: proyecciones[t]["var_esp"])
    menor_riesgo = min(validos, key=lambda t: proyecciones[t]["amplitud"])
    mayor_riesgo = max(validos, key=lambda t: proyecciones[t]["amplitud"])

    # Score compuesto: normalizar valorización y amplitud
    scores = {}
    vars_esp = {t: proyecciones[t]["var_esp"] for t in validos}
    ampls = {t: proyecciones[t]["amplitud"] for t in validos}
    min_var, max_var = min(vars_esp.values()), max(vars_esp.values())
    min_amp, max_amp = min(ampls.values()), max(ampls.values())

    for ticker in validos:
        norm_var = (vars_esp[ticker] - min_var) / (max_var - min_var + 1e-9)
        norm_amp = 1 - (ampls[ticker] - min_amp) / (max_amp - min_amp + 1e-9)
        scores[ticker] = 0.6 * norm_var + 0.4 * norm_amp

    total_score = sum(scores.values())
    pesos = {t: max(round(scores[t] / total_score * 100), 5) for t in validos}
    # Ajustar para que sumen 100
    diff = 100 - sum(pesos.values())
    pesos[max(pesos, key=pesos.get)] += diff

    # Boxes de recomendación
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        proj_bv = proyecciones[mejor_valorizacion]
        st.markdown(f"""
        <div class="rec-box best">
            <div class="rec-title">🏆 Mejor expectativa de valorización</div>
            <div class="rec-content">
                <strong style="color:#00d4aa;font-size:1.1rem;">{mejor_valorizacion}</strong><br>
                Precio esperado: <strong>${proj_bv['esperado']:.2f}</strong> 
                (actual: ${proj_bv['actual']:.2f})<br>
                Variación esperada: <strong>{'▲' if proj_bv['var_esp']>=0 else '▼'} {abs(proj_bv['var_esp']):.2f}%</strong><br>
                Rango: ${proj_bv['p5']:.2f} – ${proj_bv['p95']:.2f}<br>
                Modelo: {proj_bv['ganador']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        proj_mr = proyecciones[menor_riesgo]
        st.markdown(f"""
        <div class="rec-box safe" style="margin-top:1rem;">
            <div class="rec-title">🛡️ Menor incertidumbre en rango proyectado</div>
            <div class="rec-content">
                <strong style="color:#818cf8;font-size:1.1rem;">{menor_riesgo}</strong><br>
                Amplitud del rango (P5–P95): <strong>{proj_mr['amplitud']:.1f}%</strong><br>
                Rango: ${proj_mr['p5']:.2f} – ${proj_mr['p95']:.2f}<br>
                Modelo: {proj_mr['ganador']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r2:
        proj_mr2 = proyecciones[mayor_riesgo]
        st.markdown(f"""
        <div class="rec-box risk">
            <div class="rec-title">⚠️ Mayor riesgo (mayor amplitud de rango)</div>
            <div class="rec-content">
                <strong style="color:#f43f5e;font-size:1.1rem;">{mayor_riesgo}</strong><br>
                Amplitud del rango (P5–P95): <strong>{proj_mr2['amplitud']:.1f}%</strong><br>
                Rango: ${proj_mr2['p5']:.2f} – ${proj_mr2['p95']:.2f}<br>
                Mayor dispersión entre las simulaciones.<br>
                Modelo: {proj_mr2['ganador']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="rec-box" style="margin-top:1rem;">
            <div class="rec-title">📊 Asignación sugerida de portafolio</div>
            <div class="rec-content">
                {''.join([f'<div style="margin:0.3rem 0;"><strong style="color:{colores[validos.index(t)]};">{t}</strong>: {pesos[t]}% — ${proyecciones[t]["esperado"]:.2f} esp.</div>' for t in validos])}
                <div style="font-size:0.7rem;color:#64748b;margin-top:0.5rem;">
                    Ponderación por score compuesto: 60% retorno esperado + 40% estabilidad del rango.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tabla resumen completa
    st.markdown('<div class="section-header">📋 Tabla Final de <span>Resultados</span></div>', unsafe_allow_html=True)

    rows_final = []
    for ticker in validos:
        proj = proyecciones[ticker]
        d = resultados[ticker]
        rows_final.append({
            "Acción": ticker,
            "Modelo Ganador": proj["ganador"],
            "RMSE GBM": round(d["rmse_gbm"], 4),
            "RMSE Heston": round(d["rmse_hes"], 4),
            "RMSE Merton": round(d["rmse_mer"], 4),
            "Precio Actual": f"${proj['actual']:.2f}",
            "Rango P5%–P95%": f"${proj['p5']:.2f} – ${proj['p95']:.2f}",
            "Precio Esperado": f"${proj['esperado']:.2f}",
            "Var. Esperada": f"{'▲' if proj['var_esp']>=0 else '▼'} {abs(proj['var_esp']):.2f}%",
            "Peso Sugerido": f"{pesos[ticker]}%"
        })

    df_final = pd.DataFrame(rows_final)
    st.dataframe(df_final, use_container_width=True, hide_index=True)

    # Pie de asignación
    st.markdown('<div class="section-header">🥧 Distribución de <span>Portafolio</span></div>', unsafe_allow_html=True)
    fig_pie = go.Figure(go.Pie(
        labels=validos,
        values=[pesos[t] for t in validos],
        marker_colors=colores[:4],
        hole=0.55,
        textinfo="label+percent",
        textfont=dict(family="DM Mono", size=12),
        hovertemplate="<b>%{label}</b><br>Peso: %{value}%<extra></extra>"
    ))
    fig_pie.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono"),
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1)
    )
    fig_pie.add_annotation(text="Portafolio<br>Sugerido", x=0.5, y=0.5,
                           showarrow=False, font=dict(family="Syne", size=13, color="#fff"))
    st.plotly_chart(fig_pie, use_container_width=True)

    # Advertencia de límites
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Advertencia sobre los límites de los modelos</strong><br><br>
        Esta aplicación utiliza modelos cuantitativos de simulación (GBM, Heston y Merton) 
        para estimar <em>escenarios probabilísticos</em> de precios. Los rangos presentados 
        <strong>no constituyen una promesa de rentabilidad ni una garantía de resultado</strong>.<br><br>
        • Los modelos se calibran con datos históricos y asumen que el futuro comparte 
        características estadísticas del pasado — lo cual puede no cumplirse.<br>
        • Eventos exógenos (cambios regulatorios, crisis geopolíticas, resultados 
        corporativos inesperados) pueden producir movimientos fuera de cualquier rango simulado.<br>
        • El modelo con menor RMSE en backtesting no necesariamente será el mejor 
        predictor en el siguiente periodo.<br><br>
        <em>En inversión, la tranquilidad no viene de adivinar, sino de entender el riesgo. — Don Rigoberto</em>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#64748b; font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase;">
        Universidad Externado de Colombia · Valoración de Activos · SimFolio v1.0
    </div>
    """, unsafe_allow_html=True)
