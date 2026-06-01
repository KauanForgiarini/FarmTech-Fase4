"""
FarmTech Solutions — Fase 4
Dashboard Interativo com Previsões de ML
Kauan Maciel Forgiarini | RM574005 | FIAP IA

Execute com:
    streamlit run dashboard/farmtech_dashboard_fase4.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
import os
import sys

# ─────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="FarmTech Solutions — Fase 4",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Caminhos relativos
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "..", "data",  "sensor_data_fase4.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "..", "ml",    "models")
META_PATH   = os.path.join(MODEL_DIR, "metadata.json")

# ─────────────────────────────────────────────
# CSS CUSTOMIZADO
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 50%, #40916c 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }
    .kpi-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #2d6a4f; }
    .kpi-label { font-size: 0.8rem; color: #666; margin-top: 0.2rem; }
    .pred-box {
        background: linear-gradient(135deg, #d8f3dc, #b7e4c7);
        border-left: 4px solid #40916c;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .pred-value { font-size: 1.4rem; font-weight: 700; color: #1b4332; }
    .pred-label { font-size: 0.85rem; color: #2d6a4f; }
    .alert-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1b4332;
        border-bottom: 2px solid #40916c;
        padding-bottom: 0.3rem;
        margin: 1rem 0 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────

@st.cache_data
def carregar_dados():
    df = pd.read_csv(DATA_PATH, parse_dates=["TIMESTAMP_LEITURA"])
    return df

@st.cache_resource
def carregar_modelos():
    modelos = {}
    targets = ["volume_irrigacao", "necessidade_fert", "rendimento_esperado"]
    for nome in targets:
        path = os.path.join(MODEL_DIR, f"model_{nome}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                modelos[nome] = pickle.load(f)
    return modelos

@st.cache_data
def carregar_metadata():
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def prever(modelos, umidade, ph, temperatura, chuva, n_kgha, p_kgha, k_kgha):
    entrada = pd.DataFrame([{
        "UMIDADE_SOLO_PCT":  umidade,
        "PH_SOLO":           ph,
        "TEMPERATURA_C":     temperatura,
        "CHUVA_PREVISTA_MM": chuva,
        "N_KGHA":            n_kgha,
        "P_KGHA":            p_kgha,
        "K_KGHA":            k_kgha,
    }])
    return {
        nome: round(max(0, m.predict(entrada)[0]), 2)
        for nome, m in modelos.items()
    }

def recomendar_acao(umidade, ph, chuva, volume_pred, fert_pred, rendimento_pred):
    acoes = []
    alertas = []

    if chuva > 5:
        acoes.append("🌧️ **Não irrigar** — chuva prevista acima de 5mm. Economia hídrica recomendada.")
    elif umidade < 40:
        acoes.append(f"💧 **Irrigação urgente** — solo muito seco ({umidade:.0f}%). Volume recomendado: **{volume_pred:.2f} L/m²**.")
    elif umidade < 60:
        acoes.append(f"💧 **Irrigação recomendada** — umidade abaixo do ideal. Volume sugerido: **{volume_pred:.2f} L/m²**.")
    else:
        acoes.append("✅ **Umidade adequada** — não é necessário irrigar agora.")

    if ph < 5.5:
        alertas.append(f"⚠️ **pH ácido ({ph:.1f})** — aplicar calcário para elevar o pH ao intervalo ideal (5,5–7,0).")
    elif ph > 7.0:
        alertas.append(f"⚠️ **pH alcalino ({ph:.1f})** — aplicar enxofre elementar para reduzir o pH.")
    else:
        acoes.append(f"✅ **pH ideal ({ph:.1f})** — dentro da faixa ótima para soja (5,5–7,0).")

    if fert_pred > 20:
        acoes.append(f"🌿 **Fertilização alta necessária** — aplicar **{fert_pred:.1f} kg/ha** de fertilizante balanceado.")
    elif fert_pred > 10:
        acoes.append(f"🌿 **Fertilização moderada** — aplicar **{fert_pred:.1f} kg/ha** conforme análise de solo.")
    else:
        acoes.append(f"✅ **Fertilização mínima** — solo bem nutrido ({fert_pred:.1f} kg/ha).")

    acoes.append(f"📈 **Rendimento estimado:** {rendimento_pred:.1f} sc/ha de soja.")
    if rendimento_pred < 35:
        alertas.append("📉 Rendimento abaixo da média regional (55 sc/ha). Revisar manejo.")
    elif rendimento_pred > 45:
        acoes.append("🏆 Excelente potencial produtivo para esta safra!")

    return acoes, alertas

# ─────────────────────────────────────────────
# CARREGAMENTO
# ─────────────────────────────────────────────

df      = carregar_dados()
modelos = carregar_modelos()
meta    = carregar_metadata()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🌱 FarmTech Solutions — Assistente Agrícola IA</h1>
    <p>Fase 4 · Machine Learning para Previsão e Manejo de Soja · Kauan Maciel Forgiarini RM574005</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — FILTROS E PREVISÃO
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Painel de Controle")
    st.markdown("---")

    st.markdown("### 📅 Filtro Temporal")
    datas = df["TIMESTAMP_LEITURA"].dt.date
    data_min, data_max = datas.min(), datas.max()
    intervalo = st.date_input(
        "Período de análise",
        value=(data_min, data_max),
        min_value=data_min,
        max_value=data_max,
    )

    st.markdown("---")
    st.markdown("### 🔮 Previsão em Tempo Real")
    st.caption("Ajuste os valores dos sensores para obter previsões")

    umidade_input   = st.slider("💧 Umidade do solo (%)", 20.0, 100.0, 55.0, 0.5)
    ph_input        = st.slider("🧪 pH do solo",           4.0,   9.0,  6.2, 0.1)
    temp_input      = st.slider("🌡️ Temperatura (°C)",    15.0,  40.0, 25.0, 0.5)
    chuva_input     = st.slider("🌧️ Chuva prevista (mm)",  0.0,  30.0,  0.0, 0.5)
    n_input         = st.slider("🌿 Nitrogênio (kg/ha)",   40.0, 130.0, 80.0, 1.0)
    p_input         = st.slider("🌿 Fósforo (kg/ha)",      30.0, 100.0, 60.0, 1.0)
    k_input         = st.slider("🌿 Potássio (kg/ha)",     40.0, 110.0, 75.0, 1.0)

    btn_prever = st.button("🚀 Gerar Previsão", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# FILTRAGEM
# ─────────────────────────────────────────────

if len(intervalo) == 2:
    d_ini, d_fim = pd.Timestamp(intervalo[0]), pd.Timestamp(intervalo[1])
    df_fil = df[(df["TIMESTAMP_LEITURA"] >= d_ini) & (df["TIMESTAMP_LEITURA"] <= d_fim)]
else:
    df_fil = df.copy()

# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (k1, len(df_fil), "Leituras",         ""),
    (k2, f"{df_fil['BOMBA_LIGADA'].mean()*100:.1f}%", "% Irrigação", ""),
    (k3, f"{df_fil['PH_SOLO'].mean():.2f}",           "pH Médio",    ""),
    (k4, f"{df_fil['UMIDADE_SOLO_PCT'].mean():.1f}%", "Umidade Média", ""),
    (k5, f"{df_fil['RENDIMENTO_ESPERADO_SCHA'].mean():.1f}", "Rendimento Médio (sc/ha)", ""),
]
for col, val, label, _ in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS PRINCIPAIS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Previsões IA",
    "📊 Séries Temporais",
    "🔗 Correlações",
    "🤖 Desempenho dos Modelos",
    "📥 Dados Brutos",
])

# ══════════════════════════════════════════════
# TAB 1 — PREVISÕES IA
# ══════════════════════════════════════════════

with tab1:
    if btn_prever or True:  # mostra sempre após slider
        previsoes = prever(
            modelos, umidade_input, ph_input, temp_input,
            chuva_input, n_input, p_input, k_input
        )
        acoes, alertas = recomendar_acao(
            umidade_input, ph_input, chuva_input,
            previsoes["volume_irrigacao"],
            previsoes["necessidade_fert"],
            previsoes["rendimento_esperado"],
        )

        st.markdown('<div class="section-title">🔮 Previsões do Modelo de IA</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label">💧 Volume de Irrigação</div>
                <div class="pred-value">{previsoes['volume_irrigacao']:.2f} L/m²</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label">🌿 Necessidade de Fertilização</div>
                <div class="pred-value">{previsoes['necessidade_fert']:.2f} kg/ha</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label">📈 Rendimento Estimado</div>
                <div class="pred-value">{previsoes['rendimento_esperado']:.2f} sc/ha</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">💡 Recomendações de Manejo</div>', unsafe_allow_html=True)
        for acao in acoes:
            st.markdown(f"- {acao}")

        if alertas:
            st.markdown('<div class="section-title">⚠️ Alertas</div>', unsafe_allow_html=True)
            for alerta in alertas:
                st.markdown(f'<div class="alert-box">{alerta}</div>', unsafe_allow_html=True)

        # Gauge de rendimento
        st.markdown('<div class="section-title">📊 Gauge de Rendimento Esperado</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=previsoes["rendimento_esperado"],
            delta={"reference": 55, "valueformat": ".1f"},
            title={"text": "Rendimento (sc/ha) vs média regional (55)"},
            gauge={
                "axis": {"range": [0, 80]},
                "bar":  {"color": "#40916c"},
                "steps": [
                    {"range": [0,  30], "color": "#ffb3b3"},
                    {"range": [30, 50], "color": "#fff3b0"},
                    {"range": [50, 80], "color": "#d8f3dc"},
                ],
                "threshold": {
                    "line":  {"color": "#1b4332", "width": 3},
                    "thickness": 0.75,
                    "value": 55,
                },
            },
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — SÉRIES TEMPORAIS
# ══════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-title">📈 Séries Temporais dos Sensores</div>', unsafe_allow_html=True)

    # Umidade
    fig_umid = go.Figure()
    fig_umid.add_trace(go.Scatter(
        x=df_fil["TIMESTAMP_LEITURA"], y=df_fil["UMIDADE_SOLO_PCT"],
        name="Umidade (%)", line=dict(color="#2196f3", width=1.5),
    ))
    fig_umid.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Mín ideal (60%)")
    fig_umid.add_hline(y=80, line_dash="dash", line_color="red",    annotation_text="Máx ideal (80%)")
    fig_umid.update_layout(title="Umidade do Solo (%)", height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig_umid, use_container_width=True)

    # pH
    fig_ph = go.Figure()
    fig_ph.add_trace(go.Scatter(
        x=df_fil["TIMESTAMP_LEITURA"], y=df_fil["PH_SOLO"],
        name="pH", line=dict(color="#9c27b0", width=1.5),
    ))
    fig_ph.add_hrect(y0=5.5, y1=7.0, fillcolor="green", opacity=0.08,
                     annotation_text="Faixa ideal soja (5,5–7,0)")
    fig_ph.update_layout(title="pH do Solo", height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig_ph, use_container_width=True)

    # Rendimento
    fig_rend = go.Figure()
    fig_rend.add_trace(go.Scatter(
        x=df_fil["TIMESTAMP_LEITURA"], y=df_fil["RENDIMENTO_ESPERADO_SCHA"],
        name="Rendimento (sc/ha)", fill="tozeroy",
        line=dict(color="#40916c", width=1.5),
        fillcolor="rgba(64,145,108,0.15)",
    ))
    fig_rend.add_hline(y=55, line_dash="dot", line_color="#1b4332",
                       annotation_text="Média regional (55 sc/ha)")
    fig_rend.update_layout(title="Rendimento Esperado (sc/ha)", height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig_rend, use_container_width=True)

    # Temperatura
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=df_fil["TIMESTAMP_LEITURA"], y=df_fil["TEMPERATURA_C"],
        name="Temperatura (°C)", fill="tozeroy",
        line=dict(color="#ff7043", width=1.5),
        fillcolor="rgba(255,112,67,0.1)",
    ))
    fig_temp.update_layout(title="Temperatura (°C)", height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig_temp, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — CORRELAÇÕES
# ══════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-title">🔗 Matriz de Correlação</div>', unsafe_allow_html=True)

    colunas_corr = [
        "UMIDADE_SOLO_PCT", "PH_SOLO", "TEMPERATURA_C",
        "CHUVA_PREVISTA_MM", "N_KGHA", "P_KGHA", "K_KGHA",
        "VOLUME_IRRIGACAO_L", "NECESSIDADE_FERT_KGHA", "RENDIMENTO_ESPERADO_SCHA",
    ]
    corr = df_fil[colunas_corr].corr().round(2)

    labels_curtos = [
        "Umidade", "pH", "Temp", "Chuva",
        "N", "P", "K",
        "Vol.Irrig", "Fertiliz.", "Rendimento"
    ]

    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels_curtos, y=labels_curtos,
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 10},
    ))
    fig_corr.update_layout(title="Correlação de Pearson entre variáveis agrícolas",
                           height=500, margin=dict(t=50, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown('<div class="section-title">📊 Scatter: Umidade × Rendimento</div>', unsafe_allow_html=True)
    fig_sc = px.scatter(
        df_fil, x="UMIDADE_SOLO_PCT", y="RENDIMENTO_ESPERADO_SCHA",
        color="PH_SOLO", color_continuous_scale="RdYlGn",
        labels={
            "UMIDADE_SOLO_PCT":          "Umidade (%)",
            "RENDIMENTO_ESPERADO_SCHA":  "Rendimento (sc/ha)",
            "PH_SOLO":                   "pH",
        },
        title="Relação Umidade × Rendimento (cor = pH)",
        trendline="ols",
    )
    fig_sc.update_layout(height=420, margin=dict(t=50, b=20))
    st.plotly_chart(fig_sc, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_npk = px.bar(
            x=["Nitrogênio", "Fósforo", "Potássio"],
            y=[df_fil["N_KGHA"].mean(), df_fil["P_KGHA"].mean(), df_fil["K_KGHA"].mean()],
            color=["Nitrogênio", "Fósforo", "Potássio"],
            color_discrete_map={"Nitrogênio": "#2d6a4f", "Fósforo": "#52b788", "Potássio": "#95d5b2"},
            title="Médias de NPK (kg/ha)",
            labels={"x": "Nutriente", "y": "kg/ha"},
        )
        fig_npk.update_layout(showlegend=False, height=320, margin=dict(t=50, b=20))
        st.plotly_chart(fig_npk, use_container_width=True)
    with c2:
        motivos = df_fil["MOTIVO_DECISAO"].value_counts()
        fig_pie = px.pie(
            values=motivos.values, names=motivos.index,
            title="Distribuição de Decisões da Bomba",
            color_discrete_sequence=px.colors.sequential.Greens_r,
        )
        fig_pie.update_layout(height=320, margin=dict(t=50, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — DESEMPENHO DOS MODELOS
# ══════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-title">🤖 Métricas dos Modelos de Regressão</div>', unsafe_allow_html=True)

    if meta:
        resultados = meta.get("resultados", {})
        nomes_alvo = {
            "volume_irrigacao":    "Volume de Irrigação (L/m²)",
            "necessidade_fert":    "Fertilização (kg/ha)",
            "rendimento_esperado": "Rendimento (sc/ha)",
        }

        for chave, titulo in nomes_alvo.items():
            if chave not in resultados:
                continue
            st.markdown(f"#### 📌 {titulo}")
            info    = resultados[chave]
            modelos_res = info["modelos"]

            rows = []
            for nome_m, met in modelos_res.items():
                rows.append({
                    "Modelo":       nome_m,
                    "MAE":          met["MAE"],
                    "MSE":          met["MSE"],
                    "RMSE":         met["RMSE"],
                    "R²":           met["R2"],
                    "CV-R² (média)": met["CV_R2_mean"],
                    "CV-R² (±std)": met["CV_R2_std"],
                    "Melhor":       "✅" if nome_m == info["melhor_modelo"] else "",
                })
            df_met = pd.DataFrame(rows)
            st.dataframe(df_met, use_container_width=True, hide_index=True)

            # Gráfico R²
            fig_r2 = px.bar(
                df_met, x="Modelo", y="R²",
                color="Modelo",
                color_discrete_sequence=px.colors.sequential.Greens,
                title=f"Comparativo R² — {titulo}",
                range_y=[0, 1],
                text="R²",
            )
            fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_r2.update_layout(showlegend=False, height=320, margin=dict(t=50, b=20))
            st.plotly_chart(fig_r2, use_container_width=True)

            # Feature Importance
            fi = info.get("feature_importance")
            if fi:
                fi_df = pd.DataFrame(
                    {"Feature": list(fi.keys()), "Importância": list(fi.values())}
                ).sort_values("Importância", ascending=True)

                fig_fi = px.bar(
                    fi_df, x="Importância", y="Feature", orientation="h",
                    title=f"Importância das Variáveis — {titulo}",
                    color="Importância", color_continuous_scale="Greens",
                )
                fig_fi.update_layout(showlegend=False, height=300, margin=dict(t=50, b=20))
                st.plotly_chart(fig_fi, use_container_width=True)

            st.markdown("---")
    else:
        st.warning("Metadados dos modelos não encontrados. Execute `ml/train_models.py` primeiro.")

# ══════════════════════════════════════════════
# TAB 5 — DADOS BRUTOS
# ══════════════════════════════════════════════

with tab5:
    st.markdown('<div class="section-title">📥 Dados dos Sensores — Fase 4</div>', unsafe_allow_html=True)

    col_busca, col_download = st.columns([3, 1])
    with col_busca:
        motivo_filtro = st.multiselect(
            "Filtrar por motivo de decisão",
            options=df_fil["MOTIVO_DECISAO"].unique().tolist(),
            default=df_fil["MOTIVO_DECISAO"].unique().tolist(),
        )
    df_show = df_fil[df_fil["MOTIVO_DECISAO"].isin(motivo_filtro)]

    with col_download:
        st.markdown("<br>", unsafe_allow_html=True)
        csv_bytes = df_show.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv_bytes,
            file_name="farmtech_fase4_filtrado.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.dataframe(
        df_show.reset_index(drop=True),
        use_container_width=True,
        height=420,
    )
    st.caption(f"Exibindo {len(df_show)} de {len(df_fil)} registros no período selecionado.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "FarmTech Solutions · Fase 4 · FIAP IA 2026 · "
    "Kauan Maciel Forgiarini RM574005"
    "</p>",
    unsafe_allow_html=True,
)
