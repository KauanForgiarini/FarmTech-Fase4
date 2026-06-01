"""
FarmTech Solutions — Fase 4
Pipeline de Machine Learning — Regressão Agrícola
Kauan Maciel Forgiarini | RM574005 | FIAP IA

Modelos de regressão para prever:
  1. Volume de irrigação (L/m²)
  2. Necessidade de fertilização (kg/ha)
  3. Rendimento esperado de soja (sc/ha)

Métricas avaliadas: MAE, MSE, RMSE, R²
"""

import pandas as pd
import numpy as np
import pickle
import os
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "sensor_data_fase4.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Features de entrada (sensores disponíveis)
FEATURES = [
    "UMIDADE_SOLO_PCT",
    "PH_SOLO",
    "TEMPERATURA_C",
    "CHUVA_PREVISTA_MM",
    "N_KGHA",
    "P_KGHA",
    "K_KGHA",
]

# Variáveis alvo
TARGETS = {
    "volume_irrigacao":    "VOLUME_IRRIGACAO_L",
    "necessidade_fert":    "NECESSIDADE_FERT_KGHA",
    "rendimento_esperado": "RENDIMENTO_ESPERADO_SCHA",
}

X = df[FEATURES].copy()


# ─────────────────────────────────────────────
# 2. TREINAMENTO DE MODELOS POR VARIÁVEL ALVO
# ─────────────────────────────────────────────

resultados_gerais = {}

for nome_target, coluna_target in TARGETS.items():
    print(f"\n{'='*60}")
    print(f"📊 ALVO: {coluna_target}")
    print(f"{'='*60}")

    y = df[coluna_target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=574005
    )

    modelos = {
        "Regressão Linear":    LinearRegression(),
        "Ridge Regression":    Ridge(alpha=1.0),
        "Random Forest":       RandomForestRegressor(n_estimators=100, random_state=574005),
        "Gradient Boosting":   GradientBoostingRegressor(n_estimators=100, random_state=574005),
    }

    resultados_alvo = {}
    melhor_r2 = -np.inf
    melhor_nome = None
    melhor_pipeline = None

    for nome_modelo, modelo in modelos.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  modelo),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)

        cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")

        resultados_alvo[nome_modelo] = {
            "MAE":  round(mae,  4),
            "MSE":  round(mse,  4),
            "RMSE": round(rmse, 4),
            "R2":   round(r2,   4),
            "CV_R2_mean": round(cv_scores.mean(), 4),
            "CV_R2_std":  round(cv_scores.std(),  4),
        }

        print(f"\n  🤖 {nome_modelo}")
        print(f"     MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f} | CV-R²={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

        if r2 > melhor_r2:
            melhor_r2      = r2
            melhor_nome    = nome_modelo
            melhor_pipeline = pipe

    print(f"\n  ✅ Melhor modelo: {melhor_nome} (R²={melhor_r2:.4f})")

    # Salvar melhor modelo
    model_path = os.path.join(MODEL_DIR, f"model_{nome_target}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(melhor_pipeline, f)

    # Feature importance (RandomForest/GradientBoosting)
    importances = None
    if hasattr(melhor_pipeline.named_steps["model"], "feature_importances_"):
        fi = melhor_pipeline.named_steps["model"].feature_importances_
        importances = dict(zip(FEATURES, [round(v, 4) for v in fi]))

    resultados_gerais[nome_target] = {
        "coluna":       coluna_target,
        "melhor_modelo": melhor_nome,
        "melhor_r2":     round(melhor_r2, 4),
        "modelos":       resultados_alvo,
        "feature_importance": importances,
    }

# ─────────────────────────────────────────────
# 3. SALVAR METADADOS PARA O DASHBOARD
# ─────────────────────────────────────────────

meta = {
    "features": FEATURES,
    "targets":  TARGETS,
    "resultados": resultados_gerais,
}

meta_path = os.path.join(MODEL_DIR, "metadata.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"\n\n{'='*60}")
print("✅ TREINAMENTO CONCLUÍDO")
print(f"   Modelos salvos em: {MODEL_DIR}")
print(f"   Metadados: {meta_path}")
print(f"{'='*60}")


# ─────────────────────────────────────────────
# 4. FUNÇÃO DE PREDIÇÃO (usada pelo dashboard)
# ─────────────────────────────────────────────

def carregar_modelos():
    """Carrega todos os modelos treinados do disco."""
    modelos = {}
    for nome in TARGETS.keys():
        path = os.path.join(MODEL_DIR, f"model_{nome}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                modelos[nome] = pickle.load(f)
    return modelos


def prever(umidade, ph, temperatura, chuva, n_kgha, p_kgha, k_kgha):
    """
    Realiza previsões para os 3 alvos dado um conjunto de leituras de sensores.

    Retorna dict com volume_irrigacao, necessidade_fert e rendimento_esperado.
    """
    modelos = carregar_modelos()
    entrada = pd.DataFrame([{
        "UMIDADE_SOLO_PCT":  umidade,
        "PH_SOLO":           ph,
        "TEMPERATURA_C":     temperatura,
        "CHUVA_PREVISTA_MM": chuva,
        "N_KGHA":            n_kgha,
        "P_KGHA":            p_kgha,
        "K_KGHA":            k_kgha,
    }])

    previsoes = {}
    for nome, modelo in modelos.items():
        pred = modelo.predict(entrada)[0]
        previsoes[nome] = round(max(0, pred), 2)

    return previsoes


if __name__ == "__main__":
    # Teste rápido de predição
    print("\n🧪 Teste de predição com valores exemplo:")
    resultado = prever(
        umidade=52.0, ph=6.2, temperatura=27.5,
        chuva=0.0, n_kgha=80.0, p_kgha=60.0, k_kgha=75.0
    )
    for k, v in resultado.items():
        print(f"   {k}: {v}")
