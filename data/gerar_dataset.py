"""
FarmTech Solutions — Fase 4
Geração de Dataset Simulado para Regressão
Kauan Maciel Forgiarini | RM574005 | FIAP IA

Gera sensor_data_fase4.csv com 200 leituras simuladas de sensores ESP32
Variáveis incluem: umidade, pH, NPK, temperatura, chuva, volume_irrigacao,
rendimento_esperado e necessidade_fertilizacao para treinamento de modelos de regressão.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Seed para reprodutibilidade (RM574005)
np.random.seed(574005)

N_AMOSTRAS = 200
inicio = datetime(2025, 1, 1, 6, 0, 0)

registros = []

for i in range(N_AMOSTRAS):
    timestamp = inicio + timedelta(hours=3 * i)

    # Sensores base (continuidade com Fase 2 e 3)
    umidade_pct  = round(np.random.uniform(30, 95), 1)
    ph_solo      = round(np.random.uniform(4.5, 8.0), 2)
    temperatura  = round(np.random.uniform(18, 35), 1)
    chuva_mm     = round(max(0, np.random.exponential(3)), 1)
    n_presente   = int(np.random.choice([0, 1], p=[0.3, 0.7]))
    p_presente   = int(np.random.choice([0, 1], p=[0.35, 0.65]))
    k_presente   = int(np.random.choice([0, 1], p=[0.3, 0.7]))

    # Valores contínuos dos nutrientes (para regressão — novidade Fase 4)
    n_kgha = round(np.random.uniform(40, 130), 1)
    p_kgha = round(np.random.uniform(30, 100), 1)
    k_kgha = round(np.random.uniform(40, 110), 1)

    # ---------- VARIÁVEL ALVO 1: Volume de irrigação (L/m²) ----------
    # Regra baseada na lógica da Fase 2, agora contínua
    deficit_umidade = max(0, 70 - umidade_pct)  # quanto falta para 70%
    fator_ph = 1.0 if 5.5 <= ph_solo <= 7.0 else 0.5
    fator_chuva = max(0, 1 - chuva_mm / 10)
    fator_temp = 1 + (temperatura - 25) * 0.02  # mais calor → mais água
    volume_irrigacao = round(
        deficit_umidade * 0.15 * fator_ph * fator_chuva * fator_temp
        + np.random.normal(0, 0.3),
        2
    )
    volume_irrigacao = max(0, volume_irrigacao)

    # ---------- VARIÁVEL ALVO 2: Necessidade de fertilização (kg/ha) ----------
    # Baseado nos níveis de NPK e pH
    deficit_n = max(0, 100 - n_kgha)
    deficit_p = max(0, 75 - p_kgha)
    deficit_k = max(0, 90 - k_kgha)
    correcao_ph = abs(6.2 - ph_solo) * 5
    necessidade_fert = round(
        (deficit_n * 0.4 + deficit_p * 0.35 + deficit_k * 0.25) * 0.5
        + correcao_ph
        + np.random.normal(0, 1.5),
        2
    )
    necessidade_fert = max(0, necessidade_fert)

    # ---------- VARIÁVEL ALVO 3: Rendimento esperado (sacas/ha) ----------
    # Soja: média 55 sc/ha, influenciado por todos os fatores
    base_rendimento = 55
    fator_umidade  = 1 - abs(umidade_pct - 70) * 0.008
    fator_ph_rend  = 1 - abs(ph_solo - 6.2) * 0.1
    fator_n        = min(1, n_kgha / 100)
    fator_p        = min(1, p_kgha / 75)
    fator_k        = min(1, k_kgha / 90)
    fator_temp_r   = 1 - abs(temperatura - 24) * 0.015
    rendimento = round(
        base_rendimento
        * fator_umidade
        * fator_ph_rend
        * ((fator_n + fator_p + fator_k) / 3)
        * max(0.5, fator_temp_r)
        + np.random.normal(0, 2),
        2
    )
    rendimento = max(10, min(80, rendimento))

    # Motivo da decisão (consistência com Fase 3)
    if chuva_mm > 5:
        motivo = "chuva_prevista"
        bomba  = 0
    elif umidade_pct >= 80:
        motivo = "umidade_adequada"
        bomba  = 0
    elif ph_solo < 5.5 or ph_solo > 7.0:
        motivo = "ph_fora_faixa"
        bomba  = 0
    elif umidade_pct < 60 and (n_presente or k_presente):
        motivo = "irrigando"
        bomba  = 1
    else:
        motivo = "aguardando"
        bomba  = 0

    registros.append({
        "ID": i + 1,
        "TIMESTAMP_LEITURA": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "CULTURA": "Soja",
        "N_PRESENTE": n_presente,
        "P_PRESENTE": p_presente,
        "K_PRESENTE": k_presente,
        "N_KGHA": n_kgha,
        "P_KGHA": p_kgha,
        "K_KGHA": k_kgha,
        "PH_SOLO": ph_solo,
        "UMIDADE_SOLO_PCT": umidade_pct,
        "TEMPERATURA_C": temperatura,
        "CHUVA_PREVISTA_MM": chuva_mm,
        "BOMBA_LIGADA": bomba,
        "MOTIVO_DECISAO": motivo,
        # Variáveis alvo para regressão (NOVIDADE FASE 4)
        "VOLUME_IRRIGACAO_L": volume_irrigacao,
        "NECESSIDADE_FERT_KGHA": necessidade_fert,
        "RENDIMENTO_ESPERADO_SCHA": rendimento,
    })

df = pd.DataFrame(registros)
df.to_csv("sensor_data_fase4.csv", index=False)
print(f"✅ Dataset gerado: {len(df)} registros")
print(f"\n📊 Estatísticas das variáveis alvo:")
print(df[["VOLUME_IRRIGACAO_L", "NECESSIDADE_FERT_KGHA", "RENDIMENTO_ESPERADO_SCHA"]].describe().round(2))
print(f"\n💧 Distribuição de motivos:")
print(df["MOTIVO_DECISAO"].value_counts())
