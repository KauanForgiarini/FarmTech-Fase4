# 🌱 FarmTech Solutions — Fase 4: Previsão Inteligente na Agricultura

- **FIAP — Inteligência Artificial**
- **Aluno:** Kauan Maciel Forgiarini | **RM:** 574005
- **Aluno:** Wagner Adriano De Souza Silva Junio | **RM:** 569431
- **Aluno:** Thiago Lucas da Costa Bessa | **RM:** 570367
- **Aluna:** Beatriz de Oliveira Ossola Ribeiro | **RM:** 570190
- **Aluno:** Willian Kauê Tobias do Carmo | **RM:** 570038

---

## 📋 Sumário

1. [Visão Geral](#1-visão-geral)
2. [Arquitetura do Projeto](#2-arquitetura-do-projeto)
3. [Dataset — Fase 4](#3-dataset--fase-4)
4. [Parte 1 — Pipeline ML + Dashboard Streamlit](#4-parte-1--pipeline-ml--dashboard-streamlit)
5. [Parte 2 — Previsões e Métricas de Regressão](#5-parte-2--previsões-e-métricas-de-regressão)
6. [Sugestões de Manejo em C++](#6-sugestões-de-manejo-em-c)
7. [Banco de Dados Oracle](#7-banco-de-dados-oracle)
8. [Estrutura de Pastas](#8-estrutura-de-pastas)
9. [Como Executar](#9-como-executar)
10. [Referências](#10-referências)
11. [Vídeo Demonstrativo](#11-vídeo-demonstrativo)

---

## 1. Visão Geral

A Fase 4 da FarmTech Solutions representa a consolidação do ecossistema de **Agricultura Cognitiva**, integrando todos os componentes desenvolvidos ao longo do semestre:

| Fase | Entregável |
|------|-----------|
| Fase 1 | Sistema de irrigação em Python (terminal) |
| Fase 2 | Hardware ESP32 no Wokwi + OpenWeather API + análise em R |
| Fase 3 | Oracle SQL + Dashboard Streamlit + 5 modelos de classificação ML |
| **Fase 4** | **Regressão Scikit-Learn + Dashboard completo + Sugestões C++** |

### Objetivos da Fase 4

- Modelos de **regressão supervisionada** para prever volume de irrigação, fertilização e rendimento
- **Dashboard interativo** em Streamlit com visualizações em tempo real
- **Motor de decisão em C++** que traduz previsões do Python em ações de manejo
- **Banco de dados Oracle** atualizado com as novas variáveis-alvo de regressão

---

## 2. Arquitetura do Projeto

```
Sensores ESP32 (Wokwi)
        │
        ▼
sensor_data_fase4.csv ◄── gerar_dataset.py
        │
        ▼
train_models.py (Scikit-Learn)
   ├── Gradient Boosting Regressor → model_volume_irrigacao.pkl
   ├── Gradient Boosting Regressor → model_necessidade_fert.pkl
   └── Gradient Boosting Regressor → model_rendimento_esperado.pkl
        │
        ├──► farmtech_dashboard_fase4.py (Streamlit)
        │         └── Previsões em tempo real via sliders
        │
        └──► farmtech_suggestions.cpp (C++)
                  └── Engine de recomendações de manejo
```

---

## 3. Dataset — Fase 4

Arquivo: `data/sensor_data_fase4.csv` — **200 leituras** simuladas do ESP32 (seed=574005)

### Colunas

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| ID | int | Identificador único |
| TIMESTAMP_LEITURA | datetime | A cada 3h, a partir de 01/01/2025 |
| CULTURA | str | Soja (fixo) |
| N/P/K_PRESENTE | 0/1 | Nutriente presente ou ausente (binário) |
| N/P/K_KGHA | float | **Valor contínuo do nutriente (kg/ha) — NOVO Fase 4** |
| PH_SOLO | float | pH simulado via LDR (4,5–8,0) |
| UMIDADE_SOLO_PCT | float | Umidade simulada via DHT22 (%) |
| TEMPERATURA_C | float | Temperatura do ar (°C) |
| CHUVA_PREVISTA_MM | float | Previsão de chuva via OpenWeather (mm) |
| BOMBA_LIGADA | 0/1 | Decisão de irrigação (lógica Fase 2) |
| MOTIVO_DECISAO | str | Motivo da decisão (consistência Fase 3) |
| **VOLUME_IRRIGACAO_L** | float | **Alvo: volume a irrigar (L/m²)** |
| **NECESSIDADE_FERT_KGHA** | float | **Alvo: fertilização necessária (kg/ha)** |
| **RENDIMENTO_ESPERADO_SCHA** | float | **Alvo: rendimento esperado soja (sc/ha)** |

---

## 4. Parte 1 — Pipeline ML + Dashboard Streamlit

### Pipeline de Machine Learning

Arquivo: `ml/train_models.py`

**4 modelos** treinados para cada uma das 3 variáveis-alvo:

| Modelo | Biblioteca |
|--------|-----------|
| Regressão Linear | `sklearn.linear_model.LinearRegression` |
| Ridge Regression | `sklearn.linear_model.Ridge` |
| Random Forest | `sklearn.ensemble.RandomForestRegressor` |
| **Gradient Boosting** | `sklearn.ensemble.GradientBoostingRegressor` |

**Features de entrada** (7 variáveis dos sensores):
`UMIDADE_SOLO_PCT`, `PH_SOLO`, `TEMPERATURA_C`, `CHUVA_PREVISTA_MM`, `N_KGHA`, `P_KGHA`, `K_KGHA`

**Pipeline com StandardScaler:**
```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model",  GradientBoostingRegressor(n_estimators=100, random_state=574005)),
])
```

**Validação:** `train_test_split` 80/20 + Cross-Validation 5-fold

### Dashboard Streamlit

Arquivo: `dashboard/farmtech_dashboard_fase4.py`

| Aba | Conteúdo |
|-----|---------|
| 🔮 Previsões IA | Sliders para entrada dos sensores → previsões em tempo real + gauge |
| 📊 Séries Temporais | Umidade, pH, rendimento e temperatura ao longo do tempo |
| 🔗 Correlações | Heatmap de correlação de Pearson + scatter Umidade×Rendimento |
| 🤖 Desempenho dos Modelos | Métricas MAE/MSE/RMSE/R² + feature importance |
| 📥 Dados Brutos | Tabela filtrável + exportação CSV |

---

## 5. Parte 2 — Previsões e Métricas de Regressão

### Resultados dos Modelos

#### 🎯 Alvo 1 — Volume de Irrigação (L/m²)

| Modelo | MAE | RMSE | R² | CV-R² |
|--------|-----|------|-----|-------|
| Regressão Linear | 0.6133 | 0.7945 | 0.4908 | 0.5798 |
| Ridge Regression | 0.6123 | 0.7941 | 0.4914 | 0.5803 |
| Random Forest | 0.3819 | 0.5684 | 0.7394 | 0.7826 |
| **Gradient Boosting** | **0.3664** | **0.4839** | **0.8112** | **0.8313** |

#### 🎯 Alvo 2 — Necessidade de Fertilização (kg/ha)

| Modelo | MAE | RMSE | R² | CV-R² |
|--------|-----|------|-----|-------|
| Regressão Linear | 2.8831 | 3.7717 | 0.6254 | 0.6166 |
| Ridge Regression | 2.8817 | 3.7697 | 0.6258 | 0.6169 |
| Random Forest | 2.9162 | 3.6340 | 0.6523 | 0.7324 |
| **Gradient Boosting** | **2.2154** | **2.8903** | **0.7800** | **0.8016** |

#### 🎯 Alvo 3 — Rendimento Esperado (sc/ha)

| Modelo | MAE | RMSE | R² | CV-R² |
|--------|-----|------|-----|-------|
| Regressão Linear | 3.7152 | 4.4857 | 0.5774 | 0.3846 |
| Ridge Regression | 3.7183 | 4.4884 | 0.5768 | 0.3851 |
| Random Forest | 3.0735 | 4.1245 | 0.6427 | 0.6504 |
| **Gradient Boosting** | **2.7025** | **3.7914** | **0.6981** | **0.7067** |

**Melhor modelo em todos os alvos:** Gradient Boosting Regressor

### Interpretação das métricas

- **R² = 0.81** para volume de irrigação: o modelo explica 81% da variância
- **MAE = 0.37 L/m²** para irrigação: erro médio de menos de meio litro por m²
- **R² = 0.70** para rendimento: sólido considerando a complexidade agronômica real

---

## 6. Sugestões de Manejo em C++

Arquivo: `c_suggestions/farmtech_suggestions.cpp`

O programa recebe as leituras dos sensores ESP32 e as previsões do modelo Python e gera um relatório de ações de manejo com 4 dimensões:

```
[IRRIGACAO]      → LIGAR/NÃO LIGAR bomba com volume calculado
[CORRECAO DE PH] → Calagem ou acidificação com dose calculada
[FERTILIZACAO]   → Nível (alta/moderada/mínima) com kg/ha
[RENDIMENTO]     → Estimativa vs média regional com desvio %
[ALERTAS]        → Lista de alertas críticos priorizados
```

**Compilar e executar:**
```bash
g++ -std=c++17 -o farmtech_suggestions c_suggestions/farmtech_suggestions.cpp
./farmtech_suggestions
```

---

## 7. Banco de Dados Oracle

Arquivo: `db/farmtech_oracle_fase4.sql`

### Tabela `SENSOR_FARMTECH_F4`

Evolução direta da `SENSOR_FARMTECH` da Fase 3, com adição de:
- `N_KGHA`, `P_KGHA`, `K_KGHA` — valores contínuos dos nutrientes
- `VOLUME_IRRIGACAO_L` — variável-alvo de regressão
- `NECESSIDADE_FERT_KGHA` — variável-alvo de regressão
- `RENDIMENTO_ESPERADO_SCHA` — variável-alvo de regressão

### Conexão Oracle (FIAP)

```
Host    : oracle.fiap.com.br
Porta   : 1521
SID     : ORCL
Usuário : RM574005
Senha   : DDMMYY (data de nascimento)
```

### 10 Consultas SQL Implementadas

| # | Consulta |
|---|----------|
| 1 | SELECT * — todas as leituras |
| 2 | Leituras com bomba acionada |
| 3 | Médias dos parâmetros do solo |
| 4 | Médias das variáveis-alvo de regressão |
| 5 | Distribuição de motivos de decisão (%) |
| 6 | Totais de irrigação e volume acumulado |
| 7 | pH fora da faixa com recomendação de correção |
| 8 | Correlação entre faixa de umidade e rendimento |
| 9 | Top 10 leituras com maior rendimento |
| 10 | Tendência mensal de rendimento |

---

## 8. Estrutura de Pastas

```
FarmTech-Fase4/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── gerar_dataset.py              ← Gera sensor_data_fase4.csv
│   └── sensor_data_fase4.csv         ← 200 leituras simuladas (seed=574005)
│
├── ml/
│   ├── train_models.py               ← Pipeline ML + treinamento + métricas
│   └── models/
│       ├── model_volume_irrigacao.pkl
│       ├── model_necessidade_fert.pkl
│       ├── model_rendimento_esperado.pkl
│       └── metadata.json             ← Métricas e metadados dos modelos
│
├── dashboard/
│   └── farmtech_dashboard_fase4.py   ← Dashboard Streamlit completo
│
├── c_suggestions/
│   └── farmtech_suggestions.cpp      ← Engine de sugestões em C++
│
└── db/
    └── farmtech_oracle_fase4.sql     ← Criação de tabela + 10 queries
```

---

## 9. Como Executar

### Passo 1 — Instalar dependências Python

```bash
pip install -r requirements.txt
```

### Passo 2 — Gerar o dataset

```bash
python data/gerar_dataset.py
```

### Passo 3 — Treinar os modelos

```bash
python ml/train_models.py
```

### Passo 4 — Rodar o dashboard

```bash
streamlit run dashboard/farmtech_dashboard_fase4.py
```

### Passo 5 — Compilar e rodar o C++

```bash
g++ -std=c++17 -o farmtech_suggestions c_suggestions/farmtech_suggestions.cpp
./farmtech_suggestions
```

### Passo 6 — Banco de Dados Oracle

```
1. Conectar no Oracle SQL Developer (oracle.fiap.com.br | 1521 | ORCL | RM574005)
2. Executar db/farmtech_oracle_fase4.sql para criar a tabela
3. Importar data/sensor_data_fase4.csv como tabela SENSOR_FARMTECH_F4
4. Executar as queries analíticas do mesmo arquivo
```

---

## 10. Referências

- EMBRAPA. *Tecnologias de Produção de Soja — Região Central do Brasil 2023*. Disponível em: <https://www.embrapa.br>
- Scikit-learn. *Gradient Boosting Regressor*. Disponível em: <https://scikit-learn.org>
- Streamlit Documentation. Disponível em: <https://docs.streamlit.io>
- Plotly Documentation. Disponível em: <https://plotly.com/python>
- Oracle SQL Developer. Disponível em: <https://www.oracle.com/database/sqldeveloper/>
- FIAP. *Material de Aula — Inteligência Artificial, Fase 4*. 2026.

---

## 11. Vídeo Demonstrativo

> 🎥 **[Clique aqui para assistir no YouTube](#)** ← *substituir pelo link após gravar*

O vídeo demonstra:
- Execução do `train_models.py` com métricas MAE, RMSE, R² dos 4 modelos
- Dashboard Streamlit com previsões em tempo real via sliders
- Gráficos de correlação, séries temporais e feature importance
- Programa C++ com engine de recomendações de manejo
- Banco Oracle com a tabela da Fase 4 e as 10 queries

---

*Desenvolvado por **Kauan Maciel Forgiarini** — RM574005 | FIAP IA — Fase 4 | 2026*
