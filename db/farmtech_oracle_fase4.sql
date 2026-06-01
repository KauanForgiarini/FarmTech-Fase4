-- ============================================================
-- FarmTech Solutions — Fase 4
-- Script Oracle SQL — Banco de Dados Agrícola com ML
-- Kauan Maciel Forgiarini | RM574005 | FIAP IA
--
-- Conexão: oracle.fiap.com.br | Porta 1521 | SID ORCL
-- Usuário: RM574005 | Senha: DDMMYY (data de nascimento)
-- ============================================================


-- ─────────────────────────────────────────────
-- 1. CRIAÇÃO DA TABELA PRINCIPAL (Fase 4)
-- ─────────────────────────────────────────────

-- Remove tabela anterior se existir
BEGIN
    EXECUTE IMMEDIATE 'DROP TABLE SENSOR_FARMTECH_F4';
EXCEPTION
    WHEN OTHERS THEN NULL;
END;
/

CREATE TABLE SENSOR_FARMTECH_F4 (
    ID                        NUMBER        PRIMARY KEY,
    TIMESTAMP_LEITURA         VARCHAR2(25)  NOT NULL,
    CULTURA                   VARCHAR2(20)  DEFAULT 'Soja',

    -- Sensores booleanos (continuidade Fase 2/3)
    N_PRESENTE                NUMBER(1)     CHECK (N_PRESENTE IN (0,1)),
    P_PRESENTE                NUMBER(1)     CHECK (P_PRESENTE IN (0,1)),
    K_PRESENTE                NUMBER(1)     CHECK (K_PRESENTE IN (0,1)),

    -- Valores contínuos de NPK (NOVIDADE Fase 4 — para regressão)
    N_KGHA                    NUMBER(6,1),
    P_KGHA                    NUMBER(6,1),
    K_KGHA                    NUMBER(6,1),

    -- Sensores analógicos
    PH_SOLO                   NUMBER(4,2),
    UMIDADE_SOLO_PCT          NUMBER(5,1),
    TEMPERATURA_C             NUMBER(5,1),
    CHUVA_PREVISTA_MM         NUMBER(6,1),

    -- Decisão da bomba
    BOMBA_LIGADA              NUMBER(1)     CHECK (BOMBA_LIGADA IN (0,1)),
    MOTIVO_DECISAO            VARCHAR2(50),

    -- Variáveis alvo dos modelos de regressão (NOVIDADE Fase 4)
    VOLUME_IRRIGACAO_L        NUMBER(7,2),
    NECESSIDADE_FERT_KGHA     NUMBER(7,2),
    RENDIMENTO_ESPERADO_SCHA  NUMBER(7,2)
);

COMMENT ON TABLE SENSOR_FARMTECH_F4 IS 
    'Leituras dos sensores ESP32 com variáveis-alvo de regressão — FarmTech Fase 4';


-- ─────────────────────────────────────────────
-- 2. IMPORTAÇÃO DO CSV
-- ─────────────────────────────────────────────
-- Após criar a tabela, importe o arquivo sensor_data_fase4.csv:
--   Clique com botão direito em SENSOR_FARMTECH_F4
--   → Importar Dados → selecionar sensor_data_fase4.csv
--   → verificar mapeamento das colunas → Finalizar
-- ─────────────────────────────────────────────


-- ─────────────────────────────────────────────
-- 3. CONSULTAS ANALÍTICAS
-- ─────────────────────────────────────────────

-- Query 1: Todas as leituras ordenadas por ID
SELECT * FROM SENSOR_FARMTECH_F4 ORDER BY ID;


-- Query 2: Leituras com bomba acionada
SELECT
    ID,
    TIMESTAMP_LEITURA,
    UMIDADE_SOLO_PCT,
    PH_SOLO,
    VOLUME_IRRIGACAO_L,
    MOTIVO_DECISAO
FROM SENSOR_FARMTECH_F4
WHERE BOMBA_LIGADA = 1
ORDER BY ID;


-- Query 3: Médias gerais dos parâmetros do solo
SELECT
    ROUND(AVG(PH_SOLO), 2)                    AS MEDIA_PH,
    ROUND(AVG(UMIDADE_SOLO_PCT), 1)           AS MEDIA_UMIDADE_PCT,
    ROUND(AVG(TEMPERATURA_C), 1)              AS MEDIA_TEMP_C,
    ROUND(AVG(CHUVA_PREVISTA_MM), 1)          AS MEDIA_CHUVA_MM,
    ROUND(AVG(N_KGHA), 1)                     AS MEDIA_N_KGHA,
    ROUND(AVG(P_KGHA), 1)                     AS MEDIA_P_KGHA,
    ROUND(AVG(K_KGHA), 1)                     AS MEDIA_K_KGHA
FROM SENSOR_FARMTECH_F4;


-- Query 4: Médias das variáveis-alvo de regressão (NOVA — Fase 4)
SELECT
    ROUND(AVG(VOLUME_IRRIGACAO_L), 2)         AS MEDIA_VOL_IRRIGACAO,
    ROUND(MAX(VOLUME_IRRIGACAO_L), 2)         AS MAX_VOL_IRRIGACAO,
    ROUND(AVG(NECESSIDADE_FERT_KGHA), 2)     AS MEDIA_FERTILIZACAO,
    ROUND(AVG(RENDIMENTO_ESPERADO_SCHA), 2)  AS MEDIA_RENDIMENTO,
    ROUND(MAX(RENDIMENTO_ESPERADO_SCHA), 2)  AS MAX_RENDIMENTO,
    ROUND(MIN(RENDIMENTO_ESPERADO_SCHA), 2)  AS MIN_RENDIMENTO
FROM SENSOR_FARMTECH_F4;


-- Query 5: Distribuição de motivos de decisão
SELECT
    MOTIVO_DECISAO,
    COUNT(*)                                  AS OCORRENCIAS,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM SENSOR_FARMTECH_F4), 1) AS PCT
FROM SENSOR_FARMTECH_F4
GROUP BY MOTIVO_DECISAO
ORDER BY OCORRENCIAS DESC;


-- Query 6: Totais de irrigação
SELECT
    COUNT(*)                                  AS TOTAL_LEITURAS,
    SUM(BOMBA_LIGADA)                         AS VEZES_IRRIGOU,
    ROUND(AVG(BOMBA_LIGADA) * 100, 1)        AS PCT_IRRIGANDO,
    ROUND(SUM(VOLUME_IRRIGACAO_L), 2)        AS VOLUME_TOTAL_L
FROM SENSOR_FARMTECH_F4;


-- Query 7: Leituras com pH fora da faixa ideal (5,5–7,0)
SELECT
    ID,
    TIMESTAMP_LEITURA,
    PH_SOLO,
    MOTIVO_DECISAO,
    RENDIMENTO_ESPERADO_SCHA,
    CASE
        WHEN PH_SOLO < 5.5 THEN 'ACIDO — aplicar calcario'
        WHEN PH_SOLO > 7.0 THEN 'ALCALINO — aplicar enxofre'
        ELSE 'IDEAL'
    END AS STATUS_PH
FROM SENSOR_FARMTECH_F4
WHERE PH_SOLO < 5.5 OR PH_SOLO > 7.0
ORDER BY PH_SOLO;


-- Query 8: Correlação entre umidade e rendimento por faixa
SELECT
    CASE
        WHEN UMIDADE_SOLO_PCT < 40  THEN '1 - Critica (<40%)'
        WHEN UMIDADE_SOLO_PCT < 60  THEN '2 - Baixa (40-59%)'
        WHEN UMIDADE_SOLO_PCT < 80  THEN '3 - Ideal (60-79%)'
        ELSE                              '4 - Saturada (>=80%)'
    END AS FAIXA_UMIDADE,
    COUNT(*)                              AS QTD,
    ROUND(AVG(RENDIMENTO_ESPERADO_SCHA), 2) AS MEDIA_RENDIMENTO,
    ROUND(AVG(VOLUME_IRRIGACAO_L), 2)    AS MEDIA_VOL_IRRIGACAO
FROM SENSOR_FARMTECH_F4
GROUP BY
    CASE
        WHEN UMIDADE_SOLO_PCT < 40  THEN '1 - Critica (<40%)'
        WHEN UMIDADE_SOLO_PCT < 60  THEN '2 - Baixa (40-59%)'
        WHEN UMIDADE_SOLO_PCT < 80  THEN '3 - Ideal (60-79%)'
        ELSE                              '4 - Saturada (>=80%)'
    END
ORDER BY FAIXA_UMIDADE;


-- Query 9: Top 10 leituras com maior rendimento estimado
SELECT *
FROM SENSOR_FARMTECH_F4
ORDER BY RENDIMENTO_ESPERADO_SCHA DESC
FETCH FIRST 10 ROWS ONLY;


-- Query 10: Tendência mensal de rendimento (agrupado por mês)
SELECT
    SUBSTR(TIMESTAMP_LEITURA, 1, 7)            AS MES,
    COUNT(*)                                   AS LEITURAS,
    ROUND(AVG(RENDIMENTO_ESPERADO_SCHA), 2)   AS MEDIA_RENDIMENTO,
    ROUND(AVG(UMIDADE_SOLO_PCT), 1)            AS MEDIA_UMIDADE,
    ROUND(AVG(PH_SOLO), 2)                     AS MEDIA_PH
FROM SENSOR_FARMTECH_F4
GROUP BY SUBSTR(TIMESTAMP_LEITURA, 1, 7)
ORDER BY MES;
