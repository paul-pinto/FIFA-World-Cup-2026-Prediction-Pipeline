# FIFA World Cup 2026 Prediction Pipeline

Sistema automatizado para generar pronósticos diarios de partidos de la **FIFA World Cup 2026**, combinando modelos de Machine Learning, ratings Elo, distribución de goles con Poisson/Dixon-Coles, simulaciones Monte Carlo, cuotas de mercado y detección de value bets.

El objetivo del proyecto es producir, de forma diaria y reproducible:

* Probabilidades 1X2: local, empate, visitante.
* Goles esperados por equipo.
* Probabilidades de marcador exacto.
* Top scores más probables.
* Probabilidades de Over/Under 2.5.
* Probabilidad de BTTS.
* Simulaciones Monte Carlo.
* Comparación contra cuotas reales.
* Cálculo de edge, EV y value bets.
* Reportes diarios en CSV, JSON, Excel y Markdown.
* Notificaciones automáticas por Telegram.
* Evaluación post-partido.
* Reentrenamiento incremental con resultados nuevos.

---

## Tabla de contenidos

* [Descripción general](#descripción-general)
* [Arquitectura](#arquitectura)
* [Metodologías utilizadas](#metodologías-utilizadas)
* [Estructura del proyecto](#estructura-del-proyecto)
* [Fuentes de datos](#fuentes-de-datos)
* [Instalación local](#instalación-local)
* [Variables de entorno](#variables-de-entorno)
* [Flujo principal](#flujo-principal)
* [Entrenamiento inicial](#entrenamiento-inicial)
* [Predicción diaria](#predicción-diaria)
* [Odds y consenso no-vig](#odds-y-consenso-no-vig)
* [Value bets](#value-bets)
* [Evaluación post-partido](#evaluación-post-partido)
* [Reentrenamiento](#reentrenamiento)
* [Pipeline completo](#pipeline-completo)
* [Telegram](#telegram)
* [GitHub Actions](#github-actions)
* [DigitalOcean](#digitalocean)
* [Outputs generados](#outputs-generados)
* [Métricas](#métricas)
* [Notas operativas](#notas-operativas)
* [Limitaciones](#limitaciones)
* [Roadmap](#roadmap)
* [Disclaimer](#disclaimer)

---

# Descripción general

Este proyecto implementa un pipeline predictivo para partidos de fútbol internacional, optimizado para la FIFA World Cup 2026.

La idea central no es simplemente predecir un ganador, sino producir una distribución probabilística completa del partido:

```text
HOME WIN
DRAW
AWAY WIN
Exact scores
Expected goals
Over/Under
BTTS
Value opportunities
```

El sistema funciona como un pipeline diario:

```text
resultados previos
    ↓
evaluación
    ↓
sincronización de resultados
    ↓
reentrenamiento
    ↓
descarga de odds
    ↓
predicción de la jornada
    ↓
exportación de reportes
    ↓
notificación por Telegram
```

El proyecto no depende de una aplicación web. El core es un conjunto de scripts Python ejecutables desde CLI, GitHub Actions, cron o Task Scheduler.

---

# Arquitectura

El sistema se divide en varias capas:

```text
DATA LAYER
- histórico internacional
- fixtures del Mundial
- resultados manuales
- cuotas reales
- snapshots de odds

FEATURE LAYER
- Elo pre-partido
- forma reciente
- goles a favor/en contra
- puntos recientes
- fuerza ofensiva/defensiva

MODEL LAYER
- Machine Learning 1X2
- Machine Learning goles esperados
- Machine Learning Over 2.5
- Machine Learning BTTS
- Poisson
- Dixon-Coles
- Monte Carlo
- Ensemble final

MARKET LAYER
- The Odds API
- consenso no-vig multi-bookmaker
- cuotas sintéticas
- implied probabilities
- edge
- expected value

OUTPUT LAYER
- CSV
- JSON
- Excel
- Markdown
- Telegram
```

---

# Metodologías utilizadas

## 1. Elo dinámico

Se calcula un rating Elo pre-partido para cada selección usando el histórico internacional.

El Elo se actualiza después de cada partido considerando:

* Resultado.
* Diferencia de goles.
* Peso de la competición.
* Fuerza relativa de ambos equipos.

El modelo genera:

```text
elo_home_pre
elo_away_pre
elo_diff_pre
```

---

## 2. Features de forma

Para cada equipo se calculan estadísticas previas al partido usando ventanas móviles:

```text
últimos 5 partidos
últimos 10 partidos
últimos 20 partidos
```

Features principales:

```text
home_gf_5
home_ga_5
away_gf_5
away_ga_5
home_points_5
away_points_5
goal_diff_form_5
points_form_diff_5
attack_diff_5
defense_diff_5
```

Estas features se calculan sin leakage: solo usan partidos anteriores al partido objetivo.

---

## 3. Machine Learning

Se entrenan varios modelos con `scikit-learn`:

### Modelo 1X2

Clasificador multiclase:

```text
0 = HOME
1 = DRAW
2 = AWAY
```

Modelo usado:

```text
HistGradientBoostingClassifier
```

Salida:

```text
p_home_ml
p_draw_ml
p_away_ml
```

### Modelo de goles esperados

Dos regresores:

```text
model_home_goals.pkl
model_away_goals.pkl
```

Salida:

```text
lambda_home_ml
lambda_away_ml
```

### Modelo Over 2.5

Clasificador binario:

```text
target_over25 = total_goals > 2.5
```

### Modelo BTTS

Clasificador binario:

```text
target_btts = home_score > 0 and away_score > 0
```

---

## 4. Poisson / Dixon-Coles

A partir de los goles esperados:

```text
lambda_home
lambda_away
```

se construye una matriz de probabilidades de marcador exacto:

```text
P(0-0)
P(1-0)
P(1-1)
P(2-1)
...
```

El ajuste Dixon-Coles corrige la dependencia en marcadores bajos:

```text
0-0
1-0
0-1
1-1
```

Esto mejora el comportamiento del modelo en partidos cerrados.

---

## 5. Monte Carlo

Se simulan miles de partidos usando la matriz de scores como distribución base.

Por defecto:

```text
N_SIM=200000
```

La simulación produce:

```text
home_win
draw
away_win
over25
under25
btts_yes
btts_no
avg_home_goals
avg_away_goals
avg_total_goals
```

---

## 6. Ensemble final

El sistema combina distintas fuentes:

```text
ML
Dixon-Coles
Mercado
```

Cuando hay cuotas reales disponibles, el ensemble incorpora el consenso de mercado:

```text
final_probability =
    market component
  + ML component
  + Dixon-Coles component
```

Cuando no hay odds, el sistema usa fallback:

```text
ML + Dixon-Coles
```

---

# Estructura del proyecto

```text
worldcup_model/
│
├── data/
│   ├── raw/
│   │   ├── international_results/
│   │   │   └── results.csv
│   │   └── odds_api/
│   │
│   ├── master/
│   │   ├── worldcup_2026_fixtures.csv
│   │   ├── team_aliases.csv
│   │   ├── manual_odds.csv
│   │   ├── manual_results.csv
│   │   └── worldcup_results.csv
│   │
│   └── processed/
│       ├── matches.parquet
│       ├── matches_with_elo.parquet
│       ├── training_dataset.parquet
│       └── odds_snapshots.csv
│
├── models/
│   ├── model_1x2.pkl
│   ├── model_home_goals.pkl
│   ├── model_away_goals.pkl
│   ├── model_over25.pkl
│   ├── model_btts.pkl
│   ├── feature_columns.json
│   └── training_metrics.json
│
├── outputs/
│   ├── daily/
│   │   ├── predictions_YYYY-MM-DD.csv
│   │   ├── predictions_YYYY-MM-DD.json
│   │   ├── predictions_YYYY-MM-DD.xlsx
│   │   ├── evaluation_YYYY-MM-DD.csv
│   │   └── evaluation_YYYY-MM-DD.json
│   │
│   └── reports/
│       ├── daily_report_YYYY-MM-DD.md
│       └── evaluation_report_YYYY-MM-DD.md
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── elo.py
│   ├── features.py
│   ├── ml_models.py
│   ├── poisson_dc.py
│   ├── montecarlo.py
│   ├── predictor.py
│   ├── market.py
│   ├── odds_api.py
│   ├── value.py
│   ├── exporter.py
│   ├── run_daily.py
│   ├── results.py
│   ├── evaluator.py
│   ├── sync_worldcup_results.py
│   ├── retrain.py
│   ├── pipeline.py
│   ├── telegram_notify.py
│   └── telegram_bot.py
│
├── scripts/
│   └── run_daily.ps1
│
├── .github/
│   └── workflows/
│       └── daily.yml
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

# Fuentes de datos

## Histórico internacional

El entrenamiento usa un dataset histórico de partidos internacionales con columnas:

```csv
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
```

Archivo esperado:

```text
data/raw/international_results/results.csv
```

Este dataset es la base para:

* Elo.
* Forma.
* Machine Learning.
* Goles esperados.
* Over/Under.
* BTTS.

---

## Fixtures del Mundial 2026

Archivo:

```text
data/master/worldcup_2026_fixtures.csv
```

Formato:

```csv
match_id,date_utc,stage,group,home_team,away_team,venue,city,country,neutral
WC2026-001,2026-06-11T19:00:00Z,Group Stage,A,Mexico,South Africa,Mexico City Stadium,Mexico City,Mexico,0
```

El sistema trabaja en UTC.

---

## Team aliases

Archivo:

```text
data/master/team_aliases.csv
```

Sirve para normalizar nombres entre distintas fuentes:

```csv
canonical,alias
Czechia,Czech Republic
Bosnia and Herzegovina,Bosnia & Herzegovina
United States,USA
South Korea,Korea Republic
```

Esto evita errores al matchear:

```text
The Odds API ↔ fixtures ↔ dataset histórico
```

---

## Odds

Las cuotas pueden venir de dos fuentes:

### 1. Manual

Archivo:

```text
data/master/manual_odds.csv
```

Formato:

```csv
match_id,odds_home,odds_draw,odds_away,odds_over25,odds_under25
WC2026-001,1.55,4.00,6.50,1.95,1.85
```

### 2. The Odds API

Script:

```bash
python -m src.odds_api --date 2026-06-12
```

La API descarga odds reales y actualiza:

```text
data/master/manual_odds.csv
data/processed/odds_snapshots.csv
data/raw/odds_api/
```

---

# Instalación local

## 1. Crear entorno virtual

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 2. Instalar dependencias

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 3. Verificar instalación

```bash
python -c "import pandas, numpy, scipy, sklearn; print('OK')"
```

---

# Variables de entorno

Crear archivo `.env`:

```env
THE_ODDS_API_KEY=your_odds_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

ODDS_SPORT_KEY=soccer_fifa_world_cup
ODDS_REGIONS=us,uk,eu
ODDS_MARKETS=h2h,totals
ODDS_FORMAT=decimal

N_SIM=200000
MAX_GOALS=10
SEED=42
```

## Variables principales

| Variable             | Descripción                        |
| -------------------- | ---------------------------------- |
| `THE_ODDS_API_KEY`   | API key para The Odds API          |
| `TELEGRAM_BOT_TOKEN` | Token del bot de Telegram          |
| `TELEGRAM_CHAT_ID`   | Chat ID autorizado                 |
| `ODDS_SPORT_KEY`     | Sport key usado por The Odds API   |
| `ODDS_REGIONS`       | Regiones de bookmakers             |
| `ODDS_MARKETS`       | Mercados solicitados               |
| `N_SIM`              | Número de simulaciones Monte Carlo |
| `MAX_GOALS`          | Máximo de goles en matriz de score |
| `SEED`               | Semilla reproducible               |

---

# Flujo principal

El pipeline completo sigue esta secuencia:

```text
1. Evaluar predicciones anteriores
2. Sincronizar resultados al dataset
3. Reentrenar modelos
4. Descargar cuotas reales
5. Generar predicciones
6. Exportar reportes
7. Enviar Telegram
```

Comando principal:

```bash
python -m src.pipeline full \
  --eval-date 2026-06-11 \
  --predict-date 2026-06-12 \
  --fetch-odds \
  --telegram
```

---

# Entrenamiento inicial

## 1. Procesar histórico

```bash
python -m src.data_loader
```

Genera:

```text
data/processed/matches.parquet
```

---

## 2. Calcular Elo

```bash
python -m src.elo
```

Genera:

```text
data/processed/matches_with_elo.parquet
```

---

## 3. Calcular features

```bash
python -m src.features
```

Genera:

```text
data/processed/training_dataset.parquet
```

---

## 4. Entrenar modelos ML

```bash
python -m src.ml_models
```

Genera:

```text
models/model_1x2.pkl
models/model_home_goals.pkl
models/model_away_goals.pkl
models/model_over25.pkl
models/model_btts.pkl
models/feature_columns.json
models/training_metrics.json
```

---

# Predicción diaria

Para predecir una fecha:

```bash
python -m src.run_daily --date 2026-06-12
```

Esto lee:

```text
data/master/worldcup_2026_fixtures.csv
data/master/manual_odds.csv
models/*.pkl
```

y genera:

```text
outputs/daily/predictions_2026-06-12.csv
outputs/daily/predictions_2026-06-12.json
outputs/daily/predictions_2026-06-12.xlsx
outputs/reports/daily_report_2026-06-12.md
```

---

# Odds y consenso no-vig

El módulo:

```text
src/odds_api.py
```

descarga cuotas reales desde The Odds API.

Comando:

```bash
python -m src.odds_api --date 2026-06-12
```

El sistema:

```text
1. Descarga odds por bookmaker.
2. Guarda raw JSON.
3. Extrae h2h y totals.
4. Convierte odds a probabilidades implícitas.
5. Remueve vig por bookmaker.
6. Promedia probabilidades no-vig.
7. Convierte el consenso a cuotas sintéticas.
8. Actualiza manual_odds.csv.
9. Guarda odds_snapshots.csv.
```

## Ejemplo conceptual

Bookmaker odds:

```text
Home: 2.10
Draw: 3.30
Away: 3.50
```

Probabilidades implícitas:

```text
1 / odds
```

Luego se normalizan para remover overround:

```text
p_home + p_draw + p_away = 1
```

Finalmente se promedia entre bookmakers para generar el consenso.

---

# Value bets

El módulo:

```text
src/value.py
```

calcula:

```text
implied_probability = 1 / odds
edge = model_probability - implied_probability
EV = model_probability * odds - 1
```

Reglas iniciales:

```text
MIN_EV = 0.03
MIN_EDGE = 0.02
```

Un mercado se marca como value si:

```text
EV >= 3%
edge >= 2%
```

Los mercados evaluados son:

```text
HOME
DRAW
AWAY
OVER_2_5
UNDER_2_5
```

El reporte incluye:

```text
best_value_market
best_value_odds
best_value_probability
best_value_edge
best_value_ev
```

---

# Evaluación post-partido

Los resultados manuales se cargan en:

```text
data/master/manual_results.csv
```

Formato:

```csv
match_id,home_score,away_score,status
WC2026-001,2,0,FINISHED
WC2026-002,1,1,FINISHED
```

Luego se evalúa una fecha:

```bash
python -m src.evaluator --date 2026-06-12
```

Genera:

```text
outputs/daily/evaluation_2026-06-12.csv
outputs/daily/evaluation_2026-06-12.json
outputs/reports/evaluation_report_2026-06-12.md
```

---

# Reentrenamiento

Para incorporar resultados nuevos al dataset:

```bash
python -m src.sync_worldcup_results --print
python -m src.retrain
```

El flujo es:

```text
manual_results.csv
    ↓
worldcup_results.csv
    ↓
matches.parquet
    ↓
matches_with_elo.parquet
    ↓
training_dataset.parquet
    ↓
models/*.pkl
```

Comando completo:

```bash
python -m src.retrain
```

---

# Pipeline completo

El módulo principal de orquestación es:

```text
src/pipeline.py
```

## Predicción simple

```bash
python -m src.pipeline predict --date 2026-06-12
```

## Evaluar fecha

```bash
python -m src.pipeline evaluate --date 2026-06-12
```

## Descargar odds

```bash
python -m src.pipeline fetch-odds --date 2026-06-12
```

## Reentrenar

```bash
python -m src.pipeline retrain
```

## Pipeline full

```bash
python -m src.pipeline full \
  --eval-date 2026-06-11 \
  --predict-date 2026-06-12 \
  --fetch-odds \
  --telegram
```

---

# Telegram

El sistema tiene dos formas de usar Telegram.

## 1. Notificación saliente

Manda el reporte generado al chat configurado.

```bash
python -m src.telegram_notify send --date 2026-06-12
```

Status:

```bash
python -m src.telegram_notify status
```

## 2. Bot interactivo

El bot puede escuchar comandos:

```bash
python -m src.telegram_bot
```

Comandos disponibles:

```text
/help
/status
/hoy
/manana
/ayer
/fecha YYYY-MM-DD
/ultimo
```

Importante:

```text
telegram_notify.py = envío puntual
telegram_bot.py    = listener interactivo
```

Para producción, el bot interactivo necesita un proceso vivo 24/7, por ejemplo con DigitalOcean + systemd.

---

# GitHub Actions

El proyecto puede correr automáticamente desde GitHub Actions.

Archivo recomendado:

```text
.github/workflows/daily.yml
```

El workflow debe:

```text
1. Instalar Python.
2. Instalar requirements.
3. Calcular fechas.
4. Ejecutar pipeline full.
5. Descargar odds.
6. Mandar Telegram.
7. Commit de outputs generados.
```

Para enviar el mensaje diario a las 10:00 AM Bolivia:

```yaml
schedule:
  # 10:00 Bolivia = 14:00 UTC
  - cron: "0 14 * * *"
```

Secrets requeridos:

```text
THE_ODDS_API_KEY
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

---

# DigitalOcean

También puede desplegarse en un Droplet Ubuntu.

Recomendado:

```text
2 GB RAM mínimo
Python 3.11+
Swap si el Droplet es pequeño
cron para pipeline
systemd para telegram_bot
```

Ejemplo de ejecución en servidor:

```bash
python -m src.pipeline full \
  --eval-date 2026-06-11 \
  --predict-date 2026-06-12 \
  --fetch-odds \
  --telegram
```

---

# Outputs generados

## Predicciones

```text
outputs/daily/predictions_YYYY-MM-DD.csv
outputs/daily/predictions_YYYY-MM-DD.json
outputs/daily/predictions_YYYY-MM-DD.xlsx
outputs/reports/daily_report_YYYY-MM-DD.md
```

Columnas principales:

```text
match_id
date_utc
home_team
away_team
lambda_home
lambda_away
ml_p_home
ml_p_draw
ml_p_away
dc_p_home
dc_p_draw
dc_p_away
final_p_home
final_p_draw
final_p_away
pick
confidence
top_score_1
top_score_1_prob
best_value_market
best_value_ev
best_edge_market
best_edge_ev
```

---

## Evaluaciones

```text
outputs/daily/evaluation_YYYY-MM-DD.csv
outputs/daily/evaluation_YYYY-MM-DD.json
outputs/reports/evaluation_report_YYYY-MM-DD.md
```

Columnas principales:

```text
match_id
actual_score
actual_1x2
pick
correct_1x2
exact_score_hit
over25_hit
btts_hit
log_loss_1x2
brier_1x2
```

---

## Odds snapshots

```text
data/processed/odds_snapshots.csv
```

Contiene:

```text
snapshot_utc
target_date
match_id
odds_home
odds_draw
odds_away
odds_over25
odds_under25
p_home_consensus
p_draw_consensus
p_away_consensus
p_over25_consensus
p_under25_consensus
bookmakers_h2h
bookmakers_totals
avg_overround_h2h
avg_overround_totals
```

Esto permite medir:

```text
line movement
closing line value
market drift
opening vs closing odds
```

---

# Métricas

El sistema calcula:

## 1X2

```text
accuracy_1x2
log_loss_1x2
brier_1x2
```

## Score exacto

```text
exact_score_hit_rate
actual_score_top10_prob
```

## Mercados derivados

```text
over25_accuracy
btts_accuracy
```

## ML training

```text
accuracy_1x2
log_loss_1x2
mae_home_goals
mae_away_goals
auc_over25
auc_btts
```

---

# Notas operativas

## Fechas

El sistema trabaja en UTC.

Esto significa que un partido jugado de noche en América puede pertenecer al día siguiente UTC.

Ejemplo:

```text
2026-06-12T02:00:00Z
```

Operativamente puede sentirse como la noche del 11, pero para el sistema pertenece al 12.

---

## Resultados

Los resultados se cargan manualmente en:

```text
data/master/manual_results.csv
```

Luego el sistema los sincroniza automáticamente hacia:

```text
data/master/worldcup_results.csv
```

---

## Odds

Si The Odds API no devuelve datos, el sistema puede usar `manual_odds.csv` como fallback.

Si no hay odds para un partido, el modelo sigue funcionando con:

```text
ML + Dixon-Coles
```

---

## Telegram

El mensaje automatizado se puede enviar al final del pipeline con:

```bash
--telegram
```

Ejemplo:

```bash
python -m src.pipeline full \
  --eval-date 2026-06-11 \
  --predict-date 2026-06-12 \
  --fetch-odds \
  --telegram
```

---

# Limitaciones

Este sistema es probabilístico. No garantiza aciertos.

Limitaciones actuales:

```text
- No usa alineaciones oficiales en tiempo real.
- No usa lesiones/suspensiones automáticamente.
- No usa xG profesional.
- No simula todavía fase de grupos completa ni clasificación.
- Los resultados se cargan manualmente.
- El modelo ML se entrena con histórico internacional general.
- Las cuotas dependen de disponibilidad de The Odds API.
- El bot interactivo requiere proceso persistente.
```

---

# Roadmap

## Corto plazo

```text
- Completar fixtures del Mundial.
- Automatizar resultados.
- Mejorar aliases de equipos.
- Mejorar reportes Telegram.
- Añadir control de errores por partido.
- Añadir logs estructurados.
```

## Medio plazo

```text
- Closing line value.
- Line movement.
- Snapshots por hora.
- Ranking de value bets por EV.
- Telegram interactivo 24/7.
- Dashboard simple.
```

## Largo plazo

```text
- Simulación de grupos.
- Simulación de knockout.
- Probabilidad de clasificación.
- Simulación completa del torneo.
- Integración de lineups.
- Integración de lesiones.
- Modelos calibrados por fase del torneo.
- Calibración isotónica/Platt.
- Base de datos PostgreSQL.
```

---

# Comandos útiles

## Entrenar desde cero

```bash
python -m src.data_loader
python -m src.elo
python -m src.features
python -m src.ml_models
```

## Descargar odds

```bash
python -m src.odds_api --date 2026-06-12
```

## Predecir

```bash
python -m src.run_daily --date 2026-06-12
```

## Evaluar

```bash
python -m src.evaluator --date 2026-06-12
```

## Sincronizar resultados

```bash
python -m src.sync_worldcup_results --print
```

## Reentrenar

```bash
python -m src.retrain
```

## Pipeline completo

```bash
python -m src.pipeline full \
  --eval-date 2026-06-11 \
  --predict-date 2026-06-12 \
  --fetch-odds \
  --telegram
```

## Telegram

```bash
python -m src.telegram_notify status
python -m src.telegram_notify send --date 2026-06-12
python -m src.telegram_bot
```

---

# Disclaimer

Este proyecto tiene fines educativos, analíticos y experimentales.

Las predicciones son probabilísticas y no constituyen garantía de resultado. El uso de estas predicciones para apuestas debe hacerse de forma responsable, legal y bajo criterio propio.

El sistema intenta estimar probabilidades y detectar posibles discrepancias frente al mercado, pero el fútbol tiene alta varianza y los resultados pueden desviarse significativamente de cualquier modelo.

No se recomienda usar este sistema como única base para decisiones financieras.
