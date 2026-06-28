# src/run_daily.py

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from src.predictor import predict_match
from src.market import get_manual_odds_for_match, market_from_odds
from src.value import evaluate_value_bets
from src.knockout import estimate_advance_probabilities


ROOT = Path(__file__).resolve().parents[1]

FIXTURES_PATH = ROOT / "data" / "master" / "worldcup_2026_fixtures.csv"
OUTPUTS_DAILY = ROOT / "outputs" / "daily"
OUTPUTS_REPORTS = ROOT / "outputs" / "reports"

LOCAL_TZ = "America/La_Paz"


def load_fixtures() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES_PATH)

    required = [
        "match_id",
        "date_utc",
        "stage",
        "group",
        "home_team",
        "away_team",
        "venue",
        "city",
        "country",
        "neutral",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas en worldcup_2026_fixtures.csv: {missing}")

    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True, errors="coerce")

    bad_dates = df[df["date_utc"].isna()]

    if not bad_dates.empty:
        raise ValueError(
            "Hay filas con date_utc inválido en worldcup_2026_fixtures.csv:\n"
            + bad_dates.to_string(index=False)
        )

    df["kickoff_local"] = df["date_utc"].dt.tz_convert(LOCAL_TZ)
    df["date_local"] = df["kickoff_local"].dt.strftime("%Y-%m-%d")

    return df


def filter_fixtures_by_date(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """
    Filtra por fecha calendario Bolivia, no por fecha UTC.

    Ejemplo:
    Sweden vs Tunisia = 2026-06-15T02:00:00Z
    En Bolivia es 2026-06-14 22:00, por tanto debe entrar en --date 2026-06-14.
    """

    target_date = pd.to_datetime(target_date).strftime("%Y-%m-%d")

    today = df[df["date_local"] == target_date].copy()
    today = today.sort_values("kickoff_local")

    return today


def _safe_get(d: Any, key: str, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def _top_scores_columns(pred: dict) -> dict:
    out = {}

    top_scores = pred.get("top_scores") or []

    for i in range(1, 11):
        score = None
        prob = None

        if i <= len(top_scores):
            item = top_scores[i - 1]

            if isinstance(item, dict):
                home_goals = item.get("home_goals", item.get("home", item.get("h")))
                away_goals = item.get("away_goals", item.get("away", item.get("a")))
                prob = item.get("prob", item.get("probability", item.get("p")))

                if home_goals is not None and away_goals is not None:
                    score = f"{home_goals}-{away_goals}"

            elif isinstance(item, (list, tuple)):
                if len(item) >= 3:
                    score = f"{item[0]}-{item[1]}"
                    prob = item[2]
                elif len(item) >= 2:
                    score = str(item[0])
                    prob = item[1]

        out[f"top_score_{i}"] = score
        out[f"top_score_{i}_prob"] = prob

    return out


def flatten_prediction(pred: dict) -> dict:
    market = pred.get("market") or {}
    ml = pred.get("ml") or {}
    dc = pred.get("dixon_coles") or {}
    mc = pred.get("monte_carlo") or {}
    final = pred.get("final") or {}
    value = pred.get("value") or {}

    row = {
        "match_id": pred.get("match_id"),
        "date_utc": pred.get("date_utc"),
        "date_local": pred.get("date_local"),
        "kickoff_local": pred.get("kickoff_local"),
        "stage": pred.get("stage"),
        "group": pred.get("group"),
        "venue": pred.get("venue"),
        "city": pred.get("city"),
        "country": pred.get("country"),
        "home_team": pred.get("home_team"),
        "away_team": pred.get("away_team"),
        "neutral": pred.get("neutral"),
        "lambda_home": pred.get("lambda_home"),
        "lambda_away": pred.get("lambda_away"),

        "has_market": bool(market),

        "market_p_home": _safe_get(market, "p_home"),
        "market_p_draw": _safe_get(market, "p_draw"),
        "market_p_away": _safe_get(market, "p_away"),
        "market_p_over25": _safe_get(market, "p_over25"),
        "market_lambda_home": _safe_get(market, "lambda_home"),
        "market_lambda_away": _safe_get(market, "lambda_away"),

        "ml_p_home": _safe_get(ml, "p_home"),
        "ml_p_draw": _safe_get(ml, "p_draw"),
        "ml_p_away": _safe_get(ml, "p_away"),
        "ml_p_over25": _safe_get(ml, "p_over25"),
        "ml_p_btts_yes": _safe_get(ml, "p_btts_yes"),

        "dc_p_home": _safe_get(dc, "p_home"),
        "dc_p_draw": _safe_get(dc, "p_draw"),
        "dc_p_away": _safe_get(dc, "p_away"),
        "dc_p_over25": _safe_get(dc, "p_over25"),
        "dc_p_under25": _safe_get(dc, "p_under25"),
        "dc_p_btts_yes": _safe_get(dc, "p_btts_yes"),
        "dc_p_btts_no": _safe_get(dc, "p_btts_no"),

        "mc_p_home": _safe_get(mc, "p_home"),
        "mc_p_draw": _safe_get(mc, "p_draw"),
        "mc_p_away": _safe_get(mc, "p_away"),
        "mc_p_over25": _safe_get(mc, "p_over25"),
        "mc_p_btts_yes": _safe_get(mc, "p_btts_yes"),
        "mc_avg_home_goals": _safe_get(mc, "avg_home_goals"),
        "mc_avg_away_goals": _safe_get(mc, "avg_away_goals"),
        "mc_avg_total_goals": _safe_get(mc, "avg_total_goals"),

        "final_p_home": _safe_get(final, "p_home"),
        "final_p_draw": _safe_get(final, "p_draw"),
        "final_p_away": _safe_get(final, "p_away"),
        "pick": _safe_get(final, "pick"),
        "confidence": _safe_get(final, "confidence"),

        # Knockout / clasificación
        "knockout": pred.get("knockout"),
        "advance_available": pred.get("advance_available"),
        "p_home_advance": pred.get("p_home_advance"),
        "p_away_advance": pred.get("p_away_advance"),
        "advance_team": pred.get("advance_team"),
        "advance_confidence": pred.get("advance_confidence"),
        "shootout_edge_home": pred.get("shootout_edge_home"),
        "advance_note": pred.get("advance_note"),

        "has_value_data": bool(value),
        "best_value_market": _safe_get(value, "best_value_market"),
        "best_value_odds": _safe_get(value, "best_value_odds"),
        "best_value_probability": _safe_get(value, "best_value_probability"),
        "best_value_edge": _safe_get(value, "best_value_edge"),
        "best_value_ev": _safe_get(value, "best_value_ev"),
        "best_edge_market": _safe_get(value, "best_edge_market"),
        "best_edge_odds": _safe_get(value, "best_edge_odds"),
        "best_edge_probability": _safe_get(value, "best_edge_probability"),
        "best_edge": _safe_get(value, "best_edge"),
        "best_edge_ev": _safe_get(value, "best_edge_ev"),
    }

    row.update(_top_scores_columns(pred))

    return row


def predict_fixture(row) -> dict:
    match_id = row.get("match_id", "")

    odds = get_manual_odds_for_match(match_id)

    if odds is not None:
        market = market_from_odds(odds)
        print("    odds: OK")
    else:
        market = None
        print("    odds: NO DISPONIBLES")

    pred = predict_match(
        home_team=row["home_team"],
        away_team=row["away_team"],
        neutral=int(row.get("neutral", 1)),
        market=market,
    )

    pred["odds"] = odds
    pred["value"] = evaluate_value_bets(pred, odds)

    stage = str(row.get("stage", "")).strip()
    is_knockout = stage.lower() not in ["group stage", "group", "fase de grupos"]

    if is_knockout:
        knockout_info = estimate_advance_probabilities(pred)
        pred.update(knockout_info)

        if knockout_info.get("advance_available"):
            if knockout_info.get("advance_pick") == "home":
                pred["advance_team"] = row["home_team"]
            else:
                pred["advance_team"] = row["away_team"]
        else:
            pred["advance_team"] = None
    else:
        pred["knockout"] = False
        pred["advance_available"] = False
        pred["p_home_advance"] = None
        pred["p_away_advance"] = None
        pred["advance_pick"] = None
        pred["advance_team"] = None
        pred["advance_confidence"] = None
        pred["shootout_edge_home"] = None
        pred["advance_note"] = None

    pred["match_id"] = match_id
    pred["date_utc"] = row["date_utc"].isoformat()
    pred["date_local"] = row.get("date_local", "")

    pred["kickoff_local"] = (
        row["kickoff_local"].isoformat()
        if "kickoff_local" in row and pd.notna(row["kickoff_local"])
        else ""
    )

    pred["stage"] = stage
    pred["group"] = row.get("group", "")
    pred["venue"] = row.get("venue", "")
    pred["city"] = row.get("city", "")
    pred["country"] = row.get("country", "")
    pred["home_team"] = row["home_team"]
    pred["away_team"] = row["away_team"]
    pred["neutral"] = int(row.get("neutral", 1))

    return pred


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def write_markdown_report(path: Path, target_date: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Daily predictions - {target_date}")
    lines.append("")

    if not rows:
        lines.append("No hay partidos para esta fecha.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    for r in rows:
        lines.append(f"## {r.get('home_team')} vs {r.get('away_team')}")
        lines.append("")
        lines.append(f"- Match ID: `{r.get('match_id')}`")
        lines.append(f"- Stage: {r.get('stage')}")
        lines.append(f"- Kickoff Bolivia: {r.get('kickoff_local')}")
        lines.append("")
        lines.append("### 90 minutos")
        lines.append(f"- Local: {r.get('final_p_home')}")
        lines.append(f"- Empate: {r.get('final_p_draw')}")
        lines.append(f"- Visitante: {r.get('final_p_away')}")
        lines.append(f"- Pick 90': {r.get('pick')}")
        lines.append(f"- Confianza 90': {r.get('confidence')}")
        lines.append("")

        if r.get("knockout"):
            lines.append("### Clasificación")
            lines.append(f"- Clasifica {r.get('home_team')}: {r.get('p_home_advance')}")
            lines.append(f"- Clasifica {r.get('away_team')}: {r.get('p_away_advance')}")
            lines.append(f"- Pick clasifica: {r.get('advance_team')}")
            lines.append(f"- Confianza clasificación: {r.get('advance_confidence')}")
            lines.append("")

        if r.get("best_value_market"):
            lines.append("### Value")
            lines.append(f"- Mejor value: {r.get('best_value_market')}")
            lines.append(f"- Cuota: {r.get('best_value_odds')}")
            lines.append(f"- Probabilidad: {r.get('best_value_probability')}")
            lines.append(f"- Edge: {r.get('best_value_edge')}")
            lines.append(f"- EV: {r.get('best_value_ev')}")
            lines.append("")

        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def export_predictions(target_date: str, predictions: list[dict]) -> tuple[Path, Path, Path, Path]:
    OUTPUTS_DAILY.mkdir(parents=True, exist_ok=True)
    OUTPUTS_REPORTS.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUTS_DAILY / f"predictions_{target_date}.json"
    csv_path = OUTPUTS_DAILY / f"predictions_{target_date}.csv"
    xlsx_path = OUTPUTS_DAILY / f"predictions_{target_date}.xlsx"
    md_path = OUTPUTS_REPORTS / f"daily_report_{target_date}.md"

    write_json(json_path, predictions)

    rows = [flatten_prediction(p) for p in predictions]
    df = pd.DataFrame(rows)

    df.to_csv(csv_path, index=False)

    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"[WARN] No se pudo generar XLSX: {e}")

    write_markdown_report(md_path, target_date, rows)

    return json_path, csv_path, xlsx_path, md_path


def run_daily(target_date: str) -> list[dict]:
    target_date = pd.to_datetime(target_date).strftime("%Y-%m-%d")

    fixtures = load_fixtures()
    today = filter_fixtures_by_date(fixtures, target_date)

    print(f"[INFO] Fecha objetivo Bolivia: {target_date}")
    print(f"[INFO] Partidos encontrados: {len(today)}")

    if not today.empty:
        print(
            today[
                [
                    "match_id",
                    "date_utc",
                    "kickoff_local",
                    "home_team",
                    "away_team",
                ]
            ].to_string(index=False)
        )

    predictions = []

    for _, row in today.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        print(f"[+] Prediciendo {home} vs {away}")

        try:
            pred = predict_fixture(row)
            predictions.append(pred)
        except Exception as e:
            print(f"[ERROR] {home} vs {away}: {e}")
            traceback.print_exc()

    json_path, csv_path, xlsx_path, md_path = export_predictions(target_date, predictions)

    print("")
    print("[OK] Outputs generados:")
    print(f"- json: {json_path}")
    print(f"- csv: {csv_path}")
    print(f"- xlsx: {xlsx_path}")
    print(f"- md: {md_path}")

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Fecha calendario Bolivia YYYY-MM-DD")
    args = parser.parse_args()

    run_daily(args.date)


if __name__ == "__main__":
    main()