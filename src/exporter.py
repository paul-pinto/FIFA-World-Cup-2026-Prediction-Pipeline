# src/exporter.py
import json
import pandas as pd
from datetime import datetime, timezone
from src.config import DAILY_OUTPUTS_DIR, REPORTS_DIR


def flatten_prediction(pred: dict) -> dict:
    row = {
        "match_id": pred.get("match_id"),
        "date_utc": pred.get("date_utc"),
        "stage": pred.get("stage"),
        "group": pred.get("group"),
        "venue": pred.get("venue"),
        "city": pred.get("city"),
        "country": pred.get("country"),

        "home_team": pred["home_team"],
        "away_team": pred["away_team"],
        "neutral": pred["neutral"],

        "lambda_home": pred["lambda_home"],
        "lambda_away": pred["lambda_away"],
        "has_market": pred.get("market") is not None,
        "market_p_home": pred["market"]["p_home"] if pred.get("market") else None,
        "market_p_draw": pred["market"]["p_draw"] if pred.get("market") else None,
        "market_p_away": pred["market"]["p_away"] if pred.get("market") else None,
        "market_p_over25": pred["market"]["p_over25"] if pred.get("market") else None,
        "market_lambda_home": pred["market"]["lambda_home"] if pred.get("market") else None,
        "market_lambda_away": pred["market"]["lambda_away"] if pred.get("market") else None,

        "ml_p_home": pred["ml"]["p_home"],
        "ml_p_draw": pred["ml"]["p_draw"],
        "ml_p_away": pred["ml"]["p_away"],
        "ml_p_over25": pred["ml"]["p_over25"],
        "ml_p_btts_yes": pred["ml"]["p_btts_yes"],

        "dc_p_home": pred["dixon_coles"]["p_home"],
        "dc_p_draw": pred["dixon_coles"]["p_draw"],
        "dc_p_away": pred["dixon_coles"]["p_away"],
        "dc_p_over25": pred["dixon_coles"]["p_over25"],
        "dc_p_under25": pred["dixon_coles"]["p_under25"],
        "dc_p_btts_yes": pred["dixon_coles"]["p_btts_yes"],
        "dc_p_btts_no": pred["dixon_coles"]["p_btts_no"],

        "mc_p_home": pred["monte_carlo"]["home_win"],
        "mc_p_draw": pred["monte_carlo"]["draw"],
        "mc_p_away": pred["monte_carlo"]["away_win"],
        "mc_p_over25": pred["monte_carlo"]["over25"],
        "mc_p_btts_yes": pred["monte_carlo"]["btts_yes"],
        "mc_avg_home_goals": pred["monte_carlo"]["avg_home_goals"],
        "mc_avg_away_goals": pred["monte_carlo"]["avg_away_goals"],
        "mc_avg_total_goals": pred["monte_carlo"]["avg_total_goals"],

        "final_p_home": pred["final"]["p_home"],
        "final_p_draw": pred["final"]["p_draw"],
        "final_p_away": pred["final"]["p_away"],
        "pick": pred["final"]["pick"],
        "confidence": pred["final"]["confidence"],
        "has_value_data": pred.get("value", {}).get("has_value_data", False),
        "best_value_market": (
            pred.get("value", {}).get("best_value", {}) or {}
        ).get("market"),
        "best_value_odds": (
            pred.get("value", {}).get("best_value", {}) or {}
        ).get("odds"),
        "best_value_probability": (
            pred.get("value", {}).get("best_value", {}) or {}
        ).get("probability"),
        "best_value_edge": (
            pred.get("value", {}).get("best_value", {}) or {}
        ).get("edge"),
        "best_value_ev": (
            pred.get("value", {}).get("best_value", {}) or {}
        ).get("ev"),
        "best_edge_market": (
            pred.get("value", {}).get("best_edge", {}) or {}
        ).get("market"),
        "best_edge_odds": (
            pred.get("value", {}).get("best_edge", {}) or {}
        ).get("odds"),
        "best_edge_probability": (
            pred.get("value", {}).get("best_edge", {}) or {}
        ).get("probability"),
        "best_edge": (
            pred.get("value", {}).get("best_edge", {}) or {}
        ).get("edge"),
        "best_edge_ev": (
            pred.get("value", {}).get("best_edge", {}) or {}
        ).get("ev"),
    }

    for i, score in enumerate(pred["top_scores"][:10], start=1):
        row[f"top_score_{i}"] = score["score"]
        row[f"top_score_{i}_prob"] = score["probability"]

    return row


def export_daily(predictions: list[dict], run_date: str | None = None) -> dict:
    if run_date is None:
        run_date = datetime.now(timezone.utc).date().isoformat()

    rows = [flatten_prediction(p) for p in predictions]

    df = pd.DataFrame(rows)

    json_path = DAILY_OUTPUTS_DIR / f"predictions_{run_date}.json"
    csv_path = DAILY_OUTPUTS_DIR / f"predictions_{run_date}.csv"
    xlsx_path = DAILY_OUTPUTS_DIR / f"predictions_{run_date}.xlsx"
    md_path = REPORTS_DIR / f"daily_report_{run_date}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_excel(xlsx_path, index=False)

    report = build_markdown_report(predictions, run_date)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "xlsx": str(xlsx_path),
        "md": str(md_path),
    }


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_markdown_report(predictions: list[dict], run_date: str) -> str:
    lines = []

    lines.append(f"# World Cup 2026 - Daily Predictions")
    lines.append("")
    lines.append(f"Fecha: `{run_date}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    for pred in predictions:
        home = pred["home_team"]
        away = pred["away_team"]

        lines.append(f"## {home} vs {away}")
        lines.append("")
        lines.append(f"- Match ID: `{pred.get('match_id', '')}`")
        lines.append(f"- Fecha UTC: `{pred.get('date_utc', '')}`")
        lines.append(f"- Sede: {pred.get('venue', '')} - {pred.get('city', '')}, {pred.get('country', '')}")
        lines.append("")

        lines.append("### Final 1X2")
        lines.append("")
        lines.append(f"- {home}: **{pct(pred['final']['p_home'])}**")
        lines.append(f"- Empate: **{pct(pred['final']['p_draw'])}**")
        lines.append(f"- {away}: **{pct(pred['final']['p_away'])}**")
        lines.append(f"- Pick: **{pred['final']['pick']}** ({pct(pred['final']['confidence'])})")
        lines.append("")

        lines.append("### Goles esperados")
        lines.append("")
        lines.append(f"- {home}: `{pred['lambda_home']:.3f}`")
        lines.append(f"- {away}: `{pred['lambda_away']:.3f}`")
        lines.append("")

        lines.append("### Top scores")
        lines.append("")
        for i, s in enumerate(pred["top_scores"][:5], start=1):
            lines.append(f"{i}. `{s['score']}` → **{pct(s['probability'])}**")
        lines.append("")

        lines.append("### Mercados derivados")
        lines.append("")
        lines.append(f"- Over 2.5: **{pct(pred['dixon_coles']['p_over25'])}**")
        lines.append(f"- Under 2.5: **{pct(pred['dixon_coles']['p_under25'])}**")
        lines.append(f"- BTTS Sí: **{pct(pred['dixon_coles']['p_btts_yes'])}**")
        lines.append(f"- BTTS No: **{pct(pred['dixon_coles']['p_btts_no'])}**")
        lines.append("")
        
        value = pred.get("value", {})
        best_value = value.get("best_value")
        best_edge = value.get("best_edge")

        lines.append("### Value")
        lines.append("")

        if best_value:
            lines.append(f"- Mejor value: **{best_value['market']}**")
            lines.append(f"- Cuota: `{best_value['odds']:.2f}`")
            lines.append(f"- Probabilidad modelo: **{pct(best_value['probability'])}**")
            lines.append(f"- Edge: **{best_value['edge']:+.3f}**")
            lines.append(f"- EV: **{best_value['ev']:+.3f}**")
        elif best_edge:
            lines.append("- No hay value claro según umbrales.")
            lines.append(f"- Mejor edge: **{best_edge['market']}**")
            lines.append(f"- Cuota: `{best_edge['odds']:.2f}`")
            lines.append(f"- Probabilidad modelo: **{pct(best_edge['probability'])}**")
            lines.append(f"- Edge: **{best_edge['edge']:+.3f}**")
            lines.append(f"- EV: **{best_edge['ev']:+.3f}**")
        else:
            lines.append("- Sin datos de cuotas.")

        lines.append("")

        lines.append("### Monte Carlo")
        lines.append("")
        lines.append(f"- Simulaciones: `{pred['monte_carlo']['n_sim']:,}`")
        lines.append(f"- {home}: **{pct(pred['monte_carlo']['home_win'])}**")
        lines.append(f"- Empate: **{pct(pred['monte_carlo']['draw'])}**")
        lines.append(f"- {away}: **{pct(pred['monte_carlo']['away_win'])}**")
        lines.append(f"- Goles promedio: `{pred['monte_carlo']['avg_total_goals']:.3f}`")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)