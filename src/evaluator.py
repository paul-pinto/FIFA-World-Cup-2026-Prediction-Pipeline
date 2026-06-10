# src/evaluator.py
import argparse
import json
import math
import pandas as pd
from datetime import datetime, timezone

from src.config import DAILY_OUTPUTS_DIR, REPORTS_DIR
from src.results import (
    load_manual_results,
    result_1x2,
    result_over25,
    result_btts,
    exact_score,
)


EPS = 1e-15


def safe_prob(p: float) -> float:
    return max(EPS, min(1.0 - EPS, float(p)))


def brier_1x2(row: pd.Series, actual: str) -> float:
    y_home = 1.0 if actual == "HOME" else 0.0
    y_draw = 1.0 if actual == "DRAW" else 0.0
    y_away = 1.0 if actual == "AWAY" else 0.0

    return (
        (row["final_p_home"] - y_home) ** 2
        + (row["final_p_draw"] - y_draw) ** 2
        + (row["final_p_away"] - y_away) ** 2
    )


def log_loss_1x2(row: pd.Series, actual: str) -> float:
    if actual == "HOME":
        p = row["final_p_home"]
    elif actual == "DRAW":
        p = row["final_p_draw"]
    else:
        p = row["final_p_away"]

    return -math.log(safe_prob(p))


def predicted_exact_score(row: pd.Series) -> str:
    return str(row.get("top_score_1", ""))


def actual_probability(row: pd.Series, actual_score: str) -> float:
    for i in range(1, 11):
        score_col = f"top_score_{i}"
        prob_col = f"top_score_{i}_prob"

        if score_col in row and str(row[score_col]) == actual_score:
            return float(row.get(prob_col, 0.0))

    return 0.0


def evaluate_date(target_date: str) -> tuple[pd.DataFrame, dict]:
    predictions_path = DAILY_OUTPUTS_DIR / f"predictions_{target_date}.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"No existe {predictions_path}. Primero corre src.run_daily para esa fecha."
        )

    preds = pd.read_csv(predictions_path)
    results = load_manual_results()

    if results.empty:
        print("[WARN] No hay resultados FINISHED en manual_results.csv")
        return pd.DataFrame(), {}

    df = preds.merge(results, on="match_id", how="inner")

    if df.empty:
        print("[WARN] No hay match_id coincidentes entre predicciones y resultados.")
        return pd.DataFrame(), {}

    rows = []

    for _, row in df.iterrows():
        hs = int(row["home_score"])
        aw = int(row["away_score"])

        actual_1x2 = result_1x2(hs, aw)
        actual_ou25 = result_over25(hs, aw)
        actual_btts = result_btts(hs, aw)
        actual_score = exact_score(hs, aw)

        pick = str(row["pick"])
        top_score = predicted_exact_score(row)
        p_actual_score = actual_probability(row, actual_score)

        evaluated = {
            "match_id": row["match_id"],
            "date_utc": row.get("date_utc", ""),
            "home_team": row["home_team"],
            "away_team": row["away_team"],

            "home_score": hs,
            "away_score": aw,
            "actual_score": actual_score,
            "actual_1x2": actual_1x2,

            "pick": pick,
            "correct_1x2": int(pick == actual_1x2),

            "top_score_1": top_score,
            "exact_score_hit": int(top_score == actual_score),
            "actual_score_top10_prob": p_actual_score,

            "actual_over25": actual_ou25,
            "pred_over25_prob": float(row["dc_p_over25"]),
            "over25_pick": int(float(row["dc_p_over25"]) >= 0.5),
            "over25_hit": int((float(row["dc_p_over25"]) >= 0.5) == bool(actual_ou25)),

            "actual_btts": actual_btts,
            "pred_btts_prob": float(row["dc_p_btts_yes"]),
            "btts_pick": int(float(row["dc_p_btts_yes"]) >= 0.5),
            "btts_hit": int((float(row["dc_p_btts_yes"]) >= 0.5) == bool(actual_btts)),

            "final_p_home": float(row["final_p_home"]),
            "final_p_draw": float(row["final_p_draw"]),
            "final_p_away": float(row["final_p_away"]),

            "log_loss_1x2": log_loss_1x2(row, actual_1x2),
            "brier_1x2": brier_1x2(row, actual_1x2),
        }

        rows.append(evaluated)

    eval_df = pd.DataFrame(rows)

    metrics = {
        "date": target_date,
        "evaluated_matches": int(len(eval_df)),
        "accuracy_1x2": float(eval_df["correct_1x2"].mean()),
        "exact_score_hit_rate": float(eval_df["exact_score_hit"].mean()),
        "over25_accuracy": float(eval_df["over25_hit"].mean()),
        "btts_accuracy": float(eval_df["btts_hit"].mean()),
        "avg_log_loss_1x2": float(eval_df["log_loss_1x2"].mean()),
        "avg_brier_1x2": float(eval_df["brier_1x2"].mean()),
        "avg_actual_score_top10_prob": float(eval_df["actual_score_top10_prob"].mean()),
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
    }

    return eval_df, metrics


def export_evaluation(eval_df: pd.DataFrame, metrics: dict, target_date: str) -> dict:
    csv_path = DAILY_OUTPUTS_DIR / f"evaluation_{target_date}.csv"
    json_path = DAILY_OUTPUTS_DIR / f"evaluation_{target_date}.json"
    md_path = REPORTS_DIR / f"evaluation_report_{target_date}.md"

    eval_df.to_csv(csv_path, index=False, encoding="utf-8")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "rows": eval_df.to_dict(orient="records"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    report = build_evaluation_report(eval_df, metrics)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "md": str(md_path),
    }


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_evaluation_report(eval_df: pd.DataFrame, metrics: dict) -> str:
    lines = []

    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"Fecha: {metrics.get('date', '')}")
    lines.append(f"Evaluado UTC: {metrics.get('evaluated_utc', '')}")
    lines.append("")

    lines.append("## Métricas")
    lines.append("")
    lines.append(f"- Partidos evaluados: **{metrics.get('evaluated_matches', 0)}**")
    lines.append(f"- Accuracy 1X2: **{pct(metrics.get('accuracy_1x2', 0))}**")
    lines.append(f"- Exact score hit rate: **{pct(metrics.get('exact_score_hit_rate', 0))}**")
    lines.append(f"- Over 2.5 accuracy: **{pct(metrics.get('over25_accuracy', 0))}**")
    lines.append(f"- BTTS accuracy: **{pct(metrics.get('btts_accuracy', 0))}**")
    lines.append(f"- Avg log loss 1X2: **{metrics.get('avg_log_loss_1x2', 0):.4f}**")
    lines.append(f"- Avg Brier 1X2: **{metrics.get('avg_brier_1x2', 0):.4f}**")
    lines.append(f"- Prob. promedio del score real en top10: **{pct(metrics.get('avg_actual_score_top10_prob', 0))}**")
    lines.append("")

    lines.append("## Partidos")
    lines.append("")

    for _, row in eval_df.iterrows():
        lines.append(f"### {row['home_team']} vs {row['away_team']}")
        lines.append("")
        lines.append(f"- Resultado real: **{row['actual_score']}** ({row['actual_1x2']})")
        lines.append(f"- Pick modelo: **{row['pick']}**")
        lines.append(f"- Correcto 1X2: **{'Sí' if row['correct_1x2'] else 'No'}**")
        lines.append(f"- Score sugerido: {row['top_score_1']}")
        lines.append(f"- Score exacto acertado: **{'Sí' if row['exact_score_hit'] else 'No'}**")
        lines.append(f"- Log loss: {row['log_loss_1x2']:.4f}")
        lines.append(f"- Brier: {row['brier_1x2']:.4f}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        required=True,
        help="Fecha objetivo YYYY-MM-DD",
    )

    args = parser.parse_args()

    eval_df, metrics = evaluate_date(args.date)

    if eval_df.empty:
        print("[WARN] Nada para exportar.")
        return

    paths = export_evaluation(eval_df, metrics, args.date)

    print(json.dumps(metrics, indent=2))

    print()
    print("[OK] Evaluación generada:")
    for k, v in paths.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
