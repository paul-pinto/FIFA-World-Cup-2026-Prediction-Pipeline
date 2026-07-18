from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FIXTURES_PATH = ROOT / "data" / "master" / "worldcup_2026_fixtures.csv"
RESULTS_PATH = ROOT / "data" / "master" / "manual_results.csv"
DAILY_DIR = ROOT / "outputs" / "daily"
EVAL_DIR = ROOT / "outputs" / "evaluation"


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def norm_prob(value) -> float:
    v = safe_float(value, 0.0)
    if v > 1.0:
        v = v / 100.0
    return max(0.0, min(1.0, v))


def pick_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_predictions() -> pd.DataFrame:
    files = sorted(DAILY_DIR.glob("predictions_*.csv"))
    frames = []

    for path in files:
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        except Exception as exc:
            print(f"[WARN] No se pudo leer {path}: {exc}")
            continue

        if df.empty:
            continue

        if "match_id" not in df.columns:
            print(f"[WARN] {path.name} no tiene match_id, se omite.")
            continue

        date_str = path.stem.replace("predictions_", "")
        df["prediction_file"] = path.name
        df["prediction_date"] = date_str
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_truth() -> pd.DataFrame:
    fixtures = pd.read_csv(FIXTURES_PATH)
    results = pd.read_csv(RESULTS_PATH)

    needed = {"match_id", "home_score", "away_score", "status"}
    missing = needed - set(results.columns)
    if missing:
        raise ValueError(f"manual_results.csv no tiene columnas necesarias: {missing}")

    truth = fixtures.merge(results, on="match_id", how="inner", suffixes=("", "_result"))
    truth = truth[truth["status"].astype(str).str.upper().eq("FINISHED")].copy()

    truth["home_score"] = truth["home_score"].astype(int)
    truth["away_score"] = truth["away_score"].astype(int)

    def actual_1x2(row):
        if row["home_score"] > row["away_score"]:
            return "HOME"
        if row["home_score"] < row["away_score"]:
            return "AWAY"
        return "DRAW"

    truth["actual_1x2"] = truth.apply(actual_1x2, axis=1)
    truth["actual_over25"] = (truth["home_score"] + truth["away_score"] > 2.5).astype(int)
    truth["actual_btts"] = ((truth["home_score"] > 0) & (truth["away_score"] > 0)).astype(int)
    truth["actual_score"] = truth["home_score"].astype(str) + "-" + truth["away_score"].astype(str)

    # Si más adelante agregamos advance_team real en manual_results.csv,
    # lo usamos para evaluar correctamente penales.
    if "advance_team" not in truth.columns:
        truth["advance_team"] = pd.NA

    return truth


def compute_log_loss(row) -> float:
    actual = row["actual_1x2"]

    if actual == "HOME":
        p = row["p_home"]
    elif actual == "DRAW":
        p = row["p_draw"]
    else:
        p = row["p_away"]

    p = max(1e-15, min(1.0 - 1e-15, p))
    return -math.log(p)


def compute_brier_1x2(row) -> float:
    y_home = 1.0 if row["actual_1x2"] == "HOME" else 0.0
    y_draw = 1.0 if row["actual_1x2"] == "DRAW" else 0.0
    y_away = 1.0 if row["actual_1x2"] == "AWAY" else 0.0

    return (
        (row["p_home"] - y_home) ** 2
        + (row["p_draw"] - y_draw) ** 2
        + (row["p_away"] - y_away) ** 2
    ) / 3.0


def best_1x2_pick(row) -> str:
    probs = {
        "HOME": row["p_home"],
        "DRAW": row["p_draw"],
        "AWAY": row["p_away"],
    }
    return max(probs, key=probs.get)


def add_over25_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    # Si existe una probabilidad final, usamos esa.
    # Si no existe, construimos una probabilidad combinada desde ML + Dixon-Coles + Monte Carlo + mercado.
    over_col = pick_existing_col(df, ["final_p_over25", "p_over25", "prob_over25"])
    under_col = pick_existing_col(df, ["final_p_under25", "p_under25", "prob_under25"])

    if over_col:
        df["p_over25_eval"] = df[over_col].apply(norm_prob)
    else:
        over_sources = []

        if "ml_p_over25" in df.columns:
            over_sources.append(("ml_p_over25", 0.20))

        if "dc_p_over25" in df.columns:
            over_sources.append(("dc_p_over25", 0.45))

        if "mc_p_over25" in df.columns:
            over_sources.append(("mc_p_over25", 0.35))

        # Mercado con peso bajo adicional si existe.
        # Ojo: si casi todos son NaN, no aporta.
        if "market_p_over25" in df.columns:
            over_sources.append(("market_p_over25", 0.20))

        if over_sources:
            weighted = pd.Series(0.0, index=df.index)
            total_weight = pd.Series(0.0, index=df.index)

            for col, weight in over_sources:
                values = df[col].apply(lambda x: pd.NA if pd.isna(x) else norm_prob(x))
                valid = values.notna()

                weighted.loc[valid] = weighted.loc[valid] + values.loc[valid].astype(float) * weight
                total_weight.loc[valid] = total_weight.loc[valid] + weight

            df["p_over25_eval"] = pd.NA
            valid_total = total_weight > 0
            df.loc[valid_total, "p_over25_eval"] = weighted.loc[valid_total] / total_weight.loc[valid_total]
        else:
            df["p_over25_eval"] = pd.NA

    if df["p_over25_eval"].notna().any():
        if under_col:
            df["p_under25_eval"] = df[under_col].apply(norm_prob)
        elif "dc_p_under25" in df.columns:
            df["p_under25_eval"] = df["dc_p_under25"].apply(norm_prob)
        else:
            df["p_under25_eval"] = 1.0 - df["p_over25_eval"].astype(float)

        df["pick_over25"] = (df["p_over25_eval"].astype(float) >= df["p_under25_eval"].astype(float)).astype(int)
        df["hit_over25"] = (df["pick_over25"] == df["actual_over25"]).astype(int)
    else:
        df["hit_over25"] = pd.NA

    return df


def add_btts_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    btts_yes_col = pick_existing_col(df, ["final_btts_yes", "p_btts_yes", "btts_yes"])
    btts_no_col = pick_existing_col(df, ["final_btts_no", "p_btts_no", "btts_no"])

    if btts_yes_col:
        df["p_btts_yes_eval"] = df[btts_yes_col].apply(norm_prob)

        if btts_no_col:
            df["p_btts_no_eval"] = df[btts_no_col].apply(norm_prob)
        else:
            df["p_btts_no_eval"] = 1.0 - df["p_btts_yes_eval"]

        df["pick_btts"] = (df["p_btts_yes_eval"] >= df["p_btts_no_eval"]).astype(int)
        df["hit_btts"] = (df["pick_btts"] == df["actual_btts"]).astype(int)
    else:
        df["hit_btts"] = pd.NA

    return df


def add_exact_score_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    score_cols = [c for c in df.columns if c.startswith("top_score_") and not c.endswith("_prob")]

    def score_order(col: str) -> int:
        try:
            return int(col.replace("top_score_", ""))
        except Exception:
            return 999

    score_cols = sorted(score_cols, key=score_order)

    if score_cols:
        df["hit_exact_top1"] = (df[score_cols[0]].astype(str) == df["actual_score"].astype(str)).astype(int)

        top3_cols = score_cols[:3]
        top5_cols = score_cols[:5]

        df["hit_exact_top3"] = df.apply(
            lambda r: int(str(r["actual_score"]) in [str(r[c]) for c in top3_cols]),
            axis=1,
        )
        df["hit_exact_top5"] = df.apply(
            lambda r: int(str(r["actual_score"]) in [str(r[c]) for c in top5_cols]),
            axis=1,
        )
    else:
        df["hit_exact_top1"] = pd.NA
        df["hit_exact_top3"] = pd.NA
        df["hit_exact_top5"] = pd.NA

    return df


def add_advance_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    # Evaluación de clasificación knockout.
    # Si manual_results tiene advance_team, lo usamos.
    # Si no, solo evaluamos partidos knockout que NO terminaron empatados.
    if "advance_team_pred" in df.columns:
        pred_advance_col = "advance_team_pred"
    elif "advance_team" in df.columns:
        pred_advance_col = "advance_team"
    else:
        df["actual_advance_team"] = pd.NA
        df["hit_advance"] = pd.NA
        return df

    stage_col = "stage_actual" if "stage_actual" in df.columns else "stage"
    home_col = "home_team_actual" if "home_team_actual" in df.columns else "home_team"
    away_col = "away_team_actual" if "away_team_actual" in df.columns else "away_team"

    is_knockout = ~df[stage_col].astype(str).str.lower().isin(["group stage", "group", "fase de grupos"])
    df["actual_advance_team"] = pd.NA

    # Si el resultado real trae advance_team, lo priorizamos.
    if "advance_team_actual" in df.columns:
        mask_real_advance = df["advance_team_actual"].notna()
        df.loc[mask_real_advance, "actual_advance_team"] = df.loc[mask_real_advance, "advance_team_actual"]

    if "advance_team" in df.columns and "advance_team_actual" not in df.columns:
        # Caso raro por nombres sin suffix.
        mask_real_advance = df["advance_team"].notna()
        df.loc[mask_real_advance, "actual_advance_team"] = df.loc[mask_real_advance, "advance_team"]

    # Para partidos knockout no empatados, el que gana en goles clasifica.
    mask_home = is_knockout & df["actual_advance_team"].isna() & (df["actual_1x2"] == "HOME")
    mask_away = is_knockout & df["actual_advance_team"].isna() & (df["actual_1x2"] == "AWAY")

    df.loc[mask_home, "actual_advance_team"] = df.loc[mask_home, home_col]
    df.loc[mask_away, "actual_advance_team"] = df.loc[mask_away, away_col]

    df["hit_advance"] = pd.NA
    mask_eval = is_knockout & df["actual_advance_team"].notna()

    df.loc[mask_eval, "hit_advance"] = (
        df.loc[mask_eval, pred_advance_col].astype(str)
        == df.loc[mask_eval, "actual_advance_team"].astype(str)
    ).astype(int)

    return df


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = load_predictions()
    truth = load_truth()

    if preds.empty:
        raise RuntimeError("No hay predicciones en outputs/daily/predictions_*.csv")

    if "match_id" not in preds.columns:
        raise RuntimeError("Los archivos de predicción no tienen columna match_id")

    df = preds.merge(
        truth[
            [
                "match_id",
                "stage",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "actual_1x2",
                "actual_over25",
                "actual_btts",
                "actual_score",
                "advance_team",
            ]
        ],
        on="match_id",
        how="inner",
        suffixes=("_pred", "_actual"),
    )

    if df.empty:
        raise RuntimeError("No hubo cruce entre predicciones y resultados reales por match_id")

    home_col = pick_existing_col(df, ["final_p_home", "p_home", "prob_home", "home_prob"])
    draw_col = pick_existing_col(df, ["final_p_draw", "p_draw", "prob_draw", "draw_prob"])
    away_col = pick_existing_col(df, ["final_p_away", "p_away", "prob_away", "away_prob"])

    if not home_col or not draw_col or not away_col:
        raise RuntimeError(
            "No encuentro columnas de probabilidad 1X2. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    df["p_home"] = df[home_col].apply(norm_prob)
    df["p_draw"] = df[draw_col].apply(norm_prob)
    df["p_away"] = df[away_col].apply(norm_prob)

    total = df["p_home"] + df["p_draw"] + df["p_away"]
    total = total.replace(0, 1)

    df["p_home"] = df["p_home"] / total
    df["p_draw"] = df["p_draw"] / total
    df["p_away"] = df["p_away"] / total

    df["pick_1x2"] = df.apply(best_1x2_pick, axis=1)
    df["hit_1x2"] = (df["pick_1x2"] == df["actual_1x2"]).astype(int)
    df["log_loss_1x2"] = df.apply(compute_log_loss, axis=1)
    df["brier_1x2"] = df.apply(compute_brier_1x2, axis=1)

    df = add_exact_score_evaluation(df)
    df = add_over25_evaluation(df)
    df = add_btts_evaluation(df)
    df = add_advance_evaluation(df)

    summary_rows = []

    def mean_or_na(series: pd.Series):
        s = series.dropna()
        if s.empty:
            return pd.NA
        return s.mean()

    def add_summary(name: str, part: pd.DataFrame):
        row = {
            "segment": name,
            "matches": len(part),
            "accuracy_1x2": mean_or_na(part["hit_1x2"]),
            "log_loss_1x2": mean_or_na(part["log_loss_1x2"]),
            "brier_1x2": mean_or_na(part["brier_1x2"]),
            "exact_top1": mean_or_na(part["hit_exact_top1"]),
            "exact_top3": mean_or_na(part["hit_exact_top3"]),
            "exact_top5": mean_or_na(part["hit_exact_top5"]),
            "accuracy_over25": mean_or_na(part["hit_over25"]),
            "accuracy_btts": mean_or_na(part["hit_btts"]),
            "accuracy_advance": mean_or_na(part["hit_advance"]),
        }
        summary_rows.append(row)

    add_summary("ALL", df)

    stage_col = "stage_actual" if "stage_actual" in df.columns else "stage"
    if stage_col in df.columns:
        for stage, part in df.groupby(stage_col):
            add_summary(str(stage), part)

    summary = pd.DataFrame(summary_rows)

    # Orden cómodo para leer detalle.
    preferred_cols = [
        "match_id",
        "prediction_date",
        "stage_actual",
        "home_team_actual",
        "away_team_actual",
        "home_score",
        "away_score",
        "actual_1x2",
        "pick_1x2",
        "hit_1x2",
        "p_home",
        "p_draw",
        "p_away",
        "log_loss_1x2",
        "brier_1x2",
        "actual_score",
        "top_score_1",
        "top_score_2",
        "top_score_3",
        "top_score_4",
        "top_score_5",
        "hit_exact_top1",
        "hit_exact_top3",
        "hit_exact_top5",
        "actual_over25",
        "p_over25_eval",
        "pick_over25",
        "hit_over25",
        "actual_btts",
        "p_btts_yes_eval",
        "pick_btts",
        "hit_btts",
        "advance_team_pred",
        "actual_advance_team",
        "hit_advance",
        "prediction_file",
    ]

    existing_preferred = [c for c in preferred_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in existing_preferred]
    df = df[existing_preferred + remaining]

    return df, summary


def print_console_report(summary: pd.DataFrame, detail: pd.DataFrame) -> None:
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print()
        print("=== RESUMEN ===")
        print(summary)

    print()
    print("=== LECTURA RÁPIDA ===")

    all_row = summary[summary["segment"] == "ALL"].iloc[0]

    def pct(value):
        if pd.isna(value):
            return "N/A"
        return f"{float(value) * 100:.2f}%"

    def num(value):
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.4f}"

    print(f"Partidos evaluados: {int(all_row['matches'])}")
    print(f"Accuracy 1X2: {pct(all_row['accuracy_1x2'])}")
    print(f"Log Loss 1X2: {num(all_row['log_loss_1x2'])}")
    print(f"Brier 1X2: {num(all_row['brier_1x2'])}")
    print(f"Marcador exacto Top 1: {pct(all_row['exact_top1'])}")
    print(f"Marcador exacto Top 3: {pct(all_row['exact_top3'])}")
    print(f"Marcador exacto Top 5: {pct(all_row['exact_top5'])}")
    print(f"Accuracy Over 2.5: {pct(all_row['accuracy_over25'])}")
    print(f"Accuracy BTTS: {pct(all_row['accuracy_btts'])}")
    print(f"Accuracy clasificación: {pct(all_row['accuracy_advance'])}")

    missing_pred = detail["match_id"].isna().sum() if "match_id" in detail.columns else 0
    if missing_pred:
        print(f"[WARN] Hay {missing_pred} filas sin match_id.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", help="Imprime resumen en consola")
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    detail, summary = evaluate()

    detail_path = EVAL_DIR / "prediction_evaluation_detail.csv"
    summary_path = EVAL_DIR / "prediction_evaluation_summary.csv"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"[OK] Detalle guardado: {detail_path}")
    print(f"[OK] Resumen guardado: {summary_path}")

    if args.print:
        print_console_report(summary, detail)


if __name__ == "__main__":
    main()