# src/predictor.py
import json
import joblib
import pandas as pd
import numpy as np

from src.config import PROCESSED_DIR, MODELS_DIR, N_SIM, MAX_GOALS, SEED
from src.poisson_dc import (
    build_score_matrix,
    summarize_score_matrix,
    clamp_lambda,
)
from src.montecarlo import run_monte_carlo


TARGET_LABELS = {
    0: "HOME",
    1: "DRAW",
    2: "AWAY",
}


def load_models():
    return {
        "model_1x2": joblib.load(MODELS_DIR / "model_1x2.pkl"),
        "model_home_goals": joblib.load(MODELS_DIR / "model_home_goals.pkl"),
        "model_away_goals": joblib.load(MODELS_DIR / "model_away_goals.pkl"),
        "model_over25": joblib.load(MODELS_DIR / "model_over25.pkl"),
        "model_btts": joblib.load(MODELS_DIR / "model_btts.pkl"),
    }


def load_feature_columns():
    with open(MODELS_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def latest_team_snapshot(team: str, df: pd.DataFrame) -> dict:
    """
    Obtiene el último estado conocido de un equipo desde training_dataset.
    Esto sirve para armar features de partidos futuros.
    """
    home_rows = df[df["home_team"] == team].copy()
    away_rows = df[df["away_team"] == team].copy()

    candidates = []

    if not home_rows.empty:
        r = home_rows.sort_values("date").iloc[-1]
        candidates.append({
            "date": r["date"],
            "team": team,
            "elo": r["elo_home_pre"],
            "gf_5": r["home_gf_5"],
            "ga_5": r["home_ga_5"],
            "gf_10": r["home_gf_10"],
            "ga_10": r["home_ga_10"],
            "gf_20": r["home_gf_20"],
            "ga_20": r["home_ga_20"],
            "points_5": r["home_points_5"],
            "points_10": r["home_points_10"],
            "points_20": r["home_points_20"],
            "matches_hist": r["home_matches_hist"],
        })

    if not away_rows.empty:
        r = away_rows.sort_values("date").iloc[-1]
        candidates.append({
            "date": r["date"],
            "team": team,
            "elo": r["elo_away_pre"],
            "gf_5": r["away_gf_5"],
            "ga_5": r["away_ga_5"],
            "gf_10": r["away_gf_10"],
            "ga_10": r["away_ga_10"],
            "gf_20": r["away_gf_20"],
            "ga_20": r["away_ga_20"],
            "points_5": r["away_points_5"],
            "points_10": r["away_points_10"],
            "points_20": r["away_points_20"],
            "matches_hist": r["away_matches_hist"],
        })

    if not candidates:
        raise ValueError(f"No hay historial para el equipo: {team}")

    candidates = sorted(candidates, key=lambda x: x["date"])
    return candidates[-1]


def build_future_match_features(
    home_team: str,
    away_team: str,
    neutral: int = 1,
) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "training_dataset.parquet")

    home = latest_team_snapshot(home_team, df)
    away = latest_team_snapshot(away_team, df)

    row = {
        "neutral": int(neutral),

        "elo_home_pre": home["elo"],
        "elo_away_pre": away["elo"],
        "elo_diff_pre": home["elo"] - away["elo"],

        "home_gf_5": home["gf_5"],
        "home_ga_5": home["ga_5"],
        "away_gf_5": away["gf_5"],
        "away_ga_5": away["ga_5"],

        "home_gf_10": home["gf_10"],
        "home_ga_10": home["ga_10"],
        "away_gf_10": away["gf_10"],
        "away_ga_10": away["ga_10"],

        "home_gf_20": home["gf_20"],
        "home_ga_20": home["ga_20"],
        "away_gf_20": away["gf_20"],
        "away_ga_20": away["ga_20"],

        "home_points_5": home["points_5"],
        "away_points_5": away["points_5"],
        "home_points_10": home["points_10"],
        "away_points_10": away["points_10"],
        "home_points_20": home["points_20"],
        "away_points_20": away["points_20"],

        "goal_diff_form_5": (home["gf_5"] - home["ga_5"]) - (away["gf_5"] - away["ga_5"]),
        "goal_diff_form_10": (home["gf_10"] - home["ga_10"]) - (away["gf_10"] - away["ga_10"]),
        "goal_diff_form_20": (home["gf_20"] - home["ga_20"]) - (away["gf_20"] - away["ga_20"]),

        "points_form_diff_5": home["points_5"] - away["points_5"],
        "points_form_diff_10": home["points_10"] - away["points_10"],
        "points_form_diff_20": home["points_20"] - away["points_20"],

        "home_attack_strength_5": home["gf_5"],
        "away_attack_strength_5": away["gf_5"],
        "home_defense_weakness_5": home["ga_5"],
        "away_defense_weakness_5": away["ga_5"],

        "attack_diff_5": home["gf_5"] - away["gf_5"],
        "defense_diff_5": away["ga_5"] - home["ga_5"],

        "home_matches_hist": home["matches_hist"],
        "away_matches_hist": away["matches_hist"],
    }

    return pd.DataFrame([row])


def ensemble_1x2(
    p_ml: np.ndarray,
    p_dc: dict,
    weights: dict | None = None,
) -> dict:
    if weights is None:
        weights = {
            "ml": 0.55,
            "dc": 0.45,
        }

    p_home = weights["ml"] * p_ml[0] + weights["dc"] * p_dc["p_home"]
    p_draw = weights["ml"] * p_ml[1] + weights["dc"] * p_dc["p_draw"]
    p_away = weights["ml"] * p_ml[2] + weights["dc"] * p_dc["p_away"]

    total = p_home + p_draw + p_away

    return {
        "p_home": p_home / total,
        "p_draw": p_draw / total,
        "p_away": p_away / total,
    }


def predict_match(
    home_team: str,
    away_team: str,
    neutral: int = 1,
    market: dict | None = None,
) -> dict:
    models = load_models()
    feature_columns = load_feature_columns()

    X = build_future_match_features(
        home_team=home_team,
        away_team=away_team,
        neutral=neutral,
    )

    X = X[feature_columns]

    p_ml = models["model_1x2"].predict_proba(X)[0]

    lambda_home_ml = clamp_lambda(models["model_home_goals"].predict(X)[0])
    lambda_away_ml = clamp_lambda(models["model_away_goals"].predict(X)[0])

    p_over25_ml = float(models["model_over25"].predict_proba(X)[0][1])
    p_btts_ml = float(models["model_btts"].predict_proba(X)[0][1])

    # =========================
    # Lambdas finales
    # =========================
    if market is not None:
        lambda_home_final = clamp_lambda(
            0.55 * market["lambda_home"] + 0.45 * lambda_home_ml
        )
        lambda_away_final = clamp_lambda(
            0.55 * market["lambda_away"] + 0.45 * lambda_away_ml
        )
    else:
        lambda_home_final = lambda_home_ml
        lambda_away_final = lambda_away_ml

    matrix = build_score_matrix(
        lambda_home=lambda_home_final,
        lambda_away=lambda_away_final,
        max_goals=MAX_GOALS,
        rho=-0.10,
        use_dixon_coles=True,
    )

    dc_summary = summarize_score_matrix(matrix)

    mc = run_monte_carlo(
        score_matrix=matrix,
        n_sim=N_SIM,
        seed=SEED,
    )

    # =========================
    # Ensemble 1X2
    # =========================
    if market is not None:
        final_1x2 = {
            "p_home": (
                0.40 * market["p_home"]
                + 0.35 * float(p_ml[0])
                + 0.25 * dc_summary["p_home"]
            ),
            "p_draw": (
                0.40 * market["p_draw"]
                + 0.35 * float(p_ml[1])
                + 0.25 * dc_summary["p_draw"]
            ),
            "p_away": (
                0.40 * market["p_away"]
                + 0.35 * float(p_ml[2])
                + 0.25 * dc_summary["p_away"]
            ),
        }

        total = final_1x2["p_home"] + final_1x2["p_draw"] + final_1x2["p_away"]
        final_1x2 = {k: v / total for k, v in final_1x2.items()}
    else:
        final_1x2 = ensemble_1x2(
            p_ml=p_ml,
            p_dc=dc_summary,
        )

    top_scores = dc_summary["top_scores"][:10]

    candidates = {
        "HOME": final_1x2["p_home"],
        "DRAW": final_1x2["p_draw"],
        "AWAY": final_1x2["p_away"],
    }

    pick = max(candidates, key=candidates.get)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "neutral": neutral,

        "lambda_home": float(lambda_home_final),
        "lambda_away": float(lambda_away_final),

        "market": market,

        "ml": {
            "p_home": float(p_ml[0]),
            "p_draw": float(p_ml[1]),
            "p_away": float(p_ml[2]),
            "p_over25": p_over25_ml,
            "p_btts_yes": p_btts_ml,
            "lambda_home": float(lambda_home_ml),
            "lambda_away": float(lambda_away_ml),
        },

        "dixon_coles": {
            "p_home": float(dc_summary["p_home"]),
            "p_draw": float(dc_summary["p_draw"]),
            "p_away": float(dc_summary["p_away"]),
            "p_over25": float(dc_summary["p_over25"]),
            "p_under25": float(dc_summary["p_under25"]),
            "p_btts_yes": float(dc_summary["p_btts_yes"]),
            "p_btts_no": float(dc_summary["p_btts_no"]),
        },

        "monte_carlo": mc,

        "final": {
            "p_home": float(final_1x2["p_home"]),
            "p_draw": float(final_1x2["p_draw"]),
            "p_away": float(final_1x2["p_away"]),
            "pick": pick,
            "confidence": float(candidates[pick]),
        },

        "top_scores": [
            {
                "score": s["score"],
                "probability": float(s["probability"]),
            }
            for s in top_scores
        ],
    }
    models = load_models()
    feature_columns = load_feature_columns()

    X = build_future_match_features(
        home_team=home_team,
        away_team=away_team,
        neutral=neutral,
    )

    X = X[feature_columns]

    p_ml = models["model_1x2"].predict_proba(X)[0]

    lambda_home_ml = clamp_lambda(models["model_home_goals"].predict(X)[0])
    lambda_away_ml = clamp_lambda(models["model_away_goals"].predict(X)[0])

    p_over25_ml = float(models["model_over25"].predict_proba(X)[0][1])
    p_btts_ml = float(models["model_btts"].predict_proba(X)[0][1])

    matrix = build_score_matrix(
        lambda_home=lambda_home_ml,
        lambda_away=lambda_away_ml,
        max_goals=MAX_GOALS,
        rho=-0.10,
        use_dixon_coles=True,
    )

    dc_summary = summarize_score_matrix(matrix)

    mc = run_monte_carlo(
        score_matrix=matrix,
        n_sim=N_SIM,
        seed=SEED,
    )

    final_1x2 = ensemble_1x2(
        p_ml=p_ml,
        p_dc=dc_summary,
    )

    top_scores = dc_summary["top_scores"][:10]

    candidates = {
        "HOME": final_1x2["p_home"],
        "DRAW": final_1x2["p_draw"],
        "AWAY": final_1x2["p_away"],
    }

    pick = max(candidates, key=candidates.get)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "neutral": neutral,

        "lambda_home": float(lambda_home_ml),
        "lambda_away": float(lambda_away_ml),

        "ml": {
            "p_home": float(p_ml[0]),
            "p_draw": float(p_ml[1]),
            "p_away": float(p_ml[2]),
            "p_over25": p_over25_ml,
            "p_btts_yes": p_btts_ml,
        },

        "dixon_coles": {
            "p_home": float(dc_summary["p_home"]),
            "p_draw": float(dc_summary["p_draw"]),
            "p_away": float(dc_summary["p_away"]),
            "p_over25": float(dc_summary["p_over25"]),
            "p_under25": float(dc_summary["p_under25"]),
            "p_btts_yes": float(dc_summary["p_btts_yes"]),
            "p_btts_no": float(dc_summary["p_btts_no"]),
        },

        "monte_carlo": mc,

        "final": {
            "p_home": float(final_1x2["p_home"]),
            "p_draw": float(final_1x2["p_draw"]),
            "p_away": float(final_1x2["p_away"]),
            "pick": pick,
            "confidence": float(candidates[pick]),
        },

        "top_scores": [
            {
                "score": s["score"],
                "probability": float(s["probability"]),
            }
            for s in top_scores
        ],
    }


def print_prediction(result: dict):
    home = result["home_team"]
    away = result["away_team"]

    print()
    print("=" * 70)
    print(f"{home.upper()} vs {away.upper()}")
    print("=" * 70)

    print()
    print("GOLES ESPERADOS")
    print(f"- {home}: {result['lambda_home']:.3f}")
    print(f"- {away}: {result['lambda_away']:.3f}")

    print()
    print("ML 1X2")
    print(f"- {home}: {result['ml']['p_home']:.2%}")
    print(f"- Empate: {result['ml']['p_draw']:.2%}")
    print(f"- {away}: {result['ml']['p_away']:.2%}")

    print()
    print("DIXON-COLES 1X2")
    print(f"- {home}: {result['dixon_coles']['p_home']:.2%}")
    print(f"- Empate: {result['dixon_coles']['p_draw']:.2%}")
    print(f"- {away}: {result['dixon_coles']['p_away']:.2%}")

    print()
    print("FINAL 1X2")
    print(f"- {home}: {result['final']['p_home']:.2%}")
    print(f"- Empate: {result['final']['p_draw']:.2%}")
    print(f"- {away}: {result['final']['p_away']:.2%}")
    print(f"- Pick: {result['final']['pick']} ({result['final']['confidence']:.2%})")

    print()
    print("MERCADOS")
    print(f"- Over 2.5: {result['dixon_coles']['p_over25']:.2%}")
    print(f"- Under 2.5: {result['dixon_coles']['p_under25']:.2%}")
    print(f"- BTTS Sí: {result['dixon_coles']['p_btts_yes']:.2%}")
    print(f"- BTTS No: {result['dixon_coles']['p_btts_no']:.2%}")

    print()
    print("MONTE CARLO")
    print(f"- Simulaciones: {result['monte_carlo']['n_sim']:,}")
    print(f"- {home}: {result['monte_carlo']['home_win']:.2%}")
    print(f"- Empate: {result['monte_carlo']['draw']:.2%}")
    print(f"- {away}: {result['monte_carlo']['away_win']:.2%}")
    print(f"- Goles promedio: {result['monte_carlo']['avg_total_goals']:.3f}")

    print()
    print("TOP SCORES")
    for i, s in enumerate(result["top_scores"], start=1):
        print(f"{i}. {s['score']} -> {s['probability']:.2%}")


def main():
    # Test inicial. Luego lo reemplazaremos por fixtures reales.
    result = predict_match(
        home_team="Mexico",
        away_team="South Africa",
        neutral=0,
    )

    print_prediction(result)


if __name__ == "__main__":
    main()