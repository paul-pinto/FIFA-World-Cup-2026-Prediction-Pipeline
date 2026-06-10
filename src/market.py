# src/market.py
import math
import pandas as pd
from src.config import MASTER_DIR


def no_vig_3way(odds_home: float, odds_draw: float, odds_away: float) -> dict:
    raw_home = 1.0 / odds_home
    raw_draw = 1.0 / odds_draw
    raw_away = 1.0 / odds_away

    total = raw_home + raw_draw + raw_away

    return {
        "p_home": raw_home / total,
        "p_draw": raw_draw / total,
        "p_away": raw_away / total,
        "overround": total - 1.0,
    }


def no_vig_2way(odds_a: float, odds_b: float) -> dict:
    raw_a = 1.0 / odds_a
    raw_b = 1.0 / odds_b

    total = raw_a + raw_b

    return {
        "p_a": raw_a / total,
        "p_b": raw_b / total,
        "overround": total - 1.0,
    }


def poisson_cdf(k: int, lam: float) -> float:
    return sum(
        (lam ** i) * math.exp(-lam) / math.factorial(i)
        for i in range(k + 1)
    )


def over25_probability(lambda_total: float) -> float:
    return 1.0 - poisson_cdf(2, lambda_total)


def estimate_lambda_total_from_over25(p_over25: float) -> float:
    best_lam = 2.50
    best_error = float("inf")

    for i in range(50, 601):
        lam = i / 100
        p = over25_probability(lam)
        err = abs(p - p_over25)

        if err < best_error:
            best_error = err
            best_lam = lam

    return best_lam


def split_market_lambdas(
    lambda_total: float,
    p_home: float,
    p_away: float,
    alpha: float = 0.65,
) -> tuple[float, float]:
    raw_home_share = p_home / (p_home + p_away)

    home_share = 0.5 * (1 - alpha) + raw_home_share * alpha
    away_share = 1 - home_share

    return lambda_total * home_share, lambda_total * away_share


def load_manual_odds() -> pd.DataFrame:
    path = MASTER_DIR / "manual_odds.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def get_manual_odds_for_match(match_id: str) -> dict | None:
    df = load_manual_odds()

    if df.empty:
        return None

    row = df[df["match_id"] == match_id]

    if row.empty:
        return None

    r = row.iloc[0].to_dict()

    required = [
        "odds_home",
        "odds_draw",
        "odds_away",
        "odds_over25",
        "odds_under25",
    ]

    for col in required:
        if col not in r or pd.isna(r[col]):
            return None

    return {
        "odds_home": float(r["odds_home"]),
        "odds_draw": float(r["odds_draw"]),
        "odds_away": float(r["odds_away"]),
        "odds_over25": float(r["odds_over25"]),
        "odds_under25": float(r["odds_under25"]),
    }


def market_from_odds(odds: dict) -> dict:
    p_1x2 = no_vig_3way(
        odds_home=odds["odds_home"],
        odds_draw=odds["odds_draw"],
        odds_away=odds["odds_away"],
    )

    p_ou = no_vig_2way(
        odds_a=odds["odds_over25"],
        odds_b=odds["odds_under25"],
    )

    p_over25 = p_ou["p_a"]

    lambda_total = estimate_lambda_total_from_over25(p_over25)

    lambda_home, lambda_away = split_market_lambdas(
        lambda_total=lambda_total,
        p_home=p_1x2["p_home"],
        p_away=p_1x2["p_away"],
        alpha=0.65,
    )

    return {
        "p_home": p_1x2["p_home"],
        "p_draw": p_1x2["p_draw"],
        "p_away": p_1x2["p_away"],
        "overround_1x2": p_1x2["overround"],

        "p_over25": p_over25,
        "p_under25": p_ou["p_b"],
        "overround_ou25": p_ou["overround"],

        "lambda_total": lambda_total,
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
    }