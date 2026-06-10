# src/elo.py
import math
import pandas as pd
from src.config import PROCESSED_DIR


BASE_ELO = 1500.0


COMPETITION_WEIGHTS = {
    "FIFA World Cup": 1.00,
    "World Cup": 1.00,
    "UEFA Euro": 0.90,
    "Copa América": 0.85,
    "African Cup of Nations": 0.80,
    "AFC Asian Cup": 0.75,
    "Gold Cup": 0.70,
    "FIFA World Cup qualification": 0.70,
    "World Cup qualification": 0.70,
    "UEFA Nations League": 0.55,
    "Friendly": 0.25,
}


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def actual_score(home_goals: int, away_goals: int) -> tuple[float, float]:
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def margin_multiplier(home_goals: int, away_goals: int, elo_diff: float) -> float:
    goal_diff = abs(home_goals - away_goals)

    if goal_diff <= 1:
        return 1.0

    return math.log(goal_diff + 1) * (2.2 / ((abs(elo_diff) * 0.001) + 2.2))


def competition_k(tournament: str) -> float:
    weight = COMPETITION_WEIGHTS.get(str(tournament), 0.50)
    return 30.0 * weight


def build_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()

    ratings: dict[str, float] = {}

    home_elos = []
    away_elos = []
    elo_diffs = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        rh = ratings.get(home, BASE_ELO)
        ra = ratings.get(away, BASE_ELO)

        home_elos.append(rh)
        away_elos.append(ra)
        elo_diffs.append(rh - ra)

        eh = expected_score(rh, ra)
        ea = 1 - eh

        ah, aa = actual_score(row["home_score"], row["away_score"])

        k = competition_k(row["tournament"])
        mult = margin_multiplier(row["home_score"], row["away_score"], rh - ra)

        ratings[home] = rh + k * mult * (ah - eh)
        ratings[away] = ra + k * mult * (aa - ea)

    df["elo_home_pre"] = home_elos
    df["elo_away_pre"] = away_elos
    df["elo_diff_pre"] = elo_diffs

    return df


def main():
    path = PROCESSED_DIR / "matches.parquet"

    if not path.exists():
        raise FileNotFoundError("Primero corre: python -m src.data_loader")

    df = pd.read_parquet(path)
    df = build_elo_features(df)

    out = PROCESSED_DIR / "matches_with_elo.parquet"
    df.to_parquet(out, index=False)

    print(df[[
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "elo_home_pre",
        "elo_away_pre",
        "elo_diff_pre",
    ]].tail())

    print(f"Guardado en: {out}")


if __name__ == "__main__":
    main()