# src/results.py
import pandas as pd
from src.config import MASTER_DIR


def load_manual_results() -> pd.DataFrame:
    path = MASTER_DIR / "manual_results.csv"

    if not path.exists():
        return pd.DataFrame(columns=[
            "match_id",
            "home_score",
            "away_score",
            "status",
        ])

    df = pd.read_csv(path)

    required = [
        "match_id",
        "home_score",
        "away_score",
        "status",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"manual_results.csv no tiene columnas requeridas: {missing}")

    df = df[df["status"].astype(str).str.upper() == "FINISHED"].copy()

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    df = df.dropna(subset=["home_score", "away_score"])

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    return df


def result_1x2(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "HOME"

    if home_score < away_score:
        return "AWAY"

    return "DRAW"


def result_over25(home_score: int, away_score: int) -> int:
    return int((home_score + away_score) > 2.5)


def result_btts(home_score: int, away_score: int) -> int:
    return int(home_score > 0 and away_score > 0)


def exact_score(home_score: int, away_score: int) -> str:
    return f"{home_score}-{away_score}"
