# src/sync_worldcup_results.py
import argparse
import pandas as pd

from src.config import MASTER_DIR
from src.results import load_manual_results


def load_fixtures() -> pd.DataFrame:
    path = MASTER_DIR / "worldcup_2026_fixtures.csv"

    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")

    df = pd.read_csv(path)

    required = [
        "match_id",
        "date_utc",
        "stage",
        "home_team",
        "away_team",
        "city",
        "country",
        "neutral",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"worldcup_2026_fixtures.csv faltan columnas: {missing}")

    return df


def build_worldcup_results() -> pd.DataFrame:
    fixtures = load_fixtures()
    results = load_manual_results()

    if results.empty:
        return pd.DataFrame(columns=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "city",
            "country",
            "neutral",
        ])

    merged = fixtures.merge(
        results,
        on="match_id",
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame(columns=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "city",
            "country",
            "neutral",
        ])

    out = pd.DataFrame()

    out["date"] = pd.to_datetime(merged["date_utc"], utc=True).dt.date.astype(str)
    out["home_team"] = merged["home_team"]
    out["away_team"] = merged["away_team"]
    out["home_score"] = merged["home_score"].astype(int)
    out["away_score"] = merged["away_score"].astype(int)
    out["tournament"] = "FIFA World Cup"
    out["city"] = merged["city"]
    out["country"] = merged["country"]
    out["neutral"] = merged["neutral"].astype(bool)

    out = out.drop_duplicates(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
        ],
        keep="last",
    )

    out = out.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    return out


def save_worldcup_results(df: pd.DataFrame) -> str:
    path = MASTER_DIR / "worldcup_results.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print",
        action="store_true",
        help="Imprime el dataframe generado",
    )

    args = parser.parse_args()

    df = build_worldcup_results()
    path = save_worldcup_results(df)

    print(f"[OK] worldcup_results.csv generado: {path}")
    print(f"[OK] filas: {len(df):,}")

    if args.print:
        print(df)


if __name__ == "__main__":
    main()