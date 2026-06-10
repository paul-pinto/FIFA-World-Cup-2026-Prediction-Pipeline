# src/data_loader.py
import pandas as pd
from pathlib import Path
from src.config import RAW_DIR, MASTER_DIR, PROCESSED_DIR


def load_team_aliases() -> dict:
    path = MASTER_DIR / "team_aliases.csv"

    if not path.exists():
        return {}

    df = pd.read_csv(path)

    aliases = {}

    for _, row in df.iterrows():
        canonical = str(row["canonical"]).strip()
        alias = str(row["alias"]).strip()
        aliases[alias.lower()] = canonical
        aliases[canonical.lower()] = canonical

    return aliases


def canonical_team(name: str, aliases: dict) -> str:
    if pd.isna(name):
        return ""

    clean = str(name).strip()
    return aliases.get(clean.lower(), clean)


def load_international_results() -> pd.DataFrame:
    path = RAW_DIR / "international_results" / "results.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Guarda ahí el histórico internacional como results.csv"
        )

    df = pd.read_csv(path)

    required = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas en results.csv: {missing}")

    aliases = load_team_aliases()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["home_team"] = df["home_team"].apply(lambda x: canonical_team(x, aliases))
    df["away_team"] = df["away_team"].apply(lambda x: canonical_team(x, aliases))

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    df = df.dropna(subset=["home_score", "away_score"])

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df["neutral"] = df["neutral"].astype(bool)

    df = df.sort_values("date").reset_index(drop=True)

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def target_1x2(row):
        if row["home_score"] > row["away_score"]:
            return 0  # HOME
        if row["home_score"] == row["away_score"]:
            return 1  # DRAW
        return 2      # AWAY

    df["target_1x2"] = df.apply(target_1x2, axis=1)
    df["target_home_goals"] = df["home_score"]
    df["target_away_goals"] = df["away_score"]
    df["target_total_goals"] = df["home_score"] + df["away_score"]
    df["target_over25"] = (df["target_total_goals"] > 2.5).astype(int)
    df["target_btts"] = ((df["home_score"] > 0) & (df["away_score"] > 0)).astype(int)

    return df


def save_processed_matches(df: pd.DataFrame) -> Path:
    out = PROCESSED_DIR / "matches.parquet"
    df.to_parquet(out, index=False)
    return out


def build_matches_dataset() -> pd.DataFrame:
    df = load_international_results()
    df = add_targets(df)
    save_processed_matches(df)
    return df


if __name__ == "__main__":
    df = build_matches_dataset()
    print(df.head())
    print(f"Partidos procesados: {len(df):,}")
    print(f"Guardado en: {PROCESSED_DIR / 'matches.parquet'}")