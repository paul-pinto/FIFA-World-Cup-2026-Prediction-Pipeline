# src/features.py
import pandas as pd
from src.config import PROCESSED_DIR


def compute_team_form_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()

    history = {}
    rows = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        home_hist = history.get(home, [])
        away_hist = history.get(away, [])

        feat = row.to_dict()

        for prefix, hist in [("home", home_hist), ("away", away_hist)]:
            last5 = hist[-5:]
            last10 = hist[-10:]
            last20 = hist[-20:]

            def avg(items, key):
                if not items:
                    return 0.0
                return sum(x[key] for x in items) / len(items)

            feat[f"{prefix}_gf_5"] = avg(last5, "gf")
            feat[f"{prefix}_ga_5"] = avg(last5, "ga")
            feat[f"{prefix}_gf_10"] = avg(last10, "gf")
            feat[f"{prefix}_ga_10"] = avg(last10, "ga")
            feat[f"{prefix}_gf_20"] = avg(last20, "gf")
            feat[f"{prefix}_ga_20"] = avg(last20, "ga")

            feat[f"{prefix}_points_5"] = avg(last5, "points")
            feat[f"{prefix}_points_10"] = avg(last10, "points")
            feat[f"{prefix}_points_20"] = avg(last20, "points")

            feat[f"{prefix}_matches_hist"] = len(hist)

        rows.append(feat)

        home_goals = int(row["home_score"])
        away_goals = int(row["away_score"])

        if home_goals > away_goals:
            home_points, away_points = 3, 0
        elif home_goals < away_goals:
            home_points, away_points = 0, 3
        else:
            home_points, away_points = 1, 1

        history.setdefault(home, []).append({
            "gf": home_goals,
            "ga": away_goals,
            "points": home_points,
        })

        history.setdefault(away, []).append({
            "gf": away_goals,
            "ga": home_goals,
            "points": away_points,
        })

    return pd.DataFrame(rows)


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["neutral"] = df["neutral"].astype(int)

    df["goal_diff_form_5"] = (
        (df["home_gf_5"] - df["home_ga_5"])
        - (df["away_gf_5"] - df["away_ga_5"])
    )

    df["goal_diff_form_10"] = (
        (df["home_gf_10"] - df["home_ga_10"])
        - (df["away_gf_10"] - df["away_ga_10"])
    )

    df["goal_diff_form_20"] = (
        (df["home_gf_20"] - df["home_ga_20"])
        - (df["away_gf_20"] - df["away_ga_20"])
    )

    df["points_form_diff_5"] = df["home_points_5"] - df["away_points_5"]
    df["points_form_diff_10"] = df["home_points_10"] - df["away_points_10"]
    df["points_form_diff_20"] = df["home_points_20"] - df["away_points_20"]

    df["home_attack_strength_5"] = df["home_gf_5"]
    df["away_attack_strength_5"] = df["away_gf_5"]

    df["home_defense_weakness_5"] = df["home_ga_5"]
    df["away_defense_weakness_5"] = df["away_ga_5"]

    df["attack_diff_5"] = df["home_gf_5"] - df["away_gf_5"]
    df["defense_diff_5"] = df["away_ga_5"] - df["home_ga_5"]

    return df


def main():
    path = PROCESSED_DIR / "matches_with_elo.parquet"

    if not path.exists():
        raise FileNotFoundError("Primero corre: python -m src.elo")

    df = pd.read_parquet(path)
    df = compute_team_form_features(df)
    df = add_basic_features(df)

    out = PROCESSED_DIR / "training_dataset.parquet"
    df.to_parquet(out, index=False)

    print(df[[
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "elo_diff_pre",
        "home_gf_5",
        "away_gf_5",
        "goal_diff_form_5",
        "points_form_diff_5",
    ]].tail())

    print(f"Dataset final guardado en: {out}")
    print(f"Filas: {len(df):,}")
    print(f"Columnas: {len(df.columns):,}")


if __name__ == "__main__":
    main()