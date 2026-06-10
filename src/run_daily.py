# src/run_daily.py
import argparse
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.config import MASTER_DIR
from src.predictor import predict_match
from src.exporter import export_daily
from src.market import get_manual_odds_for_match, market_from_odds


def load_fixtures() -> pd.DataFrame:
    path = MASTER_DIR / "worldcup_2026_fixtures.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Crea data/master/worldcup_2026_fixtures.csv"
        )

    df = pd.read_csv(path)
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True)

    return df


def filter_fixtures_by_date(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    date = pd.to_datetime(target_date).date()
    return df[df["date_utc"].dt.date == date].copy()


def predict_fixture(row) -> dict:
    match_id = row.get("match_id", "")

    odds = get_manual_odds_for_match(match_id)

    if odds is not None:
        market = market_from_odds(odds)
        print(f"    odds: OK")
    else:
        market = None
        print(f"    odds: NO DISPONIBLES")

    pred = predict_match(
        home_team=row["home_team"],
        away_team=row["away_team"],
        neutral=int(row.get("neutral", 1)),
        market=market,
    )

    pred["match_id"] = match_id
    pred["date_utc"] = row["date_utc"].isoformat()
    pred["stage"] = row.get("stage", "")
    pred["group"] = row.get("group", "")
    pred["venue"] = row.get("venue", "")
    pred["city"] = row.get("city", "")
    pred["country"] = row.get("country", "")

    return pred

    pred["match_id"] = row.get("match_id", "")
    pred["date_utc"] = row["date_utc"].isoformat()
    pred["stage"] = row.get("stage", "")
    pred["group"] = row.get("group", "")
    pred["venue"] = row.get("venue", "")
    pred["city"] = row.get("city", "")
    pred["country"] = row.get("country", "")

    return pred


def run_daily(target_date: str):
    fixtures = load_fixtures()
    today = filter_fixtures_by_date(fixtures, target_date)

    if today.empty:
        print(f"[WARN] No hay partidos para {target_date}")
        return []

    predictions = []

    for _, row in today.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        print(f"[+] Prediciendo {home} vs {away}")

        try:
            pred = predict_fixture(row)
            predictions.append(pred)
        except Exception as e:
            print(f"[ERROR] {home} vs {away}: {e}")

    paths = export_daily(predictions, run_date=target_date)

    print()
    print("[OK] Outputs generados:")
    for k, v in paths.items():
        print(f"- {k}: {v}")

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).date().isoformat(),
        help="Fecha objetivo YYYY-MM-DD",
    )

    args = parser.parse_args()

    run_daily(args.date)


if __name__ == "__main__":
    main()