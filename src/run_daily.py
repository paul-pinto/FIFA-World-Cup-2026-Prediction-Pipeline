# src/run_daily.py
import argparse
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from src.config import MASTER_DIR
from src.predictor import predict_match
from src.exporter import export_daily
from src.market import get_manual_odds_for_match, market_from_odds
from src.value import evaluate_value_bets


LOCAL_TZ = "America/La_Paz"


def load_fixtures() -> pd.DataFrame:
    path = MASTER_DIR / "worldcup_2026_fixtures.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Crea data/master/worldcup_2026_fixtures.csv"
        )

    df = pd.read_csv(path)
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True, errors="coerce")

    if df["date_utc"].isna().any():
        bad = df[df["date_utc"].isna()]
        raise ValueError(
            "Hay filas con date_utc inválido en worldcup_2026_fixtures.csv:\n"
            + bad.to_string(index=False)
        )

    return df


def filter_fixtures_by_date(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """
    Filtra por fecha calendario Bolivia, no por fecha UTC.

    Ejemplo:
    Sweden vs Tunisia = 2026-06-15T02:00:00Z
    En Bolivia es 2026-06-14 22:00, por tanto debe entrar en --date 2026-06-14.
    """
    target_date = pd.to_datetime(target_date).strftime("%Y-%m-%d")

    df = df.copy()
    df["kickoff_local"] = df["date_utc"].dt.tz_convert(LOCAL_TZ)
    df["date_local"] = df["kickoff_local"].dt.strftime("%Y-%m-%d")

    today = df[df["date_local"] == target_date].copy()
    today = today.sort_values("kickoff_local")

    return today


def predict_fixture(row) -> dict:
    match_id = row.get("match_id", "")

    odds = get_manual_odds_for_match(match_id)

    if odds is not None:
        market = market_from_odds(odds)
        print("    odds: OK")
    else:
        market = None
        print("    odds: NO DISPONIBLES")

    pred = predict_match(
        home_team=row["home_team"],
        away_team=row["away_team"],
        neutral=int(row.get("neutral", 1)),
        market=market,
    )

    pred["odds"] = odds
    pred["value"] = evaluate_value_bets(pred, odds)

    pred["match_id"] = match_id
    pred["date_utc"] = row["date_utc"].isoformat()
    pred["date_local"] = row.get("date_local", "")
    pred["kickoff_local"] = (
        row["kickoff_local"].isoformat()
        if "kickoff_local" in row and pd.notna(row["kickoff_local"])
        else ""
    )
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
        print(f"[WARN] No hay partidos para {target_date} en horario Bolivia ({LOCAL_TZ})")
        return []

    print(f"[INFO] Fecha objetivo Bolivia: {target_date}")
    print(f"[INFO] Partidos encontrados: {len(today)}")
    print(
        today[
            ["match_id", "date_utc", "kickoff_local", "home_team", "away_team"]
        ].to_string(index=False)
    )

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
        default=datetime.now(ZoneInfo(LOCAL_TZ)).date().isoformat(),
        help="Fecha objetivo local Bolivia YYYY-MM-DD",
    )

    args = parser.parse_args()

    run_daily(args.date)


if __name__ == "__main__":
    main()