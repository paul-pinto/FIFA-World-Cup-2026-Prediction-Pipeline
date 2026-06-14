# src/odds_api.py
import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from src.data_loader import load_team_aliases, canonical_team
from src.config import PROCESSED_DIR

import pandas as pd
import requests

from src.config import (
    THE_ODDS_API_KEY,
    ODDS_SPORT_KEY,
    ODDS_REGIONS,
    ODDS_MARKETS,
    ODDS_FORMAT,
    MASTER_DIR,
    RAW_DIR,
    PROCESSED_DIR,
)


API_BASE = "https://api.the-odds-api.com/v4"


def utc_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_odds(
    sport_key: str,
    commence_from: datetime,
    commence_to: datetime,
) -> list[dict]:
    if not THE_ODDS_API_KEY:
        raise RuntimeError("Falta THE_ODDS_API_KEY en .env")

    url = f"{API_BASE}/sports/{sport_key}/odds"

    params = {
        "apiKey": THE_ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "markets": ODDS_MARKETS,
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": "iso",
        "commenceTimeFrom": utc_z(commence_from),
        "commenceTimeTo": utc_z(commence_to),
    }

    response = requests.get(url, params=params, timeout=30)

    print(f"[odds-api] status={response.status_code}")
    print(f"[odds-api] requests-used={response.headers.get('x-requests-used')}")
    print(f"[odds-api] requests-remaining={response.headers.get('x-requests-remaining')}")
    print(f"[odds-api] requests-last={response.headers.get('x-requests-last')}")

    if response.status_code != 200:
        raise RuntimeError(f"The Odds API error {response.status_code}: {response.text}")

    return response.json()


def save_raw_odds(events: list[dict], target_date: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%H-%M-%S")
    out_dir = RAW_DIR / "odds_api" / target_date
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{ts}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    return path


def load_fixtures() -> pd.DataFrame:
    path = MASTER_DIR / "worldcup_2026_fixtures.csv"

    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")

    df = pd.read_csv(path)
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True)

    return df


def normalize_team_name(name: str) -> str:
    aliases = load_team_aliases()
    canonical = canonical_team(str(name).strip(), aliases)
    return canonical.strip().lower()


def find_fixture_match(event: dict, fixtures: pd.DataFrame) -> dict | None:
    home = normalize_team_name(event.get("home_team", ""))
    away = normalize_team_name(event.get("away_team", ""))

    if not home or not away:
        return None

    fixtures = fixtures.copy()
    fixtures["_home"] = fixtures["home_team"].apply(normalize_team_name)
    fixtures["_away"] = fixtures["away_team"].apply(normalize_team_name)

    exact = fixtures[
        (fixtures["_home"] == home)
        & (fixtures["_away"] == away)
    ]

    if not exact.empty:
        return exact.iloc[0].to_dict()

    reversed_match = fixtures[
        (fixtures["_home"] == away)
        & (fixtures["_away"] == home)
    ]

    if not reversed_match.empty:
        row = reversed_match.iloc[0].to_dict()
        row["_reversed"] = True
        return row

    return None


def extract_market(bookmaker: dict, key: str) -> dict | None:
    for market in bookmaker.get("markets", []):
        if market.get("key") == key:
            return market
    return None


def extract_h2h_prices(event: dict, bookmaker: dict) -> dict | None:
    market = extract_market(bookmaker, "h2h")

    if market is None:
        return None

    home_team = event.get("home_team")
    away_team = event.get("away_team")

    prices = {}

    for outcome in market.get("outcomes", []):
        name = outcome.get("name")
        price = outcome.get("price")

        if name is None or price is None:
            continue

        if name == home_team:
            prices["odds_home"] = float(price)
        elif name == away_team:
            prices["odds_away"] = float(price)
        elif str(name).lower() == "draw":
            prices["odds_draw"] = float(price)

    if {"odds_home", "odds_draw", "odds_away"}.issubset(prices.keys()):
        return prices

    return None


def extract_totals_prices(bookmaker: dict, point: float = 2.5) -> dict | None:
    market = extract_market(bookmaker, "totals")

    if market is None:
        return None

    prices = {}

    for outcome in market.get("outcomes", []):
        name = str(outcome.get("name", "")).lower()
        price = outcome.get("price")
        outcome_point = outcome.get("point")

        if price is None:
            continue

        if outcome_point is not None and float(outcome_point) != float(point):
            continue

        if name == "over":
            prices["odds_over25"] = float(price)
        elif name == "under":
            prices["odds_under25"] = float(price)

    if {"odds_over25", "odds_under25"}.issubset(prices.keys()):
        return prices

    return None


def no_vig_3way_from_odds(odds_home: float, odds_draw: float, odds_away: float) -> dict:
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


def no_vig_2way_from_odds(odds_over: float, odds_under: float) -> dict:
    raw_over = 1.0 / odds_over
    raw_under = 1.0 / odds_under

    total = raw_over + raw_under

    return {
        "p_over25": raw_over / total,
        "p_under25": raw_under / total,
        "overround": total - 1.0,
    }


def synthetic_odds(probability: float) -> float:
    probability = max(1e-9, min(1 - 1e-9, float(probability)))
    return 1.0 / probability


def aggregate_event_odds(event: dict) -> dict | None:
    """
    Agrega odds por evento usando consenso no-vig por bookmaker.

    Flujo:
    - extraer cuotas por bookmaker
    - convertir cada bookmaker a probabilidades implícitas
    - remover overround por bookmaker
    - promediar probabilidades no-vig
    - convertir probabilidades consenso a odds sintéticas
    """
    h2h_probs = []
    totals_probs = []

    h2h_overrounds = []
    totals_overrounds = []

    for bookmaker in event.get("bookmakers", []):
        h2h = extract_h2h_prices(event, bookmaker)
        totals = extract_totals_prices(bookmaker, point=2.5)

        if h2h is not None:
            try:
                p = no_vig_3way_from_odds(
                    odds_home=h2h["odds_home"],
                    odds_draw=h2h["odds_draw"],
                    odds_away=h2h["odds_away"],
                )
                h2h_probs.append(p)
                h2h_overrounds.append(p["overround"])
            except Exception:
                pass

        if totals is not None:
            try:
                p = no_vig_2way_from_odds(
                    odds_over=totals["odds_over25"],
                    odds_under=totals["odds_under25"],
                )
                totals_probs.append(p)
                totals_overrounds.append(p["overround"])
            except Exception:
                pass

    if not h2h_probs:
        return None

    h2h_df = pd.DataFrame(h2h_probs)

    consensus_p_home = float(h2h_df["p_home"].mean())
    consensus_p_draw = float(h2h_df["p_draw"].mean())
    consensus_p_away = float(h2h_df["p_away"].mean())

    total = consensus_p_home + consensus_p_draw + consensus_p_away

    consensus_p_home /= total
    consensus_p_draw /= total
    consensus_p_away /= total

    out = {
        "odds_home": synthetic_odds(consensus_p_home),
        "odds_draw": synthetic_odds(consensus_p_draw),
        "odds_away": synthetic_odds(consensus_p_away),

        "p_home_consensus": consensus_p_home,
        "p_draw_consensus": consensus_p_draw,
        "p_away_consensus": consensus_p_away,

        "bookmakers_h2h": int(len(h2h_probs)),
        "avg_overround_h2h": float(sum(h2h_overrounds) / len(h2h_overrounds)) if h2h_overrounds else None,
    }

    if totals_probs:
        totals_df = pd.DataFrame(totals_probs)

        consensus_p_over25 = float(totals_df["p_over25"].mean())
        consensus_p_under25 = float(totals_df["p_under25"].mean())

        total_ou = consensus_p_over25 + consensus_p_under25

        consensus_p_over25 /= total_ou
        consensus_p_under25 /= total_ou

        out["odds_over25"] = synthetic_odds(consensus_p_over25)
        out["odds_under25"] = synthetic_odds(consensus_p_under25)

        out["p_over25_consensus"] = consensus_p_over25
        out["p_under25_consensus"] = consensus_p_under25

        out["bookmakers_totals"] = int(len(totals_probs))
        out["avg_overround_totals"] = float(sum(totals_overrounds) / len(totals_overrounds)) if totals_overrounds else None
    else:
        out["odds_over25"] = None
        out["odds_under25"] = None
        out["p_over25_consensus"] = None
        out["p_under25_consensus"] = None
        out["bookmakers_totals"] = 0
        out["avg_overround_totals"] = None

    return out
    """
    Agrega odds por evento usando promedio simple de probabilidades/cuotas disponibles.
    Para V1 usamos promedio de cuotas por mercado entre bookmakers válidos.
    """
    h2h_rows = []
    totals_rows = []

    for bookmaker in event.get("bookmakers", []):
        h2h = extract_h2h_prices(event, bookmaker)
        totals = extract_totals_prices(bookmaker, point=2.5)

        if h2h is not None:
            h2h_rows.append(h2h)

        if totals is not None:
            totals_rows.append(totals)

    if not h2h_rows:
        return None

    h2h_df = pd.DataFrame(h2h_rows)

    out = {
        "odds_home": float(h2h_df["odds_home"].mean()),
        "odds_draw": float(h2h_df["odds_draw"].mean()),
        "odds_away": float(h2h_df["odds_away"].mean()),
        "bookmakers_h2h": int(len(h2h_df)),
    }

    if totals_rows:
        totals_df = pd.DataFrame(totals_rows)
        out["odds_over25"] = float(totals_df["odds_over25"].mean())
        out["odds_under25"] = float(totals_df["odds_under25"].mean())
        out["bookmakers_totals"] = int(len(totals_df))
    else:
        out["odds_over25"] = None
        out["odds_under25"] = None
        out["bookmakers_totals"] = 0

    return out


def build_manual_odds_from_api(events: list[dict], target_date: str) -> pd.DataFrame:
    fixtures = load_fixtures()
    target_day = pd.to_datetime(target_date).date()
    fixtures = fixtures[fixtures["date_utc"].dt.date == target_day].copy()

    rows = []

    for event in events:
        fixture = find_fixture_match(event, fixtures)

        if fixture is None:
            continue

        odds = aggregate_event_odds(event)

        if odds is None:
            continue

        # Si no hay totals 2.5, igual guardamos H2H.
        # El predictor puede usar 1X2 de mercado y dejar Over/Under sin consenso.
        if odds.get("odds_over25") is None or odds.get("odds_under25") is None:
            print(f"[WARN] Sin totals 2.5 para {fixture['match_id']}, se guarda solo H2H")

        rows.append({
            "match_id": fixture["match_id"],

            "odds_home": odds["odds_home"],
            "odds_draw": odds["odds_draw"],
            "odds_away": odds["odds_away"],
            "odds_over25": odds["odds_over25"],
            "odds_under25": odds["odds_under25"],

            "source": "the_odds_api_no_vig_consensus",

            "bookmakers_h2h": odds["bookmakers_h2h"],
            "bookmakers_totals": odds["bookmakers_totals"],

            "avg_overround_h2h": odds.get("avg_overround_h2h"),
            "avg_overround_totals": odds.get("avg_overround_totals"),

            "p_home_consensus": odds.get("p_home_consensus"),
            "p_draw_consensus": odds.get("p_draw_consensus"),
            "p_away_consensus": odds.get("p_away_consensus"),
            "p_over25_consensus": odds.get("p_over25_consensus"),
            "p_under25_consensus": odds.get("p_under25_consensus"),
        })

    return pd.DataFrame(rows)

def append_odds_snapshot(api_odds: pd.DataFrame, target_date: str) -> Path:
    path = PROCESSED_DIR / "odds_snapshots.csv"

    if api_odds.empty:
        return path

    snapshot_utc = datetime.now(timezone.utc).isoformat()

    df = api_odds.copy()
    df.insert(0, "snapshot_utc", snapshot_utc)
    df.insert(1, "target_date", target_date)

    if path.exists():
        current = pd.read_csv(path)
        final = pd.concat([current, df], ignore_index=True)
    else:
        final = df

    final.to_csv(path, index=False, encoding="utf-8")

    return path

def update_manual_odds_from_api(target_date: str) -> Path:
    day = pd.to_datetime(target_date).date()
    commence_from = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
    commence_to = commence_from + timedelta(days=1)

    events = fetch_odds(
        sport_key=ODDS_SPORT_KEY,
        commence_from=commence_from,
        commence_to=commence_to,
    )

    raw_path = save_raw_odds(events, target_date)
    print(f"[OK] raw odds guardado: {raw_path}")

    api_odds = build_manual_odds_from_api(events, target_date)
    
    snapshot_path = append_odds_snapshot(api_odds, target_date)
    print(f"[OK] odds snapshot actualizado: {snapshot_path}")

    manual_path = MASTER_DIR / "manual_odds.csv"

    if manual_path.exists():
        current = pd.read_csv(manual_path)
    else:
        current = pd.DataFrame()

    if current.empty:
        final = api_odds
    else:
        # Eliminamos filas del mismo match_id si vienen de API nuevas.
        if not api_odds.empty:
            current = current[~current["match_id"].isin(api_odds["match_id"])]
        final = pd.concat([current, api_odds], ignore_index=True)

    final.to_csv(manual_path, index=False, encoding="utf-8")

    print(f"[OK] manual_odds actualizado: {manual_path}")
    print(f"[OK] filas API: {len(api_odds):,}")
    print(api_odds)

    return manual_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    update_manual_odds_from_api(args.date)


if __name__ == "__main__":
    main()