# src/sportmonks_api.py
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    MASTER_DIR,
    RAW_DIR,
    SPORTMONKS_API_TOKEN,
    SPORTMONKS_BASE_URL,
)


def require_token():
    if not SPORTMONKS_API_TOKEN:
        raise RuntimeError("Falta SPORTMONKS_API_TOKEN en .env o GitHub Secrets")


def api_get(path: str, params: dict | None = None) -> dict:
    require_token()

    if params is None:
        params = {}

    params = dict(params)
    params["api_token"] = SPORTMONKS_API_TOKEN

    url = f"{SPORTMONKS_BASE_URL.rstrip('/')}/{path.lstrip('/')}"

    response = requests.get(url, params=params, timeout=40)

    print(f"[sportmonks] GET {response.url.replace(SPORTMONKS_API_TOKEN, '***')}")
    print(f"[sportmonks] status={response.status_code}")

    if response.status_code != 200:
        raise RuntimeError(f"Sportmonks error {response.status_code}: {response.text[:2000]}")

    return response.json()


def save_raw(payload: dict, category: str, date_str: str | None = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if date_str:
        out_dir = RAW_DIR / "sportmonks" / category / date_str
    else:
        out_dir = RAW_DIR / "sportmonks" / category

    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{stamp}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path


def fetch_fixtures_by_date(date_str: str) -> dict:
    """
    Sportmonks API v3 endpoint expected:
    /fixtures/date/{date}
    """
    return api_get(
        f"fixtures/date/{date_str}",
        params={
            "include": "participants;scores;state;venue;league;season;metadata",
        },
    )


def fetch_latest_updated_fixtures() -> dict:
    """
    Useful later for live/livescore changes.
    """
    return api_get(
        "fixtures/latest",
        params={
            "include": "participants;scores;state",
        },
    )

def search_leagues(query: str) -> dict:
    return api_get(
        f"leagues/search/{query}",
        params={
            "include": "currentSeason;seasons",
        },
    )


def fetch_fixtures_between(start_date: str, end_date: str) -> dict:
    return api_get(
        f"fixtures/between/{start_date}/{end_date}",
        params={
            "include": "participants;scores;state;venue;league;season",
        },
    )

def print_fixture_summary(payload: dict):
    data = payload.get("data", [])

    print(f"[sportmonks] fixtures={len(data)}")

    for item in data[:20]:
        fixture_id = item.get("id")
        name = item.get("name")
        starting_at = item.get("starting_at")
        state_id = item.get("state_id")
        result_info = item.get("result_info")

        participants = item.get("participants", [])
        teams = []

        if isinstance(participants, list):
            for p in participants:
                teams.append(p.get("name"))

        print()
        print(f"id={fixture_id}")
        print(f"name={name}")
        print(f"starting_at={starting_at}")
        print(f"state_id={state_id}")
        print(f"result_info={result_info}")
        print(f"participants={teams}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_test = sub.add_parser("test")

    p_date = sub.add_parser("fixtures-date")
    p_date.add_argument("--date", required=True)
    
    p_search_leagues = sub.add_parser("search-leagues")
    p_search_leagues.add_argument("--query", required=True)

    p_between = sub.add_parser("fixtures-between")
    p_between.add_argument("--start", required=True)
    p_between.add_argument("--end", required=True)

    p_latest = sub.add_parser("latest")

    args = parser.parse_args()

    if args.command == "test":
        payload = api_get("fixtures", params={"per_page": 1})
        path = save_raw(payload, "test")
        print(f"[OK] test raw guardado: {path}")
        print_fixture_summary(payload)

    elif args.command == "fixtures-date":
        payload = fetch_fixtures_by_date(args.date)
        path = save_raw(payload, "fixtures_date", args.date)
        print(f"[OK] fixtures raw guardado: {path}")
        print_fixture_summary(payload)

    elif args.command == "latest":
        payload = fetch_latest_updated_fixtures()
        path = save_raw(payload, "latest")
        print(f"[OK] latest raw guardado: {path}")
        print_fixture_summary(payload)
        
    elif args.command == "search-leagues":
        payload = search_leagues(args.query)
        path = save_raw(payload, "search_leagues", args.query.replace(" ", "_"))
        print(f"[OK] leagues raw guardado: {path}")
        data = payload.get("data", [])
        print(f"[sportmonks] leagues={len(data)}")
        for item in data[:20]:
            print()
            print(f"id={item.get('id')}")
            print(f"name={item.get('name')}")
            print(f"type={item.get('type')}")
            print(f"sub_type={item.get('sub_type')}")
            print(f"country_id={item.get('country_id')}")
            cs = item.get("currentseason") or item.get("currentSeason")
            if cs:
                print(f"currentSeason={cs}")

    elif args.command == "fixtures-between":
        payload = fetch_fixtures_between(args.start, args.end)
        path = save_raw(payload, "fixtures_between", f"{args.start}_{args.end}")
        print(f"[OK] fixtures raw guardado: {path}")
        print_fixture_summary(payload)

if __name__ == "__main__":
    main()
    
def search_leagues(query: str) -> dict:
    return api_get(
        f"leagues/search/{query}",
        params={
            "include": "currentSeason;seasons",
        },
    )


def fetch_fixtures_between(start_date: str, end_date: str) -> dict:
    return api_get(
        f"fixtures/between/{start_date}/{end_date}",
        params={
            "include": "participants;scores;state;venue;league;season",
        },
    )