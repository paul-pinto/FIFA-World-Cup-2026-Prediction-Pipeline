# src/value.py
from typing import Optional


MIN_EV = 0.03
MIN_EDGE = 0.02


def implied_probability(odds: float) -> float:
    if odds <= 1:
        raise ValueError("Odds debe ser > 1")
    return 1.0 / odds


def expected_value(probability: float, odds: float) -> float:
    return probability * odds - 1.0


def compute_edge(probability: float, odds: float) -> float:
    return probability - implied_probability(odds)


def evaluate_market(
    name: str,
    probability: float,
    odds: Optional[float],
    min_ev: float = MIN_EV,
    min_edge: float = MIN_EDGE,
) -> Optional[dict]:
    if odds is None:
        return None

    if odds <= 1:
        return None

    probability = float(probability)
    odds = float(odds)

    implied = implied_probability(odds)
    edge = compute_edge(probability, odds)
    ev = expected_value(probability, odds)

    return {
        "market": name,
        "probability": probability,
        "odds": odds,
        "implied_probability": implied,
        "edge": edge,
        "ev": ev,
        "is_value": bool(ev >= min_ev and edge >= min_edge),
    }


def evaluate_value_bets(
    pred: dict,
    odds: Optional[dict],
    min_ev: float = MIN_EV,
    min_edge: float = MIN_EDGE,
) -> dict:
    if odds is None:
        return {
            "has_value_data": False,
            "bets": [],
            "best_value": None,
            "best_edge": None,
        }

    candidates = [
        evaluate_market(
            name="HOME",
            probability=pred["final"]["p_home"],
            odds=odds.get("odds_home"),
            min_ev=min_ev,
            min_edge=min_edge,
        ),
        evaluate_market(
            name="DRAW",
            probability=pred["final"]["p_draw"],
            odds=odds.get("odds_draw"),
            min_ev=min_ev,
            min_edge=min_edge,
        ),
        evaluate_market(
            name="AWAY",
            probability=pred["final"]["p_away"],
            odds=odds.get("odds_away"),
            min_ev=min_ev,
            min_edge=min_edge,
        ),
        evaluate_market(
            name="OVER_2_5",
            probability=pred["dixon_coles"]["p_over25"],
            odds=odds.get("odds_over25"),
            min_ev=min_ev,
            min_edge=min_edge,
        ),
        evaluate_market(
            name="UNDER_2_5",
            probability=pred["dixon_coles"]["p_under25"],
            odds=odds.get("odds_under25"),
            min_ev=min_ev,
            min_edge=min_edge,
        ),
    ]

    bets = [b for b in candidates if b is not None]
    value_bets = [b for b in bets if b["is_value"]]

    if value_bets:
        best_value = max(value_bets, key=lambda x: x["ev"])
    else:
        best_value = None

    best_edge = max(bets, key=lambda x: x["edge"]) if bets else None

    return {
        "has_value_data": True,
        "bets": bets,
        "best_value": best_value,
        "best_edge": best_edge,
    }