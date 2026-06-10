# src/poisson_dc.py
import math
import numpy as np


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def tau_dixon_coles(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float = -0.10,
) -> float:
    h = home_goals
    a = away_goals
    lh = lambda_home
    la = lambda_away

    if h == 0 and a == 0:
        return 1 - lh * la * rho

    if h == 1 and a == 0:
        return 1 + la * rho

    if h == 0 and a == 1:
        return 1 + lh * rho

    if h == 1 and a == 1:
        return 1 - rho

    return 1.0


def build_score_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 10,
    rho: float = -0.10,
    use_dixon_coles: bool = True,
) -> np.ndarray:
    matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson_pmf(h, lambda_home) * poisson_pmf(a, lambda_away)

            if use_dixon_coles:
                p *= tau_dixon_coles(
                    home_goals=h,
                    away_goals=a,
                    lambda_home=lambda_home,
                    lambda_away=lambda_away,
                    rho=rho,
                )

            matrix[h, a] = p

    total = matrix.sum()

    if total <= 0:
        raise ValueError("La matriz de score tiene probabilidad total <= 0")

    matrix /= total

    return matrix


def summarize_score_matrix(matrix: np.ndarray) -> dict:
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    p_over25 = 0.0
    p_btts = 0.0

    rows, cols = matrix.shape

    scores = []

    for h in range(rows):
        for a in range(cols):
            p = float(matrix[h, a])

            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p

            if h + a > 2:
                p_over25 += p

            if h > 0 and a > 0:
                p_btts += p

            scores.append({
                "score": f"{h}-{a}",
                "home_goals": h,
                "away_goals": a,
                "probability": p,
            })

    scores.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over25": p_over25,
        "p_under25": 1 - p_over25,
        "p_btts_yes": p_btts,
        "p_btts_no": 1 - p_btts,
        "top_scores": scores,
    }


def clamp_lambda(value: float, low: float = 0.15, high: float = 4.50) -> float:
    return max(low, min(float(value), high))