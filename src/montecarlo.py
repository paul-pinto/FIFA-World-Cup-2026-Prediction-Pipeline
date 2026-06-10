# src/montecarlo.py
import numpy as np


def run_monte_carlo(
    score_matrix: np.ndarray,
    n_sim: int = 200_000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)

    flat = score_matrix.flatten()
    size = score_matrix.shape[0]

    samples = rng.choice(
        np.arange(len(flat)),
        size=n_sim,
        replace=True,
        p=flat,
    )

    home_goals = samples // size
    away_goals = samples % size
    total_goals = home_goals + away_goals

    return {
        "n_sim": int(n_sim),
        "home_win": float(np.mean(home_goals > away_goals)),
        "draw": float(np.mean(home_goals == away_goals)),
        "away_win": float(np.mean(home_goals < away_goals)),
        "over25": float(np.mean(total_goals > 2.5)),
        "under25": float(np.mean(total_goals <= 2.5)),
        "btts_yes": float(np.mean((home_goals > 0) & (away_goals > 0))),
        "btts_no": float(np.mean((home_goals == 0) | (away_goals == 0))),
        "avg_home_goals": float(np.mean(home_goals)),
        "avg_away_goals": float(np.mean(away_goals)),
        "avg_total_goals": float(np.mean(total_goals)),
    }