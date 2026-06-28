# src/knockout.py

def _as_probability(value):
    if value is None:
        return None

    try:
        value = float(value)
    except Exception:
        return None

    # Si viene como porcentaje 55.0, convertir a 0.55
    if value > 1.0:
        value = value / 100.0

    return value


def _find_probability(pred: dict, names: list[str]):
    """
    Busca una probabilidad tanto en la raíz del pred como dentro
    de bloques anidados: final, ml, dixon_coles, monte_carlo.
    """

    containers = [pred]

    for key in ["final", "ml", "dixon_coles", "monte_carlo", "market"]:
        value = pred.get(key)
        if isinstance(value, dict):
            containers.append(value)

    for container in containers:
        for name in names:
            if name in container and container[name] is not None:
                return _as_probability(container[name])

    return None


def estimate_advance_probabilities(pred: dict) -> dict:
    """
    Estima quién clasifica en partidos knockout.

    Usa probabilidades de 90 minutos:
    - local gana en 90'
    - empate en 90'
    - visitante gana en 90'

    Luego reparte la probabilidad de empate como proxy de prórroga/penales,
    usando fuerza relativa entre local y visitante.
    """

    p_home = _find_probability(
        pred,
        [
            "p_home",
            "p_home_win",
            "home_win",
            "home",
            "prob_home",
            "prob_home_win",
            "home_win_prob",
            "final_p_home",
            "final_home_prob",
            "home_prob",
            "p1",
            "1",
        ],
    )

    p_draw = _find_probability(
        pred,
        [
            "p_draw",
            "draw",
            "prob_draw",
            "draw_prob",
            "final_p_draw",
            "final_draw_prob",
            "px",
            "x",
            "X",
        ],
    )

    p_away = _find_probability(
        pred,
        [
            "p_away",
            "p_away_win",
            "away_win",
            "away",
            "prob_away",
            "prob_away_win",
            "away_win_prob",
            "final_p_away",
            "final_away_prob",
            "away_prob",
            "p2",
            "2",
        ],
    )

    if p_home is None or p_draw is None or p_away is None:
        return {
            "knockout": True,
            "advance_available": False,
            "p_home_advance": None,
            "p_away_advance": None,
            "advance_pick": None,
            "advance_confidence": None,
            "shootout_edge_home": None,
            "advance_note": (
                "No se encontraron probabilidades 1X2 compatibles. "
                f"Detectado p_home={p_home}, p_draw={p_draw}, p_away={p_away}."
            ),
        }

    total = p_home + p_draw + p_away

    if total <= 0:
        return {
            "knockout": True,
            "advance_available": False,
            "p_home_advance": None,
            "p_away_advance": None,
            "advance_pick": None,
            "advance_confidence": None,
            "shootout_edge_home": None,
            "advance_note": "Probabilidades 1X2 inválidas.",
        }

    # Normalizamos por seguridad.
    p_home = p_home / total
    p_draw = p_draw / total
    p_away = p_away / total

    no_draw_total = p_home + p_away

    if no_draw_total <= 0:
        shootout_edge_home = 0.50
    else:
        relative_strength_home = p_home / no_draw_total

        # Ventaja moderada para prórroga/penales.
        # Evitamos extremos absurdos.
        shootout_edge_home = 0.50 + (relative_strength_home - 0.50) * 0.50
        shootout_edge_home = max(0.40, min(0.60, shootout_edge_home))

    p_home_advance = p_home + p_draw * shootout_edge_home
    p_away_advance = p_away + p_draw * (1.0 - shootout_edge_home)

    total_adv = p_home_advance + p_away_advance

    if total_adv > 0:
        p_home_advance = p_home_advance / total_adv
        p_away_advance = p_away_advance / total_adv

    if p_home_advance >= p_away_advance:
        pick = "home"
        confidence = p_home_advance
    else:
        pick = "away"
        confidence = p_away_advance

    return {
        "knockout": True,
        "advance_available": True,
        "p_home_advance": round(p_home_advance, 4),
        "p_away_advance": round(p_away_advance, 4),
        "advance_pick": pick,
        "advance_confidence": round(confidence, 4),
        "shootout_edge_home": round(shootout_edge_home, 4),
        "advance_note": "Estimación de clasificación: 90 minutos + reparto del empate por fuerza relativa.",
    }