# src/ml_models.py
import json
import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss, accuracy_score, mean_absolute_error, roc_auc_score
from src.config import PROCESSED_DIR, MODELS_DIR


FEATURE_COLUMNS = [
    "neutral",
    "elo_home_pre",
    "elo_away_pre",
    "elo_diff_pre",

    "home_gf_5",
    "home_ga_5",
    "away_gf_5",
    "away_ga_5",
    "home_gf_10",
    "home_ga_10",
    "away_gf_10",
    "away_ga_10",
    "home_gf_20",
    "home_ga_20",
    "away_gf_20",
    "away_ga_20",

    "home_points_5",
    "away_points_5",
    "home_points_10",
    "away_points_10",
    "home_points_20",
    "away_points_20",

    "goal_diff_form_5",
    "goal_diff_form_10",
    "goal_diff_form_20",

    "points_form_diff_5",
    "points_form_diff_10",
    "points_form_diff_20",

    "home_attack_strength_5",
    "away_attack_strength_5",
    "home_defense_weakness_5",
    "away_defense_weakness_5",

    "attack_diff_5",
    "defense_diff_5",

    "home_matches_hist",
    "away_matches_hist",
]


def load_training_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "training_dataset.parquet"

    if not path.exists():
        raise FileNotFoundError("Primero corre: python -m src.features")

    df = pd.read_parquet(path)

    df = df.dropna(subset=FEATURE_COLUMNS + [
        "target_1x2",
        "target_home_goals",
        "target_away_goals",
        "target_over25",
        "target_btts",
    ])

    # Evitamos primeros partidos sin historial suficiente.
    df = df[
        (df["home_matches_hist"] >= 5)
        & (df["away_matches_hist"] >= 5)
    ].copy()

    return df


def temporal_split(df: pd.DataFrame, test_size: float = 0.20):
    df = df.sort_values("date").reset_index(drop=True)
    cut = int(len(df) * (1 - test_size))

    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()

    return train, test


def train_models():
    df = load_training_data()
    train, test = temporal_split(df)

    X_train = train[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]

    metrics = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "date_train_min": str(train["date"].min()),
        "date_train_max": str(train["date"].max()),
        "date_test_min": str(test["date"].min()),
        "date_test_max": str(test["date"].max()),
    }

    # =========================
    # Modelo 1X2
    # =========================
    y_train_1x2 = train["target_1x2"]
    y_test_1x2 = test["target_1x2"]

    model_1x2 = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=42,
    )

    model_1x2.fit(X_train, y_train_1x2)

    p_test = model_1x2.predict_proba(X_test)
    pred_test = p_test.argmax(axis=1)

    metrics["accuracy_1x2"] = float(accuracy_score(y_test_1x2, pred_test))
    metrics["log_loss_1x2"] = float(log_loss(y_test_1x2, p_test, labels=[0, 1, 2]))

    # =========================
    # Goles esperados
    # =========================
    model_home_goals = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=42,
    )

    model_away_goals = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=42,
    )

    model_home_goals.fit(X_train, train["target_home_goals"])
    model_away_goals.fit(X_train, train["target_away_goals"])

    pred_hg = model_home_goals.predict(X_test)
    pred_ag = model_away_goals.predict(X_test)

    metrics["mae_home_goals"] = float(mean_absolute_error(test["target_home_goals"], pred_hg))
    metrics["mae_away_goals"] = float(mean_absolute_error(test["target_away_goals"], pred_ag))

    # =========================
    # Over 2.5
    # =========================
    model_over25 = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=42,
    )

    model_over25.fit(X_train, train["target_over25"])
    p_over = model_over25.predict_proba(X_test)[:, 1]

    metrics["auc_over25"] = float(roc_auc_score(test["target_over25"], p_over))

    # =========================
    # BTTS
    # =========================
    model_btts = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=42,
    )

    model_btts.fit(X_train, train["target_btts"])
    p_btts = model_btts.predict_proba(X_test)[:, 1]

    metrics["auc_btts"] = float(roc_auc_score(test["target_btts"], p_btts))

    # =========================
    # Guardar modelos
    # =========================
    joblib.dump(model_1x2, MODELS_DIR / "model_1x2.pkl")
    joblib.dump(model_home_goals, MODELS_DIR / "model_home_goals.pkl")
    joblib.dump(model_away_goals, MODELS_DIR / "model_away_goals.pkl")
    joblib.dump(model_over25, MODELS_DIR / "model_over25.pkl")
    joblib.dump(model_btts, MODELS_DIR / "model_btts.pkl")

    with open(MODELS_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    with open(MODELS_DIR / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train_models()