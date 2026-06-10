# src/retrain.py
from src.data_loader import build_matches_dataset
from src.elo import build_elo_features
from src.features import compute_team_form_features, add_basic_features
from src.ml_models import train_models
from src.config import PROCESSED_DIR


def main():
    print("[1/4] Construyendo matches.parquet...")
    df = build_matches_dataset()
    print(f"    matches: {len(df):,}")

    print("[2/4] Calculando Elo...")
    df_elo = build_elo_features(df)

    out_elo = PROCESSED_DIR / "matches_with_elo.parquet"
    df_elo.to_parquet(out_elo, index=False)

    print(f"    guardado: {out_elo}")

    print("[3/4] Calculando features...")
    df_feat = compute_team_form_features(df_elo)
    df_feat = add_basic_features(df_feat)

    out_feat = PROCESSED_DIR / "training_dataset.parquet"
    df_feat.to_parquet(out_feat, index=False)

    print(f"    guardado: {out_feat}")

    print("[4/4] Entrenando modelos ML...")
    train_models()

    print("[OK] Reentrenamiento completo.")


if __name__ == "__main__":
    main()