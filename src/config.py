# src/config.py
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MASTER_DIR = DATA_DIR / "master"
PROCESSED_DIR = DATA_DIR / "processed"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
DAILY_OUTPUTS_DIR = OUTPUTS_DIR / "daily"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = ROOT_DIR / "logs"

for path in [
    DATA_DIR,
    RAW_DIR,
    MASTER_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    DAILY_OUTPUTS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")
ODDS_SPORT_KEY = os.getenv("ODDS_SPORT_KEY", "soccer_fifa_world_cup")
ODDS_REGIONS = os.getenv("ODDS_REGIONS", "us,uk,eu")
ODDS_MARKETS = os.getenv("ODDS_MARKETS", "h2h,totals")
ODDS_FORMAT = os.getenv("ODDS_FORMAT", "decimal")

N_SIM = int(os.getenv("N_SIM", "200000"))
MAX_GOALS = int(os.getenv("MAX_GOALS", "10"))
SEED = int(os.getenv("SEED", "42"))

MODEL_VERSION = "wc2026_v0.1"