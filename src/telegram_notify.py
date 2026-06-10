# src/telegram_notify.py
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.config import DAILY_OUTPUTS_DIR, REPORTS_DIR, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


MAX_MESSAGE_LEN = 3900


def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Falta TELEGRAM_BOT_TOKEN en .env")

    if not TELEGRAM_CHAT_ID:
        raise RuntimeError("Falta TELEGRAM_CHAT_ID en .env")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    chunks = split_message(text)

    for chunk in chunks:
        r = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=30,
        )

        if r.status_code != 200:
            raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")


def split_message(text: str) -> list[str]:
    if len(text) <= MAX_MESSAGE_LEN:
        return [text]

    chunks = []
    current = ""

    for line in text.splitlines():
        if len(current) + len(line) + 1 > MAX_MESSAGE_LEN:
            chunks.append(current)
            current = line
        else:
            current += "\n" + line if current else line

    if current:
        chunks.append(current)

    return chunks


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def load_predictions(date_str: str) -> pd.DataFrame:
    path = DAILY_OUTPUTS_DIR / f"predictions_{date_str}.csv"

    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")

    return pd.read_csv(path)


def format_predictions_message(date_str: str) -> str:
    df = load_predictions(date_str)

    lines = []
    lines.append(f"🏆 <b>World Cup 2026 · Pronósticos</b>")
    lines.append(f"📅 <b>{date_str}</b>")
    lines.append("")

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        lines.append(f"⚽ <b>{home} vs {away}</b>")
        lines.append(f"🏟 {row.get('venue', '')} · {row.get('city', '')}")

        lines.append("")
        lines.append("1X2:")
        lines.append(f"• {home}: <b>{pct(row['final_p_home'])}</b>")
        lines.append(f"• Empate: <b>{pct(row['final_p_draw'])}</b>")
        lines.append(f"• {away}: <b>{pct(row['final_p_away'])}</b>")
        lines.append(f"🎯 Pick: <b>{row['pick']}</b> ({pct(row['confidence'])})")

        lines.append("")
        lines.append("Top scores:")
        for i in range(1, 6):
            s_col = f"top_score_{i}"
            p_col = f"top_score_{i}_prob"
            if s_col in row and p_col in row and pd.notna(row[s_col]):
                lines.append(f"{i}. {row[s_col]} · <b>{pct(row[p_col])}</b>")

        lines.append("")
        lines.append("Mercados:")
        lines.append(f"• Over 2.5: <b>{pct(row['dc_p_over25'])}</b>")
        lines.append(f"• Under 2.5: <b>{pct(row['dc_p_under25'])}</b>")
        lines.append(f"• BTTS Sí: <b>{pct(row['dc_p_btts_yes'])}</b>")

        best_value = row.get("best_value_market", None)
        if pd.notna(best_value):
            lines.append("")
            lines.append("💰 Value:")
            lines.append(f"• Mercado: <b>{best_value}</b>")
            lines.append(f"• EV: <b>{row['best_value_ev']:+.3f}</b>")
            lines.append(f"• Edge: <b>{row['best_value_edge']:+.3f}</b>")
        else:
            best_edge = row.get("best_edge_market", None)
            if pd.notna(best_edge):
                lines.append("")
                lines.append("💰 Value:")
                lines.append("• Sin value claro")
                lines.append(f"• Mejor edge: <b>{best_edge}</b>")
                lines.append(f"• EV: <b>{row['best_edge_ev']:+.3f}</b>")

        lines.append("")
        lines.append("—")
        lines.append("")

    return "\n".join(lines)


def latest_prediction_date() -> str:
    files = sorted(DAILY_OUTPUTS_DIR.glob("predictions_*.csv"))

    if not files:
        raise FileNotFoundError("No hay archivos predictions_*.csv")

    latest = files[-1].stem.replace("predictions_", "")
    return latest


def send_predictions(date_str: str | None = None):
    if date_str is None:
        date_str = latest_prediction_date()

    msg = format_predictions_message(date_str)
    send_telegram_message(msg)

    print(f"[OK] Enviado Telegram para {date_str}")


def send_status():
    pred_files = sorted(DAILY_OUTPUTS_DIR.glob("predictions_*.csv"))
    eval_files = sorted(DAILY_OUTPUTS_DIR.glob("evaluation_*.csv"))
    report_files = sorted(REPORTS_DIR.glob("daily_report_*.md"))

    lines = []
    lines.append("🛠 <b>World Cup Predictor Status</b>")
    lines.append("")
    lines.append(f"Predictions: <b>{len(pred_files)}</b>")
    lines.append(f"Evaluations: <b>{len(eval_files)}</b>")
    lines.append(f"Reports: <b>{len(report_files)}</b>")

    if pred_files:
        lines.append("")
        lines.append(f"Última predicción:")
        lines.append(f"<code>{pred_files[-1].name}</code>")

    if eval_files:
        lines.append("")
        lines.append(f"Última evaluación:")
        lines.append(f"<code>{eval_files[-1].name}</code>")

    send_telegram_message("\n".join(lines))
    print("[OK] Status enviado")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["send", "status"],
    )
    parser.add_argument(
        "--date",
        required=False,
        help="YYYY-MM-DD",
    )

    args = parser.parse_args()

    if args.command == "send":
        send_predictions(args.date)
    elif args.command == "status":
        send_status()


if __name__ == "__main__":
    main()