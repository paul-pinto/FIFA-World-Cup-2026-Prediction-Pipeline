# src/telegram_notify.py

import argparse

import pandas as pd
import requests

from src.config import DAILY_OUTPUTS_DIR, REPORTS_DIR, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


MAX_MESSAGE_LEN = 3900


def get_telegram_chat_ids() -> list[str]:
    """
    Permite uno o varios chat_id en .env.

    Ejemplo:
    TELEGRAM_CHAT_ID=123456789,-1009876543210
    """

    if not TELEGRAM_CHAT_ID:
        raise RuntimeError("Falta TELEGRAM_CHAT_ID en .env")

    chat_ids = [
        chat_id.strip()
        for chat_id in str(TELEGRAM_CHAT_ID).split(",")
        if chat_id.strip()
    ]

    if not chat_ids:
        raise RuntimeError("TELEGRAM_CHAT_ID está vacío o inválido en .env")

    return chat_ids


def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Falta TELEGRAM_BOT_TOKEN en .env")

    chat_ids = get_telegram_chat_ids()

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    chunks = split_message(text)

    for chat_id in chat_ids:
        for chunk in chunks:
            r = requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": chunk,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=30,
            )

            if r.status_code != 200:
                raise RuntimeError(
                    f"Telegram error {r.status_code} para chat_id={chat_id}: {r.text}"
                )

        print(f"[OK] Telegram enviado a chat_id={chat_id}")


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


def is_present(value) -> bool:
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    return str(value).strip() != ""


def pct(x) -> str:
    if not is_present(x):
        return "N/A"

    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "N/A"


def fmt_signed(x) -> str:
    if not is_present(x):
        return "N/A"

    try:
        return f"{float(x):+.3f}"
    except Exception:
        return "N/A"


def fmt_decimal(x, digits: int = 2) -> str:
    if not is_present(x):
        return "N/A"

    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def load_predictions(date_str: str) -> pd.DataFrame:
    path = DAILY_OUTPUTS_DIR / f"predictions_{date_str}.csv"

    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")

    return pd.read_csv(path)


def _is_true(value) -> bool:
    if isinstance(value, bool):
        return value

    if not is_present(value):
        return False

    return str(value).strip().lower() in ["true", "1", "yes", "y", "si", "sí"]


def format_predictions_message(date_str: str) -> str:
    df = load_predictions(date_str)

    lines = []
    lines.append("🏆 <b>World Cup 2026 · Pronósticos</b>")
    lines.append(f"📅 <b>{date_str}</b>")
    lines.append("")

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        lines.append(f"⚽ <b>{home} vs {away}</b>")

        venue = row.get("venue", "")
        city = row.get("city", "")

        if is_present(venue) or is_present(city):
            lines.append(f"🏟 {venue} · {city}")

        lines.append("")
        lines.append("1X2 90 minutos:")
        lines.append(f"• {home}: <b>{pct(row.get('final_p_home'))}</b>")
        lines.append(f"• Empate: <b>{pct(row.get('final_p_draw'))}</b>")
        lines.append(f"• {away}: <b>{pct(row.get('final_p_away'))}</b>")
        lines.append(f"🎯 Pick 90': <b>{row.get('pick')}</b> ({pct(row.get('confidence'))})")

        # Knockout / clasificación
        is_knockout = _is_true(row.get("knockout", False))
        advance_team = row.get("advance_team", None)
        advance_confidence = row.get("advance_confidence", None)
        p_home_advance = row.get("p_home_advance", None)
        p_away_advance = row.get("p_away_advance", None)

        if is_knockout and is_present(advance_team) and is_present(advance_confidence):
            lines.append("")
            lines.append("🏆 Clasificación")
            lines.append(
                f"• Pick clasifica: <b>{advance_team}</b> "
                f"({pct(advance_confidence)})"
            )

            if is_present(p_home_advance) and is_present(p_away_advance):
                lines.append(
                    f"• {home}: <b>{pct(p_home_advance)}</b> | "
                    f"{away}: <b>{pct(p_away_advance)}</b>"
                )

        # Top scores
        lines.append("")
        lines.append("🎲 Top scores:")

        shown_scores = 0

        for i in range(1, 6):
            s_col = f"top_score_{i}"
            p_col = f"top_score_{i}_prob"

            if s_col in row.index and p_col in row.index and is_present(row.get(s_col)):
                lines.append(f"{i}. {row.get(s_col)} · <b>{pct(row.get(p_col))}</b>")
                shown_scores += 1

        if shown_scores == 0:
            lines.append("• N/A")

        # Mercados
        lines.append("")
        lines.append("📊 Mercados:")
        lines.append(f"• Over 2.5: <b>{pct(row.get('dc_p_over25'))}</b>")
        lines.append(f"• Under 2.5: <b>{pct(row.get('dc_p_under25'))}</b>")

        # Ambos marcan
        btts_yes = row.get("final_btts_yes", row.get("dc_p_btts_yes"))
        btts_no = row.get("final_btts_no", row.get("dc_p_btts_no"))
        btts_pick = row.get("btts_pick", None)
        btts_confidence = row.get("btts_confidence", None)

        lines.append("")
        lines.append("🤝 Ambos marcan:")
        lines.append(f"• Sí: <b>{pct(btts_yes)}</b>")
        lines.append(f"• No: <b>{pct(btts_no)}</b>")

        if is_present(btts_pick) and is_present(btts_confidence):
            lines.append(f"🎯 Pick BTTS: <b>{btts_pick}</b> ({pct(btts_confidence)})")

        # Value
        best_value = row.get("best_value_market", None)
        best_edge = row.get("best_edge_market", None)

        lines.append("")
        lines.append("💰 Value:")

        if is_present(best_value):
            lines.append(f"• Mejor value: <b>{best_value}</b>")
            lines.append(f"• Cuota: <b>{fmt_decimal(row.get('best_value_odds'), 2)}</b>")
            lines.append(f"• Probabilidad: <b>{pct(row.get('best_value_probability'))}</b>")
            lines.append(f"• Edge: <b>{fmt_signed(row.get('best_value_edge'))}</b>")
            lines.append(f"• EV: <b>{fmt_signed(row.get('best_value_ev'))}</b>")
        elif is_present(best_edge):
            lines.append("• Sin value claro")
            lines.append(f"• Mejor edge: <b>{best_edge}</b>")
            lines.append(f"• Cuota: <b>{fmt_decimal(row.get('best_edge_odds'), 2)}</b>")
            lines.append(f"• Probabilidad: <b>{pct(row.get('best_edge_probability'))}</b>")
            lines.append(f"• Edge: <b>{fmt_signed(row.get('best_edge'))}</b>")
            lines.append(f"• EV: <b>{fmt_signed(row.get('best_edge_ev'))}</b>")
        else:
            lines.append("• Sin datos de value")

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

    try:
        msg = format_predictions_message(date_str)
    except FileNotFoundError as e:
        print(f"[WARN] No hay predicciones para {date_str}. Se omite Telegram. {e}")
        return

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
        lines.append("Última predicción:")
        lines.append(f"<code>{pred_files[-1].name}</code>")

    if eval_files:
        lines.append("")
        lines.append("Última evaluación:")
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