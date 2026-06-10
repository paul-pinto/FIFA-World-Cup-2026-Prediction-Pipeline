# src/telegram_bot.py
import time
import traceback
from datetime import datetime, timezone, timedelta

import requests

from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from src.telegram_notify import (
    format_predictions_message,
    send_telegram_message,
    send_status,
    latest_prediction_date,
)


POLL_INTERVAL = 2


def telegram_api(method: str, payload: dict | None = None) -> dict:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Falta TELEGRAM_BOT_TOKEN en .env")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"

    if payload is None:
        payload = {}

    r = requests.post(url, json=payload, timeout=30)

    if r.status_code != 200:
        raise RuntimeError(f"Telegram API error {r.status_code}: {r.text}")

    return r.json()


def send_message(chat_id: int | str, text: str):
    telegram_api(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
    )


def today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def tomorrow_utc() -> str:
    return (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()


def yesterday_utc() -> str:
    return (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()


def is_authorized(chat_id: int | str) -> bool:
    """
    Seguridad básica: solo responde al TELEGRAM_CHAT_ID configurado.
    """
    if not TELEGRAM_CHAT_ID:
        return True

    return str(chat_id) == str(TELEGRAM_CHAT_ID)


def help_text() -> str:
    return """
🤖 <b>World Cup Predictor Bot</b>

Comandos:

/hoy
Pronósticos de hoy UTC.

/manana
Pronósticos de mañana UTC.

/ayer
Pronósticos de ayer UTC.

/fecha YYYY-MM-DD
Pronósticos de una fecha específica.

/ultimo
Último archivo de predicciones disponible.

/status
Estado de outputs generados.

/help
Muestra esta ayuda.
""".strip()


def handle_command(chat_id: int | str, text: str):
    text = text.strip()

    if not is_authorized(chat_id):
        send_message(chat_id, "⛔ No autorizado.")
        return

    if text in ["/start", "/help"]:
        send_message(chat_id, help_text())
        return

    if text == "/status":
        # send_status usa TELEGRAM_CHAT_ID fijo.
        # Si quieres responder al chat actual, usamos workaround simple:
        send_status()
        return

    if text == "/hoy":
        date_str = today_utc()
        msg = format_predictions_message(date_str)
        send_message(chat_id, msg)
        return

    if text == "/manana":
        date_str = tomorrow_utc()
        msg = format_predictions_message(date_str)
        send_message(chat_id, msg)
        return

    if text == "/ayer":
        date_str = yesterday_utc()
        msg = format_predictions_message(date_str)
        send_message(chat_id, msg)
        return

    if text == "/ultimo":
        date_str = latest_prediction_date()
        msg = format_predictions_message(date_str)
        send_message(chat_id, msg)
        return

    if text.startswith("/fecha"):
        parts = text.split()

        if len(parts) != 2:
            send_message(chat_id, "Uso correcto: <code>/fecha YYYY-MM-DD</code>")
            return

        date_str = parts[1].strip()
        msg = format_predictions_message(date_str)
        send_message(chat_id, msg)
        return

    send_message(chat_id, "Comando no reconocido. Usa /help")


def run_bot():
    print("[bot] starting long polling")

    offset = None

    while True:
        try:
            payload = {
                "timeout": 30,
                "allowed_updates": ["message"],
            }

            if offset is not None:
                payload["offset"] = offset

            data = telegram_api("getUpdates", payload)

            for update in data.get("result", []):
                offset = update["update_id"] + 1

                message = update.get("message", {})
                chat = message.get("chat", {})
                chat_id = chat.get("id")
                text = message.get("text", "")

                if not chat_id or not text:
                    continue

                print(f"[bot] chat={chat_id} text={text}")

                try:
                    handle_command(chat_id, text)
                except FileNotFoundError as e:
                    send_message(
                        chat_id,
                        f"⚠️ No encontré predicciones para esa fecha.\n\n<code>{e}</code>",
                    )
                except Exception as e:
                    print(traceback.format_exc())
                    send_message(
                        chat_id,
                        f"❌ Error procesando comando:\n<code>{e}</code>",
                    )

        except KeyboardInterrupt:
            print("[bot] stopping")
            break
        except Exception:
            print(traceback.format_exc())
            time.sleep(5)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_bot()