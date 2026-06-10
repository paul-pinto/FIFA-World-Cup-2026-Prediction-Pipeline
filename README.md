# Worldcup Model

Este repositorio no debe incluir credenciales ni entornos virtuales.

## Configuración

1. Crear un archivo `.env` en la raíz con estos valores:
   - `THE_ODDS_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`

2. Ejemplo:
   ```
   THE_ODDS_API_KEY=your_api_key_here
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   ```

3. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```

4. Ejecutar scripts:
   ```
   python src/run_daily.py
   ```

## Seguridad

- No subir `.env` a GitHub.
- No subir `.venv` a GitHub.
- Si ya subiste credenciales, revoca la API y cambia los tokens.
