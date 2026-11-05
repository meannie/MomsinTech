"""Find your Telegram chat ID after you've made a bot and messaged it once.

    python -m acos.telegram_setup

Reads the token you pasted into local/secrets/telegram_token (it is never
printed), looks for the most recent private message to the bot, and prints the
chat ID to put in config as channel.chat_id. Then it sends a test message.
"""

from __future__ import annotations

from . import config as cfgmod


def find_chat(updates: list[dict]) -> tuple[int, str] | None:
    for u in reversed(updates):
        chat = (u.get("message") or {}).get("chat") or {}
        if chat.get("type") == "private":
            return chat["id"], chat.get("first_name", "")
    return None


def main() -> None:
    import requests
    f = cfgmod.secret_path("telegram_token")
    if not f.exists() or not f.read_text().strip():
        raise SystemExit(f"Paste your bot token into {f} first (INSTALL step 5).")
    token = f.read_text().strip()
    api = f"https://api.telegram.org/bot{token}"
    r = requests.get(f"{api}/getUpdates", timeout=30).json()
    if not r.get("ok"):
        raise SystemExit("Telegram didn't accept the token. Check you pasted all of it.")
    hit = find_chat(r["result"])
    if not hit:
        raise SystemExit("No message found. Open your bot in Telegram, send it 'hi', then run this again.")
    chat_id, name = hit
    requests.post(f"{api}/sendMessage", json={"chat_id": chat_id, "text": "Connected. Alerts and approval questions will arrive here."}, timeout=30)
    print(f"Found your chat ({name}). Put this in local/config.yaml under channel:\n  chat_id: {chat_id}")


if __name__ == "__main__":
    main()
