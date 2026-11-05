"""One-time sign-in per inbox. The user runs this and clicks Allow; the agent
never sees a password.

    python -m acos.auth connect personal      # opens Google's or Microsoft's sign-in page
    python -m acos.auth connect tasks         # optional: lets approvals add Google Tasks
    python -m acos.auth check                 # which inboxes are connected

Google needs local/secrets/google_client.json (downloaded in INSTALL step 3).
Microsoft needs local/secrets/m365_client.json: {"client_id": "...", "tenant": "common"}.
Tokens are saved owner-only under local/secrets/ and refreshed automatically.
If you sign in as a different address than the config expects, the token is
discarded.
"""

from __future__ import annotations

import argparse
import json

from . import config as cfgmod

# Mail: label and archive. Calendar: read only (briefing and meeting prep).
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.modify",
                "https://www.googleapis.com/auth/calendar.readonly"]
TASKS_SCOPES = ["https://www.googleapis.com/auth/tasks"]
M365_SCOPES = ["Mail.ReadWrite", "Calendars.Read"]


def _google_client():
    f = cfgmod.secret_path("google_client.json")
    if not f.exists():
        raise SystemExit(f"Missing {f}. Finish INSTALL step 3 (Google sign-in setup) first.")
    return f


def google_creds(name: str, scopes: list[str]):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    f = cfgmod.secret_path(f"google_{name}.json")
    if not f.exists():
        raise SystemExit(f"'{name}' isn't connected. Run: python -m acos.auth connect {name}")
    creds = Credentials.from_authorized_user_file(str(f), scopes)
    if not creds.valid:
        creds.refresh(Request())
        cfgmod.write_secret(f, creds.to_json())
    return creds


def gmail_service(inbox_id: str):
    from googleapiclient.discovery import build
    return build("gmail", "v1", credentials=google_creds(inbox_id, GMAIL_SCOPES), cache_discovery=False)


def _msal_app():
    import msal
    f = cfgmod.secret_path("m365_client.json")
    if not f.exists():
        raise SystemExit(f"Missing {f}. Finish INSTALL step 3 (Microsoft app registration) first.")
    client = json.loads(f.read_text())
    return msal.PublicClientApplication(client["client_id"],
                                        authority=f"https://login.microsoftonline.com/{client.get('tenant', 'common')}",
                                        token_cache=_msal_cache())


def _msal_cache():
    import msal
    cache = msal.SerializableTokenCache()
    f = cfgmod.secret_path("m365_cache.json")
    if f.exists():
        cache.deserialize(f.read_text())
    return cache


def _save_msal(app) -> None:
    if app.token_cache.has_state_changed:
        cfgmod.write_secret(cfgmod.secret_path("m365_cache.json"), app.token_cache.serialize())


def calendar_service(inbox_id: str):
    from googleapiclient.discovery import build
    return build("calendar", "v3", credentials=google_creds(inbox_id, GMAIL_SCOPES), cache_discovery=False)


def m365_session(address: str):
    import requests
    app = _msal_app()
    acct = next((a for a in app.get_accounts() if a["username"].lower() == address.lower()), None)
    result = app.acquire_token_silent(M365_SCOPES, account=acct) if acct else None
    if not result or "access_token" not in result:
        raise SystemExit(f"{address} isn't connected or the sign-in expired. Run: python -m acos.auth connect <inbox>")
    _save_msal(app)
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {result['access_token']}"
    return s


def connect(cfg, name: str) -> None:
    if name == "tasks":
        from google_auth_oauthlib.flow import InstalledAppFlow
        creds = InstalledAppFlow.from_client_secrets_file(str(_google_client()), TASKS_SCOPES) \
            .run_local_server(port=0, login_hint=cfg.user_address, prompt="consent select_account")
        cfgmod.write_secret(cfgmod.secret_path("google_tasks.json"), creds.to_json())
        print("Connected Google Tasks (add tasks only).")
        return
    inbox = cfg.inbox(name)
    if inbox.provider == "gmail":
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        creds = InstalledAppFlow.from_client_secrets_file(str(_google_client()), GMAIL_SCOPES) \
            .run_local_server(port=0, login_hint=inbox.address, prompt="consent select_account")
        got = build("gmail", "v1", credentials=creds, cache_discovery=False) \
            .users().getProfile(userId="me").execute()["emailAddress"].lower()
        if got != inbox.address:
            raise SystemExit(f"Signed in as {got}, expected {inbox.address}. Nothing saved; run again and pick {inbox.address}.")
        cfgmod.write_secret(cfgmod.secret_path(f"google_{name}.json"), creds.to_json())
    else:
        app = _msal_app()
        result = app.acquire_token_interactive(M365_SCOPES, login_hint=inbox.address, prompt="select_account")
        got = (result.get("id_token_claims") or {}).get("preferred_username", "").lower()
        if "access_token" not in result or got != inbox.address:
            raise SystemExit(f"Sign-in didn't match {inbox.address} ({result.get('error_description', got)}). Nothing saved.")
        _save_msal(app)
    print(f"Connected {inbox.address} as '{name}' (mail: label and archive only; calendar: read only).")


def check(cfg) -> None:
    for i in cfg.inboxes:
        try:
            if i.provider == "gmail":
                got = gmail_service(i.id).users().getProfile(userId="me").execute()["emailAddress"].lower()
                print(f"{i.id:12} {i.address:36} {'ok' if got == i.address else 'MISMATCH ' + got}")
            else:
                m365_session(i.address)
                print(f"{i.id:12} {i.address:36} ok")
        except SystemExit as e:
            print(f"{i.id:12} {i.address:36} not connected: {e}")
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"{i.id:12} {i.address:36} FAILED: {str(e)[:80]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("connect")
    c.add_argument("name", help="an inbox id from config, or 'tasks'")
    sub.add_parser("check")
    a = p.parse_args()
    cfg = cfgmod.load()
    connect(cfg, a.name) if a.cmd == "connect" else check(cfg)
