"""How asks, alerts and briefings reach the user, and how replies come back.

  email     ask goes to the user's own address; the user replies from that address
  instinct  ask goes to the user's Instinct assistant, which texts them; their text
            comes back as an email from the assistant's address
  telegram  ask goes to the user's private chat with their own bot; they reply to it

Recipients are fixed by config (the user, or their assistant). There is no way
to send to anyone else through this file.
"""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.utils import parseaddr

from .core import Reply

PROCESSED_LABEL = "COS/Processed"


def base_domain(domain: str) -> str:
    parts = domain.lower().strip(".").split(".")
    return ".".join(parts[-2:])


def email_sender_ok(headers: dict, labels: set[str], expected_from: str, mailbox: str) -> bool:
    """True only when the message really came from `expected_from`.

    Replies the user sends from the same mailbox the ask came from carry Gmail's
    SENT label (only the account owner can create those). Replies from another
    mailbox must pass SPF and DKIM for the sender's own domain.
    """
    _, addr = parseaddr(headers.get("from", ""))
    addr = addr.lower()
    if addr != expected_from.lower():
        return False
    if addr == mailbox.lower():
        return "SENT" in labels
    auth = headers.get("authentication-results", "").lower()
    dom = re.escape(base_domain(addr.partition("@")[2]))
    return "spf=pass" in auth and re.search(rf"dkim=pass[^;]*header\.(d|i)=@?([\w.-]*\.)?{dom}\b", auth) is not None


class GmailChannel:
    """Email and Instinct channels, sent and read through one connected Gmail inbox."""

    def __init__(self, cfg, service=None):
        self.cfg = cfg
        self.kind = cfg.channel["type"]
        send_from = cfg.channel.get("send_from")
        self.mailbox = cfg.inbox(send_from) if send_from else next(i for i in cfg.inboxes if i.provider == "gmail")
        if self.mailbox.provider != "gmail":
            raise SystemExit("email and Instinct channels need a connected Gmail inbox as channel.send_from")
        if service is None:
            from .. import auth
            service = auth.gmail_service(self.mailbox.id)
        self.svc = service
        from .. import config as cfgmod
        self.sent_log = cfgmod.LOCAL / "state" / "channel_sent_ids.json"
        self.to = cfg.channel["assistant_address"].lower() if self.kind == "instinct" else cfg.user_address
        if not self.to:
            raise SystemExit("channel recipient missing: set user.address or channel.assistant_address")

    def text_prefix(self) -> str | None:
        # An assistant that reads email may act on what's in it, so tell it plainly what to do.
        return f"Please text {self.cfg.user_name}:" if self.kind == "instinct" else None

    def send(self, subject: str, body: str, thread_key: str | None = None) -> str:
        msg = MIMEText(body, "plain", "utf-8")
        msg["To"] = self.to
        msg["From"] = self.mailbox.address
        msg["Subject"] = subject[:200]
        payload = {"raw": base64.urlsafe_b64encode(msg.as_bytes()).decode()}
        if thread_key:
            payload["threadId"] = thread_key
        sent = self.svc.users().messages().send(userId="me", body=payload).execute()
        ids = self._sent_ids() | {sent["id"]}
        self.sent_log.parent.mkdir(parents=True, exist_ok=True)
        self.sent_log.write_text(json.dumps(sorted(ids)[-500:]))
        return sent["threadId"]

    def _sent_ids(self) -> set[str]:
        return set(json.loads(self.sent_log.read_text())) if self.sent_log.exists() else set()

    @property
    def processed_id(self) -> str:
        if not hasattr(self, "_pid"):
            self._pid = self._label_id()
        return self._pid

    def _label_id(self) -> str:
        labels = {l["name"]: l["id"] for l in self.svc.users().labels().list(userId="me").execute()["labels"]}
        if PROCESSED_LABEL not in labels:
            labels[PROCESSED_LABEL] = self.svc.users().labels().create(
                userId="me", body={"name": PROCESSED_LABEL}).execute()["id"]
        return labels[PROCESSED_LABEL]

    def replies(self, thread_keys: list[str]) -> list[Reply]:
        out, ours = [], self._sent_ids()
        for tk in thread_keys:
            th = self.svc.users().threads().get(userId="me", id=tk, format="full").execute()
            for m in th.get("messages", []):
                labels = set(m.get("labelIds", []))
                if m["id"] in ours or self.processed_id in labels:
                    continue                      # our own ask/follow-ups, or already handled
                h = {x["name"].lower(): x["value"] for x in m["payload"]["headers"]}
                out.append(Reply(thread_key=tk, text=_plain_text(m),
                                 sender_ok=email_sender_ok(h, labels, self.to, self.mailbox.address),
                                 received=datetime.fromtimestamp(int(m["internalDate"]) / 1000, timezone.utc),
                                 ref=m["id"]))
        return out

    def mark_processed(self, reply: Reply) -> None:
        self.svc.users().messages().modify(userId="me", id=reply.ref, body={
            "addLabelIds": [self.processed_id], "removeLabelIds": ["UNREAD"]}).execute()


class TelegramChannel:
    API = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, cfg, http=None, state_path=None):
        from .. import config as cfgmod
        self.cfg = cfg
        self.chat_id = int(cfg.channel["chat_id"])
        token_file = cfgmod.secret_path("telegram_token")
        if http is None:
            import requests
            http = requests.Session()
        self.http = http
        self.token = token_file.read_text().strip() if token_file.exists() else ""
        if not self.token and not hasattr(http, "fake"):
            raise SystemExit(f"Telegram token missing: paste it into {token_file}")
        self.state = state_path or (cfgmod.LOCAL / "state" / "telegram_offset.json")

    def text_prefix(self) -> None:
        return None

    def _call(self, method: str, **params) -> dict:
        r = self.http.post(self.API.format(token=self.token, method=method), json=params, timeout=60)
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"telegram {method} failed: {data.get('description', '')[:120]}")
        return data["result"]

    def send(self, subject: str, body: str, thread_key: str | None = None) -> str:
        text = f"{subject}\n\n{body}"[:4000]
        params = {"chat_id": self.chat_id, "text": text}
        if thread_key:
            params["reply_parameters"] = {"message_id": int(thread_key)}
        return str(self._call("sendMessage", **params)["message_id"])

    def replies(self, thread_keys: list[str]) -> list[Reply]:
        offset = json.loads(self.state.read_text())["offset"] if self.state.exists() else 0
        updates = self._call("getUpdates", offset=offset, timeout=0, allowed_updates=["message"])
        out = []
        for u in updates:
            offset = max(offset, u["update_id"] + 1)
            m = u.get("message") or {}
            chat_ok = m.get("chat", {}).get("id") == self.chat_id and m.get("from", {}).get("id") == self.chat_id
            replied_to = str((m.get("reply_to_message") or {}).get("message_id", ""))
            out.append(Reply(thread_key=replied_to, text=m.get("text", ""), sender_ok=chat_ok,
                             received=datetime.fromtimestamp(m.get("date", 0), timezone.utc),
                             ref=str(m.get("message_id", ""))))
        self.state.parent.mkdir(parents=True, exist_ok=True)
        self.state.write_text(json.dumps({"offset": offset}))
        return out

    def mark_processed(self, reply: Reply) -> None:
        pass   # the update offset already moved past it


def channel_for(cfg):
    return TelegramChannel(cfg) if cfg.channel["type"] == "telegram" else GmailChannel(cfg)


def _plain_text(msg: dict) -> str:
    def walk(p):
        if p.get("mimeType") == "text/plain" and p.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(p["body"]["data"]).decode("utf-8", "ignore")
        for c in p.get("parts", []) or []:
            r = walk(c)
            if r:
                return r
        return ""
    return walk(msg["payload"]) or msg.get("snippet", "")
