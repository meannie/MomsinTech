"""Approval loop core: codes, reply checks, the action allowlist. No network.

An approval only counts when ALL of these hold:
  - the channel verified the sender (see channels.py)
  - the reply is in the same thread as the ask (email thread, or a Telegram reply)
  - it says YES or NO followed by the ask's code
  - the code is pending, unused and not expired
Anything else in a reply is never treated as an instruction.
"""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

CODE_CHARS = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"   # no 0/O, 1/I/L
ACTION_TYPES = {"note", "task_add"}               # the most any config may allow
QUOTE_SPLIT = re.compile(r"\n(On .+wrote:|>|-{3,}|_{3,}|From: |Sent from )")


@dataclass
class Reply:
    thread_key: str        # email thread id, or the Telegram message id that was replied to
    text: str
    sender_ok: bool
    received: datetime
    ref: str = ""          # provider message id, for marking it processed


class Store:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def save(self, asks: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asks, indent=1))


def validate_action(action: dict, allowed: list[str]) -> None:
    t = action.get("type")
    if t not in ACTION_TYPES or t not in allowed:
        raise ValueError(f"action type {t!r} is not allowed through approvals")
    if t == "task_add" and not str(action.get("title", "")).strip():
        raise ValueError("task_add needs a title")
    if t == "note" and not str(action.get("text", "")).strip():
        raise ValueError("note needs text")


def new_code(existing, length: int = 3) -> str:
    while True:
        code = "".join(secrets.choice(CODE_CHARS) for _ in range(length))
        if code not in existing:
            return code


def clean_question(q: str, limit: int = 300) -> str:
    """One short plain line. Asks never carry quoted email text."""
    q = re.sub(r"\s+", " ", q.replace(">", "")).strip()
    return q[:limit]


def ask_body(question: str, code: str, hours: int, text_prefix: str | None) -> str:
    lead = f"{text_prefix}\n\n" if text_prefix else ""
    return (f"{lead}{clean_question(question)}\n\n"
            f"Reply YES {code} to approve or NO {code} to skip. (Expires in {hours} hours.)\n")


def new_reply_text(text: str) -> str:
    """Only the new part of a reply, above any quoted original."""
    return QUOTE_SPLIT.split(text or "", maxsplit=1)[0].strip()


def evaluate(reply: Reply, asks: dict, now: datetime | None = None) -> tuple[str | None, str, bool]:
    """Return (code, outcome, run_action). Mutates the matched ask's status."""
    now = now or datetime.now(timezone.utc)
    if not reply.sender_ok:
        return None, "ignored: sender checks failed", False
    code = next((k for k, v in asks.items() if v.get("thread_key") == reply.thread_key), None)
    if not code:
        return None, "no ask in this thread", False
    ask = asks[code]
    hit = re.search(r"\b(YES|NO)\s+([A-Z2-9]{3,8})\b", new_reply_text(reply.text).upper())
    if not hit or hit.group(2) != code:
        return code, "ignored: no YES/NO with this ask's code", False
    if ask["status"] != "pending":
        return code, f"already {ask['status']}", False
    ask["resolved"] = now.isoformat()
    if now > datetime.fromisoformat(ask["expires"]):
        ask["status"] = "expired"
        return code, "expired", False
    if hit.group(1) == "NO":
        ask["status"] = "declined"
        return code, "declined", False
    ask["status"] = "approved"
    return code, "approved", True


def record(asks: dict, code: str, summary: str, question: str, action: dict,
           thread_key: str, hours: int, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    asks[code] = {"summary": summary, "question": clean_question(question), "action": action,
                  "thread_key": str(thread_key), "created": now.isoformat(),
                  "expires": (now + timedelta(hours=hours)).isoformat(), "status": "pending"}
    return asks[code]
