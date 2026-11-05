"""Loads local/config.yaml (written by the interview) and fills in defaults.

Every personal value lives in that file. Nothing in this package hard-codes a
name, address or ID, so the same code works for anyone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LOCAL = Path(os.environ.get("ACOS_LOCAL", ROOT / "local"))

DEFAULT_TRIAGE = {
    "days": 2,
    "label_prefix": "COS",
    "keep_visible_domains": [],       # newsletters you still want to see
    "always_needs_you_domains": [],   # senders that always need you
    "vendor_domains": [],             # automated senders that look like people
    "protected_labels": [],           # mail carrying these labels is never archived
    "keep_newest_insurance": True,    # keep the latest insurance/benefits notice visible
    "archive": {"receipts": True, "newsletters": True, "notifications": True},
}


@dataclass
class Inbox:
    id: str
    provider: str          # "gmail" or "m365"
    address: str


@dataclass
class Config:
    user_name: str
    user_address: str
    timezone: str
    inboxes: list[Inbox]
    triage: dict
    channel: dict
    approvals: dict
    raw: dict = field(default_factory=dict)

    def inbox(self, inbox_id: str) -> Inbox:
        for i in self.inboxes:
            if i.id == inbox_id:
                return i
        raise KeyError(f"no inbox '{inbox_id}' in config")


def _merge(defaults: dict, given: dict | None) -> dict:
    out = dict(defaults)
    for k, v in (given or {}).items():
        out[k] = _merge(defaults[k], v) if isinstance(defaults.get(k), dict) and isinstance(v, dict) else v
    return out


def load(path: Path | None = None) -> Config:
    path = path or LOCAL / "config.yaml"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run the interview first (/cos-onboard, part 2).")
    raw = yaml.safe_load(path.read_text()) or {}
    return from_dict(raw)


def from_dict(raw: dict) -> Config:
    user = raw.get("user", {})
    inboxes = [Inbox(id=i["id"], provider=i["provider"], address=i["address"].lower())
               for i in raw.get("inboxes", [])]
    for i in inboxes:
        if i.provider not in ("gmail", "m365"):
            raise ValueError(f"inbox {i.id}: provider must be gmail or m365, not {i.provider!r}")
    channel = raw.get("channel", {"type": "email"})
    if channel.get("type") not in ("email", "instinct", "telegram"):
        raise ValueError("channel.type must be email, instinct or telegram")
    approvals = _merge({"allowed_actions": ["note", "task_add"], "expiry_hours": 24, "code_length": 3},
                       raw.get("approvals"))
    return Config(
        user_name=user.get("name", "you"),
        user_address=user.get("address", "").lower(),
        timezone=raw.get("timezone", "UTC"),
        inboxes=inboxes,
        triage=_merge(DEFAULT_TRIAGE, raw.get("triage")),
        channel=channel,
        approvals=approvals,
        raw=raw,
    )


def secret_path(name: str) -> Path:
    """Where a token for `name` lives. Created owner-only; never printed or committed."""
    d = LOCAL / "secrets"
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    return d / name


def write_secret(path: Path, text: str) -> None:
    old = os.umask(0o077)
    try:
        path.write_text(text)
    finally:
        os.umask(old)
    os.chmod(path, 0o600)


def get(dotted: str, default: str = "") -> str:
    """Read one value from config, for shell scripts: python -m acos.config get modules.triage"""
    try:
        node = yaml.safe_load((LOCAL / "config.yaml").read_text()) or {}
    except FileNotFoundError:
        return default
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    if isinstance(node, bool):        # YAML reads a bare off/on as false/true
        return "off" if node is False else "on"
    return str(node)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "get":
        print(get(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""))
    else:
        raise SystemExit("usage: python -m acos.config get <dotted.key> [default]")
