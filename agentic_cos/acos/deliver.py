"""Deliver a briefing or an alert to the user.

    python -m acos.deliver digest --subject "Morning briefing" --body-file out.txt
    python -m acos.deliver urgent --subject "Bill past due" --body-file alert.txt

digest  the full text. Email channel and Instinct: emailed to the user's own
        address. Telegram: sent to the bot chat (long text is split).
urgent  a short alert through the configured channel. For Instinct it opens with
        "Please text <name>:" and is cut to 300 characters, so the assistant gets
        an instruction to text you and nothing else to act on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config as cfgmod
from .approvals.channels import GmailChannel, TelegramChannel


def deliver(cfg, kind: str, subject: str, body: str, gmail_service=None, http=None) -> None:
    ctype = cfg.channel["type"]
    if ctype == "telegram":
        ch = TelegramChannel(cfg, http=http)
        text = body if kind == "digest" else body[:600]
        for i in range(0, max(len(text), 1), 3800):
            ch.send(subject if i == 0 else f"{subject} (cont.)", text[i:i + 3800])
        return
    if kind == "digest" or ctype == "email":
        # Briefings always go to the user's own inbox, never to an assistant.
        email_cfg = cfgmod.from_dict({**cfg.raw, "channel": {**cfg.channel, "type": "email"}})
        GmailChannel(email_cfg, gmail_service).send(subject, body)
        return
    ch = GmailChannel(cfg, gmail_service)
    ch.send(subject, f"{ch.text_prefix()} {' '.join(body.split())[:300]}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["digest", "urgent"])
    p.add_argument("--subject", required=True)
    p.add_argument("--body-file", required=True)
    a = p.parse_args()
    body = Path(a.body_file).read_text()
    if not body.strip():
        raise SystemExit("empty body, nothing sent")
    deliver(cfgmod.load(), a.kind, a.subject, body)


if __name__ == "__main__":
    main()
