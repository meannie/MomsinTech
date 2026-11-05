"""Approvals from the command line (and from scheduled jobs).

    python -m acos.approvals.cli send --summary "Add task" --question "Add 'Buy gift' due Fri?" --action-file a.json
    python -m acos.approvals.cli poll        # scheduled every 5 minutes
    python -m acos.approvals.cli list
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .. import config as cfgmod
from . import actions, core
from .channels import channel_for


def store() -> core.Store:
    return core.Store(cfgmod.LOCAL / "state" / "asks.json")


def send(cfg, ch, summary: str, question: str, action: dict) -> str:
    core.validate_action(action, cfg.approvals["allowed_actions"])
    st = store()
    asks = st.load()
    code = core.new_code(asks, cfg.approvals["code_length"])
    hours = cfg.approvals["expiry_hours"]
    thread = ch.send(f"Ask {code}: {summary}", core.ask_body(question, code, hours, ch.text_prefix()))
    core.record(asks, code, summary, question, action, thread, hours)
    st.save(asks)
    return code


def poll(cfg, ch, tasks_service=None, now: datetime | None = None) -> list[str]:
    st = store()
    asks = st.load()
    pending = [v["thread_key"] for v in asks.values() if v["status"] == "pending"]
    log = []
    for reply in ch.replies(pending):
        if cfg.channel["type"] == "telegram" and not reply.thread_key:
            # A private chat with the user's own bot is one thread; on a phone people
            # type "YES ABC" rather than swiping to reply, so match by the code.
            hit = core.re.search(r"\b(?:YES|NO)\s+([A-Z2-9]{3,8})\b", reply.text.upper())
            if hit and hit.group(1) in asks:
                reply.thread_key = asks[hit.group(1)]["thread_key"]
        code, outcome, go = core.evaluate(reply, asks, now)
        if go:
            ask = asks[code]
            try:
                result = actions.run(ask["action"], tasks_service)
                ask["status"] = "done"
                outcome = f"done: {result}"
                ch.send(f"Done {code}: {ask['summary']}",
                        _wrap(ch, f"Done: {ask['summary']} ({result})."), ask["thread_key"])
            except Exception as e:  # noqa: BLE001 - report, never retry silently
                ask["status"] = "failed"
                outcome = f"failed: {str(e)[:120]}"
                ch.send(f"Failed {code}: {ask['summary']}",
                        _wrap(ch, f"Couldn't do it: {ask['summary']}. It will be looked at next session."),
                        ask["thread_key"])
        elif code is None and reply.sender_ok and reply.text.strip():
            # A message from the user that isn't an answer: keep it for the next live session, as data.
            actions.append_inbox("message", reply.text)
            outcome = "saved for next session"
        ch.mark_processed(reply)
        log.append(f"{datetime.now():%Y-%m-%d %H:%M} {code or '-'}: {outcome}")
    # Expire anything past its time so it can't be approved later.
    now = now or datetime.now(timezone.utc)
    for v in asks.values():
        if v["status"] == "pending" and now > datetime.fromisoformat(v["expires"]):
            v["status"] = "expired"
    st.save(asks)
    return log


def _wrap(ch, text: str) -> str:
    prefix = ch.text_prefix()
    return f"{prefix}\n\n{text}\n" if prefix else text + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send")
    s.add_argument("--summary", required=True)
    s.add_argument("--question", required=True)
    s.add_argument("--action-file", required=True)
    sub.add_parser("poll")
    sub.add_parser("list")
    a = p.parse_args()
    cfg = cfgmod.load()
    if a.cmd == "list":
        for k, v in sorted(store().load().items(), key=lambda kv: kv[1]["created"]):
            print(f"{k}  {v['status']:9} {v['created'][:16]}  {v['summary']}")
        return
    ch = channel_for(cfg)
    if a.cmd == "send":
        print("sent ask", send(cfg, ch, a.summary, a.question, json.loads(Path(a.action_file).read_text())))
    else:
        for line in poll(cfg, ch):
            print(line)


if __name__ == "__main__":
    main()
