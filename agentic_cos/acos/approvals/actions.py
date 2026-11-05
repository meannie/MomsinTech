"""The only things an approval can make happen. Nothing here spends money,
sends to other people, publishes, signs or deletes.

  note      {"type": "note", "text": "..."}   recorded for Claude's next live session
  task_add  {"type": "task_add", "title": "...", "due": "YYYY-MM-DD"?, "list": "..."?}
"""

from __future__ import annotations

import json
from datetime import datetime

from .. import config as cfgmod


def inbox_path():
    return cfgmod.LOCAL / "state" / "approved_inbox.jsonl"


def append_inbox(kind: str, text: str) -> None:
    p = inbox_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="minutes"), "kind": kind,
                            "text": text[:2000]}) + "\n")


def run(action: dict, tasks_service=None) -> str:
    t = action["type"]
    if t == "note":
        append_inbox("approved", action["text"])
        return "recorded for your next session"
    if t == "task_add":
        if tasks_service is None:
            from googleapiclient.discovery import build
            from .. import auth
            tasks_service = build("tasks", "v1", credentials=auth.google_creds("tasks", auth.TASKS_SCOPES),
                                  cache_discovery=False)
        lists = tasks_service.tasklists().list(maxResults=100).execute().get("items", [])
        want = action.get("list")
        tl = next((l for l in lists if want and l["title"] == want), lists[0] if lists else None)
        if not tl:
            raise RuntimeError("no Google Tasks list found")
        body = {"title": action["title"][:200]}
        if action.get("due"):
            body["due"] = f"{action['due']}T00:00:00.000Z"
        tasks_service.tasks().insert(tasklist=tl["id"], body=body).execute()
        return f"task added to {tl['title']}"
    raise RuntimeError(f"unknown action {t}")
