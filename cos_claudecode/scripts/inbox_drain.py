#!/usr/bin/env python3
"""
inbox_drain.py — Pull phone-webhook entries from a GitHub Issue inbox
into session_log.yaml.

Runs at the start of every /cos session. Idempotent: tracks the last
seen comment id in .inbox_state.json and only fetches newer ones.

Comment body format (set by your iOS Shortcut — see PHONE_INBOX.md):
    ```yaml
    ts: "2026-04-29T08:42"
    type: observation
    note: "your note here"
    ```

    # Optional: grocery list entries
    ```yaml
    ts: "2026-04-29T08:42"
    type: grocery
    note: "almond milk"
    ```

Configuration (add to config.yaml):
    phone_inbox:
      repo: "yourname/yourCOSrepo"   # private GitHub repo where your inbox Issue lives
      issue_number: 1                 # issue number (default: 1)

GitHub PAT: stored at <COS_DIR>/.github_pat (gitignored — never commit this file)
COS_DIR: set via COS_DIR environment variable, or defaults to ~/cos
"""

import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# COS_DIR: override with COS_DIR env var, or default to ~/cos
COS_DIR = Path(os.environ.get("COS_DIR", os.path.expanduser("~/cos")))
SESSION_LOG = COS_DIR / "session_log.yaml"
GROCERY_LIST = COS_DIR / "grocery_list.yaml"
STATE_FILE = COS_DIR / ".inbox_state.json"
PAT_FILE = COS_DIR / ".github_pat"
CONFIG_FILE = COS_DIR / "config.yaml"


def load_config() -> dict:
    if not HAS_YAML or not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f) or {}


config = load_config()
phone_inbox_cfg = config.get("phone_inbox", {})

GITHUB_REPO = phone_inbox_cfg.get("repo", "")
INBOX_ISSUE = phone_inbox_cfg.get("issue_number", 1)

if not GITHUB_REPO:
    print(
        "phone_inbox.repo not configured in config.yaml — skipping inbox drain.\n"
        "See PHONE_INBOX.md to set this up.",
        file=sys.stderr,
    )
    print(0)
    sys.exit(0)

GITHUB_API = (
    f"https://api.github.com/repos/{GITHUB_REPO}/issues/{INBOX_ISSUE}/comments"
)


def load_pat() -> str | None:
    if PAT_FILE.exists():
        return PAT_FILE.read_text().strip()
    return None


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_seen_id": 0}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def fetch_comments(pat: str, since_id: int) -> list[dict]:
    """Fetch all issue comments, paginated; return only those with id > since_id."""
    all_comments = []
    url = f"{GITHUB_API}?per_page=100&sort=created&direction=asc"
    while url:
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {pat}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "cos-inbox-drain/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                page = json.loads(resp.read().decode("utf-8"))
                all_comments.extend(page)
                link = resp.headers.get("Link", "")
                next_url = None
                for part in link.split(","):
                    if 'rel="next"' in part:
                        m = re.search(r"<([^>]+)>", part)
                        if m:
                            next_url = m.group(1)
                url = next_url
        except urllib.error.HTTPError as e:
            print(f"GitHub API error: HTTP {e.code}", file=sys.stderr)
            return []
        except Exception as e:
            print(f"Fetch failed: {e}", file=sys.stderr)
            return []

    return [c for c in all_comments if c["id"] > since_id]


YAML_BLOCK_RE = re.compile(r"```yaml\s*\n(.*?)\n```", re.DOTALL)


def parse_comment(comment: dict) -> dict | None:
    """Extract entry from comment body's yaml block. Returns None if malformed."""
    body = comment.get("body", "")
    m = YAML_BLOCK_RE.search(body)
    if not m:
        return None
    block = m.group(1)
    entry: dict = {}
    for line in block.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        entry[k.strip()] = v.strip().strip('"')
    if "ts" not in entry or "note" not in entry:
        return None
    entry.setdefault("type", "nag")
    return entry


def append_to_session_log(entries: list[dict]):
    with open(SESSION_LOG, "a") as f:
        for e in entries:
            f.write(
                f'\n- ts: "{e["ts"]}"\n'
                f'  type: {e["type"]}\n'
                f'  note: "{e["note"]}"\n'
            )


def append_to_grocery_list(entries: list[dict]):
    if not GROCERY_LIST.exists():
        GROCERY_LIST.write_text("items:\n")
    text = GROCERY_LIST.read_text()
    if "items:" not in text:
        text += "\nitems:\n"
    if "items: []" in text:
        text = text.replace("items: []", "items:")
    block = ""
    for e in entries:
        block += (
            f'  - ts: "{e["ts"]}"\n'
            f'    item: "{e["note"]}"\n'
            f'    source: "phone"\n'
        )
    GROCERY_LIST.write_text(text.rstrip() + "\n" + block)


def main():
    pat = load_pat()
    if not pat:
        print(f"No PAT at {PAT_FILE} — skipping inbox drain", file=sys.stderr)
        print(0)
        return 0

    state = load_state()
    since_id = state.get("last_seen_id", 0)

    new_comments = fetch_comments(pat, since_id)
    if not new_comments:
        print("inbox: 0 new entries", file=sys.stderr)
        print(0)
        return 0

    log_entries = []
    grocery_entries = []
    max_id = since_id
    for c in new_comments:
        e = parse_comment(c)
        if e:
            if e["type"] == "grocery":
                grocery_entries.append(e)
            else:
                log_entries.append(e)
        if c["id"] > max_id:
            max_id = c["id"]

    if log_entries:
        append_to_session_log(log_entries)
    if grocery_entries:
        append_to_grocery_list(grocery_entries)

    save_state({"last_seen_id": max_id})

    skipped = len(new_comments) - len(log_entries) - len(grocery_entries)
    print(
        f"inbox: drained {len(log_entries)} log entries, "
        f"{len(grocery_entries)} grocery items "
        f"({skipped} unparseable skipped)",
        file=sys.stderr,
    )
    # stdout = total count, for /cos to read
    total = len(log_entries) + len(grocery_entries)
    print(total)
    return total


if __name__ == "__main__":
    sys.exit(0 if main() >= 0 else 1)
