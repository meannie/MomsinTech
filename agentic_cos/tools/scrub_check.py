"""Pre-publish privacy check for maintainers. Blocks anything that could identify
whoever built their setup from this template.

    python tools/scrub_check.py                 # scans agentic_cos/
    python tools/scrub_check.py --path ..       # scans another folder

Two kinds of checks:
  1. A private block list of real names, organizations, places, domains and
     addresses, one per line, kept OUTSIDE the repo (the list itself would
     identify you): ~/.config/agentic_cos/blocklist.txt, or $ACOS_BLOCKLIST.
  2. Patterns: email addresses not on example domains, phone numbers, calendar
     IDs, home-folder paths, API-key-looking strings.

Exit code 1 on any finding. Install as a git pre-push hook:
    printf '#!/bin/sh\\npython3 agentic_cos/tools/scrub_check.py\\n' > .git/hooks/pre-push && chmod +x .git/hooks/pre-push
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "local", "__pycache__", ".pytest_cache", "node_modules"}
SKIP_FILES = {"scrub_check.py"}
EXAMPLE_DOMAIN = re.compile(r"@([\w.-]+\.)?(example|example\.(com|org|net))$|@example\.[a-z]+$", re.I)
ALLOWED_EMAILS = {"noreply@anthropic.com"}
PATTERNS = {
    "email address": re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+"),
    "phone number": re.compile(r"(?<!\d)(\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"),
    "calendar id": re.compile(r"[\w.]+@group\.calendar\.google\.com"),
    "home folder path": re.compile(r"/(Users|home)/(?!<|\$|you\b|USER\b|name\b)[\w.-]+"),
    "key-like string": re.compile(r"\b(sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,}|ghp_[A-Za-z0-9]{30,}|xox[bp]-[\w-]{20,})\b"),
}


def blocklist() -> list[str]:
    p = Path(os.environ.get("ACOS_BLOCKLIST", Path.home() / ".config" / "agentic_cos" / "blocklist.txt"))
    if not p.exists():
        print(f"warning: no block list at {p}; only pattern checks ran", file=sys.stderr)
        return []
    return [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]


def scan(root: Path) -> list[str]:
    words = [(w, re.compile(rf"(?<![\w]){re.escape(w)}(?![\w])", re.I)) for w in blocklist()]
    findings = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name in SKIP_FILES or SKIP_DIRS & set(p.relative_to(root).parts):
            continue
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        rel = p.relative_to(root)
        for n, line in enumerate(text.splitlines(), 1):
            for w, rx in words:
                if rx.search(line):
                    findings.append(f"{rel}:{n}: block-listed term")   # never print the term itself
            for kind, rx in PATTERNS.items():
                for m in rx.finditer(line):
                    hit = m.group(0)
                    if kind == "email address" and (EXAMPLE_DOMAIN.search(hit) or hit.lower() in ALLOWED_EMAILS):
                        continue
                    findings.append(f"{rel}:{n}: {kind}")
        if words and any(rx.search(p.name) for _, rx in words):
            findings.append(f"{rel}: block-listed term in file name")
    return findings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(Path(__file__).resolve().parent.parent))
    a = ap.parse_args()
    found = scan(Path(a.path).resolve())
    for f in found:
        print(f)
    print(f"{len(found)} finding(s)" if found else "clean")
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
