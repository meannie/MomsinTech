"""Mail triage pass over every connected inbox.

    python -m acos.triage.run                  # dry run (shadow mode): report only
    python -m acos.triage.run --apply          # add labels and archive
    python -m acos.triage.run --inbox work --days 1

Reads headers only, never bodies. Writes local/reports/triage_<stamp>.md/.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from .. import config as cfgmod
from .rules import classify, keep_newest_insurance

ORDER = ["Needs you", "Bills", "FYI", "(own message)", "Receipts", "Newsletters", "Notifications"]


def provider_for(inbox, rules: dict):
    from .. import auth
    from .providers import GmailProvider, M365Provider
    if inbox.provider == "gmail":
        return GmailProvider(auth.gmail_service(inbox.id), rules["protected_labels"])
    return M365Provider(auth.m365_session(inbox.address), rules["protected_labels"])


def triage_inbox(provider, me: str, rules: dict, days: int, apply: bool) -> list[dict]:
    rows = []
    for m in provider.fetch(days):
        d = classify(m, me, rules)
        rows.append({"id": m.id, "from_name": m.from_name, "from_addr": m.from_addr,
                     "subject": m.subject[:120], "received": m.received.isoformat(),
                     "label": d.label, "archive": d.archive, "reason": d.reason,
                     "categories": m.extra.get("categories", [])})
    if rules.get("keep_newest_insurance", True):
        keep_newest_insurance(rows)
    if apply:
        provider.apply(rows, rules["label_prefix"])
    return rows


def report(results: dict, days: int, applied: bool, stamp: str) -> str:
    mode = "applied" if applied else "shadow mode, nothing changed"
    md = [f"# Mail triage {stamp} (last {days} days, {mode})"]
    for address, rows in results.items():
        verb = "archived" if applied else "would archive"
        md.append(f"\n## {address}: {len(rows)} in inbox, {sum(r['archive'] for r in rows)} {verb}")
        for lab in ORDER:
            sub = [r for r in rows if r["label"] == lab]
            if sub:
                md.append(f"\n### {lab} ({len(sub)})")
                md += [f"- [{'archive' if r['archive'] else 'keep'}] {r['from_name'] or r['from_addr']}: "
                       f"{r['subject']}  _({r['reason']})_" for r in sub]
    return "\n".join(md) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int)
    p.add_argument("--inbox", action="append", help="limit to these inbox ids (repeatable)")
    p.add_argument("--apply", action="store_true", help="add labels and archive; never trashes or deletes")
    a = p.parse_args()

    cfg = cfgmod.load()
    rules = cfg.triage
    days = a.days or rules["days"]
    inboxes = [i for i in cfg.inboxes if not a.inbox or i.id in a.inbox]
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out = cfgmod.LOCAL / "reports"
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for inbox in inboxes:
        rows = triage_inbox(provider_for(inbox, rules), inbox.address, rules, days, a.apply)
        results[inbox.address] = rows
        print(f"{inbox.id:12} {len(rows):4} in inbox, {sum(r['archive'] for r in rows):4} "
              f"{'archived' if a.apply else 'would archive'}")
    (out / f"triage_{stamp}.json").write_text(json.dumps({"applied": a.apply, "results": results}, indent=1))
    (out / f"triage_{stamp}.md").write_text(report(results, days, a.apply, stamp))
    print(f"report: {out / f'triage_{stamp}.md'}")


if __name__ == "__main__":
    main()
