"""Meeting prep: pick the meetings that need a packet and gather the material.

    python -m acos.meetings --window evening   # tomorrow's meetings (run in the evening)
    python -m acos.meetings --window soon      # meetings starting in about an hour (refresh)

Writes local/state/meeting_prep/<window>_<stamp>.json and prints its path, or
prints nothing when no meeting needs prep. The job runner hands that file to
Claude, which writes the packets. This file only reads: calendars (read-only),
mail search results (subjects and the short preview Gmail returns), and local
notes. It never contacts attendees.

Config (local/config.yaml):
  meetings:
    calendars: [{inbox: personal, id: primary}, {inbox: work, id: default}]
    important_keywords: ["board", "interview"]   # always prep these, even internal
    skip_keywords: ["focus", "lunch", "hold"]    # never prep these
    notes_dirs: ["~/notes"]                      # searched for attendee names and domains
    lookback_days: 180
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config as cfgmod

DEFAULTS = {"calendars": [], "important_keywords": [], "skip_keywords": ["focus", "hold", "block", "lunch"],
            "notes_dirs": [], "lookback_days": 180}


def own_domains(cfg) -> set[str]:
    return {i.address.partition("@")[2] for i in cfg.inboxes} | {cfg.user_address.partition("@")[2]}


def needs_prep(ev: dict, mine: set[str], rules: dict) -> tuple[bool, str]:
    """ev: {title, all_day, declined, recurring, attendees: [{email, resource}]}"""
    title = (ev.get("title") or "").lower()
    if ev.get("all_day") or ev.get("declined"):
        return False, "all-day or declined"
    if any(k.lower() in title for k in rules.get("skip_keywords", [])):
        return False, "skip keyword"
    if any(k.lower() in title for k in rules.get("important_keywords", [])):
        return True, "important keyword"
    outside = [a for a in ev.get("attendees", []) if not a.get("resource")
               and a.get("email", "").partition("@")[2].lower() not in mine]
    if outside:
        return True, f"{len(outside)} outside attendee(s)"
    return False, "internal" + (" recurring" if ev.get("recurring") else "")


def google_events(svc, cal_id: str, start: datetime, end: datetime, me: str) -> list[dict]:
    r = svc.events().list(calendarId=cal_id, timeMin=start.isoformat(), timeMax=end.isoformat(),
                          singleEvents=True, orderBy="startTime", maxResults=100).execute()
    out = []
    for e in r.get("items", []):
        atts = e.get("attendees", [])
        out.append({
            "id": e["id"], "title": e.get("summary", ""), "start": e["start"].get("dateTime") or e["start"].get("date"),
            "all_day": "date" in e["start"], "recurring": "recurringEventId" in e,
            "declined": any(a.get("self") and a.get("responseStatus") == "declined" for a in atts),
            "attendees": [{"email": a.get("email", "").lower(), "name": a.get("displayName", ""),
                           "resource": a.get("resource", False)} for a in atts if not a.get("self")],
            "description": (e.get("description") or "")[:3000], "location": e.get("location", ""),
            "links": re.findall(r"https://docs\.google\.com/\S+", e.get("description") or ""),
        })
    return out


def m365_events(session, start: datetime, end: datetime) -> list[dict]:
    url = (f"https://graph.microsoft.com/v1.0/me/calendarView?startDateTime={start.isoformat()}"
           f"&endDateTime={end.isoformat()}&$top=100&$select=id,subject,start,isAllDay,type,attendees,"
           f"bodyPreview,location,responseStatus")
    out = []
    while url:
        page = session.get(url, timeout=60).json()
        for e in page.get("value", []):
            out.append({
                "id": e["id"], "title": e.get("subject", ""), "start": e["start"]["dateTime"],
                "all_day": e.get("isAllDay", False), "recurring": e.get("type") == "occurrence",
                "declined": (e.get("responseStatus") or {}).get("response") == "declined",
                "attendees": [{"email": a["emailAddress"]["address"].lower(), "name": a["emailAddress"].get("name", ""),
                               "resource": a.get("type") == "resource"} for a in e.get("attendees", [])],
                "description": (e.get("bodyPreview") or "")[:3000],
                "location": (e.get("location") or {}).get("displayName", ""), "links": [],
            })
        url = page.get("@odata.nextLink")
    return out


def mail_history(gmail, who: str, days: int) -> list[dict]:
    """Recent threads with one person: subject, date and Gmail's short preview only."""
    r = gmail.users().threads().list(userId="me", q=f"(from:{who} OR to:{who}) newer_than:{days}d",
                                     maxResults=5).execute()
    out = []
    for t in r.get("threads", []):
        th = gmail.users().threads().get(userId="me", id=t["id"], format="metadata",
                                         metadataHeaders=["Subject", "Date"]).execute()
        last = th["messages"][-1]
        h = {x["name"].lower(): x["value"] for x in last["payload"]["headers"]}
        out.append({"subject": h.get("subject", ""), "date": h.get("date", ""), "preview": last.get("snippet", "")[:300]})
    return out


def notes_mentions(dirs: list[str], needles: list[str], limit: int = 8) -> list[dict]:
    hits = []
    pats = [re.compile(re.escape(n), re.I) for n in needles if n and len(n) > 3]
    for d in dirs:
        for p in Path(os.path.expanduser(d)).rglob("*"):
            if p.suffix.lower() not in (".md", ".txt") or not p.is_file():
                continue
            try:
                text = p.read_text(errors="ignore")
            except OSError:
                continue
            lines = [l.strip() for l in text.splitlines() if any(x.search(l) for x in pats)]
            if lines:
                hits.append({"file": str(p), "lines": lines[:5]})
            if len(hits) >= limit:
                return hits
    return hits


def window_bounds(window: str, now: datetime) -> tuple[datetime, datetime]:
    if window == "evening":
        local = now.astimezone()
        day = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return day, day + timedelta(days=1)
    return now + timedelta(minutes=45), now + timedelta(minutes=75)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--window", choices=["evening", "soon"], required=True)
    a = p.parse_args()
    cfg = cfgmod.load()
    rules = {**DEFAULTS, **(cfg.raw.get("meetings") or {})}
    mine = own_domains(cfg)
    now = datetime.now(timezone.utc)
    start, end = window_bounds(a.window, now)
    state = cfgmod.LOCAL / "state" / "meeting_prep"
    state.mkdir(parents=True, exist_ok=True)
    done_file = state / "done.json"
    done = set(json.loads(done_file.read_text())) if done_file.exists() else set()

    from . import auth
    gmail_by_inbox = {}
    packets = []
    for c in rules["calendars"]:
        inbox = cfg.inbox(c["inbox"])
        if inbox.provider == "gmail":
            events = google_events(auth.calendar_service(inbox.id), c.get("id", "primary"), start, end, inbox.address)
            gmail = gmail_by_inbox.setdefault(inbox.id, auth.gmail_service(inbox.id))
        else:
            events = m365_events(auth.m365_session(inbox.address), start, end)
            gmail = None
        for ev in events:
            key = f"{a.window}:{ev['id']}"
            ok, why = needs_prep(ev, mine, rules)
            if not ok or key in done:
                continue
            outside = [x for x in ev["attendees"] if x["email"].partition("@")[2] not in mine and not x["resource"]]
            ev["why"] = why
            ev["history"] = {x["email"]: mail_history(gmail, x["email"], rules["lookback_days"])
                             for x in outside[:6]} if gmail else {}
            needles = [x["name"] for x in outside] + [x["email"] for x in outside] + \
                      sorted({x["email"].partition("@")[2] for x in outside})
            ev["notes"] = notes_mentions(rules["notes_dirs"], needles)
            packets.append(ev)
            done.add(key)

    if not packets:
        return
    out = state / f"{a.window}_{now:%Y%m%d_%H%M}.json"
    out.write_text(json.dumps({"window": a.window, "generated": now.isoformat(), "meetings": packets}, indent=1))
    done_file.write_text(json.dumps(sorted(done)[-1000:]))
    print(out)


if __name__ == "__main__":
    main()
