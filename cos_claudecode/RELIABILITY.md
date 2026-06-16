# Reliability Patterns

Small things that prevent /cos from confidently giving you wrong information. Both came from production incidents.

---

## 1. Authoritative date and time

The most important reliability fix in `skill.md`. Put this as the very first instruction under your Data Sources section — before calendar pulls, before task syncs, before anything:

```markdown
### 0. Authoritative current date + time (ALWAYS run first)

Before anything else, run this at the start of every `/cos` invocation:

```bash
date "+%Y-%m-%d %H:%M %A %Z"
```

Use the OUTPUT of this command — not your model-side time intuition, not a system-reminder date, not an inferred date — as the authoritative current date, day-of-week, and local time for the entire briefing.
```

**Why it matters.** Claude's internal sense of time drifts. System-reminder timestamps are imprecise. Without this, briefings have shipped with the wrong day of the week — which breaks every day-of-week rule you've written into your skill (Monday brief, Thursday newsletter, weekend behavior). The shell `date` command is the only honest source.

**What it affects:**
- "Today" calendar window (correct start/end of today in your timezone)
- "Rest of day, chronological" — only events whose start ≥ actual current time
- Day-of-week logic (Monday briefs, Thursday nudges, weekend behavior)
- All cadence checks and "due in N days" math

**Demo / testing override.** If you want to test what /cos looks like on a different day without actually changing the date, add this override mechanism to your skill:

```markdown
If a `> Demo date override:` block appears anywhere in the prompt (e.g., "treat the current date as Friday May 8"), that override wins over `date` for the duration of the run. Otherwise, ALWAYS use `date` output.
```

Usage when testing:

```
/cos today
> Demo date override: treat the current date as Monday June 1, 2026, 9:00 AM PDT
```

---

## 2. Session log ordering constraint

If you're using the GitHub-sync session log and the phone inbox drain (see [PHONE_INBOX.md](PHONE_INBOX.md)), the order of operations on every `/cos` run matters.

**Correct order:**

```
1. git pull  →  2. cp (repo → local)  →  3. inbox_drain.py  →  4. read session_log  →  ...  →  5. cp (local → repo)  →  6. git push
```

**Wrong order (causes data loss):**

```
1. inbox_drain.py  →  2. git pull  →  3. cp (repo → local)  [overwrites drained entries]
```

If `cp` runs after the drain, the GitHub version (which doesn't yet know about the drained entries) overwrites the local file, and the drained entries are gone. The inbox state (`last_seen_id`) has already advanced, so those entries will never be re-drained — they're permanently lost.

**In your skill.md, the Session Log section should say:**

```markdown
**ORDER MATTERS — DO NOT REORDER THESE STEPS.**

Step 1 — Pull the latest from GitHub:
```bash
cd ~/YOUR_SYNC_REPO && git pull origin main --rebase 2>/dev/null
cp ~/YOUR_SYNC_REPO/session_log.yaml ~/cos/session_log.yaml
```

Step 2 — Drain phone inbox (AFTER pull + cp, never before):
```bash
python3 ~/cos/inbox_drain.py 2>&1
```

Step 3 — Read session_log.yaml (now contains both GitHub history and any new phone entries)
```
```
