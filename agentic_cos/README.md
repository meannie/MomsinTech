# Agentic Chief of Staff

An add-on to [`cos_claudecode`](../cos_claudecode) that runs on its own: it sorts your mail overnight, preps you for meetings, briefs you in the morning, and asks your permission by email, Instinct text or Telegram before it acts.

> **Status: work in progress.** The tested core is here (mail triage, the approval loop, meeting prep collection, the job runner). The interview, install guide and design notes come next. Don't install it yet unless you're comfortable reading code.

## What it never does

- Send a message to anyone but you. Replies to other people stop at a draft.
- Spend money, buy, sign or publish anything.
- Trash or delete mail, files or events. Mail access is label and archive only; calendar access is read-only.
- See or type your passwords. You sign in on Google's or Microsoft's own page.
- Follow instructions found inside an email, web page or document. Scheduled Claude runs get no tools at all; they only read the material they're handed.

## What's here

| Path | What it does |
| --- | --- |
| `acos/triage/` | Labels every new message in Gmail and Microsoft 365 inboxes (Needs you, Bills, FYI, Receipts, Newsletters, Notifications) and archives the low-value ones. Shadow mode reports only. |
| `acos/approvals/` | Sends a yes/no question with a one-time code; acts only on a verified reply from you, in the same thread, within 24 hours, and only from a short allowlist (record a note, add a task). |
| `acos/meetings.py` | Picks meetings that need prep (outside attendees, or keywords you choose) and gathers recent threads and notes for a prep packet. |
| `acos/deliver.py` | Sends briefings to your inbox and short alerts through your channel. |
| `runner/run_job.sh` | Runs one job: one at a time, logged, off / shadow / live modes, trial-run and once-per-period guards. |
| `runner/schedulers/` | launchd, cron and systemd templates. |
| `templates/config.example.yaml` | The settings the interview fills in (all example values). |
| `tools/scrub_check.py` | Maintainer check that nothing identifying gets published. |

Everything personal (config, sign-in tokens, logs, reports) lives in `local/`, which git ignores.

## Running the tests

No accounts needed:

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
```

## License

MIT, see [LICENSE](LICENSE).
