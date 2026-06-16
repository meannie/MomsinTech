# /cos — Chief of Staff for Claude Code

A personal Chief of Staff agent built as a Claude Code skill. Surfaces what needs your attention across all your projects, tasks, and calendar — and remembers what you said you'd do.

Built and battle-tested by [Annie Tsai](https://annietsai.co) (COO, [Interact](https://tryinteract.com) and [Moms in Tech](https://momsintech.com)), open-sourced for the MinTs community.

## What it does

`/cos` is a slash command in [Claude Code](https://claude.com/claude-code) that runs in five modes:

| Mode | What it does |
|------|-------------|
| `/cos today` | "What should I focus on right now?" — today's calendar, open windows, suggested focus |
| `/cos week` | Weekly briefing — day-by-day calendar, deadlines, downstream flags, relationship check-ins |
| `/cos review` | Weekly review — what you said vs. what happened, project health dashboard |
| `/cos focus` | Right-now decision — given the next open block, what should you work on? |
| `/cos update <note>` | Tell it what you just did. Updates the session log so future runs have context. |

## Why it's different from a normal task list

Three things:

1. **It has memory.** A `session_log.yaml` file tracks your stated intents, completed work, observations, and unresolved nags across sessions. The agent reads it on every run, so it knows what you promised yesterday and whether you actually did it.

2. **It connects dots across systems.** Pulls from Google Tasks, all your Google Calendars, file modification times in your project directories, and the session log — then synthesizes a single view of "what's actually going on."

3. **It nags.** If you said you'd ship something three days ago and haven't, it tells you. With escalating intensity. Most personal productivity tools let things silently drop. This one doesn't.

## Architecture

```
~/.claude/skills/cos/
  └── skill.md                    # The Claude Code skill (your customized version)

~/.claude/skills/new-topic/       # Optional subagent skill (see SUBAGENTS.md)
  └── skill.md

<your_cos_dir>/                   # e.g. ~/Desktop/cos/
  ├── tasks_sync.py               # Pulls Google Tasks → tasks_snapshot.json
  ├── gcal_sync.py                # Pulls Google Calendar → gcal_snapshot.json
  ├── inbox_drain.py              # Drains phone inbox from GitHub Issue (see PHONE_INBOX.md)
  ├── config.yaml                 # Your calendars, timezone (gitignored)
  ├── session_log.yaml            # The agent's memory (gitignored)
  ├── oauth_credentials.json       # Google OAuth client (gitignored)
  ├── tasks_token.json             # Google Tasks token (gitignored, auto-generated)
  ├── gcal_token.json              # Google Calendar token (gitignored, auto-generated)
  ├── tasks_snapshot.json          # Latest tasks pull (gitignored, auto-generated)
  └── gcal_snapshot.json           # Latest calendar pull (gitignored, auto-generated)
```

The skill markdown is the most important file — it's the instructions Claude follows when you run `/cos`. The Python sync scripts just refresh the local snapshots that the skill reads.

## Quick start

See [SETUP.md](SETUP.md) for the full walkthrough. The short version:

1. **Install Claude Code** if you haven't already: https://claude.com/claude-code
2. **Clone this repo** and copy the templated files into your own setup
3. **Customize `skill.md.template`** with your name, priorities, relationships, etc.
4. **Set up Google OAuth** for Tasks + Calendar (instructions in SETUP.md)
5. **Connect Google Calendar MCP** in Claude.ai (or use the local sync script)
6. **Run `/cos today`** in Claude Code

## Files in this repo

### Core (read these first)

| File | What it is |
|------|-----------|
| [`skill.md.template`](skill.md.template) | The Claude Code skill, with `{{PLACEHOLDERS}}` for your personal context |
| [`SETUP.md`](SETUP.md) | Step-by-step setup walkthrough (start here) |
| [`CUSTOMIZING.md`](CUSTOMIZING.md) | How to make `/cos` actually feel like *your* chief of staff |
| [`ADVANCED.md`](ADVANCED.md) | Multi-skill chains, congestion detection, MCP dual-path pattern, time-based reminders |
| [`MOBILE.md`](MOBILE.md) | Run /cos directly from your phone via a GitHub-synced sync repo (full skill, same session log) |
| [`PHONE_INBOX.md`](PHONE_INBOX.md) | Log anything from your iPhone via a Shortcut → GitHub Issue → session log drain |
| [`RELIABILITY.md`](RELIABILITY.md) | Two non-obvious fixes that prevent /cos from giving you confidently wrong information |
| [`SUBAGENTS.md`](SUBAGENTS.md) | Build COS subagent skills that share the session log (includes `/new-topic`) |
| [`SLACK.md`](SLACK.md) | Optional: run /cos from Slack on your phone (Socket Mode bot — chat interface alternative) |

### Scripts

| File | What it does |
|------|-----------|
| [`scripts/gcal_sync.py`](scripts/gcal_sync.py) | Google Calendar → local JSON snapshot (read) |
| [`scripts/tasks_sync.py`](scripts/tasks_sync.py) | Google Tasks → local JSON snapshot (read) |
| [`scripts/tasks_add.py`](scripts/tasks_add.py) | Write tasks back to Google Tasks (separate write-scope OAuth token) |
| [`scripts/inbox_drain.py`](scripts/inbox_drain.py) | Phone inbox → session log drain (reads GitHub Issue #1 comments) |
| [`scripts/sync_session_log.py`](scripts/sync_session_log.py) | Sync session_log.yaml between local /cos and a private GitHub repo (mobile + multi-device continuity) |
| [`scripts/gdocs_push.py`](scripts/gdocs_push.py) | Markdown → Google Doc push (create new or update existing). Lets /cos write structured artifacts to Drive |
| [`scripts/gmail_helper.py`](scripts/gmail_helper.py) | Optional: multi-account Gmail OAuth helper (search/read/send across accounts via `--account <name>` flag) |
| [`scripts/cos_slack_bot.py`](scripts/cos_slack_bot.py) | Optional: /cos as a Slack bot for mobile access (see SLACK.md) |

### Skills

| File | What it does |
|------|-----------|
| [`skills/new-topic/skill.md`](skills/new-topic/skill.md) | `/new-topic` subagent — log session status to COS and compact conversation (see SUBAGENTS.md) |

### Config + setup

| File | What it is |
|------|-----------|
| [`config.example.yaml`](config.example.yaml) | Calendar list + timezone config template |
| [`session_log.example.yaml`](session_log.example.yaml) | Empty session log starter with format examples |
| [`requirements.txt`](requirements.txt) | Python dependencies |

## What it costs

- **Claude Code subscription** — required to run the skill
- **Google Cloud project** — free tier covers all API calls
- **Time** — about 30 minutes for initial setup, then 5-10 minutes/week to tune the skill as you use it

## New in this release

- **[RELIABILITY.md](RELIABILITY.md)** — Two production-discovered fixes: the authoritative `date` shell command (prevents wrong-day-of-week briefings) and the session log ordering constraint (prevents data loss when using phone inbox)
- **[PHONE_INBOX.md](PHONE_INBOX.md)** — Log tasks and notes from your iPhone via an iOS Shortcut + GitHub Issue drain. No server, no bot. Step-by-step Shortcut setup included.
- **[SUBAGENTS.md](SUBAGENTS.md)** — Pattern for building COS subagent skills. Includes `/new-topic`: run it at end of a session to synthesize what was done, write it to the session log, and signal the conversation is ready to compact.
- **Congestion detection** — Teach /cos to lead the briefing when your day architecture is broken (3+ meetings with <30 min gaps, zero morning deep work, etc.). Details in [ADVANCED.md](ADVANCED.md).
- **MCP dual-path pattern** — Try MCP tools first, fall back to local sync scripts, never mix paths in one run. Details in [ADVANCED.md](ADVANCED.md).
- **Time-based reminders** — Pattern for use-or-lose credits, expiring benefits, and any deadline that benefits from N-days-before surfacing. Details in [ADVANCED.md](ADVANCED.md).

## A note on customization

The default `skill.md.template` is a starting point. The real magic happens when you tune it to your specific patterns — your energy curve, your strategic priorities, the people you want to track, the recurring obligations the agent should remind you about.

See [CUSTOMIZING.md](CUSTOMIZING.md) for examples of how to extend it for things like:
- Tax/compliance deadline reminders specific to your business entity structure
- Travel benefits / credit card use-or-lose tracking
- Recurring writing or content cadences (newsletter, column, podcast)
- Industry-specific recurring obligations
- Family logistics integration

See [ADVANCED.md](ADVANCED.md) for the multi-skill chain pattern, congestion detection, the MCP dual-path pattern, and time-based reminders.

See [PHONE_INBOX.md](PHONE_INBOX.md) for logging from your iPhone — full step-by-step iOS Shortcut setup included. No server required.

See [MOBILE.md](MOBILE.md) for direct mobile access — full /cos on your phone via a private GitHub-synced sync repo. Updates from your phone show up on your laptop and vice versa; both devices read the same session log and snapshots.

See [SLACK.md](SLACK.md) for the optional Slack bot — same /cos, reachable from your phone via Slack DM if you prefer a chat-style interface.

See [SUBAGENTS.md](SUBAGENTS.md) for the `/new-topic` skill and the general pattern for building COS subagents — skills that share the session log infrastructure for specific handoffs and transitions.

The goal is for the agent to know enough about your life that its recommendations actually fit your reality — not generic productivity advice.

## License

MIT

## Built by

[Annie Tsai](https://annietsai.co) — COO at Interact, COO of Moms in Tech, columnist for the San Mateo Daily Journal. Built for her own use over 2026 Q1, then templatized for the MinTs community.
