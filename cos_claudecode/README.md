# /cos — Chief of Staff for Claude Code

A personal Chief of Staff agent built as a Claude Code skill. Surfaces what needs your attention across all your projects, tasks, and calendar — and remembers what you said you'd do.

Built and battle-tested by [Annie Tsai](https://annietsai.co), open-sourced for the [Moms in Tech](https://momsintech.com) community.

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

<your_cos_dir>/                   # e.g. ~/cos/
  ├── tasks_sync.py               # Pulls Google Tasks → tasks_snapshot.json
  ├── gcal_sync.py                # Pulls Google Calendar → gcal_snapshot.json
  ├── config.yaml                 # Your calendars, timezone (gitignored)
  ├── session_log.yaml            # The agent's memory (gitignored)
  ├── oauth_credentials.json      # Google OAuth client (gitignored)
  ├── tasks_token.json            # Google Tasks token (gitignored, auto-generated)
  ├── gcal_token.json             # Google Calendar token (gitignored, auto-generated)
  ├── tasks_snapshot.json         # Latest tasks pull (gitignored, auto-generated)
  └── gcal_snapshot.json          # Latest calendar pull (gitignored, auto-generated)
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

| File | What it is |
|------|-----------|
| [`skill.md.template`](skill.md.template) | The Claude Code skill, with `{{PLACEHOLDERS}}` for your personal context |
| [`scripts/tasks_sync.py`](scripts/tasks_sync.py) | Google Tasks → local JSON snapshot |
| [`scripts/gcal_sync.py`](scripts/gcal_sync.py) | Google Calendar → local JSON snapshot |
| [`config.example.yaml`](config.example.yaml) | Calendar list + timezone config template |
| [`session_log.example.yaml`](session_log.example.yaml) | Empty session log starter with format examples |
| [`requirements.txt`](requirements.txt) | Python dependencies |
| [`SETUP.md`](SETUP.md) | Step-by-step setup walkthrough |
| [`CUSTOMIZING.md`](CUSTOMIZING.md) | How to make `/cos` actually feel like *your* chief of staff |

## What it costs

- **Claude Code subscription** — required to run the skill
- **Google Cloud project** — free tier covers all API calls
- **Time** — about 30 minutes for initial setup, then 5-10 minutes/week to tune the skill as you use it

## A note on customization

The default `skill.md.template` is a starting point. The real magic happens when you tune it to your specific patterns — your energy curve, your strategic priorities, the people you want to track, the recurring obligations the agent should remind you about. 

See [CUSTOMIZING.md](CUSTOMIZING.md) for examples of how to extend it for things like:
- Tax/compliance deadline reminders specific to your business entity structure
- Travel benefits / credit card use-or-lose tracking
- Recurring writing or content cadences (newsletter, column, podcast)
- Industry-specific recurring obligations
- Family logistics integration

The goal is for the agent to know enough about your life that its recommendations actually fit your reality — not generic productivity advice.

## License

MIT

## Built by

[Annie Tsai](https://annietsai.co) — COO at Interact, founder of Moms in Tech, columnist for the San Mateo Daily Journal. Built for her own use over 2026 Q1, then templatized for the MinTs community.
