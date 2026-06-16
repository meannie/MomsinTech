# COS Subagents

A COS subagent is a Claude Code skill that shares the session log infrastructure — pull, append, push — but is invoked separately for a specific purpose.

The base `/cos` skill handles briefings, calendar, tasks, and general updates. Subagents handle specific transitions or handoffs that are frequent enough to deserve their own skill but don't need the full COS context every time.

---

## The pattern

A COS subagent follows this structure:

```markdown
---
name: your-subagent-name
description: What it does in one line
allowed-tools: Bash, Read, Write
---

# Subagent Name

[What it does and why]

## Session Log Infrastructure

[Copy the pull → append → push block from skill.md verbatim]

## Steps

1. Pull session log from GitHub
2. [Subagent-specific work]
3. Append 1-3 entries to session_log.yaml
4. Push to GitHub
5. [Output to user]
```

Any skill that reads or writes `session_log.yaml` via the git sync repo is a COS subagent. The session log is the shared state.

---

## `/new-topic` — session handoff subagent

The canonical COS subagent. Run it at the end of any working session to capture what was accomplished and what's still open before you compact the conversation.

**What it does:**
1. Reads the conversation context to synthesize what was built, fixed, or decided
2. Identifies open items — anything unresolved, at risk of regression, or needing follow-up
3. Writes 1 `update` entry + 0-3 `observation` entries to `session_log.yaml`
4. Pushes to GitHub so the entries survive across context windows
5. Signals that the conversation is ready to compact

**Why it matters.** Claude's context window is finite. Without `/new-topic`, anything you built or debugged in a long session exists only in that session's context — when the context compacts or a new conversation starts, that institutional memory is gone. The session log is the handoff mechanism. `/new-topic` writes to it systematically.

**Installation:**

```bash
mkdir -p ~/.claude/skills/new-topic
cp /path/to/cos_claudecode/skills/new-topic/skill.md ~/.claude/skills/new-topic/skill.md
```

**Usage:**

```
/new-topic
/new-topic shell-e imap fix
/new-topic momsintech events feature
```

With no argument, the skill infers the topic from conversation context. With an argument, it uses that as the label prefix in the session log entries.

**Example output:**

```
Logged to COS:

[update] Shell-E IMAP auth fixed — helpme_app_password re-added to VM creds.json,
cron running every 5 min. Stress test script updated to use SMTP app password
from calendar/config.yaml.

[observation] creds.json on VM may be overwritten by git pull — key disappeared
twice today. Check git ls-files creds.json on VM; remove from git tracking if tracked.

Session logged. You can now compact this conversation and start fresh.
```

---

## Building your own subagent

Any repeating operation that touches the session log is a candidate:

| What you're doing | Subagent idea |
|-------------------|---------------|
| Ending a focused work session | `/new-topic` (covered above) |
| Finishing a weekly review | `/week-close` — writes review summary to session log |
| Preparing for a specific meeting | `/meeting-prep` — reads session log for context, writes a prep note |
| Logging an async decision | `/log-decision` — quick entry with decision + reasoning |

**The one hard constraint:** a subagent MUST follow the pull → append → push sequence. Never write to `session_log.yaml` without first pulling the latest from GitHub — skipping the pull risks clobbering entries from another device or from phone inbox entries not yet visible locally.

---

## Subagent vs `/cos update`

Use `/cos update` for quick conversational updates during a session:

```
/cos update just finished the column draft
/cos update moved the project to testing phase
```

Use `/new-topic` when you want synthesized, structured entries with open-item tracking — at the end of a working session before compacting conversation history.

The entries go to the same place. The difference is how much synthesis happens before writing.
