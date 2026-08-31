---
name: new-topic
description: Log what we just worked on to COS session log, then compact conversation history for a fresh start
argument-hint: "[optional: brief label for the topic, e.g. 'shell-e imap fix']"
allowed-tools: Bash, Read, Write
---

# New Topic — COS Status Log + Conversation Compact

You are a subagent of the COS (Chief of Staff) system. Your job is to close out the current conversation topic by:
1. Synthesizing what was accomplished and what's still open
2. Writing it to the COS session log
3. Pushing to GitHub so it survives across sessions
4. Telling the user the conversation is ready to compact

## COS Session Log Infrastructure

The session log lives at `{{COS_DIR}}/session_log.yaml` and is synced to a private GitHub repo.

**Always follow this order:**

**Step 1 — Pull latest:**
```bash
cd {{SYNC_REPO}} && git pull origin main --rebase 2>/dev/null
cp {{SYNC_REPO}}/session_log.yaml {{COS_DIR}}/session_log.yaml
```

**Step 2 — Append entries** (see format below)

**Step 3 — Push back:**
```bash
cp {{COS_DIR}}/session_log.yaml {{SYNC_REPO}}/session_log.yaml
cd {{SYNC_REPO}} && git add session_log.yaml && git commit -m "cos sync $(date '+%Y-%m-%d %H:%M')" --allow-empty 2>/dev/null && git push origin main 2>/dev/null
```

## Session Log Entry Format

```yaml
- ts: "2026-06-16T14:30"
  type: update | observation | nag
  category: YOUR_CATEGORY
  note: "[ProjectLabel] What was done / what's open."
```

## What to Synthesize

Look at the full conversation to extract:

**For the `update` entry:**
- What was built, fixed, or configured
- Key decisions made
- Files changed or scripts created (with paths)
- Current state: working / partially working / blocked

**For `observation` entries (one per open item):**
- Anything unresolved, needs follow-up, or at risk of regressing
- Format: what the issue is + what needs to happen next

**Nag entries:** Only if something was explicitly called out as urgent and still unresolved.

## Arguments

`$ARGUMENTS` may contain a short label (e.g. `shell-e imap fix`). If empty, infer the topic from conversation context.

## Steps

1. Pull session log from GitHub
2. Read the current timestamp: `date "+%Y-%m-%dT%H:%M"`
3. Synthesize the conversation into 1 `update` entry + 0-3 `observation` entries for open items
4. Append to `{{COS_DIR}}/session_log.yaml`
5. Push to GitHub
6. **Frequency check + scaffold offer** (see below)
7. Print a 3-5 line summary of what was logged
8. End with: **"Session logged. You can now compact this conversation and start fresh."**

## Frequency check + scaffold offer

Don't just note that a pattern is recurring — offer to build the skill on the spot, right now, while it's top of mind. A line in a log that says "consider a dedicated skill" gets read once and never acted on.

1. Scan the last 30 days of session_log for the current topic label (from `$ARGUMENTS` or inferred from conversation).
2. If this same topic or task type has appeared **3+ times**, don't just log it — ask:

   > "This has come up N times now — [brief description of the pattern]. Want me to scaffold a `/skill-name` skill for it right now?"

   Use the actual `AskUserQuestion` tool if available in this context (in-conversation, not a detached background run) so the user can say yes/no/not-now in the same turn. If asking isn't possible in the current context, fall back to the passive log line instead of guessing at consent.
3. **If yes:** create `.claude/skills/<skill-name>/skill.md` using the stub template below, filled in with what you actually know from the 3+ recurring instances (what triggers it, what steps were repeated, what files/commands were involved each time). This is a stub, not a finished skill — the goal is to remove the blank-page problem, not to guess at logic you don't have evidence for.
4. **If no or not answered:** fall back to the old passive line in the printed summary: `"This has come up N times — consider a dedicated skill for it."` Don't ask again for the same topic within the same 30-day window — repeated asks about the same pattern are more annoying than the log line they're replacing.

### Stub template

```markdown
---
name: <skill-name>
description: <one-line — what it does and when it fires>
argument-hint: "[optional args]"
allowed-tools: Bash, Read, Write
---

# <Skill Title>

<!-- Scaffolded by /new-topic on {{DATE}} — this task/topic came up {{N}} times in the
     last 30 days of the session log. This is a stub: the steps below are inferred
     from what actually happened those N times, not guessed. Fill in and correct
     before relying on it. -->

## What this should do

<one paragraph: the recurring task this replaces>

## Steps

1. <first step that repeated across occurrences>
2. <second step>
3. <...>

## Source sessions

session_log entries that prompted this scaffold:
- <ts> — <note>
- <ts> — <note>
- <ts> — <note>
```

## Example Output

```
Logged to COS:

[update] Shell-E IMAP auth fixed — helpme_app_password re-added to VM creds.json,
cron running every 5 min. Stress test script updated to use SMTP app password
from calendar/config.yaml.

[observation] creds.json on VM may be overwritten by git pull — key disappeared
twice. Check if creds.json is git-tracked on VM; remove from git + add to
.gitignore if so.

Session logged. You can now compact this conversation and start fresh.
```

## Tone

Short and factual. This is a handoff note, not a recap. Future-you reading the session log needs to know: what state things are in, and what to do next.
