---
name: cos-onboard
description: Interview the user and self-configure a personalized /cos skill from skill.md.template — run once, at setup
argument-hint: "[optional: path to cos_claudecode repo, if not run from inside it]"
allowed-tools: Bash, Read, Write, Glob, Grep, AskUserQuestion, mcp__claude_ai_Google_Calendar__list_calendars
---

# COS Onboarding — Interview & Self-Configure

This skill replaces manually editing `skill.md.template` by hand (SETUP.md Steps 1-3). It interviews the user, then writes a fully personalized `skill.md`, `config.yaml`, and `session_log.yaml` directly — no placeholder-hunting required.

**Run this once**, at first setup, or again later to re-run the interview and regenerate the skill (e.g. a new strategic priority quarter, a new relationship to track, a new calendar).

## Ground rules for the interview

- **Batch questions, don't interrogate one field at a time.** Group related fields into one conversational turn (see Steps below for the batching). Nobody wants twelve round-trips to answer "what's your timezone."
- **Use `AskUserQuestion` for closed choices** (yes/no, pick-one, pick-several) and **plain conversation for open text** (names, descriptions, priorities) — `AskUserQuestion` is the wrong tool for "describe your strategic priorities," right tool for "do you use msgvault?"
- **Infer before asking.** Before asking for the projects directory or COS_DIR, check common locations (`~/cos`, `~/Documents/cos`, current working directory) and confirm rather than asking blind. Before asking for calendars, try `mcp__claude_ai_Google_Calendar__list_calendars` and present what comes back for the user to label/select, rather than asking them to type IDs from scratch.
- **Nothing is final until confirmed.** Show the assembled skill.md.template substitutions back as a summary before writing any files, and let the user correct anything before it's written.
- **It's fine to under-fill.** Not everyone has 5 strategic priorities or 3 tracked relationships on day one. 2 priorities and 1 relationship is a valid, complete answer — don't pad with placeholders to hit a round number, and don't block on fields the user says to skip.

## Steps

### 1. Locate the template and target locations

- Find `skill.md.template` and `config.example.yaml` — either in the directory this skill was invoked from, `$ARGUMENTS` if given, or by asking where the cloned/forked `cos_claudecode` repo lives.
- Ask (batched, one turn): where should the working `cos` directory live (this becomes `{{COS_DIR}}` — holds `config.yaml`, `session_log.yaml`, and any local sync scripts)? Suggest `~/cos` as the default. Check if it already exists — if so, confirm before touching anything inside it (this may be a re-run).
- Ask: what directory holds the projects/code this person wants file-activity inference to scan (`{{PROJECTS_DIR}}`)? Suggest their home directory's most obviously code-shaped subfolder if one is easy to spot, otherwise ask directly.

### 2. Identity & context (one batched turn)

Ask conversationally, not via AskUserQuestion (all free text):
- First name (`{{USER_FIRST_NAME}}`)
- One or two sentences on what they do and the domains they operate across — company, side projects, civic/board work, family, anything else relevant. This becomes `{{USER_CONTEXT_PARAGRAPH}}`; you can either take their sentences directly or, if they give you fragments, assemble them into the template's example sentence structure (`"{{USER_FIRST_NAME}} operates across many domains simultaneously: running {{COMPANY_NAME}}..."`) and read it back for confirmation.
- Timezone (`{{TIMEZONE}}`, IANA format) — infer from system locale as a default guess, confirm rather than ask blind.

### 3. Energy map (one batched turn)

Explain briefly what this is for (when the agent should suggest deep work vs. admin vs. protect family time), then ask for four time ranges in one turn:
- Peak hours (`{{PEAK_HOURS}}`) — deep work window
- Valley hours (`{{VALLEY_HOURS}}`) — lighter/admin window
- Family/protected hours (`{{FAMILY_HOURS}}`) — non-negotiable block, plus what it's protecting (`{{FAMILY_LOGISTICS_DESCRIPTION}}`, e.g. "kid pickups, dinner, homework")
- Evening/second-wind hours (`{{EVENING_HOURS}}`), if they have one — this row can be dropped from the table entirely if they don't work evenings; don't force a value

Also ask, same turn:
- Which day(s) coffee/optional meetings should default to (`{{COFFEE_MEETING_DAYS}}`)
- Any recurring blackout periods for deep work (`{{BLACKOUT_PERIODS}}` — e.g. "travel weeks," "the last week of every quarter") — okay to say none

If the user describes a recurring drop-off/wait/pickup obligation (a commute, a kid activity, an appointment) with a real gap in the middle, treat it as a transport-window candidate and offer to add the transport-window-mining pattern from CUSTOMIZING.md §1a as a custom addition to the generated skill, rather than leaving it implicit.

### 4. Strategic priorities

Ask: what are the 2-7 things that actually matter this year — specific, not generic ("get X operational by Q2," not "grow the business"). Take as many as they give, in priority order. Populates `{{PRIORITY_N_NAME}}` / `{{PRIORITY_N_DESCRIPTION}}` — pad or trim the template's 5 numbered slots to match how many they actually gave (don't leave empty numbered slots, don't force a 6th if they only have 5).

### 5. Relationship tracking

Ask: who should the agent actively track and flag if they go quiet — direct reports, a co-founder, a partner, close friends, key clients? For each: name, one-line context, what "needs attention" looks like for that person. Same flexible-count handling as priorities (`{{PERSON_N_NAME}}` / `{{PERSON_N_CONTEXT}}` / `{{PERSON_N_WATCH_FOR}}`).

Ask one follow-up: are any of these people reachable through an always-on channel (a group chat, a shared Slack channel) where silence doesn't mean disengagement? If so, note that as a silence-exception per CUSTOMIZING.md's guidance rather than flagging them for every quiet stretch.

### 6. Calendars

Try `mcp__claude_ai_Google_Calendar__list_calendars` first (load its schema via `ToolSearch` if deferred). If it succeeds, present the returned calendars and ask which to monitor and what label each should get (`{{CALENDAR_LIST}}`, and mirrored into `config.yaml`'s `calendars:` list). If MCP isn't available, ask the user to paste calendar IDs directly (point them to Google Calendar → Settings → Integrate calendar → Calendar ID for anything beyond `primary`).

### 7. Optional integrations (batched yes/no via AskUserQuestion)

Ask in one batch:
- Multiple Gmail/Workspace accounts, or just one? If multiple: point to `gmail_helper.py` (Tier 2) as a next step post-onboarding, not something this interview configures directly (it needs its own per-account OAuth flow).
- Use or want msgvault (local multi-account email/iMessage archive)? If yes, keep the `<!-- OPTIONAL: msgvault -->` blocks in the generated skill.md and note MSGVAULT.md as the next setup step; if no, strip those blocks entirely rather than leaving them commented out.
- Any recurring reminders to bake in (tax/compliance dates, use-or-lose benefits, a content cadence)? Free text, becomes `{{CUSTOM_REMINDERS}}` — okay to leave empty.
- Any key recurring meetings worth the agent always knowing about (`{{KEY_RECURRING_MEETINGS}}`)? Okay to leave empty.
- Internal team members who communicate on a non-email channel, so their threads shouldn't be flagged as "dropped" (`{{INTERNAL_TEAM_LIST}}` / `{{INTERNAL_TEAM_CHANNEL}}`) — only relevant if msgvault dropped-thread detection is being kept.

### 8. Assemble and confirm

Print a summary of every value collected, organized by section, and ask for confirmation or corrections before writing anything. This is the one mandatory checkpoint — don't skip straight to file writes even if the interview felt complete.

### 9. Write the files

Once confirmed:
1. Read `skill.md.template`, substitute every `{{PLACEHOLDER}}` with the collected value. For any optional block (msgvault) the user declined, remove the entire `<!-- OPTIONAL: ... -->` ... `<!-- END OPTIONAL -->` region rather than leaving it with unfilled placeholders.
2. Write the result to `~/.claude/skills/cos/skill.md` (confirm this path — ask if they want it somewhere else).
3. Write `{{COS_DIR}}/config.yaml` from `config.example.yaml`, with the real timezone and calendar list substituted.
4. If `{{COS_DIR}}/session_log.yaml` doesn't already exist, copy `session_log.example.yaml` to it as an empty starter. If it already exists (a re-run), leave it untouched.
5. Print what was written and where, and the immediate next steps: connect the Google Calendar/Gmail MCP connectors if not already connected (claude.ai Settings → Connectors), then run `/cos today` to confirm it works end to end.

## Re-running

If `~/.claude/skills/cos/skill.md` already exists, say so up front and ask whether this is a full re-interview (regenerate everything) or a targeted update (e.g. "just add a new strategic priority" or "add a relationship"). For a targeted update, read the existing skill.md, make the specific edit requested, and skip the rest of the interview — don't force a full re-run for a one-line change.
