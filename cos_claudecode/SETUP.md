# Setup Guide

Getting `/cos` running on your machine in about 15 minutes (MCP path) or 30 minutes (with OAuth fallback).

> ## ⚠️ Make your fork private BEFORE adding any personal context
>
> If you forked this repo to set up your own /cos, the fork is **public by default**. As soon as you customize `skill.md.template` with your strategic priorities, relationships, calendar IDs, recurring obligations, or session log entries, that information becomes public if the fork visibility is still public.
>
> **Make your fork private right now, before doing any of the steps below:**
>
> 1. Go to your forked repo on GitHub
> 2. Settings → scroll to bottom → **Danger Zone**
> 3. **Change repository visibility** → **Make private** → confirm
>
> This is the most common foot-gun on this repo. Do it before Step 1.

## Prerequisites

- **Claude Code** installed and working — https://claude.com/claude-code
- **A Google account** with Tasks and Calendar you want to sync

For the OAuth fallback path (optional), you'll also need:
- **Python 3.10+** installed
- **Basic terminal comfort** — a few `python` commands

## Step 1: Pick your `cos` directory

Choose a directory where session logs and (optionally) sync scripts will live. We'll call this `<COS_DIR>` for the rest of the guide.

```bash
mkdir -p ~/cos
cd ~/cos
```

Copy the base files:

```bash
cp /path/to/cos_claudecode/session_log.example.yaml ~/cos/
cp /path/to/cos_claudecode/config.example.yaml ~/cos/
```

## Step 2: Initialize the session log

```bash
cp session_log.example.yaml session_log.yaml
```

Delete the example entries. Leave the file empty (just the comments at the top). The agent will start writing to it on the first `/cos` run.

## Step 3: Customize `skill.md.template` for you

This is the most important step, and you have two ways to do it:

> **Recommended: let `/cos-onboard` do it for you.** Install `skills/onboarding/skill.md` into Claude Code the same way you'll install `cos` itself (see Step 5 below, do this one first), then run `/cos-onboard`. It interviews you — batched, conversational questions, not a wall of blanks to fill — and writes a fully substituted `skill.md`, `config.yaml`, and `session_log.yaml` directly. It also pulls your actual calendar list via MCP instead of asking you to hand-copy IDs. Skip the rest of this step if you use it, and go straight to Step 4.

The manual path — open `skill.md.template` and find every `{{PLACEHOLDER}}` token yourself:

The bare minimum to fill in:
- `{{USER_FIRST_NAME}}` — your name
- `{{USER_CONTEXT_PARAGRAPH}}` — 2-3 sentences about who you are and what you do
- `{{COS_DIR}}` — the absolute path to your cos directory (e.g., `/Users/jane/cos`)
- `{{SYNC_REPO}}` — absolute path to your local clone of the private sync repo (e.g., `/Users/jane/personalCOS`)
- `{{PROJECTS_DIR}}` — where your code/work projects live
- `{{PEAK_HOURS}}`, `{{VALLEY_HOURS}}`, `{{FAMILY_HOURS}}`, `{{EVENING_HOURS}}` — your energy curve
- `{{FAMILY_LOGISTICS_DESCRIPTION}}` — one sentence on family logistics (e.g., "school pickup at 3pm, fencing Tue/Thu")
- `{{COFFEE_MEETING_DAYS}}` — which days of the week you allow coffee meetings (e.g., "Thursday or Friday")
- `{{BLACKOUT_PERIODS}}` — recurring windows you want protected from scheduling (e.g., "competition weekends")
- `{{TIMEZONE}}` — same as in config.yaml
- `{{CALENDAR_LIST}}` — bulleted list of your calendars (see Step 4 for finding calendar IDs)
- `{{KEY_RECURRING_MEETINGS}}` — one-line list of standing meetings Claude should know about
- At least 3 entries under "Strategic Priorities"
- At least 2 entries under "Relationship Tracking"

The richer you make this, the better `/cos` performs. See [CUSTOMIZING.md](CUSTOMIZING.md) for ideas.

Save the customized version as `skill.md` (drop the `.template`).

## Step 4: Connect Google MCPs (recommended)

The fastest way to give `/cos` access to your Google data. Enable these connectors in your Claude account:

### Google Calendar
**Settings → Connectors → Google Calendar**

When connected, the agent calls `mcp__claude_ai_Google_Calendar__list_events` directly — live data, no snapshot staleness, and can create events, check availability, etc.

To find calendar IDs for your `skill.md`:
- **Your primary calendar:** use `primary`
- **Other calendars you own:** use the email address (`you@yourcompany.com`)
- **Shared/group calendars:** Google Calendar in browser → 3 dots next to calendar name → **Settings and sharing** → **Integrate calendar** → copy **Calendar ID** (looks like `c_xxxxxxxx@group.calendar.google.com`)

### Gmail
**Settings → Connectors → Gmail**

When connected, the agent can search your inbox for event invites (Luma, Eventbrite, Partiful), action items, and context. When not connected, inbox scanning is skipped gracefully.

### Google Tasks
**No MCP connector exists for Google Tasks yet.** If you use Google Tasks and want `/cos` to read them, you'll need the OAuth fallback (Step 7). If you don't use Google Tasks, skip it — `/cos` works fine without it.

## Step 5: Install the skill in Claude Code

```bash
mkdir -p ~/.claude/skills/cos
cp skill.md ~/.claude/skills/cos/skill.md
```

## Step 6: Run it

Open Claude Code and type:

```
/cos today
```

You should get a briefing. If calendar data is missing, check that the Google Calendar MCP connector is enabled and authorized (Step 4).

**If this works, you're done.** Steps 7+ are only needed if you want Google Tasks integration or an offline fallback for calendar data.

---

## Step 7: (Optional) OAuth fallback for Google Tasks and offline calendar

The local sync scripts give you two things the MCPs can't:
1. **Google Tasks** — read/write task lists (no MCP exists)
2. **Offline calendar snapshots** — if you prefer not to grant MCP connector access, or want a local backup

### 7a: Copy sync scripts and install dependencies

```bash
cp /path/to/cos_claudecode/scripts/tasks_sync.py ~/cos/
cp /path/to/cos_claudecode/scripts/gcal_sync.py ~/cos/
cp /path/to/cos_claudecode/requirements.txt ~/cos/

cd ~/cos
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 7b: Set up Google Cloud OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project (or use an existing one) — call it something like "cos-personal"
3. Go to **APIs & Services → Library** and enable:
   - **Google Tasks API**
   - **Google Calendar API** (only needed if you're NOT using the Calendar MCP)
4. Go to **APIs & Services → Credentials** and click **Create Credentials → OAuth client ID**
5. If prompted to set up OAuth consent screen first:
   - User type: **External** (unless you have a Google Workspace, then Internal)
   - App name: "cos personal" (or whatever)
   - User support email: your email
   - Developer contact: your email
   - Scopes: skip (the script declares them)
   - Test users: **add your own Google email** (required for External + unverified apps)
6. Application type: **Desktop app**
7. Name: "cos local"
8. Click **Create**, then **Download JSON**
9. Save the downloaded file as `oauth_credentials.json` in `<COS_DIR>`

### 7c: Configure calendars (only if NOT using Calendar MCP)

If you're using the Calendar MCP for live data, skip this — it's only for the local `gcal_sync.py` fallback.

```bash
cp config.example.yaml config.yaml
```

Open `config.yaml` and set your `timezone` and `calendars:` list.

### 7d: Authenticate

```bash
python3 tasks_sync.py --auth
```

A browser tab opens. Sign in with the Google account that owns the calendars/tasks. You'll see a "Google hasn't verified this app" warning — click **Advanced → Go to (your project name)** to proceed (expected for personal-use OAuth clients).

If you're also using the calendar fallback:

```bash
python3 gcal_sync.py --auth
```

After both succeed, you'll have `tasks_token.json` and `gcal_token.json` in `<COS_DIR>`. **These are gitignored** — don't commit them.

### 7e: Test

```bash
python3 tasks_sync.py --list
```

You should see your task lists printed. If using the calendar fallback too:

```bash
python3 gcal_sync.py --list
```

## Step 8: (Optional) Schedule local sync scripts

The skill instructs Claude to run `tasks_sync.py` itself before reading the snapshot, so this is technically optional. But if you want fresh data without waiting:

```cron
# Refresh cos snapshots every 30 minutes during work hours
*/30 8-22 * * * cd ~/cos && /usr/bin/python3 tasks_sync.py >/dev/null 2>&1
*/30 8-22 * * * cd ~/cos && /usr/bin/python3 gcal_sync.py --days 7 >/dev/null 2>&1
```

## How MCP + OAuth coexist at runtime

The skill tries MCP first for each service. If MCP is connected, it uses live data. If not, it falls back to the local sync script and snapshot JSON. It won't mix paths within a single `/cos` run.

| Service | MCP connected | MCP not connected |
|---------|--------------|-------------------|
| Calendar | Live via MCP | `gcal_sync.py` → `gcal_snapshot.json` |
| Gmail | Live via MCP | Inbox scanning skipped |
| Tasks | *No MCP exists* | `tasks_sync.py` → `tasks_snapshot.json` |

## Troubleshooting

**Calendar MCP shows no events** — Make sure you authorized the correct Google account in Settings → Connectors → Google Calendar. Also verify the calendar IDs in your `skill.md` are correct.

**"OAuth credentials not found"** — You forgot to save `oauth_credentials.json` in `<COS_DIR>`. Re-download from Google Cloud Console.

**"Token has been expired or revoked"** — Delete `tasks_token.json` (or `gcal_token.json`) and re-run `--auth`.

**"Google hasn't verified this app"** — Expected. Click Advanced → Go to (project name). For personal OAuth clients you don't need verification.

**Calendar sync returns 0 events but you have events** — Check the calendar ID in `config.yaml`. The shared calendar IDs are long strings with `@group.calendar.google.com`, not the friendly name.

**`/cos` runs but says "no session log found"** — You need to create `session_log.yaml` (Step 2). Empty is fine.

## What's next

Read [CUSTOMIZING.md](CUSTOMIZING.md) for the real value: making the agent know enough about your specific life that its recommendations actually fit. The default skill is generic — yours shouldn't be.
