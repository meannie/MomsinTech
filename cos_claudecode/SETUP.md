# Setup Guide

Getting `/cos` running on your machine in about 30 minutes.

## Prerequisites

- **Claude Code** installed and working — https://claude.com/claude-code
- **Python 3.10+** installed
- **A Google account** with Tasks and Calendar you want to sync
- **Basic terminal comfort** — you'll be running a few `python` commands

## Step 1: Pick your `cos` directory

Choose a directory where the sync scripts and snapshots will live. `~/cos/` works well, but you can put it anywhere stable. We'll call this `<COS_DIR>` for the rest of the guide.

```bash
mkdir -p ~/cos
cd ~/cos
```

Copy the contents of this repo's `scripts/` directory into `<COS_DIR>`:

```bash
cp /path/to/cos_claudecode/scripts/tasks_sync.py ~/cos/
cp /path/to/cos_claudecode/scripts/gcal_sync.py ~/cos/
cp /path/to/cos_claudecode/config.example.yaml ~/cos/
cp /path/to/cos_claudecode/session_log.example.yaml ~/cos/
cp /path/to/cos_claudecode/requirements.txt ~/cos/
```

## Step 2: Install Python dependencies

From `<COS_DIR>`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 3: Set up Google Cloud OAuth

You need an OAuth client to read your Google Tasks and Calendar.

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project (or use an existing one) — call it something like "cos-personal"
3. Go to **APIs & Services → Library** and enable:
   - **Google Tasks API**
   - **Google Calendar API**
4. Go to **APIs & Services → Credentials** and click **Create Credentials → OAuth client ID**
5. If prompted to set up OAuth consent screen first:
   - User type: **External** (unless you have a Google Workspace, then Internal)
   - App name: "cos personal" (or whatever)
   - User support email: your email
   - Developer contact: your email
   - Scopes: skip (the script declares them)
   - Test users: **add your own Google email** (this is required for External + unverified apps)
6. Application type: **Desktop app**
7. Name: "cos local"
8. Click **Create**, then **Download JSON**
9. Save the downloaded file as `oauth_credentials.json` in `<COS_DIR>`

## Step 4: Configure your calendars

Copy the example config to a real one and edit it:

```bash
cp config.example.yaml config.yaml
```

Open `config.yaml` and:
- Set your `timezone` (IANA format like `America/New_York`, `Europe/London`, etc.)
- Add each calendar you want monitored under `calendars:`

To find a calendar's ID:
- **Your primary calendar:** use `primary` as the ID
- **Other calendars you own:** use the email address (`you@yourcompany.com`)
- **Shared/group calendars:** Open Google Calendar in browser → click the 3 dots next to the calendar name → **Settings and sharing** → scroll to **Integrate calendar** → copy **Calendar ID** (looks like `c_xxxxxxxx@group.calendar.google.com`)

Example:

```yaml
timezone: America/Los_Angeles

calendars:
  - id: primary
    label: Personal
  - id: jane@acmecorp.com
    label: Work
  - id: c_abc123xyz@group.calendar.google.com
    label: Family
```

## Step 5: First-run authentication

This opens a browser window where you'll grant access to your Google account. You only do this once per script.

```bash
python3 tasks_sync.py --auth
```

A browser tab opens. Sign in with the Google account that owns the calendars/tasks you configured. You'll see a "Google hasn't verified this app" warning — click **Advanced → Go to (your project name)** to proceed (this is expected for personal-use OAuth clients).

Then:

```bash
python3 gcal_sync.py --auth
```

Same flow. After both succeed, you'll have `tasks_token.json` and `gcal_token.json` in `<COS_DIR>`. **These are gitignored** — don't commit them.

## Step 6: Test the sync scripts

```bash
python3 tasks_sync.py --list
python3 gcal_sync.py --list
```

You should see your task lists and today's calendar events printed. If yes, the data layer works.

## Step 7: Initialize the session log

```bash
cp session_log.example.yaml session_log.yaml
```

Open `session_log.yaml` and delete the example entries. Leave the file empty (just the comments at the top). The agent will start writing to it on the first `/cos` run.

## Step 8: Customize `skill.md.template` for you

This is the most important step. Open `skill.md.template` and find every `{{PLACEHOLDER}}` token.

The bare minimum to fill in:
- `{{USER_FIRST_NAME}}` — your name
- `{{USER_CONTEXT_PARAGRAPH}}` — 2-3 sentences about who you are and what you do
- `{{COS_DIR}}` — the absolute path to your cos directory (e.g., `/Users/jane/cos`)
- `{{PROJECTS_DIR}}` — where your code/work projects live
- `{{PEAK_HOURS}}`, `{{VALLEY_HOURS}}`, `{{FAMILY_HOURS}}`, `{{EVENING_HOURS}}` — your energy curve
- `{{TIMEZONE}}` — same as in config.yaml
- `{{CALENDAR_LIST}}` — bulleted list of your calendars
- At least 3 entries under "Strategic Priorities"
- At least 2 entries under "Relationship Tracking"

The richer you make this, the better `/cos` performs. See [CUSTOMIZING.md](CUSTOMIZING.md) for ideas.

Save the customized version as `skill.md` (drop the `.template`).

## Step 9: Install the skill in Claude Code

Claude Code reads skills from `~/.claude/skills/`. Create the cos skill directory:

```bash
mkdir -p ~/.claude/skills/cos
cp skill.md ~/.claude/skills/cos/skill.md
```

## Step 10: (Optional but recommended) Connect Google Calendar MCP

The skill works two ways for calendar data:
1. **Local sync** (`gcal_sync.py`) — pulls into a local JSON snapshot
2. **Google Calendar MCP** — Claude.ai's Google Calendar connector lets the agent query calendar data live

For best results, enable the **Google Calendar MCP connector** in your Claude account so the agent can query your calendar directly during a `/cos` run instead of relying only on the snapshot. Settings → Connectors → Google Calendar.

Both work — MCP is faster and live, the local script is a fallback if you don't want to grant Claude.ai connector access.

## Step 11: Run it

Open Claude Code and type:

```
/cos today
```

You should get a briefing. If it's empty or wrong, check:
- Did `python3 tasks_sync.py` and `python3 gcal_sync.py` run successfully?
- Are the JSON snapshot files in `<COS_DIR>` populated?
- Did you fill in `{{COS_DIR}}` correctly in `skill.md`?

## Step 12: (Optional) Schedule the sync scripts

The skill instructs Claude to run `tasks_sync.py` itself before reading the snapshot, so this is technically optional. But if you want fresh data without waiting:

Add to your crontab (`crontab -e`):

```cron
# Refresh cos snapshots every 30 minutes during work hours
*/30 8-22 * * * cd ~/cos && /usr/bin/python3 tasks_sync.py >/dev/null 2>&1
*/30 8-22 * * * cd ~/cos && /usr/bin/python3 gcal_sync.py --days 7 >/dev/null 2>&1
```

## Troubleshooting

**"OAuth credentials not found"** — You forgot to save `oauth_credentials.json` in `<COS_DIR>`. Re-download from Google Cloud Console.

**"Token has been expired or revoked"** — Delete `tasks_token.json` (or `gcal_token.json`) and re-run `--auth`.

**"Google hasn't verified this app"** — Expected. Click Advanced → Go to (project name). For personal OAuth clients you don't need verification.

**Calendar sync returns 0 events but you have events** — Check the calendar ID in `config.yaml`. The shared calendar IDs are long strings with `@group.calendar.google.com`, not the friendly name.

**`/cos` runs but says "no session log found"** — You need to create `session_log.yaml` (Step 7). Empty is fine.

## What's next

Read [CUSTOMIZING.md](CUSTOMIZING.md) for the real value: making the agent know enough about your specific life that its recommendations actually fit. The default skill is generic — yours shouldn't be.
