# Mobile Access via GitHub-Synced Repo

Run /cos directly from your phone — same skill, same session log, no middleman bot.

This doc covers GitHub-sync mobile: your phone runs Claude Code directly against a synced copy of your /cos directory, giving you full /cos behavior (all modes, full skill instructions) on your phone.

(A Slack-bot chat-interface alternative previously lived here — removed for now, may return later.)

---

## How GitHub-sync mobile works

```
                          ┌─────────────────────────────┐
                          │  Private GitHub repo        │
                          │  (session_log.yaml +        │
                          │   gcal_snapshot.json +      │
                          │   tasks_snapshot.json)      │
                          └──────────┬──────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  Always-on Mac   │  │   Phone / iPad   │  │  Second machine  │
    │                  │  │                  │  │                  │
    │  • Has Google    │  │  • Has Claude    │  │  • Same setup as │
    │    OAuth tokens  │  │    Code installed│  │    Phone         │
    │  • Refreshes     │  │  • git pulls     │  │                  │
    │    snapshots on  │  │    before /cos   │  │                  │
    │    cron          │  │  • git pushes    │  │                  │
    │  • git pushes    │  │    after updates │  │                  │
    │    snapshots up  │  │                  │  │                  │
    └──────────────────┘  └──────────────────┘  └──────────────────┘
```

The trick: **mobile doesn't need Google API tokens.** The always-on Mac generates the snapshots, pushes them to the private GitHub repo, and the phone just reads the cached JSON. The phone DOES need to push updates back to the session log, but git auth (SSH key or HTTPS token) is enough.

---

## Step 1: Create the sync repo (if you haven't already)

This is a separate **private** GitHub repo from the one you forked the template from. It holds your living state — session log, snapshots — and acts as the shared brain across devices.

1. **Create a new PRIVATE repo** at https://github.com/new
   - Name it whatever you want (e.g. `myCOS`, `cos-state`, `personalCOS`)
   - Visibility: **Private** (this is non-negotiable — it will contain calendar events, task contents, and session log entries)
2. **Clone it to your always-on machine**:
   ```bash
   cd ~
   git clone git@github.com:YOU/myCOS.git
   ```

You'll point both your /cos directory AND your phone at this repo.

---

## Step 2: Wire the always-on Mac to refresh + push snapshots

Place these files inside your cloned sync repo:

```
~/myCOS/
  ├── session_log.yaml                    # the brain
  ├── session_log_archive.yaml            # rotated history
  ├── gcal_snapshot.json                  # refreshed by gcal_sync.py
  ├── tasks_snapshot.json                 # refreshed by tasks_sync.py
  ├── oauth_credentials.json              # GITIGNORED — never commit
  ├── tasks_token.json, gcal_token.json   # GITIGNORED — never commit
  ├── scripts/
  │     ├── gcal_sync.py
  │     ├── tasks_sync.py
  │     ├── tasks_add.py
  │     └── sync_session_log.py
  └── .gitignore                          # excludes credentials + tokens
```

Verify your `.gitignore` includes:

```
oauth_credentials.json
*_token.json
creds.json
```

(Snapshots are intentionally NOT gitignored here — they're the payload that gets shared with mobile.)

---

## Step 3: Schedule the sync on the always-on machine

You want `sync_session_log.py` to run on a regular cadence so the snapshots stay fresh on the phone. Two paths:

### macOS (launchd) — every 30 min during work hours

Create `~/Library/LaunchAgents/com.you.cos-sync.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.you.cos-sync</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/python3</string>
        <string>/Users/YOU/myCOS/scripts/sync_session_log.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/YOU/myCOS</string>
    <key>StartCalendarInterval</key>
    <array>
        <!-- Every 30 min, 7 AM – 11 PM, Mon–Fri -->
        <dict><key>Hour</key><integer>7</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Hour</key><integer>7</integer><key>Minute</key><integer>30</integer></dict>
        <!-- ...add more entries through 23:30, weekdays only via DayOfWeek key... -->
    </array>
    <key>StandardOutPath</key><string>/tmp/cos-sync.log</string>
    <key>StandardErrorPath</key><string>/tmp/cos-sync.err</string>
</dict>
</plist>
```

Load:
```bash
launchctl load ~/Library/LaunchAgents/com.you.cos-sync.plist
```

(For frequency that simple, a `StartInterval` of 1800 seconds and a sleep-aware launchd setting is also fine; the calendar-interval pattern is more controlled.)

### Linux / WSL (cron)

```bash
crontab -e
# Add:
*/30 7-23 * * 1-5 cd ~/myCOS && /usr/bin/python3 scripts/sync_session_log.py >> /tmp/cos-sync.log 2>&1
```

Verify with:
```bash
tail -f /tmp/cos-sync.log
```

You should see periodic "Pulling from GitHub... Snapshots refreshed... Pushed to GitHub" entries.

---

## Step 4: Set up the phone

You need three things on your phone:

1. **Claude Code mobile** (or, alternatively, SSH client to a remote machine — see Section 6)
2. **Git access to the private sync repo** — most easily done with an SSH key generated on the phone, added to your GitHub account
3. **A clone of the sync repo** in a location Claude Code mobile can access

### iOS path (using Working Copy + Claude Code mobile)

[Working Copy](https://workingcopy.app/) is the standard iOS git client.

1. **Install Working Copy** from the App Store (free for read; one-time purchase to enable push)
2. **Generate an SSH key** in Working Copy → Settings → SSH Keys → New Key. Copy the public key.
3. **Add the public key** to your GitHub account: https://github.com/settings/keys
4. **Clone the sync repo** in Working Copy:
   - Repositories → `+` → Clone Repository → SSH URL `git@github.com:YOU/myCOS.git`
5. **Open the cloned directory in Claude Code mobile** as your working directory. The skill auto-loads from `~/.claude/skills/` on your phone, but the working data (session log, snapshots) lives in the cloned Working Copy directory.
6. **Configure Claude Code mobile** to git-pull before /cos runs and git-push after — same skill instruction below works on mobile.

### Android path (using Termux + Claude Code or SSH)

1. **Install Termux** from F-Droid or the Play Store
2. Inside Termux: `pkg install git openssh python` and generate an SSH key (`ssh-keygen -t ed25519`)
3. **Add the public key** to GitHub
4. **Clone the repo**: `git clone git@github.com:YOU/myCOS.git`
5. **Run /cos** via whichever LLM CLI you use (Claude Code, an Anthropic SDK script, etc.) pointing at the cloned dir

---

## Step 5: Wire the skill to git-pull at the start of every run

The /cos skill should ALREADY be doing this (it's in `skill.md.template`'s Session Log section). Verify your skill includes a block like:

```markdown
**At the START of every /cos run**, pull the latest from GitHub:
```bash
cd ~/myCOS && git pull origin main --rebase 2>/dev/null
cp ~/myCOS/session_log.yaml <your_cos_working_dir>/session_log.yaml
```

After presenting the briefing, append new observations/nags and push:
```bash
cp <your_cos_working_dir>/session_log.yaml ~/myCOS/session_log.yaml
cd ~/myCOS && git add session_log.yaml && git commit -m "cos sync $(date)" --allow-empty 2>/dev/null && git push origin main 2>/dev/null
```
```

If you don't already have this pattern in `skill.md`, add it now. Without it, devices won't see each other's updates.

---

## Step 6: (Alternative) SSH from your phone to the always-on Mac

If you don't want Claude Code mobile (or it's not available for your platform), the simplest pattern is:

1. **Set up Tailscale** on both your Mac and your phone (free, https://tailscale.com)
2. **Enable SSH on the Mac** (System Settings → Sharing → Remote Login)
3. **SSH client on the phone** — Termius (iOS/Android), Blink Shell (iOS), Termux (Android)
4. From the phone: `ssh you@your-mac.tailnet.ts.net` and just run `claude` — you're in the same Claude Code your Mac uses, no syncing needed

Trade-off: this doesn't work when your Mac is asleep. The GitHub-sync pattern works regardless of Mac state.

---

## Verifying the round-trip

End-to-end test:

1. On your laptop, run `/cos update test from desktop`
2. Wait 30 seconds (the next sync cycle)
3. On your phone, open Claude Code (or SSH session), run `/cos today`
4. You should see "test from desktop" in the session log context
5. From the phone, run `/cos update test from phone`
6. Wait for the phone-side push (immediate, after the run completes)
7. Back on your laptop, run `/cos today`
8. You should see "test from phone" in the context

If the round-trip works, you have full multi-device /cos.

---

## Troubleshooting

**Phone can't push** → SSH key not added to GitHub correctly. Test with: `ssh -T git@github.com` from the phone — should print "Hi YOU! You've successfully authenticated..."

**Phone sees stale data** → The phone is reading the snapshot at clone time. Confirm your skill's instruction to `git pull` is being followed. Check the timestamp inside `gcal_snapshot.json` (`synced_at` field) — should be recent.

**Mac sync job not running** → Check `tail -f /tmp/cos-sync.log`. If empty, launchd isn't firing. Try `launchctl unload && load` of the plist; verify ProgramArguments paths are correct.

**Conflict on git push** → Two devices wrote to session_log at the same time. The script does `--rebase` on pull, so most cases auto-resolve. If you hit a hard conflict, manually merge the YAML and push.

**Don't want snapshots in the repo** → Remove them from the sync (edit `sync_session_log.py` — the `git_push` function lists which files get added). Trade-off: phone won't have current calendar/tasks data unless it can run the sync scripts itself (which means OAuth tokens on the phone, which is messier).

---

## Why this is "clutch"

Most personal-AI-assistant tools force you to be at a particular machine, or require you to deploy a server you maintain. This pattern uses GitHub as a free, persistent, multi-device sync layer that you already trust with your code. Your phone sees the same brain your laptop sees. When you log a thought from the bus, it's there when you get to your desk. When you make a commitment in a meeting, it's there when you check during dinner.

The cost is a few minutes of setup and the discipline of keeping the sync repo private. The payoff is a chief-of-staff who follows you across devices.
