# msgvault Integration (Optional)

msgvault is a local email and iMessage archive tool. This page explains why it's useful alongside the Gmail MCP connector, how to set it up, and how to wire it into your `/cos` skill.

**This entire integration is optional.** If you only have one Gmail account and the Gmail MCP connector covers everything you need, skip this page.

---

## Why msgvault complements the Gmail MCP

The Gmail MCP connector (`mcp__claude_ai_Gmail__search_threads`) covers **only the primary Google account** connected to your claude.ai session. If you have additional Gmail accounts, Workspace accounts, or IMAP accounts (Outlook, iCloud, etc.), the MCP can't reach them.

msgvault fills this gap:
- Syncs all your email accounts (any mix of Gmail OAuth + IMAP) to a local SQLite database
- Exposes a single `msgvault search` command that searches across all of them
- Runs continuously as a background daemon, so data is always current
- Also imports iMessage history for relationship context

With both tools running, `/cos` can:
- Scan for event invites across every inbox (not just the primary one)
- Detect dropped threads with personal/external contacts using full email history
- Surface iMessage context for relationship tracking

---

## Installation

### 1. Install the msgvault binary

Download the latest release from the [msgvault GitHub releases page](https://github.com/Inbox-Sage/msgvault/releases) and place it at `~/.local/bin/msgvault`:

```bash
mkdir -p ~/.local/bin
# download the binary for your platform, then:
mv ~/Downloads/msgvault ~/.local/bin/msgvault
chmod +x ~/.local/bin/msgvault
```

Verify:
```bash
msgvault --version
```

### 2. Create the config directory

```bash
mkdir -p ~/.msgvault
```

### 3. Configure accounts

Create `~/.msgvault/config.toml`. The minimum configuration:

```toml
[data]
  data_dir = "/Users/yourname/.msgvault"

[[accounts]]
email = "you@gmail.com"
schedule = "0 */6 * * *"
enabled = true

# Add one [[accounts]] block per email address.
# Gmail accounts use OAuth (see step 4).
# IMAP accounts use the imaps:// URL format:
# email = "imaps://username@mail.example.com:993"

[oauth]
  client_secrets = "/path/to/your/gmail_client_secret.json"
```

See `~/.msgvault/config.toml` comments or the msgvault docs for the full schema (vector search, server config, etc.).

### 4. Gmail OAuth setup

msgvault uses its own OAuth credentials to access Gmail (separate from the MCP connector).

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or reuse a project
3. Enable **Gmail API** under APIs & Services → Library
4. Create an OAuth client ID (Desktop app type)
5. Download the client secret JSON
6. Set `client_secrets` in `~/.msgvault/config.toml` to the path of that JSON file

For each Gmail account, run:
```bash
msgvault auth --email you@gmail.com
```

A browser tab opens for OAuth consent. After auth, msgvault stores the token in `~/.msgvault/`.

### 5. Start the sync daemon

msgvault runs as a background daemon that syncs on the schedule you set in config.toml:

```bash
msgvault serve
```

To run it automatically on login, set up a launchd agent. Create `~/Library/LaunchAgents/com.msgvault.serve.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.msgvault.serve</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/yourname/.local/bin/msgvault</string>
        <string>serve</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/yourname/.msgvault/serve.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/yourname/.msgvault/serve.log</string>
</dict>
</plist>
```

Load it:
```bash
launchctl load ~/Library/LaunchAgents/com.msgvault.serve.plist
```

### 6. Verify

```bash
msgvault stats
```

You should see message counts per account. The first full sync may take several minutes per account.

---

## iMessage nightly sync (optional)

msgvault can import your iMessage history from the macOS Messages database, giving `/cos` relationship context for personal contacts.

### Export your contacts

1. Open Contacts.app
2. File → Export → Export vCard
3. Save to a stable path — `~/Downloads/Contacts.vcf` works well (avoid Desktop/iCloud Drive, which shell tools can't always access)

### Set up the nightly launchd agent

Copy the template and fill in your paths:

```bash
cp /path/to/cos_claudecode/scripts/imessage-sync.plist.template \
   ~/Library/LaunchAgents/com.msgvault.imessage-sync.plist
```

Edit the plist to replace:
- `{{MSGVAULT_BINARY_PATH}}` → `/Users/yourname/.local/bin/msgvault`
- `{{CONTACTS_VCF_PATH}}` → `/Users/yourname/Downloads/Contacts.vcf`
- `{{MSGVAULT_DATA_DIR}}` → `/Users/yourname/.msgvault`

Load it:
```bash
launchctl load ~/Library/LaunchAgents/com.msgvault.imessage-sync.plist
```

This runs at 3am nightly. To run immediately:
```bash
launchctl start com.msgvault.imessage-sync
```

**Note on contacts matching:** msgvault matches contacts by email address. Phone-number matching requires E.164 format in the vCard (`+14155551234`), which the macOS Contacts export doesn't always produce. Most Apple ID users can be matched by email regardless.

**Note on permissions:** The first import requires Full Disk Access for the terminal (or the binary itself). System Settings → Privacy & Security → Full Disk Access.

---

## Wiring into /cos skill.md

Once msgvault is running, add these two optional blocks to your `skill.md` (after the API table in the Data Sources section):

### Block 1 — Multi-account event invite scanning

```markdown
<!-- OPTIONAL: msgvault multi-account event invite scanning
     Remove if you only have one email account or don't use msgvault.
     Gmail MCP covers only the primary account. msgvault covers all others. -->

#### Event Invite Scanning — Multi-Account (msgvault)

Run this **in parallel** with the Gmail MCP search — do not use as a fallback:

```bash
# Search for event invites across all msgvault accounts (last 14 days)
~/.local/bin/msgvault search \
  "subject:(invitation OR invite OR \"you're invited\" OR RSVP OR \"join us\" OR calendar) OR from:(calendar-notification OR noreply@calendar)" \
  --since 14d --format json 2>/dev/null
```

- Gmail MCP → event invites sent to the primary Gmail account
- msgvault → event invites sent to all other accounts

Merge both result sets and deduplicate by subject + sender before composing the briefing. Prefer the MCP result when the same event appears in both (richer metadata). Flag any invite that requires an RSVP and doesn't have one.
<!-- END OPTIONAL: msgvault event invite scanning -->
```

### Block 2 — Relationship dropped-thread detection

Add this to the `week` mode's Relationship Check-In section:

```markdown
<!-- OPTIONAL: msgvault dropped-thread detection
     Skip the Interact team and anyone whose primary channel is Slack/chat.
     Focus on personal contacts and external collaborators. -->

#### Dropped Thread Detection (msgvault)

For personal and external contacts, check for threads that were active but fell off:

```bash
# Threads with external contacts that went quiet in the last 30 days
~/.local/bin/msgvault search \
  "from:({{PERSONAL_CONTACT_DOMAINS}}) has:thread" \
  --since 60d --until 30d --format json 2>/dev/null
```

Surface as: "You had a thread going with [person] about [topic] — it fell off around [date]. Worth picking back up?"

Do NOT flag:
- {{INTERNAL_TEAM_LIST}} — they communicate on {{INTERNAL_TEAM_CHANNEL}}, not email
- Threads you ended intentionally (check session log for "closed" or "done" notes about the person)
<!-- END OPTIONAL: msgvault dropped-thread detection -->
```

Replace the placeholders:
- `{{PERSONAL_CONTACT_DOMAINS}}` — domains of personal/external contacts you want to track (e.g., `gmail.com OR personalfriend.com`)
- `{{INTERNAL_TEAM_LIST}}` — names of team members whose primary channel isn't email
- `{{INTERNAL_TEAM_CHANNEL}}` — their actual channel (e.g., Slack, Teams)

---

## Troubleshooting

**"OAuth client secrets file not accessible"** — The path in `config.toml` is wrong or the file was moved. Check the absolute path.

**Sync not running** — `launchctl list | grep msgvault` to see if the daemon is loaded. Check `~/.msgvault/serve.log` for errors.

**iMessage import says 0 contacts matched** — This is usually fine if your contacts use email. Phone-number matching requires E.164 format in the vCard, which macOS doesn't always export. The data still imports; contacts just appear by phone number instead of name.

**First sync is slow** — Expected. Full history syncs take time. Let it run; subsequent syncs are incremental.
