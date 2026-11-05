# Install

About an hour, most of it in steps 3 and 4. Claude runs every command. You only do the steps marked **You**. Each step ends with what success looks like; Claude checks it before moving on.

Nothing acts for real until step 8. Before that, every job runs in shadow mode: it reports what it would do and changes nothing.

---

## 1. Before you start

**You:** have Claude Code installed and signed in, and the base `/cos` set up (`/cos-onboard` part 1).

**Claude:** creates the Python environment for this folder:

```
cd agentic_cos
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
```

**Success looks like:** every test passes. They use no accounts.

## 2. Interview

**Claude:** runs [INTERVIEW.md](INTERVIEW.md), shows you the summary, and writes `local/config.yaml` after you say it's right.

**Success looks like:** `local/config.yaml` exists, every module in it says `shadow`, and `approvals` says `live` (it only ever acts on your own YES).

## 3. Sign-in setup for your kind of inbox

This creates a small private "app" that only you use, so your own inboxes can give your own computer access. You do it once.

### Gmail or Google Workspace (about 10 minutes)

**You**, at console.cloud.google.com:

1. Create a new project. Any name, like "My chief of staff".
2. Turn on three APIs (search each name, click Enable): **Gmail API**, **Google Calendar API**, and **Google Tasks API** (only if you said yes to tasks).
3. Set up the **OAuth consent screen**:
   - Workspace account: pick **Internal**.
   - Personal Gmail: pick **External**, and add yourself as a test user.
4. Personal Gmail only: on the consent screen page, click **Publish app** (moving it to "In production").
   - Without this, Google signs you out every 7 days.
   - Google will warn the app is "unverified". That's expected for an app only you use. You'll click through the warning when you sign in.
5. Go to **Credentials → Create credentials → OAuth client ID → Desktop app**, then click **Download JSON**.

**Claude:** moves the downloaded file to `local/secrets/google_client.json` and checks that it's a desktop client. It doesn't print the file's contents.

### Microsoft 365 / Outlook (about 10 minutes)

**You**, at entra.microsoft.com:

1. Go to **App registrations → New registration**.
   - Name: anything.
   - Supported account types: pick the option that includes **personal Microsoft accounts**, unless this is only for a work account.
   - Redirect URI: pick **Public client/native** and enter `http://localhost`.
2. On the new app's **Authentication** page, turn on **Allow public client flows**.
3. Under **API permissions → Add a permission → Microsoft Graph → Delegated**, add **Mail.ReadWrite** and **Calendars.Read**.
4. Copy the **Application (client) ID** from the Overview page.

**Work account:** your IT admin may need to approve the app. If the sign-in in step 4 says "Need admin approval", send them this: *"I registered a personal app that reads my own mail and calendar (Mail.ReadWrite, Calendars.Read, delegated). Could you grant consent for it?"*

**Claude:** writes `local/secrets/m365_client.json` with `{"client_id": "<the ID>", "tenant": "common"}`. The client ID isn't a secret, so you can paste it in chat.

**Success looks like:** the client file exists under `local/secrets/`.

## 4. Connect each inbox

**Claude:** runs this for each inbox, one at a time:

```
.venv/bin/python -m acos.auth connect <inbox id>
```

**You:** a browser page opens. Pick the right account and click **Allow**.
- Gmail asks for mail (read, label, archive) and calendar (read only).
- Microsoft asks for the same.

If you said yes to tasks, Claude also runs `connect tasks`.

**Claude:** runs `.venv/bin/python -m acos.auth check`.

**Success looks like:** every inbox shows `ok`. A wrong account is rejected and nothing is saved, so just run it again and pick the right one.

## 5. Set up your channel

- **Email:** nothing to do.
- **Instinct:** **You** confirm your assistant's email address. It's already in config from the interview.
- **Telegram** (about 2 minutes):
  1. **You:** in Telegram, message **@BotFather**, send `/newbot`, and pick any name and username. It replies with a token.
  2. **You:** paste the token into `local/secrets/telegram_token`. Claude creates the empty file and opens it for you. Don't paste it in chat.
  3. **You:** open your new bot and send it `hi`.
  4. **Claude:** runs `.venv/bin/python -m acos.telegram_setup`, puts the chat ID it prints into config, and sends a test message.

**Claude then sends a test ask:**

```
echo '{"type": "note", "text": "install test"}' > local/test_action.json
.venv/bin/python -m acos.approvals.cli send --summary "Install test" --question "Reply YES with the code to confirm approvals reach you." --action-file local/test_action.json
```

**You:** reply `YES` and the code.

**Claude:** runs `.venv/bin/python -m acos.approvals.cli poll`.

**Success looks like:** the poll prints `done`, and you get a "Done" message back.

## 6. Schedule it

**Claude:** runs `.venv/bin/python -m acos.schedule`. It writes one schedule file per job and prints the commands that turn them on.

**You:** say yes before Claude runs those commands. On a Mac, if asked, allow the Terminal or Claude Code to run in the background.

**You, on a laptop:** in System Settings → Battery → Options, turn on **Prevent automatic sleeping on power adapter when the display is off**. Keep it plugged in overnight. Jobs that fall due while it's asleep run the next time it wakes.

**Success looks like:** `launchctl list | grep acos` (Mac) or `crontab -l` (Linux) shows every job.

## 7. Trial week (shadow mode)

For seven days nothing changes in your inbox:
- Each night, the mail sort writes a report of what it would have done to `local/reports/`.
- Meeting prep packets arrive marked `[shadow]`.

Each morning, skim the report with Claude and correct anything wrong, like "that sender is a person" or "keep that newsletter". Claude turns each correction into a setting in `local/config.yaml`.

## 8. Switch on

**You:** say which modules to switch to live. Mail sorting first is the usual order.

**Claude:** changes `modules.<name>` from `shadow` to `live` in `local/config.yaml`.

## 9. Final check

- [ ] `acos.auth check` shows every inbox `ok`
- [ ] A test approval went out and came back `done`
- [ ] Last night's mail report exists in `local/reports/`
- [ ] A meeting prep packet arrived for a meeting with an outside attendee
- [ ] Every job is listed in the scheduler
- [ ] `local/logs/` has no errors from the last day

---

## Troubleshooting

| What you see | What to do |
| --- | --- |
| No report this morning | The laptop slept or was off. Check step 6's sleep setting. The next run catches up on the last 2 days. |
| "isn't connected or the sign-in expired" | Run `acos.auth connect <inbox>` again. Personal Gmail expiring weekly means step 3's **Publish app** was skipped. |
| The test ask never showed up (email channel) | It's an email from you to you, and some mail apps fold those into Sent. Search for "Ask" plus the code. To avoid it, set `channel.send_from` to a different connected inbox than the one you read. |
| An approval reply did nothing | It has to be YES or NO plus the exact code, sent within 24 hours, in reply to the ask. `approvals.cli list` shows each ask's status. |
| Briefing or packet went to spam | Mark it "not spam" once. It comes from your own address. |
| Something got archived that shouldn't have | Find it under its COS label and move it back. Tell Claude the sender so the rule is fixed. |

**Turning everything off:** `.venv/bin/python -m acos.schedule --remove` prints the commands to stop every job. Nothing is deleted. To revoke access too, remove the app at myaccount.google.com/permissions or account.microsoft.com/privacy/app-access.
