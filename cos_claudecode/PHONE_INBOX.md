# Phone Inbox: Log Anything from Your iPhone

The phone inbox lets you capture tasks, notes, intentions, and observations into your COS session log from your iPhone — without opening a laptop, without a bot, without a server.

The pattern: your iPhone posts a comment to a private GitHub Issue. On the next `/cos` run, a drain script pulls those comments and appends them to your session log. The sync repo (already handling your session log) carries the entries to your desktop.

---

## The use case

You're on the bus, in a meeting, or picking up kids. Something comes up that you want to remember:

- A task you committed to
- A decision that was made
- A project update you want to capture before you forget
- Something you want `/cos` to nag you about

On your phone: open Shortcuts → tap your COS shortcut → dictate or type the note → done. It's in your session log by the next `/cos` run.

---

## How it works end-to-end

```
iPhone Shortcut
   ↓  POST (JSON payload)
GitHub Issue #1 in your private sync repo
   ↓  fetched on every /cos run
inbox_drain.py
   ↓  parses + appends
session_log.yaml
   ↓  git push
Private sync repo → available on all devices
```

1. Your iPhone Shortcut captures input (voice or text)
2. The Shortcut POSTs a YAML-formatted comment to Issue #1 in your private sync repo
3. On every `/cos` run, `inbox_drain.py` fetches new comments since `last_seen_id`
4. Each comment is parsed as a session log entry and appended to `session_log.yaml`
5. `last_seen_id` is saved to `.inbox_state.json` — nothing is processed twice

---

## Setup

### Step 1: Create the inbox issue

In your private sync repo (the one holding `session_log.yaml`):

1. Go to the repo on GitHub → **Issues** → **New issue**
2. Title it "COS Phone Inbox" (the content doesn't matter, the issue number does)
3. Make sure it's **Issue #1** — the drain script reads comments from issue number 1 by default. If you already have other issues open, close them first or adjust the issue number in `inbox_drain.py`

### Step 2: Create a GitHub Personal Access Token

The Shortcut needs a token to post comments. Generate one with minimum permissions:

1. GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens**
2. Repository access: **Only select repositories** → your private sync repo
3. Repository permissions: **Issues** → **Read and write** (that's all you need)
4. Click **Generate token** → copy it immediately (shown once)

Save the token where `/cos` can read it:

```bash
echo "ghp_YOUR_TOKEN_HERE" > ~/cos/.github_pat
chmod 600 ~/cos/.github_pat
```

Make sure `.gitignore` includes `.github_pat` so it's never committed.

### Step 3: Set up `inbox_drain.py`

Copy `inbox_drain.py` to your cos directory and edit the constants at the top:

```python
GITHUB_OWNER = "YOUR_USERNAME"
GITHUB_REPO  = "YOUR_SYNC_REPO_NAME"
ISSUE_NUMBER = 1
PAT_FILE     = Path.home() / "cos" / ".github_pat"
STATE_FILE   = Path.home() / "cos" / ".inbox_state.json"
SESSION_LOG  = Path.home() / "cos" / "session_log.yaml"
```

Test it:

```bash
python3 ~/cos/inbox_drain.py
# Expected: "Drained 0 new entries." (nothing in inbox yet)
```

### Step 4: Add the drain to your skill.md

In your skill's Session Log section, add the drain as Step 2 — AFTER the git pull and cp, NEVER before:

```markdown
**Step 1 — Pull latest from GitHub:**
```bash
cd ~/YOUR_SYNC_REPO && git pull origin main --rebase 2>/dev/null
cp ~/YOUR_SYNC_REPO/session_log.yaml ~/cos/session_log.yaml
```

**Step 2 — Drain phone inbox (always after pull + cp):**
```bash
python3 ~/cos/inbox_drain.py 2>&1
```

If 1+ entries were drained, mention it in the briefing: "N new from phone since last run."
```

Ordering matters. See [RELIABILITY.md](RELIABILITY.md) for why.

---

## iPhone Shortcut setup

### Basic version (text input)

1. Open the **Shortcuts** app on your iPhone
2. Tap **+** to create a new shortcut
3. Add action: **Ask for Input**
   - Prompt: "What do you want to log?"
   - Input type: Text
4. Add action: **Get Contents of URL**
   - URL: `https://api.github.com/repos/YOUR_USERNAME/YOUR_SYNC_REPO/issues/1/comments`
   - Method: **POST**
   - Headers:
     - `Authorization` → `token YOUR_GITHUB_PAT`
     - `Content-Type` → `application/json`
     - `Accept` → `application/vnd.github+json`
   - Request Body: **JSON**
     - Add key: `body`
     - Value: tap the variable picker → select the **Provided Input** variable from step 3
5. (Optional) Add action: **Show Notification** → "Logged to COS"
6. Rename it "Log to COS" → Add to Home Screen

### Voice version (dictation)

Same as above, but in step 3 change Input type to **Dictation**. Your iPhone will open the dictation interface and transcribe before posting.

### Formatting the body for inbox_drain.py

The drain script accepts two formats:

**Structured (recommended):** Type or dictate raw YAML:

```yaml
type: intent
note: "pick up dry cleaning before 6pm"
category: Family
```

**Plain text fallback:** If the input doesn't parse as YAML, the drain wraps it automatically:

```yaml
- ts: "2026-06-16T14:30"
  type: intent
  category: Everything Else
  note: "[phone] whatever you typed"
```

You can dictate naturally ("pick up dry cleaning before 6pm") and it lands as an intent. For structured entries, type the YAML.

### Valid entry types and categories

| Field | Options |
|-------|---------|
| `type` | `intent`, `update`, `observation`, `nag` |
| `category` | Your category list from skill.md |

---

## Verifying the round-trip

1. Open Shortcuts on your iPhone → tap "Log to COS" → type "test from phone"
2. Check GitHub: your sync repo → Issues → #1 → new comment should appear
3. On your desktop, run any `/cos` mode
4. Look for "N new from phone since last run" in the output
5. The entry should appear in `session_log.yaml`

---

## Troubleshooting

**Shortcut fails with auth error** — Check that your GitHub PAT has Issues read/write permission on the correct repo. Regenerate if expired.

**Comment posts but drain doesn't pick it up** — Verify `GITHUB_OWNER`, `GITHUB_REPO`, and `ISSUE_NUMBER` in `inbox_drain.py`. Check `.inbox_state.json` has the correct `last_seen_id` (set to 0 to drain from the beginning).

**Entries appear twice** — The drain ran before the git pull/cp. Fix the ordering in your skill.md. See [RELIABILITY.md](RELIABILITY.md).

---

## Privacy note

Issue comments are not end-to-end encrypted. Entries are visible to anyone with access to the repo. Since the repo is private, that means you and any collaborators. Don't log credentials or information you wouldn't want in a private GitHub repo.
