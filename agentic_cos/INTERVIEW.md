# The interview (part 2 of /cos-onboard)

Claude runs this to fill in `local/config.yaml`. It follows the same ground rules as part 1:
- Ask two or three questions at a time, in conversation.
- Use `AskUserQuestion` only for closed choices.
- Suggest a sensible default for every question.
- Accept "skip" anywhere.
- Read the whole summary back before writing anything.

**Reuse part 1.** Name, timezone, energy map, priorities, people and the calendar list already live in the /cos skill and its `config.yaml`. Read them and confirm them; don't ask again.

Each round below lists what to ask, why it matters (say this in one sentence), and the config it fills.

---

## Round 1: Your inboxes

**Ask:**
- Which email addresses should it look after?
- For each one, is it Gmail or Google Workspace, or Microsoft 365 / Outlook?
- If they don't know which: it's Microsoft if they sign in at outlook.office.com or use Outlook for work.

**Say why:** each inbox gets its own one-time sign-in, and the two kinds sign in differently.

**Fills:** `inboxes:` with a short `id` per inbox (`personal`, `work`, ...), its `provider` (`gmail` or `m365`) and its `address`. The user's main address also becomes `user.address`.

## Round 2: What can be filed away

**Explain the six labels** in one breath:
- **Needs you** and **Bills** stay in the inbox.
- **FYI** stays too.
- **Receipts**, **Newsletters** and **Notifications** get archived. Archived mail stays searchable under its label; nothing is ever deleted.

**Ask:**
- Is it fine to archive receipts, newsletters and notifications? Most people say yes, since they read on their phone anyway. Any of the three can be kept in the inbox instead.
- Any newsletters you actually read and want kept visible? (A kids' activity club, a professional newsletter.) Get the sender's domain.
- Any senders that always need you, even when they look automated? (A school office, an accountant, a client domain.)
- Do you use a label or category for mail that must never be moved? (Fine to say no.)

**Fills:** `triage.archive`, `triage.keep_visible_domains`, `triage.always_needs_you_domains`, `triage.protected_labels`.

## Round 3: How to reach you

**Ask** (`AskUserQuestion`): where should urgent alerts and yes/no approval questions go?
- **Email**: they arrive in your own inbox; you reply YES or NO with a short code. Nothing else to set up.
- **Instinct**: your Instinct assistant gets the email and texts you; you text back. Needs your assistant's email address.
- **Telegram**: a private chat with your own bot. About 2 minutes to set up during install.

**Then say plainly:**
- Instinct and Telegram pass message text through their own servers.
- Telegram bot chats aren't end-to-end encrypted.
- So alerts are kept short, and the detail stays in your email.

**Also ask:**
- Morning briefings and meeting prep packets go to your email inbox. Which connected Gmail inbox should send them? It sends only to you.

**Fills:** `channel.type`, `channel.send_from`, `channel.assistant_address` (Instinct), `channel.chat_id` (Telegram, filled during install).

## Round 4: What it may do on its own

**Walk through four tiers** with an example each, and ask them to move anything they disagree with:

| Tier | Examples | Default |
| --- | --- | --- |
| Just do it | Label and archive mail per the rules above | Yes, after the trial week |
| Do it, then tell you | Morning briefing, meeting prep packets | Yes |
| Ask first (by text or email) | Add a task to your to-do list; save a note for your next session | Yes |
| Always yours | Sending anything to another person, spending, buying, signing, publishing, deleting | Never automated |

The last row can't be changed. If they ask for it, explain that the code has no way to do those things.

**Ask:** should approved asks be able to add Google Tasks? If yes, install has one extra sign-in.

**Fills:** `approvals.allowed_actions` (`note`, plus `task_add` if yes), `approvals.expiry_hours` (default 24).

## Round 5: Meeting prep

**Ask:**
- Should it make prep packets? Default: yes, for meetings with anyone outside your own email domains.
- Any words in a meeting title that always mean "prep this", even when it's internal? (board, interview, review)
- Any that mean "never prep this"? (focus, lunch, hold)
- Is there a folder of notes it should search for the people you're meeting? Get the path. Fine to skip.
- When do you want packets? Default: 7pm the evening before, plus a short reminder about an hour before each meeting.

**Fills:** `modules.meeting_prep`, `meetings.calendars` (from part 1's calendar list, tagged with the inbox each belongs to), `meetings.important_keywords`, `meetings.skip_keywords`, `meetings.notes_dirs`, `schedule.meeting_prep_evening`.

## Round 6: Other jobs

**Offer each with one line** (`AskUserQuestion`, multi-select). Each becomes a prompt in `local/prompts/` plus a schedule time.
- **Weekly status:** what moved, what stalled and what's due, per area of your life.
- **Monthly security check:** looks for passwords or keys saved in the wrong place.
- **Household check:** parties that need a gift, family birthdays, trips that need drive time and someone assigned.
- **A recurring piece of work:** describe it. Claude writes a prompt that does the prep and stops before anything goes out.

**Fills:** `schedule.custom`, `modules.<name>: shadow`, and one `local/prompts/<name>.md` each.

## Round 7: When it runs

**Explain:**
- Jobs run on this computer while it's on and awake.
- Overnight mail sorting needs the laptop plugged in with sleep turned off while charging.
- A cloud server option is coming later.

**Ask:** is the 11pm mail sort OK, or is another time better?

**Fills:** `schedule.triage`.

---

## Before writing

Show one summary grouped by round, with every value. Ask: "Anything to change?" Then write `local/config.yaml`, using `templates/config.example.yaml` as the shape. Every module starts at `shadow`, except `approvals: live` (it only ever acts on the user's own YES, so there's nothing to trial).

## Re-running

"Add an inbox", "change where alerts go", "stop meeting prep": read `local/config.yaml`, run only the matching round, show the change, write it. For a new inbox, connect it (INSTALL step 4). For a channel change, set it up (INSTALL step 5).
