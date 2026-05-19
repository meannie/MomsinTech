# /wr — Work Receipts for Claude Code

A Claude Code skill that captures the metrics and outcomes of your work **in real time** — before you forget them.

Built and open-sourced by [Annie Tsai](https://annietsai.co) (COO, [Interact](https://tryinteract.com) and [Moms in Tech](https://momsintech.com)).

## The problem

Most people remember *what* they shipped. They forget the numbers. By the time a layoff hits, a review comes around, or a recruiter calls — the specifics are gone. Work Receipts fixes that by prompting you to log wins with enough detail to turn them into resume bullets, interview stories, and LinkedIn content — on demand.

## Who this is for

- Tech workers who've been laid off and need to reconstruct their impact
- Anyone preparing for a performance review
- People job searching while still employed
- Consultants and freelancers building case studies
- Anyone who has ever stared at a blank resume and thought *"I know I did good work, I just can't remember the specifics"*

## What it does

| Command | What it does |
|---|---|
| `/wr` or `/wr log` | Capture a new win. Skill prompts for any missing metrics. |
| `/wr resume` | Format your entries as resume bullets. |
| `/wr interview` | Format entries as STAR stories for interviews. |
| `/wr linkedin` | Draft a LinkedIn post or weekly outcomes narrative. |
| `/wr list` | Show all logged entries. Incomplete ones are flagged. |
| `/wr nudge` | Daily reflection prompt: "What moved a needle today?" |

## Setup

### 1. Install Claude Code

[Claude Code](https://claude.com/claude-code) is Anthropic's CLI for Claude. Install it and make sure you have an active Claude subscription.

### 2. Add the skill

Copy the `WorkReceipts/` folder into your Claude skills directory:

```bash
cp -r WorkReceipts/ ~/.claude/skills/work-receipts/
```

### 3. Create your receipts file

Work Receipts stores everything locally. It never leaves your machine.

```bash
touch ~/work_receipts.yaml
```

You can put this anywhere — update the `RECEIPTS_FILE` path at the top of `skill.md` to match.

### 4. Use it

In Claude Code, type `/wr` to start logging.

## What gets stored

Each entry captures:

```yaml
- id: wr_001
  ts: 2026-05-19
  company: Acme Corp
  title: Redesigned onboarding flow
  description: Led redesign of the new user onboarding experience
  metrics:
    before: 34% 7-day activation rate
    after: 51% 7-day activation rate
    delta: +17pp
  scope: 12,000 new users/month
  role: sole designer + PM, eng team of 3
  category: product
  status: complete   # or: needs_metrics
```

`needs_metrics` entries stay alive — the skill resurfaces them and asks you to fill in the numbers once you have them. **Not having the number today is not a blocker.**

## Output examples

**Resume bullet**
> Redesigned new user onboarding flow, improving 7-day activation rate from 34% to 51% (+17pp) across 12,000 monthly new users

**STAR interview story**
> *Situation:* Our 7-day activation rate was 34% — most new users weren't reaching the first value moment...
> *Task:* I owned the full redesign of the onboarding flow...
> *Action:* Ran moderated user sessions, identified 3 drop-off points, redesigned the flow in Figma...
> *Result:* Activation improved to 51% within 6 weeks — a 17-point lift

**LinkedIn hook**
> I shipped something last quarter that I almost forgot to document. 7-day activation up 17 points. Here's why that almost didn't make it onto my resume — and how I'm fixing that.

## Integrating with the Personal COS Skill

If you're using the [/cos Chief of Staff skill](https://github.com/meannie/MomsinTech/tree/main/cos_claudecode), you can wire Work Receipts in so that every `/cos today` morning run automatically prompts for yesterday's wins.

### Part 1: Daily morning capture (`/cos today`)

Add the following to your COS `skill.md` after the **Suggested Focus** block in the `today` mode output template:

```
### Work Receipts — Morning Capture
At the end of every `/cos today` run, append this section after Suggested Focus:

---
Look at yesterday's session log entries (type: update), completed tasks, and recent file
activity. Identify 1-2 things the user completed yesterday that could have measurable
business impact — shipped features, closed deals, improved metrics, led a decision,
unblocked a team, etc.

Surface them by name:
  "Yesterday you [specific thing from session log]. Worth logging metrics on that one?"

If nothing specific is identifiable from the log, ask openly:
  "Anything you shipped or moved yesterday worth capturing before the week buries it?"

If the user engages: run the /wr log flow inline — ask for missing metrics (baseline,
outcome, scope, role), save to ~/work_receipts.yaml, confirm saved.
If the user says skip or nothing: drop it. Do not nag. Do not re-surface the same
item the next day unless it's still showing as a recent completion.
---

Do NOT run this prompt on /cos week, /cos review, or /cos focus.
Only on /cos today.
```

### Part 2: Weekly review synthesis (`/cos review`)

Add the following to your COS `skill.md` inside the `review` mode output template, after the Category Health Dashboard:

```
### Work Receipts — This Week's Wins

Read ~/work_receipts.yaml. Filter for entries where ts is within the last 7 days.
If no entries in the last 7 days: output a single line —
  "No Work Receipts logged this week. Morning prompts will continue."
  Then move on. Do not editorialize.

If entries exist, output two blocks:

---
**Resume bullets**
One bullet per complete entry (status: complete). Format:
  [Action verb] [what you did] [quantified outcome] [scope if available]
Keep each bullet to one line. Lead with the metric if it's strong. Flag entries with
status: needs_metrics separately:
  "⚠ [title] — metrics still missing. Fill in before using on a resume."

**LinkedIn this week**
Write 3-5 sentences in first person that connect this week's wins into an outcomes
narrative. Not a list — a story. Find the thread: were these all about speed?
customer impact? a product bet paying off? Lead with the most interesting result.
End with a hook — a question, a tension, or an implication that would make someone
want to comment. This should feel like the opening of a LinkedIn post, not a review.
---

Do this section for /cos review only.
```

## Privacy

Everything stays local. `work_receipts.yaml` is your file, on your machine. Nothing is sent anywhere except to Claude for processing — and Claude doesn't retain it.

Add `work_receipts.yaml` to your `.gitignore` if you version-control your skills directory.

## Philosophy

**Not having the number today is not a blocker.** Log the intent. The skill flags it as `needs_metrics` and resurfaces it. Most metrics become visible within a few weeks — dashboards update, A/B tests conclude, quarters close.

**Log liberally.** A feature that seemed minor often has outsized metrics six weeks later. You can always delete. You can never recover what you didn't write down.

## Contributing

Open source under MIT. PRs welcome.

---

Built by [Annie Tsai](https://linkedin.com/in/annietsai) · Inspired by every laid-off tech worker who couldn't remember their own metrics.
