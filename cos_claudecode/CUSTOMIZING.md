# Customizing /cos

The default `skill.md.template` gets you running. The actual value comes from making it know your life well enough that its recommendations fit.

This guide is a menu of patterns to add. Pick what's relevant; ignore the rest.

## The principle

A real chief of staff knows your calendar, your priorities, your relationships, your obligations, and your patterns. They give advice grounded in that context. A generic productivity agent gives advice grounded in nothing.

Every section you add to `skill.md` makes the agent's recommendations more specific to your reality. The reverse is also true — a sparse skill produces sparse, generic output.

## Things worth customizing

### 1. Energy curve

Most people aren't equally productive across the day. Tell the agent when to suggest deep work, when to schedule calls, and when to leave you alone.

Example:

```markdown
| Time | Energy | Best Use |
|------|--------|----------|
| 5–8am | Quiet | Personal writing, reading, journaling — no meetings ever |
| 9am–1pm | Peak | Deep work, hard decisions, strategic thinking |
| 1–3pm | Recovery | Lunch, walks, low-stakes admin |
| 3–5pm | Collaborative | Calls, meetings, async messages |
| 5–8pm | Family | Kid pickup, dinner, homework. Non-negotiable. |
| 8–10pm | Second wind | Code work, planning, deep focus on side projects |
```

The more specific you are, the better. Include rules:
- "Coffee meetings only on Thursday/Friday"
- "Never schedule deep work on Mondays" (or whichever day is your meeting day)
- "Travel days are reduced capacity — flag conflicts"

### 1a. Transport / logistics windows

A recurring drop-off-and-wait obligation (kid activity, appointment, errand you have to sit through) isn't dead time — it's a usable pocket the agent should surface, not just note as "busy."

If you have a recurring commitment with a real gap in the middle (drop off, wait, pick up), tell the agent to mine it:

```markdown
**[Activity] transport = work window.** When [you] do [activity] drop-off/pickup, the time between them is usable — from the car, a nearby coffee shop, wherever. Treat it as: [drive time] + drop-off → [gap] min of available time (async work, email, errands, lighter tasks) → [drive time] + pickup. Flag this as a usable pocket in the schedule, especially for tasks that don't require a desk.
```

This only pays off if the agent can also tell when *you're* not the one doing the transport. If you share logistics with a partner/co-parent and use event color (or some other marker) on a shared calendar to signal who's driving, pair this with a rule like:

```markdown
When presenting logistics in briefings, only show transport events you're responsible for. If [partner] is handling a drop-off, don't list it as your commitment — just note "[partner] has it covered" so you know the window is free, but don't claim the *work* window for yourself unless you're the one driving.
```

Without that pairing, the agent will credit you with usable time from a window someone else is covering, which produces a schedule that looks free but isn't.

### 2. Strategic priorities

5-7 things you actually care about for the year. Specific, not generic.

Bad: "Grow the business."
Good: "Get DGX Spark operational and run Llama training on customer data by end of Q2."

The agent uses these to filter recommendations. If you have 30 open tasks and the priority list says nothing about marketing, the agent won't push you to work on marketing items unless they're tied to a hard deadline.

### 3. Relationship tracking

The people the agent should monitor for you. Include:
- **Who** — name and role
- **Why** — what makes them important
- **Cadence** — how often you should be in touch
- **Watch for** — what would trigger concern (e.g., "no 1:1 in 2+ weeks beyond ops")

Example:

```markdown
| Person | Context | Watch For |
|--------|---------|-----------|
| **Pat** | Co-founder | Strategic relationship beyond standups. Flag if no real 1:1 in 2+ weeks |
| **Sam** | Direct report | Weekly 1:1, monthly career conversation |
| **Jane** | Best friend | Personal — flag if 3+ weeks without contact |
| **Dr. Lee** | Therapist | Biweekly Tuesday 4pm — confirm before each session |
```

The agent will check these against your calendar and session log and tell you when relationships are going dark.

### 4. Recurring obligations & deadlines

This is where the agent earns its keep. The things you keep meaning to track but don't.

Examples to customize:

**Tax / compliance** (if you run a business):
```markdown
- **Mar 15**: CA Form 100 + Federal 1120 due Apr 15 — confirm bookkeeper has Q4
- **May 15**: CA Q2 estimated tax due Jun 15
- **Aug 15**: CA Q3 estimated tax due Sep 15
- **Nov 15**: CA Q4 estimated tax due Dec 15
- **Dec 1**: WY Annual Report due Jan 1
```

**Travel benefits / credit card credits** (if you maximize):
```markdown
- **Monthly (1st)**: Use Amex Gold Grubhub credit ($20/mo)
- **Quarterly (Jan/Apr/Jul/Oct 1)**: Hilton credit on Biz Plat ($50/qtr)
- **Mar 15, Jun 15, Sep 15, Dec 15**: Mid-quarter check on use-or-lose credits
- **Nov 1**: Year-end push — surface remaining annual credits before Dec 31
```

**Content cadences** (if you write/podcast):
```markdown
- **Tuesday column**: Submit to editor by Monday morning. Thursday/Friday = pick next week's topic. Sunday = nag if nothing chosen.
- **Monthly newsletter**: First Tuesday of month, draft by previous Friday
```

**Recurring family logistics**:
```markdown
- **Wednesday afternoons**: Pick up kid from soccer practice
- **Every other Friday**: Coordinate with co-parent for weekend logistics
```

The pattern: **the more specific the trigger and the action, the more useful the reminder.**

### 5. Tone & behavior

How you want the agent to talk to you. Some people want soft. You probably don't want soft if you're using a tool called "Chief of Staff."

Things worth defining:
- "Be direct and opinionated. Push back when I'm wrong."
- "Don't sugarcoat. If I have 33 unfinished tasks in one area, say so."
- "When I'm thinking strategically: narrative and trade-offs. When I'm executing: bullets and next steps."
- "If something keeps getting deferred, name the pattern, not just the missed task."
- "Flag overconfidence gently. Ask if I want you to pressure-test."
- "Sounding board, not solution machine. Reflect back. Ask the sharper question."

### 6. Modes you actually use

The default skill defines `today`, `week`, `review`, `focus`, and `update`. Add or remove based on what you actually run.

Maybe you want a `morning` mode that runs differently than `today`. Maybe you want a `meeting <name>` mode that pulls all context for a specific upcoming meeting. Maybe you want a `vacation` mode that checks for things needing pre-action before a trip.

The skill is just markdown. You can add sections.

### 7. Custom data sources

The default reads:
- `session_log.yaml` (your memory)
- `tasks_snapshot.json` (Google Tasks)
- `gcal_snapshot.json` (Google Calendar)
- File modification times in your projects directory

You can add more. Examples:
- A `projects.yaml` you maintain by hand with project status
- Stripe data via `mcp__claude_ai_Stripe__*` tools
- Email triage via `mcp__claude_ai_Gmail__*` tools
- HubSpot CRM data
- Your own scraped/exported data from any tool

For each new source, add a section under "Data Sources" that tells the agent what's there and how to use it.

## Iterating on it

Don't expect the first version to be right. Treat the skill like a living document.

After each `/cos` run, ask yourself: "What did the agent miss? What did it overweight? What did it not know that it should have?" Then add that to the skill.

You'll find that after 2-3 weeks of tuning, the agent starts feeling genuinely useful — not because it's smarter, but because it knows enough about your specific life to give specific advice.

## Things NOT to put in the skill

- **Secrets** — never put API keys, passwords, or credentials in `skill.md`. Use environment variables or a separate `creds.json` that's gitignored.
- **Sensitive personal info** that you wouldn't want leaked — health details, financial account numbers, etc. The skill markdown is read into Claude's context every time you run `/cos`. Treat it like a doc you'd be uncomfortable having leaked.
- **Temporary state** — if it's only relevant for a week, put it in `session_log.yaml` instead of the skill itself.

## Sharing back

If you build a useful pattern (a new mode, a clever data source, a customization that other people would benefit from), consider PRing it back to this repo with the personal data scrubbed out. The community gets better when people share.
