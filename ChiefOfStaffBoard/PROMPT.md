# The prompt

Copy everything inside the block below into Claude Code. See [README.md](README.md) for what it builds.

````markdown
Build me a "Chief of Staff Board": a private, auto-refreshing dashboard published as an Artifact, generated from my own planning files, that shows what needs my attention by strategy and lets me leave notes that you apply on your next run.

## 1. Data layer (set up whatever I don't already have)
- **Activity log** (`log.yaml`, append-only). Entries: `ts` (YYYY-MM-DDTHH:MM), `type` (update | decision | observation | nag | action_item | intent), `category`, optional `project`, `note` (start with a `[Label]` tag), and for action items: `owner`, `due`, `status: open|done`, `action: decide|do`.
- **Closing items without editing history:** to close an entry, append an `update` with `resolves: ["<ts>|<first 40 chars of its note>"]`. Also treat `status: done` or a `[SUPERSEDED]`/`[CLOSED]` tag as closed.
- **Project registry** (`projects.yaml`): `name`, `category`, `status` (active|paused|done), `started`, `last_touched`, `summary`, optional `aliases` (other labels that map to it), optional `parent` (child projects roll up to their parent strategy).
- **Strategy briefs** (`strategy_briefs.yaml`): one status per top-level strategy, **5 sentences max**, plain language, current state plus the next step.
- Optionally: my task manager export and any tool-specific queues. Ask me what I use.

## 2. Builder script (`dashboard_build.py`)
A Python script that reads those files and writes one static `dashboard/index.html`. No build step. Before writing, it runs `node --check` on every inline `<script>` and refuses to write if one doesn't parse, since a single stray quote silently kills every button.

Map each log entry to a strategy by `project`, then its `[Label]` tag, then aliases, then a keyword match on the first ~120 characters. Anything unmatched goes to an "Everything else" card.

**Page layout, top to bottom:**
1. **Header:** the title, "Updated <time>", and a **Sync now** button.
2. **Summary strip:** counts for To decide, To do, Overdue, and Done this week.
3. **Activity chart, like a GitHub contribution graph:** one row per strategy and one square per day for the last 6 weeks. The shade of blue is the number of log entries that day, on 5 levels. Hovering shows the count and date, and clicking a row jumps to that strategy's card. Below it, one chip per strategy: **green "on track"** or **red "N overdue."**
4. **Area filter chips.**
5. **Strategy cards, grouped under area headers** (all of one area's strategies together), with the areas holding the most urgent items first. Each card has:
   - its name, area, and an **Update status** button
   - the brief (5 sentences max)
   - three **collapsed dropdown lanes**: **Decide** (choices only I can make), **Do** (tasks), and **Done** (updates and decisions from the last 7 days). Each lane summary shows its count and an "N overdue" or "due today" flag.
   - each item shows its first sentence, with a "More" toggle for the rest
   - open items older than 14 days with no due date collapse into "N older items, still open?"
6. **"Not tied to a strategy":** an Everything else card, plus task-manager items due within 3 days.
7. **Quiet strategies** (collapsed): the ones with no open items and no activity this week.
8. **Infra Moves** (collapsed, at the very bottom): tooling and plumbing entries, meaning anything labeled infra or housekeeping. Keep these out of the main cards.

Decide vs. Do: use `action:` if set. Otherwise it's Decide only if the text contains decision words like "decide," "whether to," or "should we." Default to Do.

**Design:** a utilitarian working board, not a landing page.
- Color tokens on `:root`, with dark-mode overrides via `@media (prefers-color-scheme: dark) :root:not([data-theme="light"])` and `:root[data-theme="dark"]`.
- Semantic colors: purple for Decide, amber for due today, red for overdue, green for done and on-track, and a blue scale for the heatmap.
- Tabular numbers. Works at phone width, and the heatmap scrolls sideways inside its own container.

## 3. Comments that write back (`db` capability)
Publish with `capabilities: {db: {}}`. Load the artifact-capabilities skill first and follow its call contract.
- Every open item gets **Done** and **Comment** buttons. Every card gets **Update status**. Give each item a stable `data-key` (for log items, `log:<ts>|<first 40 chars>`).
- A click opens an inline textarea. Saving writes a doc to the collection `inbox`: `{key, strategy, item, kind: done|comment|status|sync, text, status: "new", created}`. **Sync now** writes `kind: "sync"`.
- Subscribe once to `inbox where status == "new"`. Mark items that have pending notes, and show a banner: "N notes waiting; the next run applies them."
- Keep the buttons hidden until `claude.use("db")` resolves, so the page is read-only where the capability isn't available.

## 4. The processing loop (add this to my planning skill, or create one)
On every run, **first** read the inbox (`ArtifactData` query, `status == "new"`) and apply each note:
- `done`: append a `resolves` update, or complete the task in the task manager.
- `comment`: log it (as an update, decision, or new action item) and act on it if it's an instruction.
- `status`: rewrite that strategy's brief.
- `sync`: a request for a full run.

Then mark each doc `{status: "applied", applied: "<one line on what you did>"}`, pinning `if_version`.

**Last**, rerun the builder and republish **to the same URL.** Always pass `url` so a new conversation doesn't create a second artifact. Record the URL in the skill.

## 5. Guardrails
- The page is private. Never put secrets in it.
- Confirm with me before any irreversible or outward-facing action a note asks for.
- After the first publish, do one functional check: list the `inbox` collection and tell me what you verified.

Start by asking me where my existing plans, tasks and projects live, and what my 4–8 top-level areas are. Then build the files, the builder and the page, publish it, and show me the link.
````

## Tips

- **It works best alongside a planning skill you already run daily**, like [`/cos`](../cos_claudecode), so the board refreshes itself on every run.
- **Sync now only queues a request.** Running your planning skill automatically needs a scheduled job on your own machine, which is a separate decision.
- **The `db` capability depends on your account.** Without it, you still get a fully useful read-only board, just without the buttons.
