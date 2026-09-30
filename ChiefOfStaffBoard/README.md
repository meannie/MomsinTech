# Chief of Staff Board

A private, self-refreshing dashboard for Claude Code that shows what needs your attention, organized by strategy. It's published as a Claude Artifact and built from your own planning files, and you can leave notes on it that Claude applies the next time it runs.

Built by [Annie Tsai](https://annietsai.co) for [Moms in Tech](https://momsintech.com). It's the visual companion to [`/cos`](../cos_claudecode), but it works with any planning skill or log you already keep.

## Why it exists

A terminal briefing is great until you find yourself asking Claude to reprint the same open items three times a day. The board keeps that picture on one page, so you can check it from your phone and act on it without starting a new conversation.

## What you get

- **A summary strip** with counts for To decide, To do, Overdue, and Done this week.
- **An activity chart** in the style of a GitHub contribution graph: one row per strategy, darker blue on days with more logged activity. Under it, a green or red chip for each strategy shows whether anything is overdue.
- **Strategy cards, grouped by area**, each with a status of five sentences or fewer and collapsible **Decide / Do / Done** lanes. Open items that sit for weeks roll up into "older items, still open?" so they don't clutter the view.
- **Buttons that write back.** Done, Comment, and Update status write to the artifact's small database. Claude reads those notes at the start of its next run, applies them to your logs, and marks them applied. **Sync now** queues a full refresh.
- **Infra Moves** at the very bottom, so tooling and plumbing work stays out of your way.

## How to use it

1. Open Claude Code in the folder where you keep your planning files, or in a new folder if you're starting from scratch.
2. Paste the prompt in [PROMPT.md](PROMPT.md).
3. Answer its questions about where your plans and tasks live and what your top-level areas are.
4. It builds the data files, a small Python builder script, and the page, then publishes it and hands you a private link.

From then on, every run of your planning skill refreshes the same link.

## Requirements

- Claude Code with Artifacts enabled.
- Python 3 with PyYAML, and Node (used only for a syntax check before publishing).
- The Artifact `db` capability for the write-back buttons. If your account doesn't have it, you still get a fully useful read-only board.

## Notes

- **The page is private by default.** Keep secrets out of your logs anyway, since the board renders what's in them.
- **Sync now only queues a request.** Having it trigger a run on its own needs a scheduled job on your machine. That's worth deciding on deliberately rather than turning on by default.
- **Notes left on the board are instructions to Claude**, so the prompt tells it to confirm with you before doing anything irreversible or outward-facing that a note asks for.

## License

MIT
