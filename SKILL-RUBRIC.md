# Skill Evaluation Rubric

Use this to evaluate whether a skill file is well-structured and ready for use. A good skill should pass all of these checks.

## Structure

- [ ] **Single responsibility.** The skill does one thing well. If it does two things (e.g., inbox triage AND email drafting), split it into two skills.
- [ ] **Clear trigger description.** The frontmatter `description` field makes it obvious WHEN this skill should activate. Include example phrases the user might say.
- [ ] **Segmented into small sections.** No wall of text. Use headers to break into: purpose, rules, workflow, data sources, tone, living document.
- [ ] **Priority hierarchy stated.** If the skill handles multiple types of work, explicitly state which type wins when they conflict.

## Content Quality

- [ ] **Rules are actionable.** Each rule says what to DO, not just what to think about. Bad: "Consider the user's energy." Good: "Morning = deep work. Afternoon = calls/admin. Don't suggest deep work after 3pm."
- [ ] **Anti-patterns included.** State what NOT to do, not just what to do. Include common mistakes the AI makes and how to avoid them.
- [ ] **Examples where ambiguous.** If a rule could be interpreted multiple ways, include an example that disambiguates.
- [ ] **No orphan knowledge.** Everything the skill needs to function is either in the skill file itself, or has a clear pointer to where to find it (file path, API endpoint, command to run).

## Operational

- [ ] **Data sources specified (read AND write).** If the skill needs external data, specify exactly how to get it AND where to persist state. Read: commands, endpoints, auth. Write: where does the skill log its output, update status, or record decisions? If there's no write path, the skill relies on memory across sessions — which is unreliable.
- [ ] **Never rely on memory.** The AI's memory across sessions is lossy. Any state that matters must be written to a file, session log, or external system. If the skill needs to "remember" something for next time, specify where it gets written.
- [ ] **Output format defined.** Specify what the output should look like — headers, structure, length. Don't leave it to the AI's default verbosity.
- [ ] **Error handling.** What should happen if a data source is unavailable? Say so explicitly rather than letting the AI silently skip it.
- [ ] **Living document clause.** Include a section explaining when and how to update the skill itself.

## Integration

- [ ] **Boundaries with other skills clear.** If this skill overlaps with another, state where one ends and the other begins.
- [ ] **Tone specified.** Even if brief — direct? warm? terse? Match the user's communication style.
- [ ] **Confirmation workflow defined.** For skills that take action (sending emails, creating files), specify when to ask for confirmation vs. proceed autonomously.

## The Living Skill Concept

Skills are not static documentation — they are living systems that evolve as you use them. Every skill should include guidance on how and when to update itself.

A good living skill section specifies:
- **What should evolve**: current priorities, key people, specific sources/channels, output tweaks based on what works
- **What should stay stable**: voice/style rules, trigger descriptions, structural format, core workflow logic
- **How updates happen**: the AI proposes a specific edit; the user confirms; the AI makes the change directly to the skill file. No silent edits. No drifting without consent.
- **When to watch for drift**: at the end of each run, note if a workstream is finished, a new one is dominating, a person is no longer relevant, or a data source has changed

The goal: every correction the user makes becomes a permanent rule. The skill gets smarter over time without the user needing to maintain a separate document of preferences.

## Red Flags (fail if any present)

- Contains PII (real email addresses, API keys, phone numbers)
- Has hardcoded dates that will become stale (use relative references or note update cadence)
- References files/systems that don't exist in this user's setup
- Longer than ~600 lines (split it)
- Duplicates guidance that's in another skill (single source of truth)
