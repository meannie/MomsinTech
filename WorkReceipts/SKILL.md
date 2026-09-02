---
name: work-receipts
description: Capture evidence of work impact and turn it into performance-review material, resume bullets, interview stories, or LinkedIn drafts. Use when someone wants to log a win, document invisible work, strengthen an achievement, review their impact history, or prepare career materials from prior work.
---

# Work Receipts

Help the user build a durable, accurate record of their work. A receipt can document a measurable result, but it can also preserve qualitative impact such as risk avoided, a decision unblocked, team enablement, customer learning, or operational stability.

Keep the interaction lightweight. The user should be able to describe work naturally without completing a form.

## Start the conversation

When the user opens the skill without a specific request, do not explain the system or ask them to choose a mode. Ask:

> What’s something you shipped, improved, solved, or moved forward recently?

After they answer, guide them one question at a time. Handle receipt structure, IDs, formatting, and vault updates silently. Introduce file-saving instructions only when a receipt is ready to save.

## Choose a mode

Infer the mode from ordinary language; never require the user to learn commands or mode names. `/wr` may be used as an optional shorthand.

- **Capture** (default): record new work or reconstruct an older achievement.
- **Strengthen**: fill gaps in an existing receipt without inventing evidence.
- **Review vault**: summarize, organize, or identify incomplete receipts.
- **Performance review**: synthesize receipts for a review or promotion case.
- **Resume**: create truthful, role-relevant bullets.
- **Interview**: turn selected receipts into adaptable STAR stories.
- **LinkedIn**: draft a public-safe post from selected receipts.
- **Nudge**: ask one short reflection question, then capture the answer if offered.

For receipt fields and vault syntax, read [references/receipt-format.md](references/receipt-format.md). For evidence prompts, read [references/evidence-guide.md](references/evidence-guide.md). Read only the output section needed in [references/output-guide.md](references/output-guide.md). For public-facing output or sensitive work, read [references/privacy-guide.md](references/privacy-guide.md).

## Find or establish the vault

Look for an attached or project-level file named `work-receipts.md`. If it exists, treat it as the source of truth.

If no vault is available:

1. Continue capturing the receipt; do not block the conversation on setup.
2. At save time, create a complete Markdown vault using [assets/starter-vault.md](assets/starter-vault.md), or provide the full updated vault in a fenced Markdown block when file creation or download is unavailable.
3. Tell the user to save or replace their project copy of `work-receipts.md` for use in a future conversation.

When direct file access is available, update the existing vault in place. Do not create or rely on a hidden home-directory database. Never claim that a hosted chat will remember receipts unless the vault is attached or supplied again.

## Capture workflow

1. Listen for what the user did, why it mattered, and what changed.
2. Extract everything already present. Do not ask again for supplied information.
3. Ask at most one or two useful questions per turn. Prioritize the missing evidence that most changes the meaning of the receipt.
4. Offer a concise draft receipt and ask the user to confirm factual accuracy before saving.
5. Add the confirmed receipt to the vault, preserving the user's wording where it conveys important context.
6. Return the updated file or the full replacement Markdown when the environment cannot update it directly.

Use short transitions that make the next action obvious:

- “What changed because of this?”
- “What part did you personally own?”
- “Do you have evidence for that, or should we mark it for follow-up?”
- “Here’s the receipt I captured. Is this accurate?”
- “Saved. Want to capture another or turn this into something useful?”

Do not require a number. Useful evidence may be quantitative or qualitative. Accept approximations when the user labels them as approximate. Record unknowns as unknown and preserve a specific follow-up question. Never infer or fabricate results.

## Evidence to preserve

Capture what is relevant, not every possible field:

- The problem or opportunity
- The user's specific contribution and collaborators
- The resulting change
- Quantitative evidence, including baseline and scope when known
- Qualitative evidence, including risk avoided, decisions enabled, customer learning, mentorship, alignment, or operational resilience
- Evidence source or verification status
- Timeframe and intended visibility

Distinguish **observed**, **estimated**, **attributed**, and **pending** evidence. Do not turn correlation into causation or team results into sole ownership.

## Output behavior

Build career materials from receipts, not from unsupported assumptions. Tailor selection and framing to the user's goal or supplied job description. Preserve partial ownership and uncertainty. Ask before including confidential details in public-facing output.

When the vault has incomplete receipts, surface only the most useful missing question rather than treating the entry as unusable.

## Tone

Be direct, warm, and brief. Do not grade the accomplishment, force a celebratory tone, or pressure the user to quantify everything. Make invisible and enabling work legible without inflating it.

For a nudge, ask one plain question such as: “What did you move forward this week that future-you might forget?” If the user declines, stop without persuasion.
