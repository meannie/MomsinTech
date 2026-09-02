# Receipt and vault format

Use `work-receipts.md` as the portable source of truth. Markdown is intentionally readable by people and importable into hosted AI projects without extra software.

## Vault header

Preserve this small metadata block near the top:

```yaml
vault: work-receipts
format_version: 1
updated: YYYY-MM-DD
```

## Receipt template

Use headings and omit optional fields that add no value.

```markdown
## WR-YYYYMMDD-01 — Short descriptive title

- **Date:** YYYY-MM-DD
- **Organization/project:** Name or Private
- **Category:** engineering | product | design | data | leadership | sales | operations | community | other
- **Visibility:** private | internal-safe | public-safe | review-needed
- **Status:** complete | follow-up
- **Contribution:** What the user specifically did
- **Outcome:** What changed or became possible
- **Evidence:** Observed or qualitative evidence
- **Scope:** People, customers, revenue, systems, geography, or duration affected
- **Collaborators:** Team context needed for accurate ownership
- **Evidence quality:** observed | estimated | attributed | pending
- **Evidence source:** Dashboard, customer feedback, decision log, manager feedback, estimate, or unknown
- **Follow-up:** One specific unanswered question, or none

**Context:** One short paragraph preserving the situation, constraints, and useful detail.
```

IDs combine the receipt date with a two-digit sequence. Never renumber existing receipts. If the date is unknown, use `WR-UNDATED-01` and increment the sequence.

## Updating the vault

- Update the header's `updated` date.
- Add new receipts above older receipts unless the vault already uses another consistent order.
- Preserve user edits and unrecognized fields.
- Edit an existing receipt rather than creating a duplicate when the user supplies follow-up evidence.
- Keep private details in the vault unless the user asks for a redacted version.

When file tools are unavailable, return the entire updated vault, not only the new entry. Label it clearly as the replacement contents for `work-receipts.md`.
