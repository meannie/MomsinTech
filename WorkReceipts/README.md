# Work Receipts

Work Receipts is an open-source AI skill that helps people capture evidence of their work before review season, a job search, or an interview makes them reconstruct it from memory.

It records measurable outcomes **and** the work traditional achievement trackers miss: risk prevented, decisions unblocked, customer learning, team enablement, operational resilience, mentorship, and community impact.

Built by [Annie Tsai](https://annietsai.co) with [Moms in Tech](https://momsintech.com).

## Start here

After installing the skill, type:

> Help me capture a work receipt.

The assistant will ask what you worked on and guide you one question at a time. You do not need to know the receipt format, choose a mode, create an ID, or have a metric ready.

You can also begin with whatever is already on your mind:

> I finally got two teams to agree on the launch plan.

> I fixed a recurring customer issue, but I don’t know the numbers yet.

> Help me remember what I accomplished this quarter.

The assistant will turn the conversation into a draft receipt, show it to you for confirmation, and then help you save it.

## What you can ask

- “Help me capture something I shipped today.”
- “I unblocked a decision, but there isn't a metric. Is that still a receipt?”
- “Review my vault and find the strongest themes for my performance review.”
- “Turn the most relevant receipts into resume bullets for this job.”
- “Build interview stories without overstating my role.”
- “Make this receipt safe to share on LinkedIn.”

`/wr` remains a convenient shorthand where slash-style commands are supported.

## Install in an AI app

### Claude

Upload the skill ZIP in **Customize → Skills**, enable it, and type “Help me capture a work receipt.” Code execution and file creation must be enabled for Claude Skills.

### ChatGPT and Codex

Standalone skills work in the ChatGPT desktop app's Codex experience, Codex CLI, and the IDE extension. For regular ChatGPT chat on web, desktop, or mobile, Work Receipts will be distributed through a plugin containing this same skill.

If your AI app does not support skills, create a Project or equivalent workspace and add `SKILL.md` as its instructions. Also add `assets/starter-vault.md`, renamed to `work-receipts.md`, as project knowledge.

The portable vault is what carries context between conversations. Keep the private master copy somewhere you control, attach it to a new conversation or Project, and replace it with the updated version after capturing receipts.

## Install in a coding agent

Copy this directory into the agent's skills directory using the folder name `work-receipts`. Coding agents with file access can update `work-receipts.md` directly. The skill does not require Python, a database, or a home-directory YAML file.

## How persistence works

Work Receipts does not pretend that every chat has permanent memory:

1. Your receipts live in a readable `work-receipts.md` file.
2. The skill uses an attached or project-level vault when one is available.
3. In a hosted chat without file editing, it returns the full updated vault for you to save or replace.
4. In a coding environment, it can update the vault in place.

The Markdown format is deliberately portable between AI products and remains useful without an AI assistant.

## Privacy

Receipts may contain confidential workplace or personal information. New receipts default to private. Before public output, the skill checks for names, proprietary metrics, unreleased plans, customer data, personnel information, and other sensitive details.

AI processing and retention depend on the provider, account, workspace, and settings you use. Review those terms before uploading sensitive material. The skill cannot guarantee local-only processing or provider deletion.

## Repository contents

```text
WorkReceipts/
├── SKILL.md
├── README.md
├── LICENSE
├── assets/
│   └── starter-vault.md
├── examples/
│   ├── career-return.md
│   ├── engineering.md
│   ├── leadership.md
│   └── operations.md
└── references/
    ├── evidence-guide.md
    ├── output-guide.md
    ├── privacy-guide.md
    └── receipt-format.md
```

## Contributing

Contributions are welcome, especially examples representing overlooked work, different roles, career breaks, accessibility needs, and nontraditional career paths. Examples must be fictionalized and contain no employer, customer, or personal secrets.

Open source under the [MIT License](LICENSE).
