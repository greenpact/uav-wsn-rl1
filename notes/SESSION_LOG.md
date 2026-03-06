# Session Log

Purpose: preserve key prompts, decisions, and outcomes across Codespace restarts.

How to use:
- At the start of a work session, add a new entry block.
- Capture important prompts and final decisions, not every chat line.
- Commit this file when you want a durable project memory in git.

## Entry Template

Date: YYYY-MM-DD
Codespace/Context: <optional>
Focus: <what you are trying to do>

Prompts:
- <important user prompt>
- <important follow-up prompt>

Decisions:
- <decision made and reason>

Actions Taken:
- <files changed / commands run>

Open Items:
- <next tasks or questions>

---

## Log Entries

Date: 2026-03-06
Codespace/Context: Initial setup
Focus: Persist chat context across restarts

Prompts:
- "how can i retrieve the last prompt from previous session?"
- "yes" (confirm setup)

Decisions:
- Keep a repository-tracked session log under `notes/SESSION_LOG.md`.

Actions Taken:
- Created this file with a reusable template.

Open Items:
- Optionally add a helper prompt/command to append entries faster.
