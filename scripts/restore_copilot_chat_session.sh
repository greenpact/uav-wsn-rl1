#!/usr/bin/env bash
set -euo pipefail

# Restores Copilot chat session resources from the most recent previous
# workspaceStorage slot into the active slot when the active slot is empty.

STORAGE_ROOT="${VSCODE_CHAT_STORAGE_ROOT:-$HOME/.vscode-remote/data/User/workspaceStorage}"
CHAT_SUBPATH="GitHub.copilot-chat/chat-session-resources"
FORCE_MERGE=0

if [[ "${1:-}" == "--force" ]]; then
  FORCE_MERGE=1
fi

if [[ ! -d "$STORAGE_ROOT" ]]; then
  echo "[restore-chat] workspaceStorage root not found: $STORAGE_ROOT"
  echo "[restore-chat] Nothing to restore."
  exit 0
fi

pick_active_workspace_dir() {
  local with_lock
  with_lock=$(find "$STORAGE_ROOT" -mindepth 2 -maxdepth 2 -name "vscode.lock" -printf '%h\n' 2>/dev/null | xargs -r ls -1dt 2>/dev/null | head -n 1 || true)
  if [[ -n "$with_lock" ]]; then
    printf '%s\n' "$with_lock"
    return
  fi

  # Fallback: most recently modified workspace storage directory.
  ls -1dt "$STORAGE_ROOT"/* 2>/dev/null | head -n 1 || true
}

ACTIVE_DIR="$(pick_active_workspace_dir)"
if [[ -z "$ACTIVE_DIR" || ! -d "$ACTIVE_DIR" ]]; then
  echo "[restore-chat] Unable to locate active workspace storage directory."
  exit 1
fi

ACTIVE_CHAT_DIR="$ACTIVE_DIR/$CHAT_SUBPATH"
mkdir -p "$ACTIVE_CHAT_DIR"

active_session_count=$(find "$ACTIVE_CHAT_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
if [[ "$active_session_count" -gt 0 && "$FORCE_MERGE" -ne 1 ]]; then
  echo "[restore-chat] Active chat session directory already has data ($active_session_count session(s))."
  echo "[restore-chat] Use --force to merge from older workspace storage entries."
  exit 0
fi

SOURCE_CHAT_DIR=""
SOURCE_SESSION_COUNT=0

while IFS= read -r candidate; do
  [[ "$candidate" == "$ACTIVE_DIR" ]] && continue
  candidate_chat_dir="$candidate/$CHAT_SUBPATH"
  [[ -d "$candidate_chat_dir" ]] || continue

  candidate_count=$(find "$candidate_chat_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
  [[ "$candidate_count" -gt 0 ]] || continue

  SOURCE_CHAT_DIR="$candidate_chat_dir"
  SOURCE_SESSION_COUNT="$candidate_count"
  break
done < <(ls -1dt "$STORAGE_ROOT"/* 2>/dev/null || true)

if [[ -z "$SOURCE_CHAT_DIR" ]]; then
  echo "[restore-chat] No previous non-empty Copilot chat session storage found."
  exit 0
fi

restored=0
skipped=0
while IFS= read -r session_dir; do
  session_name="$(basename "$session_dir")"
  destination="$ACTIVE_CHAT_DIR/$session_name"
  if [[ -e "$destination" ]]; then
    skipped=$((skipped + 1))
    continue
  fi

  cp -a "$session_dir" "$destination"
  restored=$((restored + 1))
done < <(find "$SOURCE_CHAT_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

echo "[restore-chat] Active workspace storage: $ACTIVE_DIR"
echo "[restore-chat] Source workspace storage: $(dirname "$(dirname "$SOURCE_CHAT_DIR")")"
echo "[restore-chat] Source sessions: $SOURCE_SESSION_COUNT"
echo "[restore-chat] Restored sessions: $restored"
echo "[restore-chat] Skipped existing sessions: $skipped"
