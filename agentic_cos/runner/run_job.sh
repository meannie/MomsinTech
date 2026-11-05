#!/bin/bash
# Runs one scheduled job. The scheduler (launchd, cron or systemd) calls this.
#
#   runner/run_job.sh triage
#   runner/run_job.sh approvals
#   runner/run_job.sh meeting_prep_evening
#   runner/run_job.sh meeting_prep_soon
#   runner/run_job.sh <name>                    # any job with a prompt at local/prompts/<name>.md
#   runner/run_job.sh <name> --only-on 2026-12-01   # trial run: does nothing on other days
#   runner/run_job.sh <name> --once day|week|month  # skip if it already ran this period
#
# Every job is in one of three modes, set by the interview (config: modules.<job>):
#   off     does nothing
#   shadow  reports what it would do; changes nothing (the default for a new job)
#   live    does it
#
# Claude runs with no tools at all (--tools ""): the job's material is passed in as
# text, so nothing in an email can get it to open a file or call anything.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL="${ACOS_LOCAL:-$ROOT/local}"
PY="${ACOS_PYTHON:-$ROOT/.venv/bin/python}"
CLAUDE="${ACOS_CLAUDE:-claude}"
MODEL="${ACOS_MODEL:-sonnet}"
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH" ACOS_LOCAL="$LOCAL"

JOB="${1:?usage: run_job.sh <job> [--only-on YYYY-MM-DD] [--once day|week|month]}"; shift
ONLY_ON=""; ONCE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only-on) ONLY_ON="$2"; shift 2;;
    --once) ONCE="$2"; shift 2;;
    *) echo "unknown option $1"; exit 2;;
  esac
done

mkdir -p "$LOCAL/logs" "$LOCAL/state/locks" "$LOCAL/state/runs" "$LOCAL/out"
exec >>"$LOCAL/logs/$JOB.log" 2>&1
echo "=== $(date '+%Y-%m-%d %H:%M:%S') $JOB"
cd "$ROOT" || exit 1

TODAY=$(date +%Y-%m-%d)
if [ -n "$ONLY_ON" ] && [ "$ONLY_ON" != "$TODAY" ]; then echo "only runs on $ONLY_ON"; exit 0; fi
case "$ONCE" in
  day) PERIOD="$TODAY";; week) PERIOD="$(date +%G-W%V)";; month) PERIOD="$(date +%Y-%m)";; "") PERIOD="";;
  *) echo "--once must be day, week or month"; exit 2;;
esac
MARK="$LOCAL/state/runs/$JOB.$PERIOD"
if [ -n "$PERIOD" ] && [ -f "$MARK" ]; then echo "already ran for $PERIOD"; exit 0; fi

# One run at a time per job.
LOCK="$LOCAL/state/locks/$JOB"
if ! mkdir "$LOCK" 2>/dev/null; then echo "still running from before, skipping"; exit 0; fi
trap 'rmdir "$LOCK"' EXIT

MODE=$("$PY" -m acos.config get "modules.$JOB" shadow)
case "$JOB" in meeting_prep_*) MODE=$("$PY" -m acos.config get modules.meeting_prep shadow);; esac
[ "$MODE" = "off" ] && { echo "off"; exit 0; }

# compose <prompt file> <material file> <output file>: Claude reads the material as text, no tools.
compose() {
  { cat "$2"; } | "$CLAUDE" -p "$(cat "$1")" --tools "" --strict-mcp-config \
      --no-session-persistence --model "$MODEL" > "$3"
  [ -s "$3" ] || { echo "empty output from claude"; return 1; }
}

# send <digest|urgent> <subject> <body file>: shadow mode sends to the user only, marked as a trial.
send() {
  local subj="$2"; [ "$MODE" = "shadow" ] && subj="[shadow] $2"
  if [ -n "${ACOS_DRY_SEND:-}" ]; then echo "would send $1: $subj"; cat "$3"; return 0; fi
  "$PY" -m acos.deliver "$1" --subject "$subj" --body-file "$3"
}

STAMP=$(date +%Y%m%d_%H%M)
OUT="$LOCAL/out/${JOB}_$STAMP.txt"
case "$JOB" in
  triage)
    if [ "$MODE" = "live" ]; then "$PY" -m acos.triage.run --apply; else "$PY" -m acos.triage.run; fi || exit 1
    ;;
  approvals)
    [ "$MODE" = "live" ] || { echo "approvals in shadow mode: not polling"; exit 0; }
    "$PY" -m acos.approvals.cli poll || exit 1
    ;;
  meeting_prep_evening|meeting_prep_soon)
    WINDOW=evening; [ "$JOB" = "meeting_prep_soon" ] && WINDOW=soon
    MATERIAL=$("$PY" -m acos.meetings --window "$WINDOW") || exit 1
    [ -z "$MATERIAL" ] && { echo "no meetings need prep"; exit 0; }
    compose "$ROOT/runner/prompts/meeting_prep_$WINDOW.md" "$MATERIAL" "$OUT" || exit 1
    if [ "$WINDOW" = "evening" ]; then send digest "Prep for tomorrow's meetings" "$OUT"
    else send urgent "Meeting in an hour" "$OUT"; fi || exit 1
    ;;
  *)
    PROMPT="$LOCAL/prompts/$JOB.md"
    [ -f "$PROMPT" ] || { echo "no job named $JOB (expected $PROMPT)"; exit 2; }
    MATERIAL="$LOCAL/out/${JOB}_material_$STAMP.txt"
    COLLECT="$LOCAL/prompts/$JOB.collect.sh"
    if [ -x "$COLLECT" ]; then "$COLLECT" > "$MATERIAL" || exit 1; else echo "(no material)" > "$MATERIAL"; fi
    compose "$PROMPT" "$MATERIAL" "$OUT" || exit 1
    send digest "$JOB $TODAY" "$OUT" || exit 1
    ;;
esac

[ -n "$PERIOD" ] && touch "$MARK"
echo "=== done $(date '+%H:%M:%S') ($MODE)"
