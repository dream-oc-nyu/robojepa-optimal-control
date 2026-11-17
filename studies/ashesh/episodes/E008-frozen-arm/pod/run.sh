#!/usr/bin/env bash
# Session-owned simulator-only replay; never starts on a busy GPU.
set -euo pipefail
ROOT=/workspace/robojepa-E001-20261112T003859Z
source "$ROOT/session-env.sh"
[[ "$SESSION_ROOT" == "$ROOT" ]] || exit 2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/e008/code:$ROOT/e002/code:$ROOT/upstream/robo_jepa-f93106f5f4a451fd526d51778aa8ab7c852eb824/scripts/deployment:${PYTHONPATH:-}"
JOB="$ROOT/jobs/e008-replay"
[[ "$(cd "$(dirname "$0")" && pwd)" == "$JOB" ]] || exit 2
exec 9>"$JOB/lock"
flock -n 9 || exit 3
# Defence in depth for direct invocation, not only the launcher.
for pid_file in "$ROOT"/jobs/*/pid; do
  [[ -f "$pid_file" && "$(dirname "$pid_file")" != "$JOB" ]] || continue
  [[ "$(cat "$(dirname "$pid_file")/state" 2>/dev/null)" == running ]] || continue
  pid=$(cat "$pid_file")
  if [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null; then
    printf 'Job already alive (%s, pid=%s); refusing\n' "$pid_file" "$pid"
    exit 3
  fi
done
printf '%s\n' "$$" > "$JOB/pid"
printf 'running\n' > "$JOB/state"
trap 'rc=$?; if (( rc == 0 )); then printf "complete\n" > "$JOB/state"; else printf "failed\n" > "$JOB/state"; fi' EXIT
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
[[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" == 0 ]] || { printf 'GPU busy, refusing\n'; exit 4; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || { printf 'GPU compute app present, refusing\n'; exit 4; }
[[ -f "$JOB/planner" ]] || { printf 'Missing planner file\n'; exit 2; }
planner=$(cat "$JOB/planner")
[[ "$planner" =~ ^[a-z0-9_]+$ ]] || exit 2
cd "$ROOT/e008/code"
python -u "$ROOT/e008/code/replay.py" --planner "$planner" &
child=$!
printf '%s\n' "$child" > "$JOB/python.pid"
# Signal only our own child, and wait before publishing terminal state.
trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; exit 143' TERM
trap 'kill -INT "$child" 2>/dev/null || true; wait "$child" || true; exit 130' INT
while kill -0 "$child" 2>/dev/null; do
  printf '%s heartbeat job=%s python=%s\n' "$(date -u +%FT%TZ)" "$$" "$child"
  sleep 30 &
  wait $! || true
done
wait "$child"
