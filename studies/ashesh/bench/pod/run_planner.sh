#!/usr/bin/env bash
# Run as <root>/jobs/e006-<planner>/run.sh.
set -euo pipefail
ROOT=/workspace/robojepa-E001-20261112T003859Z
# shellcheck disable=SC1091
source "$ROOT/session-env.sh"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$ROOT/e002/code:$ROOT/upstream/robo_jepa-f93106f5f4a451fd526d51778aa8ab7c852eb824/scripts/deployment:$PYTHONPATH"
JOB=$(cd "$(dirname "$0")" && pwd)
[[ "$(dirname "$JOB")" == "$ROOT/jobs" && "$(basename "$JOB")" =~ ^e006-[a-z0-9_]+$ ]] || exit 2
exec 9>"$JOB/lock"
flock -n 9 || exit 3
printf '%s\n' "$$" > "$JOB/pid"
printf 'running\n' > "$JOB/state"
trap 'rc=$?; if (( rc == 0 )); then printf "complete\n" > "$JOB/state"; else printf "failed\n" > "$JOB/state"; fi' EXIT
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
[[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" == 0 ]] || { printf 'GPU busy, refusing\n'; exit 4; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || { printf 'GPU compute app present, refusing\n'; exit 4; }
cd "$ROOT/e002/code"
python -c 'import json,os; from pathlib import Path; r=Path(os.environ["SESSION_ROOT"])/"e002"; assert json.loads((r/"budget.json").read_text())["budget_s"] > 0; assert json.loads((r/"smoke.json").read_text())["complete"]'
[[ -f "$JOB/planner" && -f "$JOB/seeds" ]] || { printf 'Missing required planner/seeds files\n'; exit 2; }
planner=$(cat "$JOB/planner")
seeds=$(cat "$JOB/seeds")
[[ "$planner" =~ ^[a-z0-9_]+$ && "$(basename "$JOB")" == "e006-$planner" ]] || exit 2
[[ "$seeds" =~ ^[0-2](,[0-2])*$ ]] || exit 2
python -u -m bench.pod.run_planner --planner "$planner" --seeds "$seeds" &
child=$!
printf '%s\n' "$child" > "$JOB/python.pid"
# Output heartbeat prevents a long reset/CEM iteration being mistaken for a stall.
while kill -0 "$child" 2>/dev/null; do
  printf '%s heartbeat job=%s python=%s\n' "$(date -u +%FT%TZ)" "$$" "$child"
  sleep 30 &
  wait $! || true
done
wait "$child"
