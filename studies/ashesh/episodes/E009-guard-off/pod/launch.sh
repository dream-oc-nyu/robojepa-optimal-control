#!/usr/bin/env bash
# From repository root: bash episodes/E009-guard-off/pod/launch.sh
set -euo pipefail
[[ $# == 0 ]] || { printf 'Usage: launch.sh\n'; exit 2; }
ROOT=/workspace/robojepa-E001-20261112T003859Z
K=(kubectl --context hsrn -n wireless)
LOCAL=episodes/E009-guard-off/pod
[[ -f bench/harness.py && -f "$LOCAL/guardoff.py" && -f "$LOCAL/test_guardoff.py" && -f "$LOCAL/run.sh" ]] || { printf 'Run from repository root\n'; exit 2; }
POD=$("${K[@]}" get pods -l app=spatial-reasoning-gpu -o jsonpath='{.items[0].metadata.name}')
[[ -n "$POD" ]] || { printf 'No matching pod\n'; exit 2; }
JOB="$ROOT/jobs/e009-guardoff"
# Check again before the launch as well as before staging any code.
CHECK='for pid_file in "$ROOT"/jobs/*/pid; do
  [[ -f "$pid_file" ]] || continue
  [[ "$(cat "$(dirname "$pid_file")/state" 2>/dev/null)" == running ]] || continue
  pid=$(cat "$pid_file")
  if [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null; then
    printf "Job already alive (%s, pid=%s); refuse overwrite/relaunch\n" "$pid_file" "$pid"
    exit 3
  fi
done'
"${K[@]}" exec "$POD" -c dev -- bash -lc "
set -euo pipefail
ROOT='$ROOT'
source \"\$ROOT/session-env.sh\"
$CHECK
mkdir -p '$ROOT/e009/code' '$ROOT/e009/runs' '$JOB'
"
# Copy only these three files; never write to e002/code or e002/runs.
for name in guardoff.py test_guardoff.py run.sh; do
  "${K[@]}" cp -c dev "$LOCAL/$name" "$POD:$ROOT/e009/code/$name"
done
"${K[@]}" exec "$POD" -c dev -- bash -lc "
set -euo pipefail
ROOT='$ROOT'
source \"\$ROOT/session-env.sh\"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH='$ROOT/e009/code:$ROOT/e002/code':\"\${PYTHONPATH:-}\"
# Local analysis/export tests are skipped when their files are not staged.
CUDA_VISIBLE_DEVICES='' python -B '$ROOT/e009/code/test_guardoff.py'
$CHECK
cp '$ROOT/e009/code/run.sh' '$JOB/run.sh'
nohup bash '$JOB/run.sh' >> '$JOB/log' 2>&1 < /dev/null & echo launcher_pid=\$!
"
sleep 60
"${K[@]}" exec "$POD" -c dev -- bash -lc "
set -euo pipefail
printf 'job=%s\n' '$JOB'
if [[ -f '$JOB/state' ]]; then printf 'state=%s\n' \"\$(cat '$JOB/state')\"; else printf 'state=not-started\n'; fi
for name in pid python.pid; do
  if [[ -f '$JOB'/\$name ]]; then
    pid=\$(cat '$JOB'/\$name)
    alive=false
    if [[ \"\$pid\" =~ ^[1-9][0-9]*$ ]] && kill -0 \"\$pid\" 2>/dev/null; then alive=true; fi
    printf '%s=%s alive=%s\n' \"\$name\" \"\$pid\" \"\$alive\"
  else
    printf '%s=missing alive=false\n' \"\$name\"
  fi
done
if [[ -f '$JOB/log' ]]; then tail -n 15 '$JOB/log'; fi
"
