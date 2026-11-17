#!/usr/bin/env bash
# From repo root: bash episodes/E008-frozen-arm/pod/launch.sh [planner]
set -euo pipefail
ROOT=/workspace/robojepa-E001-20261112T003859Z
K=(kubectl --context hsrn -n wireless)
[[ $# -le 1 ]] || { printf 'Usage: launch.sh [planner]\n'; exit 2; }
planner=${1-cem_release}
[[ "$planner" =~ ^[a-z0-9_]+$ ]] || { printf 'Invalid planner name\n'; exit 2; }
LOCAL=episodes/E008-frozen-arm/pod
[[ -f bench/harness.py && -f "$LOCAL/replay.py" && -f "$LOCAL/run.sh" ]] || { printf 'Run from repository root\n'; exit 2; }
POD=$("${K[@]}" get pods -l app=spatial-reasoning-gpu -o jsonpath='{.items[0].metadata.name}')
[[ -n "$POD" ]] || { printf 'No matching pod\n'; exit 2; }
JOB="$ROOT/jobs/e008-replay"
"${K[@]}" exec "$POD" -c dev -- bash -lc "
set -euo pipefail
source '$ROOT/session-env.sh'
for pid_file in '$ROOT'/jobs/*/pid; do
  [[ -f \"\$pid_file\" ]] || continue
  [[ \"\$(cat \"\$(dirname \"\$pid_file\")/state\" 2>/dev/null)\" == running ]] || continue
  pid=\$(cat \"\$pid_file\")
  if [[ \"\$pid\" =~ ^[1-9][0-9]*$ ]] && kill -0 \"\$pid\" 2>/dev/null; then
    printf 'Job already alive (%s, pid=%s); refuse overwrite/relaunch\n' \"\$pid_file\" \"\$pid\"
    exit 3
  fi
done
for instance in {100..107}; do
  input='$ROOT/e002/runs/$planner'/inst\${instance}-seed0/unit.json
  [[ -f \"\$input\" ]] || { printf 'Missing input: %s\n' \"\$input\"; exit 2; }
done
mkdir -p '$ROOT/e008/code' '$ROOT/e008/out' '$JOB'
"
# Never overwrite any e002 files.
"${K[@]}" cp -c dev "$LOCAL/replay.py" "$POD:$ROOT/e008/code/replay.py"
"${K[@]}" cp -c dev "$LOCAL/run.sh" "$POD:$ROOT/e008/code/run.sh"
"${K[@]}" exec "$POD" -c dev -- bash -lc "
set -euo pipefail
source '$ROOT/session-env.sh'
cp '$ROOT/e008/code/run.sh' '$JOB/run.sh'
printf '%s\n' '$planner' > '$JOB/planner'
nohup bash '$JOB/run.sh' >> '$JOB/log' 2>&1 < /dev/null & echo launcher_pid=\$!
"
sleep 45
"${K[@]}" exec "$POD" -c dev -- bash -lc "
set -euo pipefail
printf 'job=%s\n' '$JOB'
if [[ -f '$JOB/state' ]]; then printf 'state=%s\n' \"\$(cat '$JOB/state')\"; else printf 'state=not-started\n'; fi
if [[ -f '$JOB/pid' ]]; then
  pid=\$(cat '$JOB/pid')
  alive=false
  if [[ \"\$pid\" =~ ^[1-9][0-9]*$ ]] && kill -0 \"\$pid\" 2>/dev/null; then alive=true; fi
  printf 'pid=%s alive=%s\n' \"\$pid\" \"\$alive\"
else
  printf 'pid=missing alive=false\n'
fi
if [[ -f '$JOB/python.pid' ]]; then printf 'python.pid=%s\n' \"\$(cat '$JOB/python.pid')\"; else printf 'python.pid=missing\n'; fi
if [[ -f '$JOB/log' ]]; then tail -n 15 '$JOB/log'; fi
"
