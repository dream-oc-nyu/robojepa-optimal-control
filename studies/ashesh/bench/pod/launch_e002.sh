#!/usr/bin/env bash
# Stage ONLY this worker's files, then launch/resume one bounded E002 job.
# Usage from repository root: bash bench/pod/launch_e002.sh calibration|cem [seeds]
set -euo pipefail
ROOT=/workspace/robojepa-E001-20261112T003859Z
K=(kubectl --context hsrn -n wireless)
kind=${1:?Usage: launch_e002.sh calibration|cem [seeds]}
seeds=${2-0,1,2}
[[ "$kind" == calibration || "$kind" == cem ]] || exit 2
[[ "$seeds" =~ ^[0-2](,[0-2])*$ ]] || { printf 'Seeds must be comma-separated values from 0,1,2\n'; exit 2; }
[[ -f bench/interface.py && -f planners/cem_release.py ]] || { printf 'Run from repository root\n'; exit 2; }
POD=$("${K[@]}" get pods -l app=spatial-reasoning-gpu -o jsonpath='{.items[0].metadata.name}')
JOB="$ROOT/jobs/e002-$kind"
"${K[@]}" exec "$POD" -c dev -- bash -lc "source '$ROOT/session-env.sh'; mkdir -p '$ROOT/e002/code/bench/pod' '$ROOT/e002/code/planners' '$JOB'; if [[ -f '$JOB/pid' ]] && kill -0 \$(cat '$JOB/pid') 2>/dev/null; then printf 'Job already alive; refuse overwrite/relaunch\n'; exit 3; fi"
for path in bench/interface.py bench/score.py bench/test_score.py bench/test_runtime.py bench/rollout.py bench/harness.py planners/__init__.py planners/cem_release.py; do
  "${K[@]}" cp -c dev "$path" "$POD:$ROOT/e002/code/$path"
done
"${K[@]}" cp -c dev bench/pod "$POD:$ROOT/e002/code/bench/"
"${K[@]}" cp -c dev bench/pod/run.sh "$POD:$JOB/run.sh"
if [[ "$kind" == cem ]]; then
  "${K[@]}" exec "$POD" -c dev -- bash -lc "printf '%s\n' '$seeds' > '$JOB/seeds'"
fi
"${K[@]}" exec "$POD" -c dev -- bash -lc "source '$ROOT/session-env.sh'; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; nohup bash '$JOB/run.sh' >> '$JOB/log' 2>&1 < /dev/null & echo launcher_pid=\$!"
sleep 60
"${K[@]}" exec "$POD" -c dev -- bash -lc "source '$ROOT/session-env.sh'; python '$ROOT/e002/code/bench/pod/status.py'"
