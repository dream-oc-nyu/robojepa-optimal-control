#!/usr/bin/env bash
# Stage a registered dev planner only when none of our session jobs is alive.
# Usage from repository root: bash bench/pod/launch_planner.sh <planner> [seeds]
set -euo pipefail
ROOT=/workspace/robojepa-E001-20261112T003859Z
K=(kubectl --context hsrn -n wireless)
[[ $# -ge 1 && $# -le 2 ]] || { printf 'Usage: launch_planner.sh <planner> [seeds]\n'; exit 2; }
planner=$1
planner_file=$planner
seeds=${2-0}
[[ "$planner" =~ ^[a-z0-9_]+$ ]] || { printf 'Invalid planner name\n'; exit 2; }
[[ "$seeds" =~ ^[0-2](,[0-2])*$ ]] || { printf 'Seeds must be comma-separated values from 0,1,2\n'; exit 2; }
case "$planner" in
  cem_clampexec) test_module=bench.pod.test_clampexec ;;
  cem_cap10) test_module=bench.pod.test_cap10 ;;
  gn_cube_a) test_module=bench.pod.test_gn_cube_a ;;
  gn_cube_b) test_module=bench.pod.test_gn_cube_b ;;
  sat10_cem_a|bisect_cem_a|halvesat_cem_a|sat10_gna_a|bisect_gna_a|halvesat_gna_a|sat10_gnb_a|bisect_gnb_a|halvesat_gnb_a) test_module=bench.pod.test_bisect_a; planner_file=bisect_a ;;
  sat10_cem_b|bisect_cem_b|halvesat_cem_b|sat10_gna_b|bisect_gna_b|halvesat_gna_b|sat10_gnb_b|bisect_gnb_b|halvesat_gnb_b) test_module=bench.pod.test_bisect_b; planner_file=bisect_b ;;
  *) printf 'Unknown planner: %s\n' "$planner"; exit 2 ;;
esac
[[ -f bench/interface.py && -f planners/cem_release.py && -f planners/gn_cube_a.py && -f planners/gn_cube_b.py && -f planners/bisect_a.py && -f planners/bisect_b.py && -f "planners/$planner_file.py" ]] || { printf 'Run from repository root with a local planner file\n'; exit 2; }
POD=$("${K[@]}" get pods -l app=spatial-reasoning-gpu -o jsonpath='{.items[0].metadata.name}')
JOB="$ROOT/jobs/e006-$planner"
"${K[@]}" exec "$POD" -c dev -- bash -lc "
set -euo pipefail
source '$ROOT/session-env.sh'
mkdir -p '$ROOT/e002/code/bench/pod' '$ROOT/e002/code/planners' '$JOB'
for pid_file in '$ROOT'/jobs/*/pid; do
  [[ -f \"\$pid_file\" ]] || continue
  # A finished job keeps its pid file; only a job still marked running can hold the GPU (pid reuse).
  [[ \"\$(cat \"\$(dirname \"\$pid_file\")/state\" 2>/dev/null)\" == running ]] || continue
  pid=\$(cat \"\$pid_file\")
  if [[ \"\$pid\" =~ ^[1-9][0-9]*$ ]] && kill -0 \"\$pid\" 2>/dev/null; then
    printf 'Job already alive (%s, pid=%s); refuse overwrite/relaunch\n' \"\$pid_file\" \"\$pid\"
    exit 3
  fi
done
"
declare -A staged=()
for path in bench/interface.py bench/score.py bench/test_score.py bench/test_runtime.py bench/rollout.py bench/harness.py planners/__init__.py planners/cem_release.py planners/cem_clampexec.py planners/cem_cap10.py planners/gn_cube_a.py planners/gn_cube_b.py planners/bisect_a.py planners/bisect_b.py "planners/$planner_file.py"; do
  [[ -z "${staged[$path]+present}" ]] || continue
  staged[$path]=1
  "${K[@]}" cp -c dev "$path" "$POD:$ROOT/e002/code/$path"
done
"${K[@]}" cp -c dev bench/pod "$POD:$ROOT/e002/code/bench/"
"${K[@]}" cp -c dev bench/pod/run_planner.sh "$POD:$JOB/run.sh"
"${K[@]}" exec "$POD" -c dev -- bash -lc "set -euo pipefail; printf '%s\n' '$planner' > '$JOB/planner'; printf '%s\n' '$seeds' > '$JOB/seeds'"
"${K[@]}" exec "$POD" -c dev -- bash -lc "
set -euo pipefail
source '$ROOT/session-env.sh'
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd '$ROOT/e002/code'
CUDA_VISIBLE_DEVICES='' PYTHONPATH='$ROOT/e002/code:$ROOT/upstream/robo_jepa-f93106f5f4a451fd526d51778aa8ab7c852eb824/scripts/deployment':\"\$PYTHONPATH\" python -m unittest '$test_module'
"
"${K[@]}" exec "$POD" -c dev -- bash -lc "set -euo pipefail; source '$ROOT/session-env.sh'; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; nohup bash '$JOB/run.sh' >> '$JOB/log' 2>&1 < /dev/null & echo launcher_pid=\$!"
sleep 60
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
