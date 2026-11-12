#!/bin/bash
# Adapted from facebookresearch/robo_jepa, scripts/deployment/examples/reproduce_*.sh:
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved. Licensed under CC BY-NC-SA 4.0 (LICENSE).
# Changed: the four scripts merged into one (TASK), the CEM call routed through planners/run_client.py.
#
# One task, one planner, a list of setup seeds: starts the RoboCasa env server, then the official policy_client.py
# through planners/run_client.py (which swaps mpc_utils.cem for PLANNER and logs every planning call). Run it where
# the robo_jepa environment is active, on a GPU machine.
#
#   REPO=<robo_jepa clone> MODELS=<checkpoints> OUT=<outputs> TASK=push SCENE=7 PLANNER=cem \
#       bash scripts/run_task.sh <size> <step> <seed> [seed ...]
#
# MODELS/wm-<size>/ holds cooldown_<size>_<step>.pth.tar, config.json and encoder.pth.tar (README, Setup).
# Every seed listed runs in one client process, against one env server. Results go to OUT/<task>/<TAG>/ (TAG defaults
# to the planner and its budget).
#
# TASK (default reach): reach | objreach | obstacle | push, with the settings of the official
# scripts/deployment/examples/reproduce_*.sh. The seed picks the start and the goal; SCENE is the env's scene seed
# (reach, objreach, obstacle; default 0) or the target kitchen (push: 1-10, default 4, as reproduce_push.sh; larger
# values are taken mod 10, 0 = 10). ObjectReach carries OBJECT (default banana).
# The CEM settings default to the official ones (CEM_ITERS, STEP_CEM, SAMPLES, NUM_BATCHES, CUTOFF).
set -e
SIZE=$1; STEP=$2; shift 2; SEEDS="$@"
TASK=${TASK:-reach}
[ "$TASK" = push ] && SCENE=${SCENE:-4} || SCENE=${SCENE:-0}
if [ "$TASK" = obstacle ]; then
  ROLLOUT=${ROLLOUT:-5}; EMBODIMENT=${EMBODIMENT:-3}; TOTAL_STEPS=5
  TASK_FLAGS="--obstacle_reach_task --reach_clear_radius 1.0 --kitchen_split target"
  TASK_FLAGS="$TASK_FLAGS --obstacle_classes CoffeeMachine --reach_min_distance 0.40 --reach_max_distance 0.80"
elif [ "$TASK" = push ]; then
  ROLLOUT=${ROLLOUT:-5}; EMBODIMENT=${EMBODIMENT:-3}; TOTAL_STEPS=18
  K=$((SCENE % 10)); [ $K = 0 ] && K=10  # the target layouts are 1-10
  TASK_FLAGS="--object_push_task --object_push_target_kitchen_id $K --object_push_object_name flourbag001"
  TASK_FLAGS="$TASK_FLAGS --object_push_forward_cone_deg 60 --object_push_clear_radius 1.0"
elif [ "$TASK" = objreach ]; then
  # a scripted top-down grasp at reset, then carry the object to the goal
  ROLLOUT=${ROLLOUT:-1}; EMBODIMENT=${EMBODIMENT:-2}; TOTAL_STEPS=5
  OBJECT=${OBJECT:-banana}
  TASK_FLAGS="--object_reach_task --object_reach_object_name $OBJECT --object_reach_max_distance 0.15"
  TASK_FLAGS="$TASK_FLAGS --object_reach_clear_radius 1.0 --object_reach_max_retries 3"
else
  ROLLOUT=${ROLLOUT:-1}; EMBODIMENT=${EMBODIMENT:-2}; TOTAL_STEPS=5
  TASK_FLAGS="--reach_task --reach_max_distance 0.30 --reach_clear_radius 1.0"
fi
CEM_ITERS=${CEM_ITERS:-15}; STEP_CEM=${STEP_CEM:-10}; SAMPLES=${SAMPLES:-16}; NUM_BATCHES=${NUM_BATCHES:-64}
CUTOFF=${CUTOFF:-20}  # CEM's elite count
export PLANNER=${PLANNER:-cem}
TAG=${TAG:-${PLANNER}_${SAMPLES}x${NUM_BATCHES}_${CEM_ITERS}-${STEP_CEM}}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)  # the repo root
: "${REPO:?set REPO to the robo_jepa clone}" "${MODELS:?set MODELS to the checkpoint folder}" "${OUT:?set OUT}"
RUN=$OUT/$TASK/$TAG; mkdir -p "$RUN"
export PLAN_LOG=$RUN/plan_log.jsonl  # one line per planning call (planners.make_planner)
PORT=${PORT:-$((7300 + RANDOM % 600))}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONPATH="$REPO:$REPO/scripts/deployment:${PYTHONPATH:-}"
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=$((PORT + 20000))

SRC=$MODELS/wm-$SIZE
CKPT=/tmp/robojepa_ckpt_${SIZE}_${STEP}_$$; mkdir -p "$CKPT"
ln -sf "$SRC/cooldown_${SIZE}_${STEP}.pth.tar" "$CKPT/world_model.pth.tar"
ln -sf "$SRC/config.json" "$CKPT/config.json"
ln -sf "$SRC/encoder.pth.tar" "$CKPT/encoder.pth.tar"

TASKS=$(mktemp -d /tmp/robojepa_tasks_XXXXXX)  # the client runs every .json in it, in sorted file order
for s in $SEEDS; do
  if [ "$TASK" = push ]; then
    printf '{"init":{"mode":"init","object_push_task_seed":%d},"goal":{"mode":"goal","object_push_task_seed":%d}}' "$s" "$s" > "$TASKS/ts$s.json"
  elif [ "$TASK" = objreach ]; then
    printf '{"init":{"mode":"init","seed":%d,"object_reach_task_seed":%d,"object_reach_object_name":"%s"},"goal":{"mode":"goal","seed":%d,"object_reach_task_seed":%d,"object_reach_object_name":"%s"}}' "$SCENE" "$s" "$OBJECT" "$SCENE" "$s" "$OBJECT" > "$TASKS/ts$s.json"
  else
    printf '{"init":{"mode":"init","seed":%d,"reach_task_seed":%d},"goal":{"mode":"goal","seed":%d,"reach_task_seed":%d}}' "$SCENE" "$s" "$SCENE" "$s" > "$TASKS/ts$s.json"
  fi
done
export TASK_DIR=$TASKS  # planners.make_planner: which seed each planning call belongs to
VIEW_W=${VIEW_W:-0.5,0.5}  # the cost's weights of the left and the wrist camera
{ echo "seeds=$SEEDS task=$TASK steps=$TOTAL_STEPS scene=$SCENE env_flags=$TASK_FLAGS"
  echo "embodiment=$EMBODIMENT size=$SIZE step=$STEP rollout=$ROLLOUT cem=$CEM_ITERS/$STEP_CEM samples=$SAMPLES x $NUM_BATCHES cutoff=$CUTOFF"
  env | grep -E '^(PLANNER|PLAN_[A-Z_]+)=' | sort
  echo "VIEW_W=$VIEW_W"; } >> "$RUN/config.txt"

# as the official scripts: the reach tasks pass their scene seed and a placeholder inner task, ObjectReach its scene
# seed, push scene seed 0 (its kitchen is pinned by --object_push_target_kitchen_id)
SCENE_FLAGS="--seed $SCENE --task_name CloseStandMixerHead --reach_task_seed $SCENE"
[ "$TASK" = push ] && SCENE_FLAGS="--seed 0"
[ "$TASK" = objreach ] && SCENE_FLAGS="--seed $SCENE --task_name ObjectReachKitchen --object_reach_task_seed $SCENE"
cd "$REPO"
python scripts/deployment/robots/robocasa/robot_env_server.py \
  --port "$PORT" --camera_names robot0_agentview_left robot0_eye_in_hand \
  --camera_width 320 --camera_height 240 --robot PandaOmron --chunk_size 4 \
  --wm_embodiment franka_omron_state_delta $SCENE_FLAGS --blocking_control \
  --blocking_pos_tol 0.001 --blocking_max_steps 200 \
  $TASK_FLAGS > "$RUN/env_server.log" 2>&1 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null' EXIT
sleep 45
kill -0 $SERVER 2>/dev/null || { echo "env server died, see $RUN/env_server.log"; tail -20 "$RUN/env_server.log"; exit 1; }

echo "== $TAG: planner=$PLANNER cem=$CEM_ITERS/$STEP_CEM samples=$SAMPLES x $NUM_BATCHES cutoff=$CUTOFF seeds=$SEEDS"
python "$HERE/planners/run_client.py" \
  --checkpoint_path "$CKPT" --host localhost --port "$PORT" \
  --embed_dim 1664 --image_size 240 320 --use_cameras 0 1 --camera_id 0 2 \
  --embodiment_id $EMBODIMENT --task_path "$TASKS" --episodes 1 --init_ep 0 \
  --total_steps $TOTAL_STEPS --rollout $ROLLOUT --cem $CEM_ITERS --step_cem $STEP_CEM --cutoff $CUTOFF \
  --samples $SAMPLES --num_batches $NUM_BATCHES --maxnorm 0.3 --maxgripper 0.99 \
  --robot_init_std 0.3 --gripper_init_std 0.8 \
  --momentum_std 0.70 --momentum_mean_gripper 0.15 --momentum_std_gripper 0.45 \
  --execution_prefix 1 --closed_loop_context_len 2 --gamma 0.99 \
  --translation_only --use_cache --load_compiled false --action_substeps 1 \
  --bypass_input --view_weights ${VIEW_W//,/ } \
  --output_filename "$RUN/ep"
