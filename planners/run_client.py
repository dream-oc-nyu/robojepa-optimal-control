"""Run the official policy_client.py with mpc_utils.cem swapped for planners.make_planner(PLANNER).

policy_client.py calls ``mpc_utils.cem(...)`` through the module, so replacing the attribute before the script
runs is enough; the official code stays untouched. Arguments are passed through unchanged.

    REPO=<robo_jepa clone> PLANNER=cem python planners/run_client.py <policy_client.py arguments>
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ["REPO"]
CLIENT_DIR = os.path.join(REPO, "scripts", "deployment", "agents", "world_model")
sys.path[:0] = [HERE, REPO, os.path.join(REPO, "scripts", "deployment"), CLIENT_DIR]

from app.robo_jepa.mpc import mpc_utils  # noqa: E402

import planners  # noqa: E402

mpc_utils.cem = planners.make_planner(os.environ.get("PLANNER", "cem"))

# torch.hub.load("facebookresearch/vjepa2", ...) asks GitHub for the default branch on every load, before it looks at
# its cache (~/.cache/torch/hub/facebookresearch_vjepa2_main); a GitHub error then kills the episode. A repo given
# without a ref is resolved to "main" here, without the network.
import torch.hub as _hub  # noqa: E402

_parse_repo_info = _hub._parse_repo_info


def _offline_repo_info(github):
    if ":" in github:
        return _parse_repo_info(github)
    owner, name = github.split("/")
    return owner, name, "main"


_hub._parse_repo_info = _offline_repo_info

script = os.path.join(CLIENT_DIR, "policy_client.py")
sys.argv = [script] + sys.argv[1:]
os.chdir(CLIENT_DIR)
runpy.run_path(script, run_name="__main__")
