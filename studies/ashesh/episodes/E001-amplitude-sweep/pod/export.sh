#!/usr/bin/env bash
# Run from the VPS repo root; no arguments or remembered shell state required.
set -euo pipefail
exec python3 episodes/E001-amplitude-sweep/pod/export_sync.py
