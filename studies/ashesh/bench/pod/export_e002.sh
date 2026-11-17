#!/usr/bin/env bash
# From the local repository root; JSON receipts only, resumable and hash checked.
set -euo pipefail
python3 bench/pod/export_e002.py
