#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LAB_PYTHON="${LAB_PYTHON:-python3}"
case "${1:-cpu}" in
  cpu)
    "$LAB_PYTHON" labs/scheduler_sim.py
    "$LAB_PYTHON" labs/transformer_lab.py all
    "$LAB_PYTHON" labs/inference_lab.py all
    ;;
  diagnose) "$LAB_PYTHON" labs/gpu_lab.py --diagnose ;;
  dimensions) "$LAB_PYTHON" labs/dimension_lab.py ;;
  gpu) "$LAB_PYTHON" labs/gpu_lab.py --suite all --dtype fp32 ;;
  bf16) "$LAB_PYTHON" labs/gpu_lab.py --suite all --dtype bf16 --out labs/results/gpu-bf16 ;;
  *) printf '%s\n' 'Usage: bash labs/run-wsl.sh [cpu|dimensions|diagnose|gpu|bf16]'; exit 2 ;;
esac
