#!/usr/bin/env bash
# Offline CPU generate / score for the frozen v0.5.2 protocol.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PINNFORGE_V052_WORKERS="${PINNFORGE_V052_WORKERS:-4}"
mkdir -p artifacts/v052_pilot

stamp() {
  printf '%s\t%s\n' "$(date -Is)" "$*" | tee -a docs/v052/run_status.txt
}

stamp "start workers=${PINNFORGE_V052_WORKERS}"
python3 -u -m pinnforge.v052 generate
stamp "generate done"
python3 -u -m pinnforge.v052 score --workers "${PINNFORGE_V052_WORKERS}" --checkpoint artifacts/v052_pilot/score_cases.jsonl
stamp "score done"
