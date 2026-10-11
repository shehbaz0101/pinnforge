#!/usr/bin/env bash
# Offline CPU generate / score for the frozen v0.5.1 protocol.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PINNFORGE_V051_WORKERS="${PINNFORGE_V051_WORKERS:-4}"
mkdir -p artifacts/v051_pilot

stamp() {
  printf '%s\t%s\n' "$(date -Is)" "$*" | tee -a docs/v051/run_status.txt
}

stamp "start workers=${PINNFORGE_V051_WORKERS}"
python3 -u -m pinnforge.v051 generate
stamp "generate done"
python3 -u -m pinnforge.v051 score --workers "${PINNFORGE_V051_WORKERS}" --checkpoint artifacts/v051_pilot/score_cases.jsonl
stamp "score done"
