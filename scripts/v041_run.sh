#!/usr/bin/env bash
# Offline CPU generate / train / score for the frozen v0.4.1 protocol.
# Resumable: generate and train skip finished artifacts; score appends
# docs/v041/score_cases.jsonl and skips completed cases.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PINNFORGE_V041_WORKERS="${PINNFORGE_V041_WORKERS:-4}"
mkdir -p artifacts/v041_pilot

stamp() {
  printf '%s\t%s\n' "$(date -Is)" "$*" | tee -a docs/v041/run_status.txt
}

stamp "start workers=${PINNFORGE_V041_WORKERS}"
python3 -u -m pinnforge.v041 generate
stamp "generate done"
python3 -u -m pinnforge.v041 train --workers "${PINNFORGE_V041_WORKERS}"
stamp "train done"
python3 -u -m pinnforge.v041 score --workers "${PINNFORGE_V041_WORKERS}" --checkpoint docs/v041/score_cases.jsonl
stamp "score done"
