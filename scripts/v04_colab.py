"""Optional free-GPU entry for the frozen v0.4 study.

The protocol, the pass bars, and the code path are the same as
``python -m pinnforge.v04``. This file exists so a free Colab or Kaggle
GPU can run ``generate``, ``train``, and ``score`` without changing a
hyperparameter. CI never executes this script.

Usage after ``pip install -e '.[ml]'``::

    python scripts/v04_colab.py
"""

from __future__ import annotations

from pinnforge.v04.report import write_report
from pinnforge.v04.study import run_generate, run_train, run_validate_and_test


def main() -> int:
    run_generate()
    run_train()
    run_validate_and_test()
    write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
