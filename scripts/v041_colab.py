"""Optional free-GPU entry for the frozen v0.4.1 study.

The protocol, the pass bars, and the code path are the same as
``python -m pinnforge.v041``. CI never executes this script.
"""

from __future__ import annotations

from pinnforge.v041.report import write_report
from pinnforge.v041.study import run_calibrate_and_test, run_generate, run_train


def main() -> int:
    run_generate()
    run_train()
    run_calibrate_and_test()
    write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
