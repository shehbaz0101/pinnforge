# v0.4.1 reproducibility

CPU-only run in this repository. No paid GPU. No secrets. The optional [scripts/v041_colab.py](../../scripts/v041_colab.py) uses the same commands. The detached runner is [scripts/v041_run.sh](../../scripts/v041_run.sh).

The protocol [protocol.json](protocol.json) was committed in `98e618b5` before any v0.4.1 test score. It was not edited after that commit.

| file | SHA-256 |
| --- | --- |
| [protocol.json](protocol.json) | `0fd6749a4cfd43154d661d385475f5b019c12e8ff1524c90d2334968360c840a` |
| [scores.json](scores.json) | `93c3c9df62c9d879b3d66c767e4246bea88674193869e72561a381435224aee4` |
| [validation.json](validation.json) | `d03b605eb2e6dd8d5ed37166d41f8efa8f0cb649e139ddb004f3acd0eddfd550` |
| [pilot_manifest.json](pilot_manifest.json) | `c8f6ad352e3b0211888a0c763b1460ee10fc7ee157f37d8e45a9b63883efeee6` |
| [train_summary.json](train_summary.json) | `f02d7a777c85a690b1298331c33c7e08fd214a88baa915863a5d745718e9058a` |
| [score_cases.jsonl](score_cases.jsonl) | `ce7c6933b9ef9d5a435a0e5ee110789d5e113f7ce602d49546aba809ab2a87d4` |

Python 3.12.3, NumPy 2.5.4, torch 2.14.1+cpu. Those three strings are also stored on [scores.json](scores.json). Checkpoints and pilot arrays stay under `artifacts/v041_pilot/` and are not committed.

Scoring wrote one JSONL row per case. Restarts skip completed keys. A four-process pool ran cases and seeds. Noise is replayed from the serial generator (`noise_seed + int(1000 * fraction)`, then `add_noise` in instance-id order) so the viscosity estimates match a one-process run. Wall-clock seconds are per-solve `perf_counter` samples on this machine while four workers ran; they are not a serial-timing copy.

```bash
python -m pinnforge.v041 generate
python -m pinnforge.v041 train
python -m pinnforge.v041 score --checkpoint docs/v041/score_cases.jsonl
python -m pinnforge.v041 report
```
