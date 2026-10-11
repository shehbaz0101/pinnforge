# v0.5 reproducibility

CPU-only run in this repository. No paid GPU. No secrets. The detached runner is [scripts/v05_run.sh](../../scripts/v05_run.sh).

The protocol [protocol.json](protocol.json) was committed in `4aed3558` before any v0.5 test score. It was not edited after that commit.

- python: `3.12.3`
- numpy: `2.5.4`
- torch: `2.14.1+cpu`
- protocol_sha256: `f4d7f8584cf003bc3b28f5d554e645ff3f5f5160f646538d854cbdae0362e902`
- validation protocol_sha256: `f4d7f8584cf003bc3b28f5d554e645ff3f5f5160f646538d854cbdae0362e902`
- PySR used: `False`

SHA-256 of committed files:

| file | SHA-256 |
| --- | --- |
| [protocol.json](protocol.json) | `f4d7f8584cf003bc3b28f5d554e645ff3f5f5160f646538d854cbdae0362e902` |
| [scores.json](scores.json) | `cb99b1752aec9330ff598bc933909392794c573843a3bfbf926ddfef68d642cf` |
| [validation.json](validation.json) | `1f04f889fc8542eef229a095e7b3bcd4f359ab74eba2bc6b3f90665224960d62` |
| [pilot_manifest.json](pilot_manifest.json) | `470996743752402de052365b6ab79de0ab2fbaf6e89bc42bb481e1bf72f2be77` |
| [train_summary.json](train_summary.json) | `5ede79115d6cca3b2af6abc3548b6a21655f388dd4ef943bdab1f49f494f7210` |

Checkpoints, pilot arrays, and the library-matrix JSONL stay under `artifacts/v05_pilot/` and are not committed. The JSONL holds full design matrices and is too large to keep next to the scores.

```bash
python -m pinnforge.v05 generate
python -m pinnforge.v05 train
python -m pinnforge.v05 score --checkpoint artifacts/v05_pilot/score_cases.jsonl
python -m pinnforge.v05 report
```

