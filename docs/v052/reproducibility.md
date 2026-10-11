# v052 reproducibility

CPU-only run in this repository. No paid GPU. No secrets. The detached runner is [scripts/v052_run.sh](../../scripts/v052_run.sh).

The protocol [protocol.json](protocol.json) was committed before any v052 test score. It was not edited after that commit. No v0.5.1 or v0.5 artifact was edited. Burgers was not rescored.

- python: `3.12.3`
- numpy: `2.5.4`
- torch: `2.14.1+cpu`
- protocol_sha256: `0c2f95cfa15f69fb38bd4b6028b294747b63438d6d752286a4b26ec5088c2cc8`
- validation protocol_sha256: `0c2f95cfa15f69fb38bd4b6028b294747b63438d6d752286a4b26ec5088c2cc8`
- PySR used: `False`
- FNO-denoise used: `False`

SHA-256 of committed files:

| file | SHA-256 |
| --- | --- |
| [protocol.json](protocol.json) | `0c2f95cfa15f69fb38bd4b6028b294747b63438d6d752286a4b26ec5088c2cc8` |
| [scores.json](scores.json) | `4163b5d35f421e2b460b810fbc62a6d4531412dc2674a70ae8f57c82a423978e` |
| [validation.json](validation.json) | `ba20a6894c9ad50b2761d19fb058bda692c41b6b4c432e68f8b17ee6df257477` |
| [pilot_manifest.json](pilot_manifest.json) | `4dcade68605db04436419e0e5f5d20ada3c2e22159ec138f76876b45cc87bd60` |
| [diagnosis_evidence.json](diagnosis_evidence.json) | `666aedaca7ccb2f344ff85a7ea3df92b9b81f087a961241ef35f192c17e73dd4` |

Pilot arrays and the library-matrix JSONL stay under `artifacts/v052_pilot/` and are not committed.

```bash
python -m pinnforge.v052 generate
python -m pinnforge.v052 score --checkpoint artifacts/v052_pilot/score_cases.jsonl
python -m pinnforge.v052 report
```

