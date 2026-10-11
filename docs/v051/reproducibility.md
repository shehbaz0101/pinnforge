# v051 reproducibility

CPU-only run in this repository. No paid GPU. No secrets. The detached runner is [scripts/v051_run.sh](../../scripts/v051_run.sh).

The protocol [protocol.json](protocol.json) was committed before any v051 test score. It was not edited after that commit. No v0.5 artifact was edited.

- python: `3.12.3`
- numpy: `2.5.4`
- torch: `2.14.1+cpu`
- protocol_sha256: `e1c0d44d1bbd0ac94c448c852f0e069a080a172c3411842abf67b9f43533cf4f`
- validation protocol_sha256: `e1c0d44d1bbd0ac94c448c852f0e069a080a172c3411842abf67b9f43533cf4f`
- PySR used: `False`
- FNO-denoise used: `False`

SHA-256 of committed files:

| file | SHA-256 |
| --- | --- |
| [protocol.json](protocol.json) | `e1c0d44d1bbd0ac94c448c852f0e069a080a172c3411842abf67b9f43533cf4f` |
| [scores.json](scores.json) | `0af0094a69a13a18e12b408117abfd2223084c54a5be28c9736d2ee865b5c154` |
| [validation.json](validation.json) | `0b1c16021dd3600c18315941c655156d43045f9b078db0173f53bf5f0724f5fc` |
| [pilot_manifest.json](pilot_manifest.json) | `ee40794c4d452938ca41cd67f80fa9c256e304a21357ea487e5cc562c1a7b136` |

Pilot arrays and the library-matrix JSONL stay under `artifacts/v051_pilot/` and are not committed.

```bash
python -m pinnforge.v051 generate
python -m pinnforge.v051 score --checkpoint artifacts/v051_pilot/score_cases.jsonl
python -m pinnforge.v051 report
```

