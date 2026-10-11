# v0.4 reproducibility

CPU-only run in this repository. No paid GPU. The optional [scripts/v04_colab.py](../../scripts/v04_colab.py) uses the same commands.

| file | SHA-256 |
| --- | --- |
| [protocol.json](protocol.json) | `3f4f49e54010bdfcf671c07fbd6ee605573d2a92a8247b1a16c43861bf5965d7` |
| [scores.json](scores.json) | `7be72b9fb526799ade8f527b6695c5c53c1757b37d81b5b71c517e437b1575b4` |
| [validation.json](validation.json) | `40ea938196780490fa87a8af821217f481aeab0c74c063265215aab51c18fc3b` |
| [pilot_manifest.json](pilot_manifest.json) | `0ce27f657e8ee5ba6e8befe6f977aa0db7a55890ec5af1a2217e0008b57bd926` |
| [convergence.json](convergence.json) | `7d4fee0dbdf9e5a98479a30cf718c056654e39883f0522c9be0bd9167bdc13fa` |
| [train_summary.json](train_summary.json) | `840a5e57c64a1df985800db7dd92fedb0ea91a9d0420af7d4d2d4eea09ef6404` |

Python 3.12.3, NumPy 2.5.4, torch 2.14.1+cpu. Those three strings are also stored on [scores.json](scores.json). Checkpoints and pilot arrays stay under `artifacts/v04_pilot/` and are not committed.

```bash
python -m pinnforge.v04 generate
python -m pinnforge.v04 convergence
python -m pinnforge.v04 train
python -m pinnforge.v04 score
python -m pinnforge.v04 report
```
