"""Anonymous fetch of the validation files used by the identifiability sweep."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pinnforge.v06.download import download_file, file_sha256
from pinnforge.v061.identifiability import VALID_FILES

RAW = Path("artifacts/v061_raw")
EXISTING = Path("artifacts/v06_raw")
MANIFEST = Path("docs/v061/download_valid_manifest.json")


def run_download_valid() -> dict[str, Any]:
    RAW.mkdir(parents=True, exist_ok=True)
    records = []
    for relative in VALID_FILES:
        dest = RAW / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        prior = EXISTING / relative
        if dest.is_file():
            path = dest
        elif prior.is_file():
            os.link(prior, dest)
            path = dest
        else:
            path = download_file(relative, RAW)
        records.append(
            {
                "path": relative,
                "local": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
        print(f"have {relative} {records[-1]['bytes']} {records[-1]['sha256'][:12]}", flush=True)
    payload = {
        "format": "pinnforge.v061_download_valid_manifest.v1",
        "role": "validation-file provenance for the identifiability sweep. Not a test score.",
        "repo": "polymathic-ai/shear_flow",
        "anonymous": True,
        "files": records,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
