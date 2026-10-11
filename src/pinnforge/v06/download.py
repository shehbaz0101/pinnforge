"""Anonymous Hugging Face fetch for the frozen file list."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from pinnforge.reference.spectral2d.dataset import load_protocol, protocol_sha256

PROTOCOL = Path("docs/v06/protocol.json")
RAW = Path("artifacts/v06_raw")
MANIFEST = Path("docs/v06/download_manifest.json")


def listed_files(protocol: dict[str, Any]) -> list[str]:
    files = protocol["subset"]["files"]
    names: list[str] = []
    for key in ("train", "valid", "test", "ood_valid", "ood_test"):
        names.extend(str(item) for item in files[key])
    return names


def download_file(relative: str, dest: Path, *, attempts: int = 5) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    dest.mkdir(parents=True, exist_ok=True)
    delay = 4.0
    last: Exception | None = None
    for _ in range(int(attempts)):
        try:
            path = hf_hub_download(
                repo_id="polymathic-ai/shear_flow",
                repo_type="dataset",
                filename=relative,
                local_dir=str(dest),
                token=False,
            )
            return Path(path)
        except HfHubHTTPError as exc:
            last = exc
            status = getattr(exc.response, "status_code", None) if getattr(exc, "response", None) else None
            if status not in {429, 500, 502, 503, 504}:
                raise
            time.sleep(delay)
            delay *= 2.0
        except Exception as exc:  # network
            last = exc
            time.sleep(delay)
            delay *= 2.0
    raise RuntimeError(f"download failed after backoff: {relative}: {last}") from last


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_download(*, tiny: bool = False) -> dict[str, Any]:
    protocol = load_protocol(PROTOCOL)
    RAW.mkdir(parents=True, exist_ok=True)
    records = []
    names = listed_files(protocol)
    if tiny:
        names = names[:1]
    for relative in names:
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
        "format": "pinnforge.v06_download_manifest.v1",
        "protocol_sha256": protocol_sha256(PROTOCOL),
        "repo": "polymathic-ai/shear_flow",
        "anonymous": True,
        "files": records,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
