"""Claim-to-evidence checks. These tests do not import torch."""

from __future__ import annotations

from pathlib import Path

import yaml

from pinnforge.claims import check

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "docs" / "research" / "strong_baselines_scores.json"
DEMO = ROOT / "docs" / "demo" / "result.json"


def _ranking() -> dict[str, object]:
    return {
        "evidence": str(SCORES.resolve()),
        "fno_methods": ["fno_hybrid_1e-2", "fno_data_only"],
        "methods": {
            "solver_oracle_ic": "methods/solver_oracle_ic/slices/hard_ood/mean_rel_error",
            "solver_sensor_ic": "methods/solver_sensor_ic/slices/hard_ood/mean_rel_error",
            "fno_hybrid_1e-2": "fno/arms/hybrid_1e-2/slices/hard_ood/mean_rel_error",
            "fno_data_only": "fno/arms/data_only/slices/hard_ood/mean_rel_error",
            "weak_form": "methods/weak_form/slices/hard_ood/mean_rel_error",
            "closed_form_ls": "methods/closed_form_ls/slices/hard_ood/mean_rel_error",
        },
    }


def _registry(tmp_path: Path, claims: list[dict[str, object]], files: list[str]) -> Path:
    document = {
        "version": 1,
        "stronger_baselines": _ranking(),
        "scan": {"superiority": files, "numbers": files},
        "claims": claims,
    }
    path = tmp_path / "claims.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_repository_claims_match_evidence() -> None:
    failures = check(ROOT)
    assert [item.format() for item in failures] == []


def test_registered_fno_best_claim_fails_against_strong_baselines(tmp_path: Path) -> None:
    (tmp_path / "claim.md").write_text(
        "The FNO inverse is the best method on hard_ood.\n",
        encoding="utf-8",
    )
    registry = _registry(
        tmp_path,
        [
            {
                "id": "fake-fno-best",
                "class": "superiority",
                "assertion": {
                    "kind": "best",
                    "scope": "hard_ood_inverse_mean_rel",
                    "method": "fno_inverse",
                },
                "locations": [{"file": "claim.md", "pattern": "best method on hard_ood"}],
            }
        ],
        ["claim.md"],
    )
    failures = check(tmp_path, registry)
    assert len(failures) == 1
    assert failures[0].claim_id == "fake-fno-best"
    assert "solver_sensor_ic 0.0138" in failures[0].message
    assert "fno_hybrid_1e-2 0.0337" in failures[0].message
    assert "closed_form_ls 0.349" in failures[0].message


def test_unregistered_superiority_phrase_fails(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text(
        "Persistence is the reference. The fit outperforms that reference.\n",
        encoding="utf-8",
    )
    registry = _registry(
        tmp_path,
        [
            {
                "id": "names-the-reference",
                "class": "superiority",
                "assertion": {"kind": "not_a_ranking", "note": "names the reference"},
                "locations": [{"file": "note.md", "pattern": "Persistence is the reference"}],
            }
        ],
        ["note.md"],
    )
    failures = check(tmp_path, registry)
    assert [item.message for item in failures] == ["unregistered superiority phrase 'outperforms'"]


def test_overrounded_percent_fails(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text(
        "Closed form mean relative error is 0.50%.\n",
        encoding="utf-8",
    )
    registry = _registry(
        tmp_path,
        [
            {
                "id": "bad-percent",
                "class": "numeric",
                "numbers": [
                    {
                        "token": "0.50%",
                        "files": ["note.md"],
                        "evidence": str(DEMO.resolve()),
                        "path": "aggregates/closed_form/mean_rel_error",
                        "scale": 100,
                    }
                ],
            }
        ],
        ["note.md"],
    )
    failures = check(tmp_path, registry)
    assert len(failures) == 1
    assert failures[0].claim_id == "bad-percent"
    assert "0.50%" in failures[0].message
