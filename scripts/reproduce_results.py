#!/usr/bin/env python3
"""Reproduce frozen publication artifacts (Level 2) or validate Level 3 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from stellar_anomaly_detection.representations import (
    REPRESENTATION_IDS,
    representation_registry,
)

ROOT = Path(__file__).resolve().parents[1]


def split_seed(index):
    """Derive the deterministic child seed for one frozen split."""
    raw = json.dumps(
        {
            "root_seed": 42,
            "split_protocol_id": "OBJECT_LEVEL_REPEATED_HOLDOUT_V1",
            "split_index": index,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32)


def splits(object_ids):
    """Materialize the 30 deterministic object-level holdout splits."""
    ids = sorted(object_ids)
    n_eval = int(np.floor(len(ids) * 0.2 + 0.5))
    records = []
    for i in range(30):
        order = np.random.default_rng(split_seed(i)).permutation(len(ids))
        evaluation = sorted(ids[j] for j in order[:n_eval])
        records.append(
            {
                "split_id": f"split_{i:03d}",
                "split_seed": split_seed(i),
                "reference_object_ids": sorted(set(ids) - set(evaluation)),
                "evaluation_object_ids": evaluation,
            }
        )
    return records


def level2(output):
    """Copy the frozen publication tables and numerical registry."""
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "publication/tables", output / "main_tables")
    shutil.copytree(ROOT / "publication/supplementary/tables", output / "supplementary_tables")
    shutil.copy2(
        ROOT / "publication/freeze/registries/final_numerical_results_registry.json",
        output / "authoritative_numerical_results.json",
    )
    summary = {
        "level": 2,
        "scientific_recomputation": False,
        "main_figures": 6,
        "main_tables": 2,
        "supplementary_tables": 5,
        "numerical_registry": "authoritative_numerical_results.json",
        "interpretation": "Frozen publication-artifact reproduction, not a new experiment.",
    }
    (output / "level_2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def validate_full(npz_path, ids_path, output):
    """Validate aligned representation matrices and write frozen splits."""
    if npz_path is None or ids_path is None or not npz_path.is_file() or not ids_path.is_file():
        raise ValueError("full mode requires existing --representations and --object-ids")
    ids = json.loads(ids_path.read_text())
    arrays = np.load(npz_path, allow_pickle=False)
    registry = representation_registry()
    if tuple(arrays.files) != REPRESENTATION_IDS:
        raise ValueError("NPZ keys/order must be the seven frozen representation IDs")
    for key in REPRESENTATION_IDS:
        value = arrays[key]
        if (
            value.shape != (len(ids), registry[key]["dimensionality"])
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"invalid matrix for {key}")
    records = splits(ids)
    output.mkdir(parents=True, exist_ok=False)
    (output / "object_level_splits.json").write_text(
        json.dumps(
            {"protocol": "OBJECT_LEVEL_REPEATED_HOLDOUT_V1", "records": records}, indent=2
        )
        + "\n"
    )
    summary = {
        "mode": "FULL_INPUT_VALIDATION",
        "objects": len(ids),
        "views": 7,
        "splits": 30,
        "evaluation_objects_per_split": len(records[0]["evaluation_object_ids"]),
        "models_executed": False,
        "next_step": (
            "Validated inputs are ready for public IF/LOF/AE APIs; this command "
            "does not silently launch the publication corpus."
        ),
    }
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("validate", "level2", "full"), default="validate")
    p.add_argument("--output", type=Path)
    p.add_argument("--representations", type=Path)
    p.add_argument("--object-ids", type=Path)
    args = p.parse_args()
    protocol = json.loads(
        (
            ROOT / "publication/supplementary/registries/experimental_protocol_registry.json"
        ).read_text()
    )
    if args.mode == "validate":
        assert (
            protocol["splits"]["count"] == 30
            and protocol["population"]["objects"] == 33
            and len(representation_registry()) == 7
        )
        print("Frozen protocol and publication inventories are valid.")
        return 0
    if args.output is None:
        p.error("--output is required for level2/full")
    print(
        json.dumps(
            (
                level2(args.output)
                if args.mode == "level2"
                else validate_full(args.representations, args.object_ids, args.output)
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
