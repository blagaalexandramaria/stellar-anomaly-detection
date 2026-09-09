#!/usr/bin/env python3
"""Run the public feature pipeline from user-supplied observations.

The bundled Level 1 workflow is available in examples/quickstart.py. Level 3
requires a CSV plus frozen operational-context JSON; scientifically meaningful
reference decisions are never guessed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stellar_anomaly_detection.graph.features import (
    extract_graph_all,
    extract_graph_core,
    extract_graph_extended,
    validate_graph_for_feature_extraction,
)
from stellar_anomaly_detection.pisd import PhysicsInformedSpectralDescriptor
from stellar_anomaly_detection.representations import RepresentationBuilder
from stellar_anomaly_detection.spectral import LombScargleAnalyzer, analyze_spectral_window

WORKFLOW = (
    "ingestion/validation",
    "Lomb-Scargle",
    "spectral window",
    "frozen operational interpretation input",
    "PISD",
    "spectral graph",
    "Graph-Core",
    "Graph-Extended",
    "seven representation views",
)


def parser():
    """Build the command-line parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", choices=("1", "3"), default="3")
    p.add_argument(
        "--input", type=Path, help="CSV containing time, flux, and optional flux_error"
    )
    p.add_argument("--operational-context", type=Path, help="Frozen downstream context JSON")
    p.add_argument("--output", type=Path)
    p.add_argument("--describe", action="store_true")
    return p


def load_observation(path):
    """Load finite time, flux, and optional uncertainty columns from CSV."""
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError("--input must be an existing CSV")
    table = np.genfromtxt(path, delimiter=",", names=True)
    names = set(table.dtype.names or ())
    if not {"time", "flux"}.issubset(names):
        raise ValueError("input CSV requires time and flux columns")
    time = np.atleast_1d(table["time"]).astype(float)
    flux = np.atleast_1d(table["flux"]).astype(float)
    error = np.atleast_1d(table["flux_error"]).astype(float) if "flux_error" in names else None
    if len(time) < 3 or not np.isfinite(time).all() or not np.isfinite(flux).all():
        raise ValueError("time/flux require at least three finite rows")
    return time, flux, error


def main():
    args = parser().parse_args()
    if args.describe:
        print("\n".join(f"{i}. {v}" for i, v in enumerate(WORKFLOW, 1)))
        return 0
    if args.level == "1" and args.input is None:
        raise SystemExit(
            "Level 1 Quick Start: run examples/quickstart.py, "
            "or supply an --input CSV and --output directory"
        )
    if not args.input or not args.output:
        parser().error("--input and --output are required")
    time, flux, error = load_observation(args.input)
    spectrum = LombScargleAnalyzer().analyze(time, flux, error)
    window = analyze_spectral_window(time, spectrum.frequency)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "spectral").mkdir()
    (args.output / "spectral/spectrum_and_window.json").write_text(
        json.dumps(
            {"lomb_scargle": spectrum.to_dict(), "spectral_window": window.to_dict()}, indent=2
        )
        + "\n"
    )
    if args.operational_context is None:
        raise SystemExit(
            "Spectral outputs written; downstream execution requires "
            "--operational-context and was not claimed complete"
        )
    context = json.loads(args.operational_context.read_text())
    required = {
        "object_id",
        "harmonic_features",
        "stability_features",
        "graph_nodes",
        "graph_edges",
    }
    if not required.issubset(context):
        raise ValueError(f"operational context missing {sorted(required - set(context))}")
    pisd = PhysicsInformedSpectralDescriptor().transform(
        spectrum.frequency,
        spectrum.power,
        time_baseline=spectrum.baseline_days,
        harmonic_features=context["harmonic_features"],
        stability_features=context["stability_features"],
    )
    graph_nodes = list(context["graph_nodes"])
    graph_edges = list(context["graph_edges"])
    validate_graph_for_feature_extraction(graph_nodes, graph_edges)
    _, core = extract_graph_core(graph_nodes, graph_edges)
    _, extended = extract_graph_extended(graph_nodes, graph_edges)
    graph_all = extract_graph_all(graph_nodes, graph_edges)
    views = RepresentationBuilder().build(pisd, core, extended)
    for d in ("pisd", "graph", "representations"):
        (args.output / d).mkdir()
    np.save(args.output / "pisd/pisd.npy", pisd, allow_pickle=False)
    np.save(args.output / "graph/graph_core.npy", core, allow_pickle=False)
    np.save(args.output / "graph/graph_extended.npy", extended, allow_pickle=False)
    np.save(args.output / "graph/graph_all.npy", graph_all, allow_pickle=False)
    np.savez(args.output / "representations/seven_views.npz", **views)
    (args.output / "reproduction_summary.json").write_text(
        json.dumps(
            {
                "object_id": context["object_id"],
                "pisd_dimension": 42,
                "graph_dimensions": {"core": 30, "extended": 36, "all": 66},
                "representation_views": list(views),
                "source_data_supplied_by_user": True,
            },
            indent=2,
        )
        + "\n"
    )
    print("Feature pipeline complete for one user-supplied observation.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        parser().error(str(exc))
