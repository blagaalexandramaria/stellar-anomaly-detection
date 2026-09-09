#!/usr/bin/env python3
"""Run the public feature and representation APIs on bundled synthetic data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stellar_anomaly_detection.graph import (
    extract_graph_all,
    extract_graph_core,
    extract_graph_extended,
)
from stellar_anomaly_detection.pisd import PhysicsInformedSpectralDescriptor
from stellar_anomaly_detection.pisd.components import (
    HarmonicStructure,
    SpectralMorphology,
    SpectralStability,
)
from stellar_anomaly_detection.representations import RepresentationBuilder
from stellar_anomaly_detection.spectral import LombScargleAnalyzer

DATA_PATH = Path(__file__).resolve().parent / "data" / "synthetic_demo_light_curve.csv"


def demo_graph(peak_analysis, harmonic_analysis):
    """Map public peak/harmonic results to the public graph feature schema."""
    peak_count = int(peak_analysis.peak_indices.size)
    if peak_count == 0 or harmonic_analysis.fundamental_peak_position < 0:
        raise RuntimeError("Synthetic demo did not produce retained spectral peaks")

    reference_position = int(harmonic_analysis.fundamental_peak_position)
    dominant_position = int(np.argmax(peak_analysis.peak_heights))
    harmonic_orders = {reference_position: 1}
    harmonic_orders.update(
        {
            int(position): int(order)
            for position, order in zip(
                harmonic_analysis.harmonic_peak_positions,
                harmonic_analysis.harmonic_orders,
                strict=True,
            )
        }
    )
    nodes = [
        {
            "node_id": f"node:peak_{position:03d}",
            "node_index": position,
            "frequency": float(peak_analysis.peak_frequencies[position]),
            "power": float(peak_analysis.peak_heights[position]),
            "is_operational_reference": position == reference_position,
            "is_dominant": position == dominant_position,
            "is_canonical_harmonic": position in harmonic_orders,
            "canonical_harmonic_order": harmonic_orders.get(position),
            "alias_advisory_present": False,
            "is_primary_sideband_member": False,
        }
        for position in range(peak_count)
    ]
    edges = []
    for index, (position, residual) in enumerate(
        zip(
            harmonic_analysis.harmonic_peak_positions,
            harmonic_analysis.normalized_frequency_deviations,
            strict=True,
        )
    ):
        normalized_residual = float(residual)
        edges.append(
            {
                "edge_id": f"edge:harmonic_{index:03d}",
                "source_node_id": f"node:peak_{reference_position:03d}",
                "target_node_id": f"node:peak_{int(position):03d}",
                "edge_type": "HARMONIC",
                "confidence_class": "HIGH",
                "normalized_resolution_residual": normalized_residual,
                "relationship_similarity_weight": 1.0 / (1.0 + normalized_residual),
            }
        )
    return nodes, edges


def main() -> int:
    table = np.genfromtxt(DATA_PATH, delimiter=",", names=True)
    time = np.asarray(table["time"], dtype=np.float64)
    flux = np.asarray(table["flux"], dtype=np.float64)
    flux_error = np.asarray(table["flux_error"], dtype=np.float64)

    analyzer = LombScargleAnalyzer(
        minimum_frequency=1.2,
        maximum_frequency=4.0,
        false_alarm_probability=None,
    )
    spectrum = analyzer.analyze(
        time,
        flux,
        flux_error,
        flux_representation="global_relative",
    )
    morphology = SpectralMorphology()
    peaks = morphology.detect_peaks(
        spectrum.frequency,
        spectrum.power,
        spectrum.baseline_days,
    )
    harmonic_component = HarmonicStructure()
    harmonic_analysis = harmonic_component.analyze(
        peaks,
        spectrum.baseline_days,
        spectrum.maximum_frequency,
    )
    harmonic_features = harmonic_component.extract(
        peaks,
        spectrum.baseline_days,
        spectrum.maximum_frequency,
    )
    stability_features = SpectralStability(analyzer).extract(
        time,
        flux,
        flux_error,
    )
    pisd = PhysicsInformedSpectralDescriptor().transform(
        spectrum.frequency,
        spectrum.power,
        time_baseline=spectrum.baseline_days,
        harmonic_features=harmonic_features,
        stability_features=stability_features,
    )

    nodes, edges = demo_graph(peaks, harmonic_analysis)
    _, graph_core = extract_graph_core(nodes, edges)
    _, graph_extended = extract_graph_extended(nodes, edges)
    graph_all = extract_graph_all(nodes, edges)
    views = RepresentationBuilder().build(pisd, graph_core, graph_extended)

    print("Synthetic demo loaded")
    print(f"Samples: {time.size}")
    print(f"Lomb–Scargle frequencies: {spectrum.frequency_samples}")
    print(f"Dominant frequency: {spectrum.dominant_frequency:.6f}")
    print(f"Retained spectral peaks: {len(nodes)}")
    print(f"Typed graph edges: {len(edges)}")
    print(f"PISD features: {pisd.size}")
    print(
        "Graph features: "
        f"Core={graph_core.size}, Extended={graph_extended.size}, All={graph_all.size}"
    )
    print(f"Representation views: {len(views)}")
    print("Demo completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
