#!/usr/bin/env python3
"""Export or render manuscript Figures 1--6 from accepted frozen inputs."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
from pathlib import Path

from stellar_anomaly_detection.reproducibility import render_non_graph

ROOT = Path(__file__).resolve().parents[1]


def accepted_files():
    """Return the accepted six-figure, three-format artifact inventory."""
    return [ROOT / f"publication/figures/figure_1_methodology.{ext}" for ext in ("pdf", "svg", "png")] + [
        ROOT / f"publication/figures/figure_{number:02d}.{ext}"
        for number in range(2, 7)
        for ext in ("pdf", "svg", "png")
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("export", "render", "list"), default="list")
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--figures",
        nargs="*",
        choices=("1", "2", "3", "4", "5", "6"),
        default=["1", "2", "3", "4", "5", "6"],
    )
    args = p.parse_args()
    if args.mode == "list":
        print("\n".join(f"Figure {i}" for i in range(1, 7)))
        return 0
    if args.output is None:
        p.error("--output is required for export/render")
    args.output.mkdir(parents=True, exist_ok=False)
    selected = {int(x) for x in args.figures}
    created = []
    if args.mode == "export":
        for source in accepted_files():
            source_number = 1 if source.stem == "figure_1_methodology" else int(source.stem.split("_")[1])
            if source_number in selected:
                target = args.output / source.name
                shutil.copy2(source, target)
                created.append(target)
    else:
        if 1 in selected:
            previous = os.environ.get("STELLAR_FIGURE_OUTPUT_DIR")
            os.environ["STELLAR_FIGURE_OUTPUT_DIR"] = str(args.output)
            try:
                runpy.run_path(str(ROOT / "scripts/figure_1_methodology.py"), run_name="__main__")
            finally:
                if previous is None:
                    os.environ.pop("STELLAR_FIGURE_OUTPUT_DIR", None)
                else:
                    os.environ["STELLAR_FIGURE_OUTPUT_DIR"] = previous
            created.extend(args.output / f"figure_1_methodology.{ext}" for ext in ("pdf", "svg", "png"))
        non_graph = tuple(f"{i:02d}" for i in sorted(selected & {2, 3, 4, 5}))
        created.extend(render_non_graph(ROOT / "publication/payloads", args.output, non_graph))
        if 6 in selected:
            # Figure 6's accepted vector artifact is itself the frozen
            # public rendering payload. Copying them preserves the accepted topology
            # without rerunning graph construction, layout, or community analysis.
            for ext in ("pdf", "svg", "png"):
                source = ROOT / f"publication/figures/figure_06.{ext}"
                target = args.output / source.name
                shutil.copy2(source, target)
                created.append(target)
    (args.output / "rendering_summary.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "figures": sorted(selected),
                "files": [x.name for x in created],
                "figure_1_source": "frozen public methodology payload",
                "figure_2_5_source": "frozen public JSON payloads",
                "figure_6_source": "accepted frozen graph-render artifact",
                "community_detection_rerun": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{args.mode.title()} complete: {len(created)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
