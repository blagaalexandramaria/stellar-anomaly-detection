"""Read-only rendering of manuscript Figures 2--5 from frozen payloads."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

MODEL_ORDER = ("IF", "LOF", "AE")
MODEL_COLORS = {"IF": "#0072B2", "LOF": "#E69F00", "AE": "#7B2CBF"}
NEUTRAL = "#4C5966"
GRID = "#D9E0E6"
VIEW_LABELS = (
    "PISD",
    "Graph-Core",
    "Graph-Ext.",
    "Graph-All",
    "PISD+Core",
    "PISD+Ext.",
    "PISD+All",
)
CONTRAST_LABELS = ("PISD ↔ Graph-All", "PISD ↔ PISD+All", "Graph-All ↔ PISD+All")


def _setup():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/stellar-anomaly-public-mpl")
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("Install the visualization extra to render figures") from exc
    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "svg.hashsalt": "stage-21-4",
        }
    )
    return __import__("matplotlib.pyplot", fromlist=["plt"])


def _letter(ax, text):
    ax.text(
        -0.13, 1.10, text, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom"
    )


def figure_02(p):
    """Render the representation-comparison figure from its frozen payload."""
    plt = _setup()
    fig = plt.figure(figsize=(13.8, 7.55), constrained_layout=True)
    grid = fig.add_gridspec(
        2, 4, height_ratios=(1.18, 0.95), width_ratios=(1, 1, 1, 0.052), hspace=0.12, wspace=0.18
    )
    images = []
    for i, entry in enumerate(p["matrices"]):
        ax = fig.add_subplot(grid[0, i])
        vals = np.array([[v["value"] for v in row] for row in entry["matrix"]])
        images.append(ax.imshow(vals, cmap="viridis", vmin=0, vmax=1))
        ax.set(
            xticks=range(7),
            yticks=range(7),
            xticklabels=VIEW_LABELS,
            yticklabels=VIEW_LABELS,
            title={
                "IF": "A1  Isolation Forest",
                "LOF": "A2  Local Outlier Factor",
                "AE": "A3  Autoencoder",
            }[entry["model"]],
        )
        ax.title.set_fontsize(10)
        ax.title.set_fontweight("semibold")
        ax.tick_params(axis="x", rotation=38, labelsize=8.2, pad=3)
        ax.tick_params(axis="y", labelsize=8.2)
        for r in range(7):
            for c in range(7):
                ax.text(
                    c,
                    r,
                    f"{vals[r,c]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color="white" if vals[r, c] < 0.52 else "#17212B",
                )
    cb = fig.colorbar(images[0], cax=fig.add_subplot(grid[0, 3]))
    cb.ax.tick_params(labelsize=8.2)
    cb.set_label("Spearman rank agreement", fontsize=9.2, labelpad=8)
    ax = fig.add_subplot(grid[1, :3])
    _letter(ax, "B")
    for ci, contrast in enumerate(CONTRAST_LABELS):
        for mi, model in enumerate(MODEL_ORDER):
            rec = next(
                x for x in p["contrasts"] if x["contrast"] == contrast and x["model"] == model
            )
            x = ci + (mi - 1) * 0.19
            ax.vlines(
                x,
                rec["q25_spearman"],
                rec["q75_spearman"],
                color=MODEL_COLORS[model],
                lw=2.8,
                zorder=2,
            )
            ax.scatter(
                x,
                rec["mean_spearman"],
                s=55,
                color=MODEL_COLORS[model],
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
                label=model if ci == 0 else None,
            )
    ax.set(
        xticks=range(3),
        xticklabels=CONTRAST_LABELS,
        ylim=(0, 1.04),
        ylabel="Frozen Spearman rank agreement",
        title="Predeclared representation contrasts (mean and interquartile range)",
    )
    ax.tick_params(axis="x", labelsize=8.5, pad=5)
    ax.tick_params(axis="y", labelsize=8)
    ax.title.set_fontsize(10)
    ax.grid(axis="y", color=GRID, lw=0.55)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower left",
        fontsize=8.7,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.suptitle(p["working_title"], fontweight="bold", fontsize=14)
    return fig


def figure_04(p):
    """Render the ranking-robustness figure from its frozen payload."""
    plt = _setup()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), constrained_layout=True, sharey=True)
    labels = [x["label"] for x in p["scenarios"]]
    y = np.arange(len(labels))
    cols = [NEUTRAL, NEUTRAL, MODEL_COLORS["IF"], MODEL_COLORS["LOF"], MODEL_COLORS["AE"]]
    for ai, (ax, key_a, key_b, title, xlabel) in enumerate(
        (
            (
                axes[0],
                "kendall",
                "spearman",
                "Rank concordance (circle: Spearman; square: Kendall)",
                "Rank concordance",
            ),
            (
                axes[1],
                "mean_abs_displacement",
                "max_abs_displacement",
                "Rank displacement (circle: mean; square: maximum)",
                "Absolute rank displacement",
            ),
        )
    ):
        _letter(ax, "A" if ai == 0 else "B")
        for i, rec in enumerate(p["scenarios"]):
            ax.plot(
                [rec[key_a], rec[key_b]],
                [i, i],
                color=cols[i],
                lw=2.35,
                solid_capstyle="round",
                zorder=1,
            )
            ax.scatter(rec[key_a], i, color=cols[i], marker="s", s=58, zorder=2)
            ax.scatter(rec[key_b], i, color=cols[i], marker="o", s=58, zorder=3)
        ax.set(yticks=y, yticklabels=labels, xlabel=xlabel, title=title)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, lw=0.55)
        ax.tick_params(axis="both", labelsize=8.5)
        ax.title.set_fontsize(9.5)
    axes[0].set_xlim(0, 1.03)
    axes[1].set_xlim(-0.25, 9.55)
    fig.suptitle(p["working_title"], fontweight="bold", fontsize=14)
    return fig


def figure_03(p):
    """Render the cross-model rank-flow figure from its frozen payload."""
    plt = _setup()
    fig = plt.figure(figsize=(14.3, 9.6), constrained_layout=True)
    grid = fig.add_gridspec(
        2, 2, height_ratios=(1.08, 1.52), width_ratios=(1, 1.08), hspace=0.13, wspace=0.18
    )
    models = list(MODEL_ORDER)
    pairs = {
        (x["left_model"].upper(), x["right_model"].upper()): x for x in p["pairwise_statistics"]
    }
    sub = grid[0, 0].subgridspec(1, 2, wspace=0.28)
    for j, metric in enumerate(("spearman_coefficient", "kendall_coefficient")):
        ax = fig.add_subplot(sub[0, j])
        vals = np.ones((3, 3))
        for i, left in enumerate(models):
            for k, right in enumerate(models):
                if left != right:
                    vals[i, k] = (pairs.get((left, right)) or pairs[(right, left)])[metric]
        ax.imshow(vals, cmap="viridis", vmin=0, vmax=1)
        ax.set(
            xticks=range(3),
            yticks=range(3),
            xticklabels=models,
            yticklabels=models,
            title=("A  Spearman" if j == 0 else "Kendall"),
        )
        ax.title.set_fontsize(10)
        ax.tick_params(labelsize=8.5)
        for i in range(3):
            for k in range(3):
                ax.text(
                    k,
                    i,
                    f"{vals[i,k]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color="white" if vals[i, k] < 0.62 else "#17212B",
                )
    fig.text(
        0.075, 0.588, f"Global Kendall W = {p['kendall_w']:.3f}", fontsize=9.3, color=NEUTRAL
    )
    ax = fig.add_subplot(grid[0, 1])
    _letter(ax, "B")
    names = ["IF–LOF", "IF–AE", "LOF–AE", "All three"]
    cols = [MODEL_COLORS["IF"], MODEL_COLORS["LOF"], MODEL_COLORS["AE"], NEUTRAL]
    width = 0.17
    for ri, rec in enumerate(p["top_k_records"]):
        vals = [q["intersection_count"] for q in rec["pairwise"]] + [
            rec["three_model_intersection_count"]
        ]
        for bi, value in enumerate(vals):
            ax.bar(
                ri + (bi - 1.5) * width,
                value,
                width,
                color=cols[bi],
                label=names[bi] if ri == 0 else None,
            )
    ax.set(
        xticks=np.arange(3),
        xticklabels=[f"k = {k}" for k in p["top_k_values"]],
        ylabel="Intersection count",
        title="Predefined top-k overlap",
    )
    ax.grid(axis="y", color=GRID, lw=0.55)
    ax.tick_params(labelsize=8.5)
    ax.title.set_fontsize(10)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        fontsize=8.7,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    ax = fig.add_subplot(grid[1, :])
    _letter(ax, "C")
    xvals = np.arange(3)
    highlight = set(p["highlighted_object_ids"])
    rows = {r["object_id"]: r for r in p["rank_rows"]}
    for rec in p["rank_rows"]:
        ys = [rec[m] for m in models]
        is_high = rec["object_id"] in highlight
        ax.plot(
            xvals,
            ys,
            color="#BFC9D2" if not is_high else "#1B3448",
            alpha=0.32 if not is_high else 0.94,
            lw=0.55 if not is_high else 1.45,
            zorder=1 if not is_high else 2,
        )
        if is_high:
            ax.scatter(xvals, ys, s=19, color="#1B3448", zorder=3)
    # Deterministic collision-free external label slots; AE rank coordinates remain unchanged.
    # Sort label slots by their already-frozen AE endpoint rank only; no plotted
    # coordinates, ranks, or membership are altered.
    label_order = sorted(p["highlighted_object_ids"], key=lambda oid: (rows[oid]["AE"], oid))
    slot_y = np.linspace(1.0, 15.2, len(label_order))
    for oid, target_y in zip(label_order, slot_y):
        rank = rows[oid]["AE"]
        label = oid.replace("object:epic:", "EPIC ")
        ax.annotate(
            label,
            xy=(2, rank),
            xytext=(2.16, target_y),
            textcoords="data",
            fontsize=7.25,
            va="center",
            color="#1B3448",
            arrowprops={
                "arrowstyle": "-",
                "color": "#758594",
                "lw": 0.55,
                "shrinkA": 1,
                "shrinkB": 2,
            },
            annotation_clip=False,
        )
    ax.set(
        xlim=(-0.08, 2.68),
        xticks=xvals,
        xticklabels=models,
        ylim=(33.7, 0.3),
        ylabel="Frozen model-specific rank (1 = most unusual)",
        title=(
            "All 33 canonical objects; labels mark the frozen union of "
            "model-specific top-10 memberships"
        ),
    )
    ax.grid(axis="y", color=GRID, lw=0.42, alpha=0.75)
    ax.tick_params(labelsize=8.5)
    ax.title.set_fontsize(9.5)
    fig.suptitle(p["working_title"], fontweight="bold", fontsize=14)
    return fig


def figure_05(p):
    """Render representative evidence cases from their frozen payload."""
    plt = _setup()
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(14.8, 9.35),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1, 1, 0.62), "hspace": 0.18, "wspace": 0.18},
    )
    for i, case in enumerate(p["cases"]):
        oid = case["object_id"].replace("object:epic:", "EPIC ")
        axes[0, i].set_title(
            f"{case['case_type'].replace('_',' ')} — {oid}\n{FIGURE_05_ROLE_LABELS[case['case_type']]}",
            fontsize=9.2,
            fontweight="semibold",
            pad=7,
        )
        ax = axes[0, i]
        ax.plot(
            case["time_bjd_minus_2450000"],
            case["kp_magnitude"],
            ".",
            ms=1.55,
            color="#31475A",
            alpha=0.82,
        )
        ax.invert_yaxis()
        ax.set(xlabel="BJD − 2450000 [d]", ylabel="Kp [mag]" if i == 0 else "")
        ax.grid(color=GRID, lw=0.42)
        ax.tick_params(labelsize=8.5)
        ax = axes[1, i]
        ax.plot(
            case["frequency_day_inverse"],
            case["lomb_scargle_power"],
            lw=0.9,
            color=MODEL_COLORS["AE"],
        )
        ax.set(xlabel="Frequency [d⁻¹]", ylabel="LS power" if i == 0 else "")
        ax.grid(color=GRID, lw=0.42)
        ax.tick_params(labelsize=8.5)
        ax = axes[2, i]
        for j, model in enumerate(MODEL_ORDER):
            rank = case["ranks"][model]
            ax.scatter(rank, j, s=64, color=MODEL_COLORS[model], zorder=3)
            ax.text(
                rank, j - 0.19, str(rank), ha="center", va="bottom", fontsize=8.3, color=NEUTRAL
            )
        ax.set(
            xlim=(33.5, 0.5),
            yticks=range(3),
            yticklabels=MODEL_ORDER if i == 0 else [],
            xlabel="Rank (1 = most unusual)",
        )
        ax.grid(axis="x", color=GRID, lw=0.48)
        ax.tick_params(labelsize=8.5)
    for idx, (letter, row, label) in enumerate(
        (
            ("A", 0, "Frozen light curve"),
            ("B", 1, "Frozen Lomb–Scargle spectrum"),
            ("C", 2, "Frozen ordinal ranks"),
        )
    ):
        fig.text(0.008, (0.965, 0.645, 0.315)[idx], letter, fontsize=11.5, fontweight="bold", va="top")
    fig.suptitle(p["working_title"], fontweight="bold", fontsize=14)
    return fig


RENDERERS = {"02": figure_02, "03": figure_03, "04": figure_04, "05": figure_05}
PAYLOAD_MAP = {"02": "01", "03": "03", "04": "02", "05": "04"}
FIGURE_05_ROLE_LABELS = {
    "CASE_A": "Robust-high-consensus case",
    "CASE_B": "Disagreement control",
    "CASE_C": "Robust-high-consensus case",
    "CASE_D": "Robust-high-consensus case",
}


def render_non_graph(
    payload_directory: Path, output_directory: Path, numbers=("02", "03", "04", "05")
):
    """Render manuscript Figures 2--5 without modifying frozen JSON payloads."""
    output_directory.mkdir(parents=True, exist_ok=True)
    created = []
    for number in numbers:
        payload_number = PAYLOAD_MAP[number]
        payload = json.loads((payload_directory / f"figure_{payload_number}_data.json").read_text())
        fig = RENDERERS[number](payload)
        for ext in ("pdf", "svg", "png"):
            target = output_directory / f"figure_{number}.{ext}"
            fig.savefig(
                target,
                dpi=600 if ext == "png" else None,
                bbox_inches="tight",
                metadata={"Creator": "stellar-anomaly-detection"},
            )
            created.append(target)
        _setup().close(fig)
    return created
