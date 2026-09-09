from pathlib import Path

from PIL import Image

from stellar_anomaly_detection.reproducibility.publication_rendering import PAYLOAD_MAP


ROOT = Path(__file__).resolve().parents[1]


def test_manuscript_figure_numbering_and_outputs():
    assert PAYLOAD_MAP == {"02": "01", "03": "03", "04": "02", "05": "04"}
    for number in range(2, 7):
        for suffix in ("pdf", "svg", "png"):
            assert (ROOT / f"publication/figures/figure_{number:02d}.{suffix}").is_file()


def test_png_exports_are_600_dpi():
    for number in range(2, 7):
        with Image.open(ROOT / f"publication/figures/figure_{number:02d}.png") as image:
            dpi = image.info.get("dpi", (0, 0))
            assert all(abs(value - 600) < 1 for value in dpi)
