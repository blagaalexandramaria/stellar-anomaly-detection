import json
from pathlib import Path

from stellar_anomaly_detection.representations import REPRESENTATION_IDS

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_publication_inventory_and_levels():
    config = json.loads((ROOT / "configs/reproducibility.json").read_text())
    assert config["inventories"] == {
        "main_figures": 6,
        "main_tables": 2,
        "supplementary_tables": 5,
    }
    assert len(REPRESENTATION_IDS) == 7
    inventory = json.loads((ROOT / "publication/publication_inventory.json").read_text())
    assert len(inventory["figures"]) == 6
    assert all(
        (ROOT / "publication/figures" / f"{entry['basename']}.pdf").is_file()
        for entry in inventory["figures"]
    )
    assert len(list((ROOT / "publication/tables").glob("table_??_*.csv"))) == 2
    assert (
        len(list((ROOT / "publication/supplementary/tables").glob("table_s[1-5]_*.csv"))) == 5
    )


def test_level_boundaries_are_explicit():
    requirements = json.loads((ROOT / "configs/data_requirements.json").read_text())
    assert requirements["level_1"]["status"] == "OPERATIONAL_WITH_SYNTHETIC_DEMO"
    assert requirements["level_3"]["status"] == "CONDITIONAL_ON_SOURCE_DATA"
    assert requirements["level_3"]["raw_observational_corpus_bundled"] is False
