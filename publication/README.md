# Publication artifacts

This directory contains the frozen public publication layer. It is an artifact and registry authority, not a working directory for exploratory outputs.

- `figures/` contains accepted manuscript Figures 1--6 in PDF, PNG, and SVG forms.
- `tables/` contains two main tables in CSV, JSON, and TeX.
- `supplementary/tables/` contains the earlier five-table frozen results bundle used by the Level 2 artifact workflow.
- `supplementary/manuscript_tables/` contains the current globally numbered Tables S1--S27 for Supplementary Sections S1--S8, with CSV, JSON, and LaTeX exports plus source/provenance indexes.
- `supplementary/registries/` contains the experimental protocol, model, representation, PISD, graph, and representative-case registries.
- `payloads/` retains the immutable legacy-numbered JSON payloads used to render manuscript Figures 2--5. Figure 6 remains backed by its accepted vector artifact.
- `freeze/` contains final numerical, claim, terminology, figure, table, and supplementary registries and contracts.
- `captions/` contains the final figure and table captions.
- `provenance/` contains compact, non-sensitive methodology provenance and a historical-to-public PISD source-path crosswalk. It contains no observational data or per-object private artifacts.

Some immutable JSON payloads and registries retain paths, identifiers, and
formula labels containing historical `stage_*` provenance. These are legacy
internal provenance identifiers, not current project milestones. They remain
unchanged so that frozen checksums, source traceability, and runtime-to-
supplement equality are preserved.

Current inventory: **6 manuscript figures (Figures 1--6), 2 main tables, 27 current manuscript supplementary tables, and the earlier 5-table frozen supplementary results bundle**. `publication_inventory.json` is the current public main-figure, main-table, and caption authority; `supplementary/manuscript_tables/supplementary_table_index.json` is the current S1--S27 authority. The immutable payload filenames and freeze registries retain their historical numbering for provenance; `scripts/generate_figures.py` applies the semantic manuscript mapping.
