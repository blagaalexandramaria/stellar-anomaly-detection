# Publication artifacts

This directory contains the frozen public publication layer. It is an artifact and registry authority, not a working directory for exploratory outputs.

- `figures/` contains accepted manuscript Figures 1--6 in PDF, PNG, and SVG forms.
- `tables/` contains two main tables in CSV, JSON, and TeX.
- `supplementary/` contains five supplementary tables plus the experimental protocol, model, representation, PISD, graph, and representative-case registries.
- `payloads/` retains the immutable legacy-numbered JSON payloads used to render manuscript Figures 2--5. Figure 6 remains backed by its accepted vector artifact.
- `freeze/` contains final numerical, claim, terminology, figure, table, and supplementary registries and contracts.
- `captions/` contains the final figure and table captions.
- `provenance/` contains compact, non-sensitive methodology provenance and a historical-to-public PISD source-path crosswalk. It contains no observational data or per-object private artifacts.

Some immutable JSON payloads and registries retain paths, identifiers, and
formula labels containing historical `stage_*` provenance. These are legacy
internal provenance identifiers, not current project milestones. They remain
unchanged so that frozen checksums, source traceability, and runtime-to-
supplement equality are preserved.

Current inventory: **6 manuscript figures (Figures 1--6), 2 main tables, and 5 supplementary tables**. `publication_inventory.json` is the current public figure, table, and caption authority. The immutable payload filenames and freeze registries retain their historical numbering for provenance; `scripts/generate_figures.py` applies the semantic manuscript mapping.
