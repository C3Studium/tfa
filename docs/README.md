# docs/ — thinking-first local AI for coding

Put this folder at `docs/` in the repository (C3Studium/tfa). Start with `pdf/20-agent-build-prompt.pdf`.

- `pdf/00-index.pdf`                — map of everything
- `pdf/10-interface-contract.pdf`   — NORMATIVE: formats, interfaces, invariants
- `pdf/20-agent-build-prompt.pdf`   — the master prompt: document set, standing rules, phases P0–P6 with copy-paste session prompts
- `pdf/A…I-*.pdf`                   — nine knowledge packs (math, algorithms, data structures, defaults, code links)
- `markdown/`                       — the same documents as Markdown + JSON source tables (load these into agent context)
- `sources/`                        — sources.csv (61 papers), download-papers.ps1/.sh, manifest.json, BRIEF.md
- `thinking-first-local-ai-thesis.pdf` — the design document
- `tfa-p0-scaffold.zip`             — the tested P0 scaffold (optional starting point; see build prompt P0)

First run on your machine:  powershell -ExecutionPolicy Bypass -File sources/download-papers.ps1   (fetches the original papers into papers/)
