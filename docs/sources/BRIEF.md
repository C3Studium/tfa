# Extraction brief (shared by all section agents)

You are producing a **knowledge pack** for one section of a design: a local, verified, self-improving coding assistant built around small open models (worker 9B in VRAM, senior ~35B-A3B MoE in RAM, verifier cascade, explicit memory, skill library, weekly LoRA self-training). The pack will be read by coding agents in VS Code who must implement the module. They need the **math, algorithms, data structures, hyperparameters and code pointers** — not opinions.

## Sources

arXiv PDFs cannot be downloaded in this environment, but the HTML versions can be read:
- Verify each paper first: `WebFetch https://arxiv.org/abs/<id>` → confirm the title matches the expected one. If it does not match, `WebSearch` for the exact title to find the right arXiv id (or an open PDF elsewhere) and use that. For ids marked `SEARCH`, find the paper first.
- Then read the paper: `WebFetch https://arxiv.org/html/<id>` (if 404, try `<id>v1`, `<id>v2`, `<id>v3`). Use **several targeted fetches per paper** (the fetch tool answers one prompt per call; long papers need an `offset`). Suggested prompts:
  1. "State the problem, the core claim, and the headline quantitative results with exact numbers and benchmark names."
  2. "Transcribe every key equation, objective, loss, or update rule exactly as written, in LaTeX, and define each symbol. Include equation numbers and the section they appear in."
  3. "Transcribe the algorithm boxes / pseudocode step by step (Algorithm 1, 2 …). If there is none, describe the procedure as numbered steps strictly as the paper states it."
  4. "List the hyperparameters, model sizes, data sizes, compute, and the ablations with their effect sizes."
  5. "List code, data and model links, the license if stated, and the limitations or failure modes the authors state."
- If no HTML version exists and no open PDF is readable, write the entry from the abstract plus one reliable secondary source, and mark it `[details from abstract/secondary — verify in PDF]`.

## Hard rules

- **Transcribe, do not invent.** Only equations, algorithms and numbers that appear in the fetched text. If something is not in what you read, write `(not in fetched text — verify in paper §n)`.
- Give each equation in pandoc-compatible LaTeX: inline `$…$`, display `$$…$$`. No Unicode sub/superscripts. Define symbols right after each equation.
- Every paper entry ends with its PDF URL `https://arxiv.org/pdf/<id>` and the code URL if any.
- Prefer exact numbers with benchmark names over adjectives.
- Keep the structure below exactly; agents will parse the headings.

## Output

Write **one Markdown file** to the path given in your task, with this structure:

```
# <Section title>

## 0. Role in the system
<3–8 lines: what module this section informs, restated from the task; what the agents must be able to build after reading this pack>

## 1. Papers
### 1.<n> <Title> (<first author> et al., <year>) — arXiv:<id>
**Problem and claim.** …
**Headline results.** … (exact numbers, benchmarks)
**Method — equations.**
$$ … $$ with symbol definitions
**Method — algorithm.**
1. … (transcribed steps / Algorithm boxes, as pseudocode blocks where the paper gives them)
**Hyperparameters and setup.** …
**Ablations that matter.** …
**Code / data / license.** …
**Stated limitations.** …
**Use in our design.** <2–6 lines: exactly which idea the module takes, and what to change for a single-user local system>
PDF: https://arxiv.org/pdf/<id>

(repeat for every paper)

## 2. Design extraction for the module
### 2.1 Algorithms to implement (consolidated pseudocode)
### 2.2 Equations and quantities the implementation must compute
### 2.3 Data structures and file formats (as TypeScript-style type sketches)
### 2.4 Hyperparameters and defaults to start from (with the paper each comes from)
### 2.5 Evaluation: how to know the module works (metrics, from the papers)
### 2.6 Known pitfalls and failure modes (from the papers' limitations)
### 2.7 Open questions this section does not settle

## 3. Verified source table
| arXiv id | Title | Year | Venue | Code | PDF |
```

Also write a JSON file next to it (same basename, `.json`) with an array of `{ "id", "title", "authors", "year", "venue", "pdf", "html", "code", "verified": true|false, "note" }` for every paper, using the ids you actually verified.

Length: thorough. 400–900 words per paper, plus the design extraction. Do not pad; do not summarize what you did — the file is the deliverable.
