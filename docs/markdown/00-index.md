# Knowledge base for building a thinking-first local coding assistant

**Purpose.** This set of documents is the source base for coding agents (running in VS Code) that will implement the system described in *A thinking-first local AI for coding* (the thesis PDF). Each pack covers one section of the architecture and extracts, from the primary papers, the mathematics, the algorithms, the data structures, the hyperparameters, the code pointers and the stated limitations that an implementer needs. Nothing here is implementation; it is what the implementation must be faithful to.

**How the packs were made.** Every paper was verified against its arXiv abstract page, then read in its arXiv HTML version. Equations and algorithms are transcribed, not paraphrased; where the fetched text did not contain a detail, the pack says so and points to the section of the paper to check. Each pack ends with a *Design extraction* that consolidates what the module must compute, and a verified source table. The original PDFs are not bundled; `download-papers.ps1` (Windows) or `download-papers.sh` fetches all of them into `papers/<section>/`.

**How agents should use this.** Read the index, then the pack for the module being built, in full. Treat section 1 (papers) as ground truth and section 2 (design extraction) as the specification to implement. When a pack and the thesis disagree, the paper wins on facts and the thesis wins on design decisions. Cite the arXiv id in code comments wherever an algorithm or constant is taken from a paper.

## Map: architecture → packs

| Thesis section | Module | Pack | Primary papers |
|---|---|---|---|
| 5.5, 6.1, 5.7 | Notebook (facts, lessons, playbook), reflector/curator, core block, Sleep | **A** Context-level learning and memory | ACE, Reflexion, Dynamic Cheatsheet, A-MEM, Agent Workflow Memory, Generative Agents, MemGPT, Sleep-time compute |
| 5.3, 6.2 | Skill tier: verified, parameterized, tested skills; composition; library growth | **B** Skill libraries and program synthesis | Voyager, SoK Agentic Skills, DreamCoder, CodeAct, LLMs as Tool Makers |
| 5.4, 7 | Thinking tier: plan-level diversity, verified tree search, cascade, anytime stop | **C** Verified search and test-time compute | LATS, Tree of Thoughts, Self-Consistency, CodeT, AlphaCode, Large Language Monkeys, Scaling test-time compute |
| 6.4 | Weekly LoRA pockets: SFT + DPO, QLoRA in 12 GB, distillation, archive | **D** Self-training and parameter-efficient fine-tuning | STaR, ReST-EM, DPO, Absolute Zero, LoRA, QLoRA, Distillation, Self-Rewarding LMs |
| 9.3 P6 | Learning while working: neural memory, TTT, self-edits, skill vectors | **E** Test-time learning and memory architectures | SEAL, TTT on ARC, Titans, TTT layers, Transformer^2, Nested Learning / HOPE |
| 3, 6.5, 9.3 P6 | Recursive cores; harness evolution under benchmark gating; integrity | **F** Recursive reasoners and self-improving systems | HRM, TRM, CTM, Darwin Gödel Machine, Adaptive Computation Time |
| 4.2, 5.1, 5.5, 7, 8 | Serving (worker/senior, MoE offload, speculative decoding, quantization) and the Library | **G** Serving efficiency and retrieval grounding | Speculative decoding, FlashAttention, Switch, Mixtral, GPTQ, AWQ, Lost in the Middle, RAG, ColBERT |
| 5.2, 9.1, 9.2 | Gate (fast/slow routing), cost cascades, benchmark and held-out protocol | **H** Routing, cost-aware cascades and evaluation | RouteLLM, FrugalGPT, SWE-bench, On the Measure of Intelligence, SWE-agent |
| 2, 3, 11 | Background and the long-term bet: what would replace the reader | **I** Foundations and long-term research | Thousand Brains, I-JEPA, V-JEPA 2, Alberta Plan, Lake et al. 2017, BabyLM, Standard Model of the Mind |

## Reading order for a build

0. **10-interface-contract** and **20-agent-build-prompt** — the seams between modules and the order of work; read before any pack.
1. **G** (serving) and **H** (gate, benchmark) — they fix the physical constraints and the measuring stick before anything else exists.
2. **C** (verified search) — the Thinking tier is the core loop; the verifier cascade is the system's source of truth.
3. **A** (Notebook) and **B** (Skills) — the two explicit memories; read together, since lessons become tests and skills carry tests.
4. **D** (self-training) — only after months of verified trajectories exist.
5. **E**, **F**, **I** — the experimental track and the background; read for direction, not for the first build.

## Conventions used in every pack

- Equations are in LaTeX as the papers write them; symbols are defined immediately after.
- Numbers carry their benchmark and the paper they come from. Vendor self-reports are labelled as such.
- `(not in fetched text — verify in paper §n)` marks a detail the extraction could not confirm.
- *Use in our design* paragraphs and section 2 of each pack are design, not paper results, and say so.
- Every entry ends with the PDF URL; code URLs are given where the authors publish them.

## Files

- `00-index.pdf` — this document
- `10-interface-contract.pdf` — **normative**: repository layout, store formats, HTTP contract, module interfaces, event flow, invariants, config
- `20-agent-build-prompt.pdf` — the master instruction for the coding agents: document set, standing prompt, phases P0–P6 with tasks, acceptance tests and copy-paste session prompts, the human's part
- `A-…` to `I-…` `.pdf` — the nine packs (and their `.md` sources and `.json` source tables)
- `sources.csv` — one row per paper: section, arXiv id, title, PDF URL, code URL
- `download-papers.ps1`, `download-papers.sh` — fetch all original PDFs into `papers/<section>/`
- `thinking-first-local-ai-thesis.pdf` — the design document these packs support
