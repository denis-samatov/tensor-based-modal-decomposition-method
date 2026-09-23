# Agent Instructions

## Purpose
This document provides guidelines for any autonomous coding agent assisting with the Tensor-Based Modal Decomposition Method (TBMD) repository.

## Project Context
The TBMD project is a Python research library for reduced-order modeling of spatiotemporal tensor data. Core capabilities include Tucker/HOSVD decomposition, sparse sensor placement (using Tensor QR), and tensor reconstruction (ADMM/Compressive Sensing). RANS/URANS conversion and `t+1` forecasting are maintained in the separate `tbmd-forecasting` repository. **This is a numerical machine learning and scientific computing codebase. It does not implement LLMs, RAG, chatbots, or autonomous agent frameworks.**

## Repository Rules
- **No LLM/RAG hallucination**: Do not attempt to add or document LLM, GenAI, or conversational features.
- **Artifact Isolation**: Datasets, trained model files (`.npz`), and generated plots must be placed in ignored directories (`data/`, `results/`, `scripts/plots/`) and not committed.
- **No "Production-Ready" claims**: Treat the code as a research codebase. Do not make claims about scalability, production-readiness, or unmatched accuracy.

## Development Workflow
1. Read `README.md`, `docs/index.md` and `docs/architecture/overview.md` to understand the mathematical flow.
2. Distinguish reusable algorithms in `src/TBMD/core/` from experiment helpers in `src/TBMD/experiments/`, synthetic examples and the dataset-specific `studies/brugge_sparse_sensing/` benchmark.
3. Maintain test parity when modifying public configurations.

## Documentation Rules
- Write documentation in the Google Developer Documentation style: direct, verifiable, and structured.
- Always include `Validation` instructions using reproducible commands (e.g., `pytest`).
- Avoid empty `TODO` stubs. If an architectural decision is unclear, use an `Owner decision required` block.

## Testing and Validation
After making changes, run the following to ensure syntax and structural hygiene:
```bash
python -m compileall src tests examples studies/brugge_sparse_sensing/scripts
pytest tests/audit -q
pytest tests/unit -q
```

## Safety and Correctness Constraints
- Do not invent non-existent mathematical features.
- Do not modify core algorithm logic (like ADMM steps or Tucker decomposition factors) without explicit user instruction.
- Do not add arbitrary dependencies to `pyproject.toml` unless strictly necessary for an approved feature.

## Final Response Format
When your task is complete, clearly list the files modified, the tests run to validate the changes, and any edge cases that might require the repository owner's attention.

## Shared Tensors graph

Graphify is installed as a local Codex skill. The shared graph is at the parent
Tensors workspace, not in this checkout. From this checkout root, use
`graphify query "<question>" --graph ../graphify-out/graph.json`; pass the same
`--graph` path to `path`, `explain`, and `affected`. From deeper directories,
resolve this path to an absolute workspace graph path first.

Run rebuilds and semantic updates from the parent Tensors workspace using its
`.codex/skills/graphify/SKILL.md`; update code structure with
`(cd .. && graphify update .)`. These shared graph paths take precedence over
the default current-directory paths in the managed graphify section below.
Read `../graphify-out/COVERAGE.md` for extraction limits. Graph output supports
navigation and does not independently validate scientific results.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
