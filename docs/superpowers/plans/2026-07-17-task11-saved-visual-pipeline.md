# Task 11 Saved-Artifact Visual Pipeline Implementation Plan

> **For agentic workers:** Execute inline with test-driven development. Do not commit, stage, reload models, or access dataset arrays.

**Goal:** Generate and visually verify category-level t+1 comparison figures using only the immutable saved A-run prediction archive and plot metadata.

**Architecture:** Add one generic multi-model field-comparison helper to the pure evaluation module. Add a narrow CLI that validates saved category/index alignment, renders into a new temporary sibling directory, writes strict run-relative generation metadata, and atomically renames the complete directory without overwriting existing artifacts.

**Tech Stack:** Python 3.12, NumPy, Matplotlib, pytest, strict JSON.

---

### Task 1: Common-scale multi-model plotting

**Files:**
- Modify: `tests/unit/test_t_plus_one_evaluation.py`
- Modify: `src/TBMD/experiments/t_plus_one_evaluation.py`

- [ ] Add a focused test requiring aligned finite fields, lower/equal orientation, and one truth/prediction color scale.
- [ ] Run the focused test and confirm the expected missing-import RED.
- [ ] Implement the minimal reusable mapping-based plotting function with inspectable metadata and axes.
- [ ] Run the focused test and existing plotting tests to GREEN.

### Task 2: Saved-artifact-only transactional generator

**Files:**
- Create: `scripts/evaluation/generate_saved_tplus1_visuals.py`
- Modify: `tests/unit/test_navier_stokes_tplus1_contracts.py`

- [ ] Add tests for category/index validation, expected filenames, run-relative metadata, no overwrite, and partial-directory cleanup on failure.
- [ ] Run the focused tests and confirm the expected missing-script/API RED.
- [ ] Implement the minimal generator using only `selected_fields.npz` and `figures/plot_metadata.json`.
- [ ] Run focused and combined tests to GREEN.

### Task 3: Generate and inspect artifacts

**Files:**
- Create: `results/t_plus_one/20260717_ablation_a_e845b153_seed0/figures/task11_field_comparisons/*`
- Create: `results/t_plus_one/20260717_ablation_a_e845b153_seed0/figures/visual_qa.json`
- Modify: `findings.md`
- Modify: `progress.md`

- [ ] Execute the generator once against the immutable A run.
- [ ] Verify exact category/index mapping and common color limits from generated metadata.
- [ ] Open every PNG in both baseline and A runs with `view_image`.
- [ ] Record per-image pass/fail findings and exact run-relative paths in strict JSON.
- [ ] Append verified outcomes and commands to findings/progress.
- [ ] Run focused/full relevant tests, Ruff, compile, strict-JSON/path checks, and review the final status/diff without committing or staging.
