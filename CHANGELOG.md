# Changelog

## 2.2.0 - prepared, not published

- Record the revised Brugge nested-optimization and E9 property-mode benchmark, frozen selections and reproducibility tooling.
- Package canonical manuscript sources, author-confirmed declarations and current figure/table generators.
- Exclude scratch reruns and internal editorial audit material from the archival candidate.
- Preserve institutional data-access restrictions and the limits of fixed-geology validation.


All notable changes should be recorded here.

## 2.2.0-rc1 - Unreleased

- Reframed the Brugge paper as a reproducible fourth-order joint-property benchmark rather than a
  new decomposition-family or universal-superiority claim.
- Added the matched-measurement E8 joint-versus-independent-property controls and V3 shape,
  measurement-mapping and rank-deficient least-squares verification.
- Reported the negative E2/E8 findings in the main scientific narrative and tightened the stated
  conditions for TBMD/POD-E algebraic equivalence.
- Replaced unsupported pressure-unit labels by the supplied-export unit `u_p`; documented the
  confirmed t-Navigator identity and the remaining version/export/vertical-reduction/unit boundary.
- Added controlled reasonable-request access language for transformed inputs, the author-confirmed
  National Research Tomsk Polytechnic University funding statement
  (`Prioritet-2030-ISP-032-090-2026`), the confirmed GPT-4o/GPT-5-Codex disclosure,
  exclusivity metadata and a standalone competing-interest declaration.
- Regenerated manuscript macros, tables, figures, graphical abstract and the allowlisted
  submission package; consolidated the remaining external upload/archive actions.
- Scientific experiment CSV values were not retuned or altered for the revised narrative.

## 2.1.0 - 2026-09-17

- Added the Brugge sparse-sensing study package `studies/brugge_sparse_sensing` for the revised
  manuscript "Fourth-order joint-property TBMD for sparse reservoir-state reconstruction: a
  reproducible Brugge benchmark": configuration and seeds, study code,
  one-command pipeline `run_all.sh`, result files, figures, SHA-256 checksums, run logs,
  data-free verification `verify_outputs.sh`, re-run comparison and manuscript claim checker.
- Documented that the numerical Brugge results of arXiv:2607.09687 are not reproducible (audit
  scripts `scripts/e0_*` of the study) and are superseded by the study.
- Replaced the README runtime table, which timed QR placement on a random matrix, by a pointer to the
  study's measured costs.
- Linked the standalone `tbmd-forecasting` repository from the forecasting boundary statements.
- No new library algorithm was introduced in this release; the study formalises and verifies the
  existing higher-dimensional modal-processing path for fourth-order joint-property TBMD.

## 2.0.0 - 2026-08-30

- Added `CITATION.cff` and a README Citation section for arXiv:2607.09687
  (Samatov, Merzlikin, Shishaev). Set repository homepage to the paper.
- Eliminated bare `except:` clauses in the numerical decomposition path;
  re-enabled `E722` and `F401` in the ruff lint configuration.
- Expanded the top-level `TBMD` public API to re-export the primary
  decomposition, sensor-placement, reconstruction, and modal-processing
  classes instead of only `geometry`.
- Standardized public repository documentation in English.
- Added repository governance files.
- Documented configuration, testing, data, and model artifact handling.
- Clarified that local datasets and generated experiment outputs should not be committed.
- Added installation and repository-structure guides.
- Normalized documentation filenames to match lowercase links.
- Removed tracked generated experiment artifacts from `scripts/plots/`.
- Expanded audit tests for repository hygiene checks.
- Historical release notes prior to this entry were not available in the
  repository at cleanup time.
