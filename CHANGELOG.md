# Changelog

All notable changes should be recorded here.

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
