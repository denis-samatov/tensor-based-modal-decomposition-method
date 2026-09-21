# TBMD v2.2.0 — Brugge nested validation and tensor-property benchmark

Prepared release; publication is pending. This version records the computational source for the revised Brugge manuscript. The exact reviewed commit and archive digest are in the release receipt generated after the commit.

- Add nested scenario optimization, frozen outer refits and E9 independent-3D/joint-4D comparisons.
- Include frozen selection JSON, canonical derived result registries, configurations, environment lock, input checksums and reproduction commands.
- Include manuscript/supplement sources and reproducible figure/table/numerical-claim generators.
- Strengthen schema/nonfinite comparisons, headless plotting and submission build validation.
- Retain strong POD-E baselines, negative coupling controls and documented data-provenance limits.

Restricted HDF5/JSON simulation inputs are not included and are not MIT licensed. Code is MIT licensed. The previous v2.1.0 DOI must not be used as this release's DOI. The current public arXiv v1 contains superseded claims. No field, production or journal-acceptance qualification is implied.

## Validation

Run the pinned-environment unit/audit suite, claim checker, stored-output verification and `git diff --check`. Exact commands and results are in the accompanying release receipt. Fresh scientific refits from the completed audit are reused; they are not repeated for release packaging.
