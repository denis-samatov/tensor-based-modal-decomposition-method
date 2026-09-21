# Editorial Manager metadata

Author names/order, affiliation, CRediT, funding role, competing interests, institutional data-access restriction, AI wording and exclusivity were confirmed in this task. Verify the archive DOI before final upload. Use Research paper if present; preserve the transfer-prefilled type. Exact upload labels remain portal-specific. Preparation requirements were supplied by the author; HTTP 403 alone is not a blocker. This sheet is not a submission receipt.

## Title

Nested optimization of tensor-based modal decomposition for sparse reservoir-state reconstruction: accuracy–compression regimes on Brugge

## Abstract

Sparse reservoir-state reconstruction requires reduced representations that preserve pressure and saturation structure under limited observations. Whether an explicit tensor property mode helps remains unclear when capacity, measurements and recovery differ. We compare tensor-based modal decomposition (TBMD) architectures on ten Brugge control scenarios using leave-one-scenario-out testing and three-fold inner selection. Shared spatial factors with property-specific coefficients are selected in 122 of 160 sparse comparisons. At 30 common grid channels, this formulation lowers pressure/saturation anomaly-relative error from 1.167/0.844 to 0.350/0.465 and raises mask-aware anomaly structural similarity from 0.406/0.856 to 0.770/0.932. Both properties also improve at five wells under both capacity controls. Rigid joint coefficients remain inferior, and full-observation saturation gains are inconsistent. Energy-weighted proper orthogonal decomposition remains strongest at existing joint-property wells, while tensor models offer conditional accuracy–storage trade-offs. The benchmark identifies partial sharing and observability, rather than tensor order alone, as the useful mechanisms and supplies a reproducible comparison with explicit limitations.

## Keywords

Reservoir monitoring; Sensor placement; Proper orthogonal decomposition; Tucker decomposition; Compressive sensing; Brugge

## Authors

1. Denis Samatov
2. Boris Merzlikin
3. Gleb Shishaev

## Affiliations

All authors: National Research Tomsk Polytechnic University, 30 Lenina Avenue, Tomsk 634050, Russia.

## Corresponding author

Denis Samatov — dss53@tpu.ru

- Denis Samatov: https://orcid.org/0009-0000-1821-323X
- Boris Merzlikin: https://orcid.org/0000-0001-8545-9491
- Gleb Shishaev: https://orcid.org/0000-0003-2387-9217

## Highlights

Nested tuning separates tensor truncation from sparse-recovery error
Shared spatial factors help when property coefficients remain independent
Matched 3D–4D comparisons retain negative and metric-dependent results
POD-E retains the lowest errors at all existing joint-property wells
Tensor models offer conditional accuracy and storage trade-offs

## Data Availability

Data availability

The t-Navigator simulations analysed in this study are represented by two supplied, transformed inputs for ten Brugge well-control scenarios: data_exp_4_.h5 (pressure and oil-saturation arrays) and all_wells_exp_4.json (well coordinates). The current archive contains neither the original simulator deck nor the exporter and vertical-reduction configuration; consequently, the path from the original Brugge model to these HDF5 arrays is not reproducible from the archive. The HDF5 file has no unit attributes, so pressure values are reported without conversion in the supplied export unit u_p.

The transformed data supporting the findings of this study are available from the corresponding author upon reasonable request, subject to the applicable data-use and access conditions. The files are not distributed under an open licence or under the code repository's MIT licence and are excluded from the public release candidate.

The input identities are fixed by SHA-256 and verified before computation:
  data_exp_4_.h5        141,982,268 bytes  SHA-256 d55ae5e5f6a45167b67251ab21af6b864334a97c68051754f50dc7566813227e
  all_wells_exp_4.json  3,151 bytes        SHA-256 e3b8c99688efcbff4d41ba36b9714f6a32cf0d23df3933465b20387af8e425e7
The reviewed release candidate contains the fold-level performance metrics, statistics, tables, number macros and figures needed to check the article. The existing v2.1.0 archive predates E8 and V3; the present revision is prepared locally for archival as v2.2.0. The transformed input files cannot be deposited publicly because of institutional access conditions. Requests should be directed to the corresponding author and will be considered under those conditions.

## Code Availability

Computer code availability

Name of code: TBMD library with the Brugge sparse-sensing study package (studies/brugge_sparse_sensing), prepared release v2.2.0
Developer: Denis Samatov, National Research Tomsk Polytechnic University, 30 Lenina Avenue, Tomsk 634050, Russia; dss53@tpu.ru
Year first available: 2026
Hardware required: general-purpose CPU; about 2 GB of memory per parallel job; no GPU
Software required: Python 3.12; NumPy, SciPy, TensorLy, PyTorch, pandas, h5py, Matplotlib (exact versions pinned in requirements-lock.txt)
Program language: Python
Program size: recorded in the local release-candidate receipt and to be confirmed for the final archive
Licence: MIT for code; not a licence for TNO-derived data
Source code: https://github.com/denis-samatov/tensor-based-modal-decomposition-method (current public release tag v2.1.0; not the reviewed revision)
Previous archive: https://doi.org/10.5281/zenodo.22814377 (v2.1.0; predates final E8/V3, nested-optimization and E9 additions)
Current revision: prepared locally for tag v2.2.0; the final tag and archival DOI have not yet been created
How to reproduce (from studies/brugge_sparse_sensing): run_all.sh retains the legacy benchmark; run_tbmd_optimization.sh runs the broad nested study and frozen refits; run_e9_property_mode.sh --reproduce independently refits the frozen E9 selections (--full also repeats inner selection). Set TBMD_DATA_DIR to the directory containing both checksummed transformed inputs. verify_outputs.sh rebuilds tables, statistics and macros from retained outputs; field maps additionally require the inputs. scripts/check_claims.py checks the manuscript's generated numbers.

The proposed release tag is v2.2.0. The exact local release commit is recorded in RELEASE_CANDIDATE.md and the local release receipt. No new version-specific DOI has been assigned or verified.

## Funding

The research was supported by the program of the National Research Tomsk Polytechnic University (Prioritet-2030-ISP-032-090-2026). The funder participated in data collection, interpretation of the results, and the decision to submit the manuscript for publication.

## Competing Interests

Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## CRediT

Denis Samatov: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Visualization, Writing – original draft. Boris Merzlikin: Supervision, Methodology, Writing – review & editing. Gleb Shishaev: Data curation, Validation, Writing – review & editing.

## AI Disclosure

During the preparation of this work, the authors used OpenAI GPT-4o and OpenAI GPT-5-Codex to assist with manuscript drafting, language and text editing, and submission-package consistency checks. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication. AI assistance also supported source-code review and corrections to publication generators and validation scripts; it is not an independent scientific validation. Reported numerical results were produced by the documented Python workflow on the checksummed inputs, and the figures were rendered by repository scripts rather than a generative image model.

## Preprint information

Public preprint: https://arxiv.org/abs/2607.09687v1. Its original title is “Tensor-Based Modal Decomposition and Sparse Sensor Placement for the Brugge Field Simulation Model”. It contains superseded quantitative claims. A rebuilt v2 source archive is prepared locally; replacement is not submitted. Suggested version comment is in ARXIV_VERSION_COMMENT.txt.

## Transfer and submission fields

Target: Applied Computing and Geosciences. Proposed type: Research paper (confirm the exact current portal label). Transfer: Computers & Geosciences CAGEO-D-26-01439. Existing exclusivity text: “The manuscript is not currently under consideration by another journal.” Confirmed by the corresponding author in this task.

## Validation

Compare title/abstract/keywords with manuscript/main.tex and sections/abstract.tex, then run the submission builder and receipt verification after any update. Statements above are copied from canonical submission sources. Technical checks do not constitute author consent.
