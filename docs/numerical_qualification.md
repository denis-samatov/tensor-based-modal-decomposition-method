# CPU numerical qualification (2026-10-31)

This qualification tests the reusable implementation against analytic references.
It does not replay the Brugge simulation or establish universal accuracy or generalization.
The protocol was fixed before evaluation (qualification protocol SHA-256
`50d7cc57aaca75a83d694f15d013ebd998533f709753070dfec87cae499cf4a8`).
The evaluated base revision was `f32502a5d3d8a8ef968dd43cc5e0e611ff5d5805`.

## Analytic references

`tests/unit/test_numerical_oracles.py` uses CPU float64 and seed 20261031:

- A `(6,5,4)` tensor with known multilinear ranks `(2,3,2)` and QR-orthonormal
  factors is reconstructed independently with NumPy `einsum`. Factor projectors
  are compared, avoiding sign/permutation ambiguity. Absolute/relative tolerance: `1e-10`.
- A diagonal matrix with singular values `(5,3,1)` has rank-one relative residual
  `sqrt(10/35)` by the Eckart–Young theorem. A zero tensor must have finite zero error.
- A non-contiguous spatial mask selects identity design rows. For observations
  `(2,-0.5,0.05)` and L1 penalty `0.1`, the unique minimizer is `(1.9,-0.4,0)`.
  Cholesky/direct/SVD paths must converge, agree with this soft-threshold solution,
  and satisfy an independently calculated KKT residual below `2e-7`.
  Changing unobserved measurements from `1e6` to `-1e6` must not change the answer.
- Perturbed factors and incorrect/unregularized Lasso solutions are negative controls.

The new zero-measurement case exposed `0/0` in `reconstruction_error`: it returned
`NaN` for exact zero reconstruction. The metric now returns zero for that case and
raises `ValueError` for a nonzero residual against zero measurements. Nonzero-input
Torch dtype/division behavior is preserved and checked explicitly in float32.
No ADMM update, decomposition factor calculation or experiment hyperparameter changed.

Peer review then identified a separate regression in the initial zero-norm guard:
nonzero measurements/residuals can have an underflowed norm (float32 `1e-30`,
float64 `1e-200`). Three additional regression cases reproduced false-zero errors.
Exact zeros are now identified elementwise. Only a nonzero measurement vector
whose norm underflows uses float64 scaled norms, so zero prediction against a
nonzero tiny target reports relative error one. A tiny nonzero residual against
an exactly zero target still raises `ValueError`. Ordinary dtype division stays
unchanged. These are post-review boundary regressions, not retuned experiments.
A second review found that dividing the residual by the tiny target scale before
its norm could overflow even for a representable relative error (`4.08e199` or
`4.08e169`). Three more cases use Python `math.hypot` as an independent norm oracle
and check genuinely unrepresentable ratios. The exceptional path now scales each
vector independently and combines scale exponents using `frexp`/`ldexp`. Finite
representable ratios remain finite; a ratio beyond float range returns infinity.
The tests qualify these finite-input cases; they do not introduce a new NaN/input
validation policy. Ordinary Torch division remains unchanged.

## Stored-result replay is a separate gate

A fresh rerender of the committed result rows in the maintained software profile
(Python 3.13.13, NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1, Torch 2.15.0; macOS arm64)
produced six byte-identical files out of twelve compared tables/macros/statistic files.
The exact replay gate **failed**. The study's historical profile instead requires
Python 3.12, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.1 and Torch 2.14.0 (see the full
study lock and its fail-closed `verify_environment.py`). The maintained software
profile is not a replacement for that scientific profile.

Both main CSV tables and the four compared LaTeX tables matched exactly. Six
remaining files differed: `table_representation.csv`, `table_noise.csv`,
`table_l1.csv`, `stats_wilcoxon.csv`, `key_numbers.json` and `numbers.tex`.
Non-p-value numeric differences in `key_numbers.json` were at most `1.14e-13`.
Two Wilcoxon values for DG versus configured POD-E LS with all 30 wells changed:

| Protocol | Stored p | Maintained-profile rerender p |
| --- | ---: | ---: |
| P1 | 0.4453125 | 0.4921875 |
| P2 | 0.333984375 | 0.375 |

The channel was isolated without changing the stored input rows or SciPy 1.18.1.
A separate statistics-only environment with NumPy 2.5.2/pandas 3.0.5 reproduced both
stored p-values using the default CSV parser. In that same environment, selecting
`float_precision="round_trip"` reproduced the two maintained-profile values instead.
Default-parsed floats differed from Python CSV/built-in float parsing by at most
`9.72e-17` (P1) and `8.33e-17` (P2). The maintained pandas 3.1.0 parser agreed with
built-in float parsing. This is a CSV conversion/rank-tie sensitivity, not an
ADMM or fresh simulation discrepancy. This diagnostic environment used Python
3.13, so it was **not** a run of the complete historical Python 3.12 study profile.
The same statistics-only environment regenerated all twelve compared files
byte-identically. That successful bounded result does not erase the failed
maintained-profile replay or qualify raw-input scientific recomputation.

The paired errors differ by at most approximately `1.4e-14`; signed-rank statistics
can depend on numerical ties at that scale. All four p-values exceed 0.05, so
these two comparisons do not support a difference under either rerender. Failure
to reject is not evidence of equivalence. This result does not alter the other
reported comparisons, and it does not qualify a new significance or equivalence
protocol. Historical artifacts and locks have been preserved. No rounding rule,
tolerance or statistic was chosen to recover the stored p-values.

## Dedicated statistics-only gate

The [isolated replay instructions](../REPRODUCIBILITY.md#statistics-only-replay-twelve-core-artifacts)
now provide a generated hash lock for Python 3.13 with NumPy 2.5.2, pandas 3.0.5
and SciPy 1.18.1. `verify_statistics.py` checks a fixed twelve-file set after
running the same three unchanged generators over fifteen committed input files
in a new directory outside the checkout. Required missing files, byte drift,
runtime drift, subprocess failures and changed source hashes fail the gate;
receipts and logs remain available for inspection.

Fresh local acceptance on 2026-10-31 used Python 3.13.13 on macOS ARM64 and
installed all six packages from the generated hash lock in a new environment.
The checksum check and three generators exited zero, all twelve required files
matched byte for byte, and all 134 recorded source hashes were unchanged.
The dedicated boundary tests passed all fifteen cases. These are local results.

The independent Linux CI job is named `statistics-only-replay`. Its execution
and receipt, rather than its presence in YAML, determine hosted qualification.
This narrow gate does not replace `verify_outputs.sh` or its historical Python
3.12 environment guard. It excludes E8, optimization/E9, figures and raw-input
recomputation. The modern-profile 6/12 failure above is preserved unchanged.

## Cross-platform parsing diagnosis

The locked profile passed 12/12 files on macOS ARM64 but reproduced only 7/12
byte-identical files on Linux x86_64, including with the same Python 3.13.13
patch version. PR #135 therefore remains unqualified for merge. Scientific
outputs and exact-match acceptance remain unchanged.

`diagnose_statistics.py` records raw input hashes, little-endian binary64 column
hashes from the unchanged pandas reader, an independent builtin-float parsing
reference, and the inputs/intermediate pair of one preselected differing median.
It writes a new JSON file outside the checkout and runs after the strict CI gate,
even on failure. Its diagnostic reference does not replace the study parser.

```bash
python studies/brugge_sparse_sensing/scripts/diagnose_statistics.py --output-file /tmp/tbmd-statistics-inputs.json
```

Matching source bytes alone do not establish identical parsed numbers. The
cross-platform diagnostic must identify the earliest differing boundary before
any change to the scientific execution profile is proposed.

### First divergent boundary identified (2026-10-31)

[Linux CI run 41206927831](https://github.com/denis-samatov/tensor-based-modal-decomposition-method/actions/runs/41206927831)
retained a successful diagnostic after the strict replay failed 7/12. Across four
identical raw CSV files and identical shapes, 12 of 23 floating-column binary64
hashes differ between macOS ARM64 and Linux x86_64. All 23 independent
builtin-float column hashes match. Thus divergence already exists during the
unchanged pandas CSV parsing, before the statistical reductions.

In the preselected ten-row P2/grid/N10/POD-LS group, a middle element changes
from `0x1.d5165e86b84f8p+4` on macOS to `0x1.d5165e86b84f9p+4` on Linux.
The two middle values then reproduce the observed median change from
`0x1.d4f2fb174798fp+4` to `0x1.d4f2fb1747990p+4` with plain binary64 arithmetic.
Using the independent builtin-float inputs gives the same median on both hosts.
That is a diagnostic control, not an adopted replacement parser.

A bounded reference of pandas 3.0.5's
[`precise_xstrtod` decimal accumulation](https://github.com/pandas-dev/pandas/blob/v3.0.5/pandas/_libs/src/parser/tokenizer.c)
compared fused multiply-add with separate multiply/add over the retained example
tokens and the fixed median inputs. All 130 observed macOS values match the
fused reference (8 only that reference); all 130 observed Linux values match the
separate reference (12 only that reference). The union contains 136 tokens,
not every input cell. All ten median inputs and the resulting one-ULP median
difference are reproduced. Static inspection of the installed macOS parser
also finds `fmadd` in `precise_xstrtod`. Linux wheel compiler flags/instructions
and causality for every derived difference have not been independently established.
The differing `longdouble` widths in the diagnostic are not attributed as cause:
this source function uses `double` accumulation.

No scientific generator, CSV parser option, numerical method, expected artifact,
rounding or acceptance tolerance has changed. The original exact failure remains
blocking. Choosing a portable numeric ingestion contract is a separate versioned
scientific decision; this diagnosis does not silently replace historical results.

## Validation

From a source checkout with the maintained execution environment installed:

```bash
PYTHONPATH=src python -m pytest tests/unit/test_numerical_oracles.py -q
PYTHONPATH=src python -m pytest tests/audit -q
PYTHONPATH=src python -m pytest tests/unit -q
python -m compileall src tests examples studies/brugge_sparse_sensing/scripts
```

For canonical scientific replay, first satisfy the historical environment check,
then run the existing study verifier. Software test success must not be reported
as success of this distinct gate:

```bash
python studies/brugge_sparse_sensing/scripts/verify_environment.py
bash studies/brugge_sparse_sensing/verify_outputs.sh
```

Author simulation HDF5/well inputs are not redistributed. Source-row rerendering,
checksums and analytic oracles do not fill that provenance or independent physical
validation gap.
