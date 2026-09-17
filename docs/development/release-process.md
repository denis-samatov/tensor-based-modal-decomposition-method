# Release Process

## Purpose
Defines how new versions of the codebase are tagged and distributed.

## Audience
Maintainers managing the repository.

## Summary
Releases identify reproducible software revisions. A public GitHub tag/release does not by itself prove a PyPI distribution, native installation qualification or physical forecast skill.

## Details
1. **Checkpointing**: When a major experiment is completed (e.g., a paper submission), the repository should be tagged with a semantic version (e.g., `v1.0.0`).
2. **Changelog**: Update `CHANGELOG.md` with the significant methodological or algorithmic changes included in the tag.
3. **Artifact review**: Exclude restricted inputs and exploratory run artifacts. Preserve curated documentation and study assets required by the reproducibility contract.

### Distribution Strategy
For revision-pinned source installation, use the public GitHub repository. Resolve the public `main` branch before installation so that `REMOTE_SHA` contains the immutable commit revision passed to pip. This process does not assume that a version tag exists.

```bash
REMOTE_SHA=$(git ls-remote https://github.com/denis-samatov/tensor-based-modal-decomposition-method.git refs/heads/main | awk '{print $1}')
python -m pip install "git+https://github.com/denis-samatov/tensor-based-modal-decomposition-method.git@${REMOTE_SHA}"
python -c "import TBMD; assert TBMD.__version__ == '2.1.0'"
```

## Validation

From the installed checkout root:

```bash
MPLBACKEND=Agg python -m pytest tests/audit -q
python -m ruff check src tests examples
python -m mypy src/TBMD/core
```

Record the exact release/tag commit, CI scope and artifacts actually produced.
[Changelog](../../CHANGELOG.md) · [Testing](testing.md) · [Reproducibility](../../REPRODUCIBILITY.md)
