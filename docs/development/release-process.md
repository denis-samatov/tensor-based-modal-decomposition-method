# Release Process

## Purpose
Defines how new versions of the codebase are tagged and distributed.

## Audience
Maintainers managing the repository.

## Summary
As a research codebase, releases are primarily internal checkpoints for reproducible experiments rather than public software distributions.

## Details
1. **Checkpointing**: When a major experiment is completed (e.g., a paper submission), the repository should be tagged with a semantic version (e.g., `v1.0.0`).
2. **Changelog**: Update `CHANGELOG.md` with the significant methodological or algorithmic changes included in the tag.
3. **No Artifacts**: Ensure no raw data or `.npz` files are accidentally included in the tagged commit.

### Distribution Strategy
The repository is distributed as a source-only GitHub package. Resolve the public `main` branch before installation so that `REMOTE_SHA` contains the immutable commit revision passed to pip. This process does not assume that a version tag exists.

```bash
REMOTE_SHA=$(git ls-remote https://github.com/denis-samatov/tensor_based_modal_decomposition_method.git refs/heads/main | awk '{print $1}')
python -m pip install "git+https://github.com/denis-samatov/tensor_based_modal_decomposition_method.git@${REMOTE_SHA}"
python -c "import TBMD; assert TBMD.__version__ == '2.0.0'"
```
