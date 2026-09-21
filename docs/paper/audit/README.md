# Distributed scientific provenance and validators

The historical notebook archive and compact A1/A2/A6 records document correction lineage. They are not executable current pipelines or current submission approval. Internal review sheets, copied web records, build logs and full rerun duplicates are intentionally excluded from the software release.

- [Correction lineage](A1_original_numerical_claims_audit.md)
- [Generated claim registry](A2_claims_registry.md)
- [Historical independent-run record](A6_reproduction_record.md)
- [Historical comparison](A6_rerun_comparison.txt)
- [Historical timings](A6_rerun_timing.json)
- [Original notebook provenance](historical_notebooks.zip)

## Validation

```bash
python docs/paper/audit/make_claims_registry.py
python docs/paper/audit/validate_latex.py
studies/brugge_sparse_sensing/verify_outputs.sh
```

See [paper source/build](../README.md) and [study guide](../../../studies/brugge_sparse_sensing/README.md).
