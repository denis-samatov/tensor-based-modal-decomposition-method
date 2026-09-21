# Revised Brugge manuscript sources

Canonical source: `manuscript/main.tex`, with `manuscript/supplementary/supplementary.tex` and `manuscript/references.bib`. Results are imported from the single study output tree. Publication generators and source statements and an author-confirmed local Word declaration are distributed; compiled upload PDFs, editorial checklists, private audit logs and scratch reruns are excluded.

The manuscript title is “Nested optimization of tensor-based modal decomposition for sparse reservoir-state reconstruction: accuracy–compression regimes on Brugge”. Public arXiv v1 is outdated. Journal submission and author approval are separate from this software release. The Word declaration is locally generated, not an official Elsevier Declaration Tool export; replace it with the official tool output if the portal requires that form.

## Validation

```bash
python docs/paper/audit/make_claims_registry.py
python docs/paper/audit/validate_latex.py
studies/brugge_sparse_sensing/verify_outputs.sh
python docs/paper/build_submission_ready.py --arxiv
python docs/paper/audit/final_gate/verify_upload_set.py
```

Builds need Tectonic with its cached bundle, pdftotext, and the document/plot dependencies. Generated upload directories are ignored. [Study](../../studies/brugge_sparse_sensing/README.md), [scientific provenance](audit/README.md), [release notes](RELEASE_NOTES.md).
