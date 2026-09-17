# Notebook status

For the current Brugge manuscript, use
[`studies/brugge_sparse_sensing`](../studies/brugge_sparse_sensing/README.md).
Its configuration, generators, result files and checksums form the reproducible study workflow.

`experiments/exp_tbmd_2.*` are historical experiments. They are retained as provenance for the
earlier manuscript and its E0 audit, not as the current quantitative benchmark. Some use separate
`data_exp_5_*` datasets. `tutorials/` and `data_visualize.ipynb` are exploratory notebook sources;
they were not executed during workspace cleanup and retain their inline historical plotting code.
The Brugge data and well input paths in `data_visualize.ipynb` now uses the actual `brugge/` directory name.

The old HW CSV/PNG extractions are now stored only in their original ZIP files. They are not inputs
to the current study. To use the commented HW loader options, restore their exact paths using the
workspace archive guide before running the notebook. `dynamic_png_new/pressure_N` corresponds to
`pressureN` members of `dynamic_png (2).zip`; plain extraction does not preserve this mapping.

Validation of the current study: run `./verify_outputs.sh` from its directory. That check does not
qualify these notebooks or the historical numerical results.
