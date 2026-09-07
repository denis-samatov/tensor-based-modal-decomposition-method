# Captured synthetic run — 7 September 2026

Source revision before this documentation update: `dac7db07e16780d7d7f0812ab97ca6a6c5d6d1f2`.
CPU execution on Linux x86_64; no external dataset or GPU used.

```text
Python 3.12.7
PyTorch 2.14.0+cpu
NumPy 2.5.3
TensorLy 0.9.0
Matplotlib 3.11.1
```

From the repository root after package installation:

```bash
MPLBACKEND=Agg python examples/basic/04_complete_pipeline.py --spatial-points 40 --time-steps 12 --n-modes 8 --n-sensors 6 --solver admm --visualize
```

The generated PNG was copied to `docs/examples/synthetic-pipeline.png`. The [console output](synthetic-pipeline-output.txt) is captured from that run. The default random seed is 42.

This example fits its modal representation and reconstructs the same synthetic sequence. The plotted errors measure that reconstruction; they do not establish held-out temporal generalization, reproduce Brugge manuscript results, or compare against a competing algorithm. Numerical values may vary with numerical-library versions and solver behavior.
