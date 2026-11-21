# Tests

Install the development dependencies and run:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests cover:

- Natural-only, proxy-only and joint likelihood objectives.
- Frequency-mask regions and sampling ratios, proxy transforms and phase-only spectral magnitudes.
- Frozen encoder behavior and consistent training/evaluation preprocessing.
- Projection and flow parameter counts.
- Checkpoint save/reload and held-out threshold calibration.
- Data-split overlap and calibration/model mismatch rejection.
- The `main.py` training → calibration → 19-dataset evaluation workflow.
- Experiment configuration validity and reuse of the P model for emerging generators.
- Baseline-score metrics and figure rendering.

Tests use small generated image sets and lightweight encoders so they can run locally without downloading CLIP parameters. A separate component check exercises the vendored CLIP architecture with randomly initialized parameters. Scientific benchmark evaluation is run through `main.py` using the prepared datasets; test fixtures are not benchmark results.
