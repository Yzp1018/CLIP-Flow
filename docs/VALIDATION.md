# Consolidation validation

Checks executed on 2026-11-21 with Python 3.9.6 on macOS ARM64/CPU:

- **38 tests passed**, including objective-gradient directions; exact frequency-mask region/ratio; all six proxy image transforms; phase-only magnitude preservation before quantization; frozen encoder and consistent train/evaluation normalization.
- A small synthetic-image training run exercised the real FrEIA flow, checkpoint save/reload, held-out calibration, test-set inference, AP/accuracy calculation and split/calibration rejection guards. A separate test exercised the actual vendored CLIP component with a small random architecture and no downloaded pretrained weights.
- The public `main.py` entry point was also executed end to end with synthetic images: default N+P configuration, one explicitly shortened training epoch, held-out calibration, and evaluation of every one of the 19 benchmark directories. It produced a checkpoint, calibration metadata and complete-suite metrics.
- All ten projected-model parameter counts match the paper's Figure 4. The unresolved no-DR count discrepancy is separately asserted and documented, not hidden.
- All 21 configurations load and validate. The full plan contains 64 commands: 21 train/calibrate/19-dataset-evaluate chains and one emerging-model evaluation reusing P weights.
- Historical-log parsing finds 221 complete result blocks, including four reference AP rows matching all 19 displayed values. Original sources, tokenizer vocabularies and logs pass archive SHA-256 verification.
- Baseline-score evaluation rejects validation/test ID overlap. Histogram and dimension-ablation figures render from synthetic run artifacts.
- CLI help, Python syntax checks, dependency consistency (`pip check`) and token/private-key pattern checks passed.

Tested key package versions: torch 2.8.0, torchvision 0.23.0, FrEIA 0.2, numpy 2.0.2, Pillow 11.3.0, scikit-learn 1.6.1, matplotlib 3.9.4, pytest 8.4.2. Matplotlib emitted dependency deprecation warnings during plot rendering; tests passed.

These are software checks using synthetic data, not 7,200-image LSUN experiments or independent reproduction of paper results. No final CLIP-Flow checkpoints, complete benchmark datasets or GPU were available. The legacy checkpoint conversion path is strict but remains unverified with an actual author checkpoint. Historical baseline scripts are preserved sources, not covered by the supported-package runtime tests.
