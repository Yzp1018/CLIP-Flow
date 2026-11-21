# Implementation provenance and unresolved details

## Recovered material

`legacy/original/` retains relevant Python sources from the first archive. `legacy/cflowtest/` retains the second archive's evaluation code and six result text files. `legacy/UniversalFake/` retains the supplied comparison-baseline sources, without multi-gigabyte checkpoints. `ARCHIVE_MANIFEST.json` identifies each source archive member and SHA-256. `INITIAL_UPLOAD_MANIFEST.json` describes the earlier, narrower upload; paths there refer to that historical snapshot.

The supported package combines the original frozen CLIP/projection/FrEIA path with the test archive's dataset list. It repairs the inactive-proxy default, training/evaluation normalization mismatch, fixed server paths, mode selection and test-set threshold fitting. Historical originals remain unmodified. The vendored CLIP packaging import was modernized; its model/tokenizer implementation and vocabulary are otherwise preserved.

## Explicit reconstruction choices

- The objective follows natural/proxy log-likelihood directions in paper equation 6 and the training source. Per-class means replace the archive's dataset-dependent `1 / len(loader)` factor. Gaussian log-likelihood is divided by feature dimension and negated for inference. The expanded equation 7 appears to reverse signs when substituted algebraically; equation 6, the text and supplied source are the reference here.
- Frequency regions preserve the loaders' **unshifted FFT corner rectangles**: low is the first quarter on both axes, mid the middle half, high the last quarter. These are not radial bands. RGB channels share mask samples; inverse FFT uses the real part. Masks are not forced to be conjugate-symmetric, matching the source.
- Proxies default to original direct float-to-uint8 conversion (`uint8_policy=wrap`). `clip` is an explicit alternative. Resizing to 256 precedes proxy construction and cropping. Source code applies spectral operations before resizing, equivalent only for already-256-square source images.
- Phase-only proxies zero selected phase angles, preserving magnitudes before inverse transform/quantization. Supplied `loader_final.py` instead changes selected amplitudes to one; it is not evidence of the final phase-only experiment. This reconstruction has not been validated against Table 3.
- Exposed proxy defaults include noise standard deviation 5, smoothing radius 1, sharpening radius 5/percent 300 and jitter amplitudes `(0.5,0.5,0.5,0.1)`. Exact final smoothing/sharpening/phase settings were not recoverable.
- N-only uses 10 epochs as a documented default; its final schedule is unknown. Fixed seed 1000, 1,000 validation images, no dropped final mini-batch and balanced-validation-accuracy threshold selection are consolidation choices. Original split membership is not claimed.
- Calibration removes a radial frequency annulus with radii 30–100 on resized validation images. Exact validation geometry, membership and threshold search were unavailable. One threshold applies to every test dataset, bound to the checkpoint SHA-256.
- Split guards compare resolved paths, not perceptual/content identity. Copies at different paths must be checked by the data owner. The split command requires a natural-only source root and cannot establish arbitrary images' LSUN origin or class coverage.

## Discrepancies retained for review

All projected 2–768-dimensional models match Figure 4's parameter counts. The no-DR model with FrEIA 0.2 has **887,808 trainable parameters**, while the figure labels **888,576**. The unresolved difference is 768. No dummy parameters were added.

The archived P row has fixed-threshold mean accuracy 84.93. One N+P row has 19 displayed accuracies matching Table 2, but its total is 87.31 and the paper prints 87.30. Original totals and threshold trials are retained as written. New aggregates are computed from actual scores, without adjustment to a target.

## Unavailable material

- CLIP-Flow checkpoints named in logs, original split memberships and final environment.
- Original per-image benchmark scores. Text records alone do not establish checkpoint identity or independently verify results.
- Emerging-generator datasets, prompts/captioner and precise generation settings. Evaluation is supported once data are provided; their original creation process is unavailable.
- Complete training implementations/settings for all external comparison methods. Supplied UniversalFake sources and a common score evaluator are included; other methods require upstream implementations and independently produced scores.

No full LSUN/GPU experiment or paper-result reproduction was performed during consolidation. `VALIDATION.md` describes the checks actually run.
