# External-method comparison

The archives contain UniversalFake source, but not all comparison methods' implementations, weights and final configurations. This repository does not substitute a generic classifier for those algorithms.

Use upstream implementations for PaDiM, CFA, GANomaly, DSR, DRAEM, STFPM, RD, DFM, FastFlow, CFlow, CNNSpot, PatchForensics, Co-Occur, FreqSpec, DIRE, UnivFD, Freq-mask and OnlyReal. See [original references](https://arxiv.org/html/2508.09477v1#S4.SS2). Some published rows were taken from earlier papers. Fresh comparisons must record upstream revision, manifests, weights, preprocessing and score orientation.

Export anomaly scores, with larger scores meaning fake:

```csv
dataset,image_id,label,score
progan,benchmark/progan/real/image_1.png,0,-1.5
progan,benchmark/progan/fake/image_2.png,1,2.1
```

Use globally stable image IDs and a separate validation CSV. Each export must come from one method/checkpoint. Validation contains held-out natural/proxy samples on the same score scale. The evaluator selects one validation threshold for all test datasets and rejects duplicate sample keys, validation/test ID overlap and nonfinite scores. It evaluates supplied rows only: externally verify complete dataset coverage.

```bash
python -m clipflow.baselines --test-csv external/method_test.csv --validation-csv external/method_validation.csv --output runs/method/metrics.json
```

Omit validation for AP-only evaluation. Published numbers are references, not automatically generated outputs of this command.
