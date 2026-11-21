# CLIP-Flow

English | [简体中文](README_zh-CN.md)

Experiments for **CLIP-Flow: A Universal Discriminator for AI-Generated Images Inspired by Anomaly Detection** — Zhipeng Yuan, Kai Wang, Weize Quan, Dong-Ming Yan, and Tieru Wu, DFF 2025.

[Paper](https://arxiv.org/abs/2508.09477) · [Published version](https://doi.org/10.1145/3746265.3759663) · [Provenance and reproduction limits](docs/REPRODUCTION.md)

This repository consolidates the supplied `cflow-ad-master.zip` and `cflowtest.zip`. Original experiment sources and logs are retained under `legacy/`. The supported `clipflow/` package is a reconstructed, configurable implementation; it is not a recovered copy of the exact final training revision. Tests establish software behavior, not reproduction of the reported benchmark numbers.

## Experiment coverage

| Paper experiment | Configuration / command |
| --- | --- |
| Tables 1–2: natural-only, proxy-only, joint training | `configs/N.json`, `P.json`, `NP.json`; 19-dataset evaluation |
| Figure 3: score histograms | `python -m clipflow.figures histograms` |
| Table 3: proxy comparison | `proxy_smooth`, `proxy_sharpen`, `proxy_noise`, `proxy_jitter`, `proxy_phase`, `NP` |
| Table 4: frequency-region comparison | `NP`, `band_mid`, `band_high` |
| Figure 4: dimension reduction | 2, 4, 8, 16, 32, 64, 128, 256, 512, 768 dimensions and `no_dr` |
| Feature normalization ablation | `NP`, `no_normalization` |
| Table 5: emerging generators | `evaluate --suite emerging` with the same P checkpoint, without retraining |
| External comparison methods | supplied `legacy/UniversalFake/` source; common score evaluator in `clipflow.baselines` |

External baseline algorithms are not all reimplemented here. Most comparison rows in the paper come from external publications. Use each method's upstream implementation and configuration for comparisons. The common evaluator accepts independently produced baseline scores; see [docs/BASELINES.md](docs/BASELINES.md).

## Install

Python 3.9 or newer is required. Use a compatible torch/torchvision pair for your hardware.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The frozen CLIP ViT-L/14 weights are downloaded from the URL and verified against the checksum embedded in the vendored CLIP loader. `--cache-dir` controls the cache location. Use `--device cuda:0` for training on an available NVIDIA GPU; CPU execution is supported for checks.

## Prepare data

Use a directory containing only natural training-source images. The split command records a **new** deterministic split; exact original paper membership was not supplied. You can instead provide original training and validation manifests as JSON lists of paths. Relative paths are resolved against each manifest's directory.

```bash
python -m clipflow split --real-root data/lsun_real --output data/splits
```

Default manifests contain 7,200 training images and 1,000 disjoint validation natural images. The validation count is a reconstruction choice. Never include benchmark test images in either split. The CLI enforces 7,200 training images unless `--allow-other-count` explicitly marks a non-paper run.

The benchmark root contains the 19 names listed in `clipflow/settings.py`. Each dataset can use `test/good`, `test/bad`, `test/0_real`, `test/1_fake`, or class-nested `0_real`/`1_fake` directories. An optional `--mapping` JSON object maps canonical names to local folders. Emerging-model folders are `dalle3`, `deepfloyd`, `realvisxl_v4`, and `sd15`, each with 1,000 real and 1,000 fake images. Dataset creation prompts, generator settings and images were not supplied.

## Train, calibrate, evaluate

`main.py` runs the complete main experiment in one command: training, independent validation-threshold calibration, then all 19 test datasets. Default mode is N+P; `--mode P` selects proxy-only and `--mode N` natural-only. This entry point deliberately excludes the ablation sweeps.

```bash
python main.py --train-manifest data/splits/train.json --validation-manifest data/splits/validation.json --test-root data/benchmark --output runs/main_NP --device cuda:0
```

The default N+P schedule is 10 epochs with low-frequency masking ratio 0.1. P uses 30 epochs and ratio 1.0. Training/validation manifests and every benchmark folder are checked before training. `--checkpoint` skips training only for an existing consolidated checkpoint matching the selected main configuration. With data in the default locations, `python main.py --device cuda:0` is sufficient.

The equivalent individual stages remain available for checkpoint management and ablations:

```bash
python -m clipflow train --config configs/NP.json --train-manifest data/splits/train.json --output runs/NP --device cuda:0
python -m clipflow calibrate --checkpoint runs/NP/last.pt --validation-manifest data/splits/validation.json --output runs/NP/calibration.json --device cuda:0
python -m clipflow evaluate --checkpoint runs/NP/last.pt --calibration runs/NP/calibration.json --test-root data/benchmark --output runs/NP/benchmark --device cuda:0
```

Accuracy uses one threshold selected exclusively from held-out validation natural images and frequency-perturbed copies. Test labels never choose the threshold. Omitting `--calibration` produces AP-only evaluation. Missing datasets cause an error; `--datasets progan` is available for explicit subset checks and is marked as an incomplete suite. Checkpoints contain settings, membership metadata and learned projection/flow weights. Training refuses to overwrite an existing `last.pt`.

## Run all experiment families

First print and review the complete command plan:

```bash
python -m clipflow.experiments --groups all --train-manifest data/splits/train.json --validation-manifest data/splits/validation.json --test-root data/benchmark --emerging-root data/emerging --output runs/paper --device cuda:0
```

Add `--execute` to run it in a fresh output directory. Shared configurations are trained once. Table 5 evaluates `runs/paper/P/last.pt`, without extra training. Individual groups are `main`, `proxies`, `bands`, `dimensions`, `normalization`, and `emerging`.

```bash
python -m clipflow.figures histograms --p-results runs/paper/P/benchmark --np-results runs/paper/NP/benchmark --output runs/paper/figure3.png
python -m clipflow.figures dimensions --root runs/paper --output runs/paper/figure4.png
```

New runs write `training.json`, calibrated-threshold metadata, `metrics.json`, and per-dataset score arrays. Plots read those new-run artifacts. Archived numbers are never substituted for predictions.

## Historical evidence and checkpoints

The supplied test archive contains 221 complete 19-dataset result blocks. The audit finds AP vectors matching the paper's P and N+P main rows and mid/high frequency rows at all 19 displayed values. This is evidence of matching recorded results, not an independently executed reproduction. See [docs/LOG_AUDIT.json](docs/LOG_AUDIT.json).

```bash
python -m clipflow.history --root legacy/cflowtest --output runs/log-audit.json
```

To import an archived CLIP-Flow checkpoint, `import-legacy` supports the archived `encoder_state_dict`, `MLP_state_dict`, and `decoder_state_dict` format, with an explicitly selected configuration and original training manifest. Compatibility and encoder identity are checked strictly. No automatic guess of mode or threshold is made.

## Attribution

This work builds on [CFLOW-AD](https://github.com/gudovskiy/cflow-ad), [OpenAI CLIP](https://github.com/openai/CLIP), [FrEIA](https://github.com/vislearn/FrEIA), and [UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect). The archive's BSD notice remains in `LICENSE`; CLIP's MIT notice is in `models/clip/LICENSE`. Unmodified historical source hashes are recorded in `docs/ARCHIVE_MANIFEST.json`.

```bibtex
@inproceedings{yuan2025clipflow,
  title={CLIP-Flow: A Universal Discriminator for AI-Generated Images Inspired by Anomaly Detection},
  author={Yuan, Zhipeng and Wang, Kai and Quan, Weize and Yan, Dong-Ming and Wu, Tieru},
  booktitle={Proceedings of the 1st on Deepfake Forensics Workshop},
  year={2025},
  doi={10.1145/3746265.3759663}
}
```
