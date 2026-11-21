# CLIP-Flow

Source code associated with **CLIP-Flow: A Universal Discriminator for AI-Generated Images Inspired by Anomaly Detection**, by Zhipeng Yuan, Kai Wang, Weize Quan, Dong-Ming Yan, and Tieru Wu (DFF 2025).

- [Paper](https://arxiv.org/abs/2508.09477)
- [Published version](https://doi.org/10.1145/3746265.3759663)
- [Repository named in the paper](https://github.com/Yzp1018/CLIP-Flow)

## Release status

This is a selected **research source snapshot** from the author-provided `cflow-ad-master.zip`. The original Python files are preserved without algorithm changes. This snapshot has not been validated as the final configuration that produced the paper's reported results. In particular, the active data loader currently uses a color-jitter proxy rather than the paper's frequency-masking proxy; see the limitations below.

No trained CLIP-Flow checkpoint or dataset is included. The large `UniversalFake` baseline checkpoints in the archive are excluded because their correspondence to the CLIP-Flow model has not been established.

## Code and method correspondence

| Component | Source |
| --- | --- |
| Entry point and configuration | `main.py`, `config.py` |
| Frozen CLIP ViT-L/14 image features | `train_vit_512.py` |
| Linear dimension reduction, 768 to 128 dimensions, followed by L2 normalization | `train_vit_512.py` |
| Normalizing flow built with FrEIA | `model.py` |
| Natural/proxy likelihood objective | `train_vit_512.py`, `utils.py` |
| Data loading and proxy construction | `custom_datasets/loader1.py` |
| Vendored CLIP implementation and tokenizer vocabulary | `models/clip/` |
| Supporting encoders, utilities and visualization | `custom_models/`, `models/`, `resnet.py`, `utils.py`, `visualize.py` |

The active path is `main.py` -> `train_vit_512.py` -> `custom_datasets/loader1.py`. Some inherited encoder modules remain because they are imported by the original package initializers.

## Environment and use

The archive's original dependency list is retained in `requirements.txt`; it is not a tested environment lockfile. The selected source also imports FrEIA, OpenCV (`cv2`), Pillow, `ftfy`, `regex`, and `pkg_resources`/`packaging`. The vendored CLIP implementation relies on older `pkg_resources` behavior, so dependency compatibility must be checked before running it. No dependency installation or GPU training was performed for this upload.

The inherited CFLOW-AD setup notes are preserved in [docs/CFLOW_AD_README.md](docs/CFLOW_AD_README.md) for provenance. Their MVTec commands, results and figures describe the upstream project, not a CLIP-Flow reproduction recipe.

Before using this snapshot:

1. Install a compatible PyTorch/torchvision environment and the dependencies above. FrEIA is available at https://github.com/vislearn/FrEIA.
2. Replace the original server data/cache paths in `main.py` and `custom_datasets/loader1.py` with local paths. The loader currently reads real training images from `ProGAN/train/0_real` and test images from `ProGAN/test/good` and `ProGAN/test/bad`.
3. Review the CUDA device selection in `main.py` and `config.py` for the available hardware.
4. Resolve and validate the proxy configuration described below before attempting to reproduce paper results. Use `norm-train` for training or `norm-test` with a compatible checkpoint for evaluation; these are existing configuration options, not verified reproduction commands.

## Known limitations in the supplied snapshot

- In `custom_datasets/loader1.py`, the frequency-masking path is behind `if random.random() < 0`, so it is never executed. The active proxy branch uses `ColorJitter`. The dormant frequency path also needs its image-to-tensor transform reviewed before enabling it.
- The training loader duplicates a subsampled set of real images with labels 0 and 1. The active loss maximizes natural-image likelihood and minimizes proxy-image likelihood. A separate, verified proxy-only (P) training configuration is not supplied here.
- Data paths, an image-loading fallback path, cache settings and a CUDA device are specific to the original server. The `stc` option is inherited but its dataset class is not exported by the selected package initializer.
- Evaluation selects an accuracy threshold from the evaluated labels. That accuracy should not be treated as performance at an independently calibrated deployment threshold.
- Model weights, exact package versions, the full benchmark setup and full paper-result reproduction have not been verified.

## Provenance and attribution

The selected archive files and their SHA-256 checksums are listed in [docs/SOURCE_MANIFEST.json](docs/SOURCE_MANIFEST.json). Source selection excludes `UniversalFake/`, unrelated training/export variants, generated plots, logs and large checkpoint files. The original archive remains the reference for those materials.

The code builds on [CFLOW-AD](https://github.com/gudovskiy/cflow-ad), [OpenAI CLIP](https://github.com/openai/CLIP), and [UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect). The original BSD 3-Clause notice is retained in [LICENSE](LICENSE); the CLIP MIT notice is included in [models/clip/LICENSE](models/clip/LICENSE). Existing source notices are preserved.

## Citation

```bibtex
@inproceedings{yuan2025clipflow,
  title={CLIP-Flow: A Universal Discriminator for AI-Generated Images Inspired by Anomaly Detection},
  author={Yuan, Zhipeng and Wang, Kai and Quan, Weize and Yan, Dong-Ming and Wu, Tieru},
  booktitle={Proceedings of the 1st on Deepfake Forensics Workshop: Detection, Attribution, Recognition, and Adversarial Challenges in the Era of AI-Generated Media},
  year={2025},
  doi={10.1145/3746265.3759663}
}
```
