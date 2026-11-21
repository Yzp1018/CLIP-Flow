# CLIP-Flow

[English](README.md) | 简体中文

论文 **CLIP-Flow: A Universal Discriminator for AI-Generated Images Inspired by Anomaly Detection** 的实验代码。

作者：Zhipeng Yuan、Kai Wang、Weize Quan、Dong-Ming Yan、Tieru Wu。发表于 DFF 2025。

[论文预印本](https://arxiv.org/abs/2508.09477) · [正式发表版本](https://doi.org/10.1145/3746265.3759663) · [实现来源与复现说明](docs/REPRODUCTION.md)

本仓库整合了 `cflow-ad-master.zip` 与 `cflowtest.zip` 中的训练代码、测试流程和实验记录。`main.py` 是论文主实验的统一入口；消融实验由独立入口运行；原始脚本与日志保存在 `legacy/` 中。统一实现的整合选择、参数来源和差异详见复现说明。

## 方法概述

CLIP-Flow 使用冻结的 CLIP ViT-L/14 提取图像特征，经线性降维与 L2 归一化后输入流模型，以负对数似然作为异常分数。训练时使用自然图像以及由自然图像构造的代理图像，无需使用 AI 生成图像作为训练样本。

| 模式 | 训练目标 | 默认设置 |
| --- | --- | --- |
| `N` | 最大化自然图像的似然 | 仅使用自然图像，当前实现默认训练 10 轮 |
| `P` | 最小化代理图像的似然 | 低频掩码比例 1.0，训练 30 轮 |
| `NP` / `N+P` | 同时最大化自然图像、最小化代理图像的似然 | 低频掩码比例 0.1，训练 10 轮 |

默认采用 N+P 模式，特征由 768 维降至 128 维，使用 Adam 优化器，学习率为 `1e-4`，批大小为 128。N 模式的训练轮数属于当前实现的默认设置，参数来源见复现说明。

## 环境安装

需要 Python 3.9 或更新版本。请安装与设备兼容的 PyTorch 和 torchvision 版本。

```bash
git clone https://github.com/Yzp1018/CLIP-Flow.git
cd CLIP-Flow
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

上述虚拟环境激活命令适用于 Linux/macOS；Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`。

CLIP ViT-L/14 预训练参数由加载器自动下载并校验。可以通过 `--cache-dir` 指定缓存目录。有可用 NVIDIA GPU 时使用 `--device cuda:0`；也支持 `--device cpu`，用于功能检查。

## 数据准备

### 训练集与验证集

将自然图像放入单独目录，然后生成训练、验证清单：

```bash
python -m clipflow split --real-root data/lsun_real --output data/splits
```

默认选取 7,200 张自然图像用于训练、1,000 张互不重叠的自然图像用于验证。验证图像数量是当前实现的设置。此命令生成新的固定随机划分；如需使用已有实验划分，可以直接提供 JSON 清单。

清单为图像路径组成的 JSON 数组。支持绝对路径；相对路径以清单所在目录为基准。例如 `data/splits/train.json` 中的路径可以写成：

```json
[
  "../lsun_real/image_0001.png",
  "../lsun_real/image_0002.png"
]
```

训练图像和验证图像均不能包含测试集图像。路径重叠会被拒绝；不同路径下的重复图像需在数据准备时另行检查。主实验默认要求 7,200 张自然训练图像；`--allow-other-count` 仅用于明确标记的其他规模实验或小样本检查。

### 19 个测试集

主实验使用以下目录名：

```text
progan        cyclegan       biggan        stylegan       gaugan
stargan       deepfake       seeingdark    san            crn
imle          guided         ldm_200       ldm_200_cfg    ldm_100
glide_100_27  glide_50_27    glide_100_10  dalle
```

其中 `seeingdark` 对应论文中的 SITD。目录结构示例：

```text
data/
├── lsun_real/
├── splits/
│   ├── train.json
│   └── validation.json
└── benchmark/
    ├── progan/
    │   └── test/
    │       ├── 0_real/
    │       └── 1_fake/
    ├── cyclegan/
    └── ...
```

同时支持 `good` / `bad` 标签目录，以及按类别嵌套的 `0_real` / `1_fake` 结构。`test/` 这一层可省略。使用分阶段的 `evaluate` 命令时，还可以通过 `--mapping` 指定测试集名称与本地目录的映射。

## 一键运行主实验

`main.py` 依次完成三个步骤：**训练 → 独立验证集阈值校准 → 全部 19 个测试集评估**。

```bash
python main.py \
  --train-manifest data/splits/train.json \
  --validation-manifest data/splits/validation.json \
  --test-root data/benchmark \
  --output runs/main_NP \
  --device cuda:0
```

数据按默认位置准备好后，可以直接运行：

```bash
python main.py --device cuda:0
```

切换训练模式：

```bash
python main.py --mode P --output runs/main_P --device cuda:0
python main.py --mode N --output runs/main_N --device cuda:0
```

入口会在训练前检查数据划分及全部测试集目录。`--checkpoint` 可用于评估与所选主实验配置匹配的已有检查点，此时跳过训练。可用参数通过 `python main.py --help` 查看。

准确率采用独立验证集校准得到的一个固定阈值，并将同一阈值应用于所有测试集，不使用测试集标签选择阈值。最终输出 mAP、平均准确率及各测试集指标。

## 分阶段运行

如需分别管理训练、校准和评估，可以使用以下命令：

```bash
python -m clipflow train --config configs/NP.json --train-manifest data/splits/train.json --output runs/NP --device cuda:0
python -m clipflow calibrate --checkpoint runs/NP/last.pt --validation-manifest data/splits/validation.json --output runs/NP/calibration.json --device cuda:0
python -m clipflow evaluate --checkpoint runs/NP/last.pt --calibration runs/NP/calibration.json --test-root data/benchmark --output runs/NP/benchmark --device cuda:0
```

评估时省略 `--calibration`，仅计算 AP。`--datasets progan` 可用于明确指定的单测试集检查，输出会标记为非完整测试流程。训练使用新的输出目录，避免覆盖已有 `last.pt`。

## 消融实验与扩展评估

| 论文实验 | 配置或入口 |
| --- | --- |
| 表 1、表 2：N、P、N+P 主实验 | `configs/N.json`、`P.json`、`NP.json` |
| 图 3：异常分数直方图 | `python -m clipflow.figures histograms` |
| 表 3：代理图像构造方式 | `proxy_smooth`、`proxy_sharpen`、`proxy_noise`、`proxy_jitter`、`proxy_phase`、`NP` |
| 表 4：低频、中频、高频掩码 | `NP`、`band_mid`、`band_high` |
| 图 4：降维消融 | 2、4、8、16、32、64、128、256、512、768 维，以及 `no_dr` |
| 特征归一化消融 | `NP`、`no_normalization` |
| 表 5：新生成模型评估 | `evaluate --suite emerging`，复用 P 模式检查点 |

可以先查看完整执行计划：

```bash
python -m clipflow.experiments \
  --groups all \
  --train-manifest data/splits/train.json \
  --validation-manifest data/splits/validation.json \
  --test-root data/benchmark \
  --emerging-root data/emerging \
  --output runs/paper \
  --device cuda:0
```

默认只打印计划，添加 `--execute` 后执行。支持分别选择 `main`、`proxies`、`bands`、`dimensions`、`normalization`、`emerging` 实验组；共享配置只训练一次。表 5 使用同一个 P 模型，不额外训练或微调。

扩展测试集目录名为 `dalle3`、`deepfloyd`、`realvisxl_v4`、`sd15`，每组默认要求 1,000 张自然图像和 1,000 张生成图像。

绘制新实验的结果：

```bash
python -m clipflow.figures histograms --p-results runs/paper/P/benchmark --np-results runs/paper/NP/benchmark --output runs/paper/figure3.png
python -m clipflow.figures dimensions --root runs/paper --output runs/paper/figure4.png
```

外部对比方法使用各自的上游实现与配置。原包中的 UniversalFake 源码保存在 `legacy/UniversalFake/`；统一分数评估接口见 [外部基线评估说明](docs/BASELINES.md)。

## 输出与目录说明

| 路径 | 内容 |
| --- | --- |
| `main.py` | 主实验的一键入口 |
| `clipflow/` | 数据处理、代理图像、模型、训练、校准、评估与绘图 |
| `configs/` | 主实验及消融配置 |
| `legacy/` | 原始源码和实验日志 |
| `tests/` | 功能与流程测试 |
| `docs/` | 实现来源、日志核对与验证说明 |
| `runs/<实验名>/last.pt` | 训练生成的模型检查点 |
| `runs/<实验名>/training.json` | 实验设置、训练记录和参数量 |
| `runs/<实验名>/calibration.json` | 验证集阈值及对应模型信息 |
| `runs/<实验名>/benchmark/metrics.json` | 各测试集与汇总指标 |
| `runs/<实验名>/benchmark/*.npz` | 图像路径、标签和异常分数 |

绘图使用当前运行生成的分数，不会用历史日志替代模型预测。历史日志包含 221 组完整的 19 测试集记录，核对报告见 [LOG_AUDIT.json](docs/LOG_AUDIT.json)。也可以重新生成报告：

```bash
python -m clipflow.history --root legacy/cflowtest --output runs/log-audit.json
```

软件验证覆盖训练目标、代理图像、阈值校准、检查点读写，以及 `main.py` 的完整小样本流程。测试通过不等同于独立复现论文的大规模实验，具体记录见 [VALIDATION.md](docs/VALIDATION.md)。

## 致谢与引用

本项目基于 [CFLOW-AD](https://github.com/gudovskiy/cflow-ad)、[OpenAI CLIP](https://github.com/openai/CLIP)、[FrEIA](https://github.com/vislearn/FrEIA) 和 [UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect) 等工作。许可证见 `LICENSE` 与 `models/clip/LICENSE`；原始文件校验信息见 `docs/ARCHIVE_MANIFEST.json`。

```bibtex
@inproceedings{yuan2025clipflow,
  title={CLIP-Flow: A Universal Discriminator for AI-Generated Images Inspired by Anomaly Detection},
  author={Yuan, Zhipeng and Wang, Kai and Quan, Weize and Yan, Dong-Ming and Wu, Tieru},
  booktitle={Proceedings of the 1st on Deepfake Forensics Workshop},
  year={2025},
  doi={10.1145/3746265.3759663}
}
```
