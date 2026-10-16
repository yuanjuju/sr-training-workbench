# SR Training Workbench

**Image super-resolution experiments built around data preparation, cloud training, and checkpoint analysis.**

This repository contains the experiment tooling and measurements I developed while studying ×4 single-image super-resolution. I built the dataset checks, resumable download and crop workflow, staged training setup, stop guard, and metric comparisons. The SAFMN and CAMixerSR network implementations belong to their original authors; their source code and pretrained models are linked below, not copied here.

## What I built

- Resumable, range-based downloads of the DIV2K and Flickr2K archives, with size checks and SHA-256 records.
- HR/LR pair validation, ×4 dimension checks, a combined DF2K index, resumable subimage preparation, and a checked BasicSR metadata index.
- A cloud training setup with saved checkpoints every 5,000 steps and a run-specific stop guard that checks both the checkpoint and validation score before stopping.
- Evaluation records that keep dataset, color channel, crop border, and checkpoint key (`params` versus `params_ema`) explicit.

## Experiment snapshot

| Item | Setting |
| --- | --- |
| Task | Bicubic ×4 single-image super-resolution |
| Training data | DF2K: 800 DIV2K + 2,650 Flickr2K HR/LR pairs |
| Prepared data | 139,995 aligned HR/LR crop pairs |
| Validation | DIV2K validation set, 100 images; RGB PSNR, border crop 4 |
| Additional check | Set5 mirror, 5 images; Y-channel PSNR/SSIM, border crop 4 |
| Training | SAFMN, batch 64, 256-pixel HR random crops, L1 + 0.05 FFT loss |
| Hardware | One RTX 4090 24 GB; Python 3.10; PyTorch 2.1.1+cu121 |

### Measured results

| Model/checkpoint | Weights used | DIV2K val100 RGB PSNR | Set5 Y PSNR / SSIM |
| --- | --- | ---: | ---: |
| SAFMN author's pretrained ×4 | `params` | 28.9615 dB | 32.1836 dB / 0.8948 |
| SAFMN trained from scratch, 5k | `params_ema` on DIV2K; `params` on Set5 | 28.1530 dB | 31.4363 dB / 0.8828 |
| SAFMN trained from scratch, 20k | `params_ema` | 28.7439 dB | — |
| SAFMN trained from scratch, 30k | `params_ema` | 28.7915 dB | — |
| SAFMN trained from scratch, 30k | `params` | 28.6901 dB | 31.7874 dB / 0.8895 |
| CAMixerSR author's pretrained ×4 | `params_ema` | — | 32.5533 dB / 0.8997 |

The SAFMN run stopped at 30k because its EMA validation PSNR improved by only 0.0476 dB from 20k to 30k and met the preselected short-run stopping rule. The ordinary 30k weights remain 0.2714 dB below the author's pretrained weights on the same DIV2K protocol. **This is a controlled short training experiment, not a claim to have completed the authors' 500k-step training or matched their paper results.** The 500k learning-rate schedule was retained, so the 30k result should not be interpreted as a fully converged short-schedule model.

The Set5 files came from a third-party mirror and include only five test images. Set5 Y-channel scores and DIV2K RGB scores are different protocols; compare values only within their own column. See [all recorded metrics](results/metrics.csv) and the [experiment notes](docs/experiment-notes.md).

## Repository layout

```text
configs/          Training and evaluation YAMLs used in the cloud experiment
scripts/          Data preparation and the run-specific stop guard
results/metrics.csv
results/logs/     Selected raw logs supporting the table above
docs/             Data provenance, run steps, and interpretation notes
```

The YAML files intentionally preserve the absolute paths used in my cloud run (`/root/fsas/sr-repro`). Change the `dataroot_*`, `meta_info_file`, and checkpoint paths for a new machine. The stop guard also records the original process ID and paths; treat it as an experiment artifact and adapt it before another run.

## Data and code provenance

- [SAFMN](https://github.com/sunny2109/SAFMN) at `1a412066e927da640bc7e2a53ea60f9ffc134241` supplied the model and BasicSR-based training pipeline.
- [CAMixerSR](https://github.com/icandle/CAMixerSR) at `d690b12bb2a0a185dc0049538b377d569f520497` supplied the second model for pretrained inference.
- [BasicSR](https://github.com/XPixelGroup/BasicSR) at `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a` was studied separately; SAFMN contains its own BasicSR-based code.
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](https://cv.snu.ac.kr/research/EDSR/) supplied the training images. The DIV2K page describes academic research use and requests citation. Dataset images and archives are not included here.
- The Set5 evaluation used a [third-party dataset mirror](https://huggingface.co/datasets/eugenesiow/Set5), so these scores are recorded as local protocol checks rather than official benchmark claims.

The training data, upstream source trees, upstream pretrained weights, and generated image examples are not committed. The 30k model and optimizer state are retained outside this public repository; their hashes and validation evidence are documented in [experiment notes](docs/experiment-notes.md).

See [third-party attribution](THIRD_PARTY.md) for the origin of model code, configuration formats, and data.

## Running the data pipeline

The following commands show the order of the scripts. Clone SAFMN into `./SAFMN` at the pinned commit and create a project-local Python environment with the upstream dependencies first. Plan for substantial cloud storage; the recorded 100 GB data disk held the source images and prepared crops.

```bash
python scripts/download_training_data.py datasets/_archives
python scripts/extract_training_data.py datasets
python scripts/prepare_df2k_pairs.py datasets
python scripts/prepare_safmn_subimages.py datasets
python scripts/build_meta_info.py datasets/DF2K --expected-count 139995
```

The last two scripts create aligned crops and `datasets/DF2K/meta_info_HR_sub.txt`, checking that names and dimensions match. This metadata-index command packages the manual check used in the original cloud run into a reusable step. Set the paths in `configs/SAFMN_train_DF2K_x4_sub_first5000.yml`, run it from the SAFMN checkout, and use `configs/SAFMN_train_DF2K_x4_sub_cloud.yml` with BasicSR's `--auto_resume` to continue. See the [experiment notes](docs/experiment-notes.md) for the exact stages and caveats.
