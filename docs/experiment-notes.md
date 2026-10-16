# Experiment notes

## Data pipeline

1. Downloaded the official DIV2K HR/LR ×4 train and validation archives and Flickr2K archive using `scripts/download_training_data.py`. The downloader uses two concurrent HTTP range workers, retries, validates the server's `Content-Range`, and records SHA-256. The observed hashes are in [`dataset_sha256.txt`](../results/dataset_sha256.txt).
2. Extracted only the images needed for ×4 bicubic super-resolution. Checked the PNG headers, image count, and exact ×4 dimensions for 800 DIV2K training pairs, 2,650 Flickr2K pairs, and 100 DIV2K validation pairs.
3. Used the upstream SAFMN crop worker to create aligned 480×480 HR and 120×120 LR crops with strides 240/60. The recorded run produced 139,995 pairs. A BasicSR metadata index was generated after confirming HR/LR filenames matched, avoiding slow folder-pair enumeration at startup.

The source images and crop files remain on the cloud data disk and are not part of this repository. The Set5 files used for external testing are from a third-party mirror.

## Training sequence

The cloud run used a project-local `.venv`, one RTX 4090, and the SAFMN repository commit in the main README. The network settings, batch size 64, optimizer, L1/FFT losses, and 500k cosine schedule follow the pinned SAFMN configuration. The data paths and BasicSR `meta_info_file` index were adapted to the cloud dataset layout.

| Stage | Purpose | Result |
| --- | --- | --- |
| 30-step smoke run | Verify data, optimizer, validation, and save path | Checkpoint/state saved; not a quality result |
| 100-step data-loader checks | Compare full-image decoding and prepared crop loading | 1.021 versus 0.460 seconds/step in these short runs |
| 5k training stage | First interpretable checkpoint | DIV2K val100 RGB PSNR 28.1530 dB (EMA) |
| Resume from `5000.state` | Continue the same experiment | 30k EMA PSNR 28.7915 dB; stopped after checkpoint and validation |

The resume log is [`full500k_resume.stdout.log`](../results/logs/full500k_resume.stdout.log). The 30k checkpoint held both `params` and `params_ema`. Its optimizer state reported iteration 30000. The files were readable in PyTorch, and local/cloud SHA-256 values matched:

| File | SHA-256 |
| --- | --- |
| `net_g_30000.pth` | `c8f372864a1479fb555d1f2a4793b31ce5d1e6a55741ca2393b37606dc0370f3` |
| `30000.state` | `d35fabbd2c8108ce62b3c0c68a4835294e1027a0121cff469c0c050f5014ebbe` |

The experiment-specific [`stop_safmn_at_50000.py`](../scripts/stop_safmn_at_50000.py) observed each saved checkpoint and validation score. From 20k onward it would stop after a complete checkpoint if the EMA PSNR was within 0.20 dB of the pretrained baseline and its 10k gain was below 0.05 dB; 50k was the upper limit. It sent SIGTERM at 30k after PSNR 28.7915 dB was logged. Remaining orphan data-loader workers were stopped separately.

## Reading the metrics correctly

SAFMN's training validation calls `SRModel.test()`, which selects `net_g_ema` when it exists. A direct test config loads `params` unless configured otherwise. Thus the training EMA score 28.7915 dB and the direct 30k ordinary-weight score 28.6901 dB answer different questions. For a matching weight-key comparison with the author's pretrained `params`, use 28.6901 versus 28.9615 dB on the same 100 DIV2K images, RGB PSNR, border crop 4.

The 30k Set5 result for ordinary weights was 31.7874 dB PSNR / 0.8895 SSIM on the Y channel. The author's pretrained SAFMN weights scored 32.1836 / 0.8948 with the same five-image mirror and protocol. This small dataset does not establish general performance. CAMixerSR's pretrained Set5 test is a separate model check and does not show that its architecture was trained in this repository.

Stopping at 30k was a time-budget decision informed by the validation trend. The original scheduler still had a 500k period and the full author training target was not reached.
