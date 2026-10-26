# SR Training Workbench

面向 ×4 图像超分辨率的数据处理、云端训练控制与结果分析工具。

本项目整理我完成的数据检查、图块准备、训练管理、检查点核验、评估和可视化工作。SAFMN 与 CAMixerSR 的网络实现来自原作者，通过各自的 BasicSR 训练流程使用；项目贡献与第三方来源分别说明。

## 1. 项目内容

| 模块 | 我完成的内容 | 入口 |
| --- | --- | --- |
| 数据准备 | 续传下载、HR/LR 配对检查、对齐裁图、BasicSR 元数据索引 | [数据工具](scripts/README.md#1-数据准备data) |
| 云端工作流 | 项目独立环境、真实批次预检、顺序训练管理、按检查点停止 | [工作流工具](scripts/README.md) |
| 配置管理 | 按模型及训练／测试用途整理实测 YAML | [配置索引](configs/README.md) |
| 实验分析 | EMA 与普通参数区分、FFT 损失对照、统一口径评估 | [实验主题](studies/README.md) |
| 结果证据 | 指标、原始日志、来源哈希、验收与对比图 | [结果索引](results/README.md) |
| 本地学习 | 参数更新、网络尺寸、CPU 单图推理及 EMA 对比 | [学习指南](learning/README.md) |

## 2. 目录结构

```text
studies/                  按研究问题组织的实验入口
  pretrained-inference/   作者权重推理
  safmn-training-4090/     早期 SAFMN 训练、续训与复测
  camixersr-training-5090/ CAMixerSR 5k 短训
  safmn-fft-paired-30k/    同环境 FFT／无 FFT 对照
scripts/
  data/                   下载、解压、配对、裁图、索引
  environment/            项目环境准备
  preflight/              真实批次预检
  training/               队列与停止控制
  evaluation/             指标与图像评估
configs/
  safmn/{train,test}/      SAFMN 配置
  camixersr/{train,test}/  CAMixerSR 配置
results/                  指标、原始证据和结果图
learning/                 教学脚本与精选样例
docs/                     详细报告与目录约定
patches/                  上游兼容补丁
third_party/              第三方许可
```

每项实验按 **目标 → 配置与方法 → 结果 → 证据 → 限制** 阅读。后续新增材料和提交遵循 [内容组织约定](docs/repository-structure.md)。

## 3. 实验主题

| 实验 | 主要结果 | 范围 |
| --- | --- | --- |
| [作者权重推理](studies/pretrained-inference/README.md) | SAFMN / CAMixerSR Set5 Y PSNR 为 32.1836 / 32.5533 dB | 预训练加载与评估流程检查 |
| [SAFMN 4090 训练](studies/safmn-training-4090/README.md) | 30k DIV2K EMA RGB PSNR 28.7915 dB | 从早期 5k 续训，按短训规则停止 |
| [CAMixerSR 5090 短训](studies/camixersr-training-5090/README.md) | 5k DIV2K EMA RGB PSNR 27.5543 dB | 第二个 BasicSR 项目的单次短训 |
| [SAFMN FFT 配对 30k](studies/safmn-fft-paired-30k/README.md) | FFT / 无 FFT 为 28.7956 / 28.7050 dB | 同环境、共同初值的单种子对照 |

以上指标均裁边 4；Set5 Y 与 DIV2K RGB 是不同口径。不同实验的硬件、batch 和优化设置有差异，不能根据此表给网络作公平性能排名。完整标签见 [指标 CSV](results/metrics.csv)。

## 4. 最新受控实验：FFT 损失的收益

两组 SAFMN ×4 使用同一张 RTX 5090、相同 DF2K 图块、batch 64、seed 10 和共同随机初始化，各训练 30k。训练差异为 **L1 + 0.05 FFTLoss** 与 **仅 L1**；保留原 500k 余弦周期和 EMA 0.999。

| 损失 | DIV2K val100 RGB PSNR | Set5 RGB PSNR | Set5 Y PSNR / SSIM |
| --- | ---: | ---: | ---: |
| L1 + 0.05 FFTLoss | **28.7956 dB** | **30.0542 dB** | **31.9488 dB / 0.891679** |
| L1 | 28.7050 dB | 29.9539 dB | 31.8489 dB / 0.889826 |

这里全部加载 `params_ema`，裁边 4；Set5 为五图算术均值。DIV2K 的差距由 5k 时的 **+0.4089 dB** 缩小到 30k 时的 **+0.0906 dB**。该结果支持本次短训中的 FFT 收益，不代表充分收敛、多种子稳定性或完整论文训练结论。

![FFT 配对实验验证曲线](results/safmn_fft_pair_30k/results/training_curve.png)

[实验入口](studies/safmn-fft-paired-30k/README.md) · [完整报告](docs/SAFMN-FFT-paired-30k.md) · [12 条定期验证](results/safmn_fft_pair_30k/results/training_metrics.csv) · [效果对比](results/safmn_fft_pair_30k/results/comparisons/)

## 5. 从哪里开始

- **了解做了什么**：先看 [实验主题](studies/README.md)，再沿每项实验的证据链接核对。
- **学习思路和框架**：从 [学习指南](learning/README.md) 和 [精选推理图](learning/examples/README.md) 开始，已有图片无需云服务器。
- **准备数据和运行工具**：按 [脚本索引](scripts/README.md) 准备上游 checkout、数据及项目独立环境，再选 [配置](configs/README.md)。
- **查详细协议**：进入 [文档导航](docs/README.md) 或 [结果索引](results/README.md)。

配置保留原云端 `/root/fsas/sr-repro` 路径；仓库功能目录与云端 suite 部署目录分别说明。环境脚本依赖已有项目布局，使用前按工具索引核对条件。历史归档保留当时的文件与来源哈希。

## 6. 模型、框架与数据来源

- [SAFMN](https://github.com/sunny2109/SAFMN)：上游提交 `1a412066e927da640bc7e2a53ea60f9ffc134241`，网络与其 BasicSR 训练代码。
- [CAMixerSR](https://github.com/icandle/CAMixerSR)：上游提交 `d690b12bb2a0a185dc0049538b377d569f520497`，网络及 `CAModel` 扩展。
- [BasicSR](https://github.com/XPixelGroup/BasicSR)：参考提交 `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a`；具体运行所用代码与版本见各实验环境记录。
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 与 [Flickr2K](https://cv.snu.ac.kr/research/EDSR/)：训练数据来源；DF2K 准备得到 139,995 对图块。
- [Set5 镜像](https://huggingface.co/datasets/eugenesiow/Set5)：五图评估和精选效果图的来源。

仓库公开工具、配置、测量证据及精选图。数据集、环境和二进制权重单独存放；网络原作者、数据来源与许可见 [第三方说明](THIRD_PARTY.md)。
