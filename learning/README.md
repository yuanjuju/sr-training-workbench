# 本地学习：前向计算、参数更新与单图超分

目标：亲眼验证 SAFMN 的尺寸变化、参数更新，并用已训练权重处理真实图片。
练习显式使用 CPU；前两项使用合成张量，第三项读取本地图片。

## 环境

在仓库根目录运行，环境保存在 `.venv-learning/`：

```bash
python3.12 -m venv .venv-learning
.venv-learning/bin/python -m pip install --no-cache-dir -r learning/requirements.txt
```

本课固定 PyTorch 2.6.0，macOS 安装方式见 [PyTorch 官方历史版本页](https://pytorch.org/get-started/previous-versions/#v260)。
它是独立的学习环境；早期 4090 SAFMN 使用 PyTorch 2.1.1，后续 5090 SAFMN 对照与 CAMixerSR 使用 2.7.1。
前两项为计算过程练习；第三项的单图指标单独记录评估口径。

## 练习一：观察原始 SAFMN 的前向计算

默认读取仓库旁边的 `../SAFMN`，也可通过 `--checkout` 指定路径。

```bash
.venv-learning/bin/python learning/01_forward_and_update.py trace
```

默认网络为 `dim=36, n_blocks=8, scale=4`，预期参数量 239,520，输出 `[1,3,256,256]`。
脚本用 forward hook 观察模块的输入、输出形状。一次只详细打印第一个特征块，其余块仍正常参与计算。
权重是随机初始化的，这一练习验证尺寸和调用关系。

为保持依赖简单，辅助文件 `_safmn_source.py` 单独执行本地架构及注册表的源码，
跳过未使用的 `torchvision.ops` 导入和 BasicSR 的整包初始化。
原始网络类及 forward 计算保持不变；此方式仅用于本课，完整训练仍使用原项目的 BasicSR 导入路径。

先预测结果，再运行这两条命令：

```bash
.venv-learning/bin/python learning/01_forward_and_update.py trace --height 80 --width 96
.venv-learning/bin/python learning/01_forward_and_update.py trace --dim 32 --blocks 4
```

- 第一条：输出应该是 `[1,3,320,384]`。
- 第二条：第一个 SAFM 的四组各 8 通道，CCM 扩展到 64 通道；空间放大倍数仍是 4。

## 练习二：观察参数怎样改变

打开 `01_forward_and_update.py`，先读 `TinySR` 和 `inspect_update()`。
TinySR 是本课独立编写的三层卷积教学网络，共 9,728 个参数。
一张合成的 32×32 目标图经 bicubic 缩小为 8×8，形成一对固定的训练样本。

```bash
.venv-learning/bin/python learning/01_forward_and_update.py update
```

重点观察：

1. `zero_grad()` 清掉梯度，模型参数不变。
2. `model(lq)` 计算预测图，`L1` 将预测图和目标图变成一个损失标量。
3. `backward()` 计算梯度，此时参数仍不变。
4. `optimizer.step()` 使用梯度更新参数。
5. 再次前向计算，观察同一个训练样本上的损失变化。

输出中的 `grad L2` 是第一层卷积梯度张量的 L2 范数，表示本次梯度的整体大小；
`max weight change` 是这层权重更新前后绝对差的最大值。梯度大小不能直接代表图像质量。

接着只改变步数，重复使用同一对样本：

```bash
.venv-learning/bin/python learning/01_forward_and_update.py update --steps 20
```

这是学习单样本拟合的练习，打印的是训练损失；没有独立验证集，不能据此判断泛化能力。
更换学习率可能改变下降速度或稳定性，一次更新也不保证任意网络、任意学习率下的损失都下降。

2026-10-21 本地 CPU 实测（Python 3.12.14、PyTorch 2.6.0、seed=10、默认学习率 0.001）：

```text
SAFMN 参数量：239,520
SAFMN 输出：[1,3,256,256]
TinySR 参数量：9,728
一步更新的 L1：0.503319 -> 0.494489
第一层卷积梯度 L2：0.035844
第一层卷积权重的最大绝对变化：0.001000
zero_grad 后梯度为 None：True
backward 后权重仍未改变：True
optimizer.step 后权重发生变化：True
```

这些数值只描述本课的合成样本。跨平台或版本运行时，浮点数末位可能不同。

## 对应到 BasicSR

| 本课代码 | SAFMN 自带的 BasicSR |
| --- | --- |
| `lq, gt = make_teaching_pair()` | Dataset + DataLoader + `SRModel.feed_data()` |
| `sr = model(lq)` | `self.output = self.net_g(self.lq)` |
| `F.l1_loss(sr, gt)` | `self.cri_pix(self.output, self.gt)` |
| 本课只使用 L1 | 原实验还可配置 `self.cri_fft` |
| `loss.backward()` | `l_total.backward()` |
| `optimizer.step()` | `self.optimizer_g.step()` |

BasicSR 进一步组织配置解析、注册、EMA、调度器、保存和验证；本课让最核心的一次更新直接可见。

## 练习三：用自己的 30k 权重处理真实图片

这项练习的重点是运行和观察图片，无需阅读脚本实现。
使用早期 4090 SAFMN 30000 步检查点 `../results/net_g_30000.pth`，默认加载普通权重 `params`；它不同于后续 5090 paired30k 的检查点。
检查点和 Set5 数据需自行准备，未随本仓库发布；默认目录对应原实验工作区，换机器时请显式指定路径。
本课默认上游源码在 `../SAFMN`；若按主 README 将它克隆到本仓库的 `./SAFMN`，请添加 `--checkout ./SAFMN`（练习一、三均支持）。
单图入口复用上面的网络加载辅助文件和原始网络计算；数据读取、推理与结果展示由本课入口组织。

先运行一张已有 Set5 图片：

```bash
cd /path/to/sr-training-workbench
.venv-learning/bin/python learning/02_infer_image.py
```

默认读取 butterfly 的 64×64 低清图片，输出 256×256 图片，
并将低清输入放大展示、Pillow 双三次插值、SAFMN 输出、高清原图排成对照图。
下一步换一张 bird，亲自运行同一套流程：

```bash
.venv-learning/bin/python learning/02_infer_image.py --input ../datasets/Set5/LR_bicubic/X4/bird.png --gt ../datasets/Set5/HR/bird.png
```

所有相对路径均按本仓库根目录解析；原图和检查点保持在原位置。
输出分别在 `learning/outputs/butterfly_30k/` 和 `learning/outputs/bird_30k/`，包括：

- `sr.png`：网络输出；
- `bicubic.png`：Pillow 双三次插值参照；
- `comparison.png`：完整图像和中心局部的对照；
- `result.json`：权重来源与哈希、参数键、尺寸、CPU 前向耗时和可用的单图指标。

先观察边缘是否更清楚、细线是否断裂、平滑区域是否产生多余纹理，再看指标。
有匹配高清图时，记录 RGB PSNR、裁边 4、输出量化到 uint8；这是单图结果，不能替代五图或百图平均分。
Pillow 双三次插值用于直观参照，其实现不等同于官方数据生成使用的 MATLAB bicubic。

换成自己的低清图片时：

```bash
.venv-learning/bin/python learning/02_infer_image.py --input /绝对路径/你的图片.png
```

没有匹配高清原图时，仅展示输出与插值参照，不报告 PSNR。
如需切换到同一检查点内的 EMA 权重，可添加 `--weight-key params_ema`；比较时保留参数键标签。

已经运行过的 [精选图与指标](examples/README.md) 可直接查看，无需启动云服务器。
`learning/outputs/` 是本机生成目录，不纳入 Git；公开样例在 `learning/examples/`，仅将路径整理为便于移植的形式，测量值保持原样。
