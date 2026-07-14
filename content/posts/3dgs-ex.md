---
title: "从官方 3DGS Baseline 到单目转头视频静态头部重建：我踩过的坑与走出来的路"
date: 2026-04-07T12:00:00+08:00
draft: false
tags: ["3DGS", "Computer Vision", "COLMAP", "Research Log", "NeRF"]
summary: "这篇文章记录的不是一份完美成果，而是一段比较真实的工程化科研推进过程：从官方 baseline 验证，到自有视频数据处理，再到输入契约、相机模型、COLMAP 主模型选择，以及最终主线数据集的收敛。"
uid: 3dgs-monocular-head
topics:
  - 3d-vision
  - research-engineering
relations:
  - target: 3dgs-experiment-log-2
    type: extends
    note: "承接日志 #2 对数据链污染的诊断，本文进一步用原生 PINHOLE clean rebuild 和 run_05 收敛可信实验主线。"
---

# 从官方 3DGS Baseline 到单目转头视频静态头部重建：我在 AutoDL 上踩过的坑与走出来的路

## 前言

最近这段时间，我一直在推进一个具体但并不轻松的题目：
**从单目转头短视频中重建静态头部。**

一开始我以为，这件事的核心难点在“模型训练”。真正做起来以后才发现，训练反而不是最先卡住我的地方。更大的问题是：

* 官方 Gaussian Splatting 到底吃什么输入？
* 自己拍的视频，怎样才能变成一个合法可训练的数据集？
* 为什么有的实验明明跑通了，但结果却不该算“成功”？
* 到底哪条数据线，才真正符合“完整头部静态重建”这个任务？

这篇文章，记录的不是一份完美成果，而是一段比较真实的工程化科研推进过程：
从官方 baseline 验证，到自有视频数据处理，再到输入契约、相机模型、COLMAP 主模型选择，以及最终主线数据集的收敛。

如果你也在做 3DGS、NeRF、单目重建、头部重建，或者正在经历“不是模型不会跑，而是整条链路总有地方不对”的阶段，这篇记录可能会对你有帮助。

---

## 1. 我做的其实不是“直接拿视频跑 3DGS”

这个题目表面上看，是“单目转头短视频静态头部重建”。
但如果按工程视角拆开，它本质上并不是一个“视频直接输入模型”的任务。

官方 Gaussian Splatting 并不接受一个 mp4 视频作为输入。它真正要求的是标准的 COLMAP / NeRF 风格数据目录，比如：

```text
images/
sparse/0/cameras.bin
sparse/0/images.bin
sparse/0/points3D.bin
```

也就是说，我真正要解决的问题不是“怎么把视频喂给 3DGS”，而是：
**怎么把单目转头短视频，转成一个既合法、又可训练、还尽量保留完整头部语义的数据集。**

所以这条链路实际上是：

* 原始视频
* → 抽帧 / 筛帧
* → COLMAP 重建
* → 构建 3DGS 可接受的数据目录
* → `train.py`
* → `render.py`
* → `metrics.py`
* → 逐视角分析

这也是为什么我后面越来越觉得，这个题目不是单纯的“调参问题”，而是一个很强的数据链路问题 + 工程约束问题。

---

## 2. 第一件事不是追求效果，而是证明官方 baseline 能跑通

在碰自己的数据之前，我先做了一件最基础但非常必要的事：
**先把官方 baseline 跑通。**

原因很简单。
如果官方 demo 都跑不通，那后面一切自有数据实验都没有意义。你根本不知道问题到底出在环境、仓库、依赖，还是出在你自己的数据。

### 实验环境
我的实验环境大致如下：

* **平台：** AutoDL
* **显卡：** RTX 4090
* **代码仓库：** `/root/gaussian-splatting`
* **conda 环境：** `gaussian_splatting`
* **Python：** 3.7.13
* **PyTorch：** 1.12.1
* **CUDA Toolkit：** 11.6
* **COLMAP：** 3.9 with CUDA

这里还有一个后续会影响流程选择的小点：
**环境里没有 ImageMagick。**
这意味着官方某些依赖 `convert` 的预处理方式，不是我当前环境里的主路径。

### 官方 baseline 命令

```bash
python train.py \
  -s /root/autodl-tmp/gs_baseline_demo/test \
  -m /root/autodl-tmp/outputs/gs_test_run \
  --eval \
  --iterations 3000

python render.py -m /root/autodl-tmp/outputs/gs_test_run
python metrics.py -m /root/autodl-tmp/outputs/gs_test_run
```

### 结果
最终跑出了这样一组 baseline 指标：

* **SSIM：** 0.8452070
* **PSNR：** 25.9392948
* **LPIPS：** 0.1992821

这一步的重要性，不在于指标有多高，而在于它证明了三件事：
1.  我的 AutoDL 4090 环境本身没问题
2.  官方仓库链路没问题
3.  `train` → `render` → `metrics` 这一整条标准流程是通的

从这里开始，我就可以放心地把注意力集中到自己的数据上，而不是怀疑环境。

---

## 3. 早期阶段：我先做的只是“视频变图片”

在正式进入 `head_project` 之前，我有一个早期临时工作区：`head_case`。
它的作用其实很单纯，就是先验证最底层的事情：
**能不能把原始视频转成一批可以用来重建的图片。**

例如最早会有这种操作：

```bash
mkdir -p /root/autodl-tmp/head_case/input
ffmpeg -i head.mp4 -vf fps=3 /root/autodl-tmp/head_case/input/%05d.jpg
```

之后再通过脚本筛帧，只保留一部分更清晰的图片。
最终大概保留了 80 张图。

现在回头看，`head_case` 这条线的定位很明确：
* 它不是正式实验线
* 它还没有真正进入 COLMAP + 3DGS 训练
* 它只是完成了“视频 → 图像”的最低层准备

但这一步依然有意义，因为很多人会高估“数据已经准备好了”这句话的含义。
实际上，视频变图片，只是刚刚开始。

---

## 4. 项目从“脚本试验”升级到“正式工程目录”

后面我不再满足于临时目录，而是把整个项目整理进了正式工程目录 `head_project`。
目录结构大概是这样：

```text
/root/autodl-tmp/head_project/
├── raw_video/
├── A_set/
├── B_set/
├── B_set_gs_s1/
├── B_set_pinhole_clean/
├── metadata/
├── reports/
├── runs/
├── notes/
└── scripts/
```

这次整理对我来说很重要。
因为从这一刻开始，我的项目不再只是“跑脚本”，而开始具备一些真正做科研/做工程该有的东西：

* 原始数据和派生数据分开
* 不同数据集版本分开
* 每轮 run 单独落盘
* 质检图和元数据单独保存
* 后续可以回溯、复查、对比

如果说 `head_case` 阶段更像“在试试能不能做”，
那 `head_project` 阶段才真正进入“我要把这个题目往前推进”的状态。

---

## 5. A_set：第一轮自有数据 baseline，说明我已经不只是跑官方 demo 了

在正式目录里，`A_set` 是第一轮比较正式的自有数据集。
根据已有总结，这一轮对应的 baseline 结果大致是：

* **SSIM：** 0.9108
* **PSNR：** 34.73
* **LPIPS：** 0.2946

不过这里必须保持严谨。
`A_set` 这轮实验虽然有总结结果，但我目前并没有看到特别完整的原始训练日志和命令证据链。

所以它更适合作为：
**“我已经在自己的数据上做过第一轮有效 baseline”的参考**
而不适合作为当前阶段最硬的核心证据。

这也是我这段时间一个越来越明确的认识：
**科研记录里，能复述一个结果，不等于你真的拥有一条强证据链。**

---

## 6. 真正的分水岭：B_set 直接训练失败

后面我重拍了视频，并更新脚本，开始推进 `B_set` 这条线。
新的重拍视频经过 `process_videov2.py` 处理后，成功生成了：

* `B_set/images` 下的 80 张图
* 模糊度分析结果
* 帧筛选报告

这说明，我已经不是停留在“抽帧试试看”，而是在构建正式的数据集。
接着，我按比较标准的 COLMAP 流程推进：

```bash
colmap feature_extractor \
    --database_path $PROJECT/B_set/database.db \
    --image_path $PROJECT/B_set/images \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1

colmap exhaustive_matcher \
    --database_path $PROJECT/B_set/database.db \
    --SiftMatching.use_gpu 1

mkdir -p $PROJECT/B_set/sparse

colmap mapper \
    --database_path $PROJECT/B_set/database.db \
    --image_path $PROJECT/B_set/images \
    --output_path $PROJECT/B_set/sparse
```

然后我直接把 `B_set` 拿去训练：

```bash
python train.py \
    -s $PROJECT/B_set \
    -m $PROJECT/runs/run_02_reshoot_baseline \
    --eval \
    --iterations 3000
```

结果一上来就报错了：

> `AssertionError: Colmap camera model not handled:`
> `only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!`

### 这次报错为什么这么关键
因为它第一次让我非常明确地意识到：
**Gaussian Splatting 不是“只要有 COLMAP 结果就能训”。**

官方 loader 对输入是有严格约束的。它只接受：
* `PINHOLE`
* `SIMPLE_PINHOLE`

而我当时的 `B_set`，本质上还是原始 COLMAP 工程，包含畸变模型，不能直接作为官方 3DGS 输入。
这件事非常重要。

因为它说明当前主问题不是训练、不是真值、不是什么 loss，而是：
**我的数据目录与官方输入契约不兼容。**

很多时候，实验做不下去，并不是因为方法太差，而是因为你根本没有真正站在“合法输入”的起点上。

---

## 7. 第一次补救：做 undistort，确实训起来了，但问题并没有真的解决

既然原始 `B_set` 不能直接喂给 3DGS，我就自然想到一条补救路径：
**先 undistort，再构建 GS-ready 数据集。**

于是我开始做 `B_set_gs`，后面又进一步诊断出：
* `sparse/0` 质量很差
* `sparse/1` 才是 80/80 注册的健康主模型

这时我才明白，COLMAP 输出的多个 sparse 模型也不能想当然地默认 `0` 就是好的。
必须检查，必须判断，不能偷懒。

后来我基于 `sparse/1` 派生出了新的目录：`B_set_gs_s1`，并在它上面跑出了一轮真正完成的训练结果：

* **run：** `run_02_reshoot_baseline_s1`
* **SSIM：** 0.8954509
* **PSNR：** 26.3039284
* **LPIPS：** 0.3262449

而且这一次，日志和输出证据是完整的：
`train.log`、`render.log`、`metrics.log`、`results.json`、`per_view.json`、`cfg_args`

从“这是不是一次真实有效的实验”来看，它当然算。
但问题在于，它并没有真正变成我想要的主线。

### 为什么它不该成为主线
因为这条线虽然跑通了，图像语义却被裁窄了。
简单说就是：
* 训练有效
* 结果也不是空的
* **但完整脸 / 完整头部信息被破坏了**

这意味着它更像一个“中间过渡成果”，而不是“符合任务定义的最终数据线”。

我后来越来越强烈地意识到一件事：
**不是所有“能跑”的实验，都应该被叫作成功。**

如果你的目标是“完整头部静态重建”，
那一个只剩局部 patch、只剩部分正脸区域的结果，即使指标还可以，也不能算真正完成任务。

---

## 8. 真正的纠偏：从原始图像重新构建 PINHOLE clean 数据集

在经历过 `B_set_gs_s1` 这条“能跑但不该当主线”的路线后，我开始转向更彻底的办法：
**不再把 undistort 产物当主线，而是从 `B_set/images` 重新构建一个原生 PINHOLE 的 clean 数据集。**

核心命令如下：

```bash
export CLEAN=$PROJECT/B_set_pinhole_clean

rm -rf $CLEAN
mkdir -p $CLEAN/images
cp $PROJECT/B_set/images/*.jpg $CLEAN/images/

colmap feature_extractor \
  --database_path $CLEAN/database.db \
  --image_path $CLEAN/images \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model PINHOLE \
  --SiftExtraction.use_gpu 1

colmap sequential_matcher \
  --database_path $CLEAN/database.db \
  --SiftMatching.use_gpu 1 \
  --SequentialMatching.overlap 10

mkdir -p $CLEAN/sparse

colmap mapper \
  --database_path $CLEAN/database.db \
  --image_path $CLEAN/images \
  --output_path $CLEAN/sparse
```

这一步在我看来是整个项目目前最关键的一次方向收敛。
因为我不再试图“修补一个不太对的输入”，而是直接重建一条更干净的输入链。

这条 clean rebuild 的统计结果也相对健康：
* 80/80 图像注册
* 5718 个 points
* reprojection error：1.268118 px

它至少说明，这不再是一个明显有结构性问题的 COLMAP 工程。

---

## 9. run_05_bset_pinhole_smoke：第一次真正回到“完整头肩语义”

在 `B_set_pinhole_clean` 上，我做了一次 smoke run：

```bash
python train.py \
  -s $CLEAN \
  -m $RUN \
  --eval \
  --iterations 3000 \
  2>&1 | tee $RUN/train.log

python render.py -m $RUN 2>&1 | tee $RUN/render.log
python metrics.py -m $RUN 2>&1 | tee $RUN/metrics.log
```

最终得到的指标是：

* **SSIM：** 0.8784330
* **PSNR：** 26.8530388
* **LPIPS：** 0.3725319

如果只看数值，这组结果未必会让人一眼觉得“比前面更厉害”。
但对我来说，这一轮实验真正重要的地方不在于指标，而在于：
**它让我的任务语义重新回到了“完整头肩重建”。**

这点非常关键。
因为从这一步开始，我面对的失败不再是：
* 环境不对
* 输入格式不对
* 相机模型不兼容
* 图像裁得乱七八糟

而变成了更像“真正研究问题”的失败：
* 侧脸区域恢复差
* 耳后区域弱约束
* 发际线不稳定
* 大角度转头区域质量脆弱

这两类失败完全不是一个层级。
前者是工程链路错了。
**后者才是你真正开始碰到任务本身的难点。**

所以我现在会认为：
`run_05_bset_pinhole_smoke` 是我目前最接近论文主线的一次有效实验。

---

## 10. 到现在为止，我完成的最重要的事，不是“跑通”，而是“收敛主线”

如果要总结目前进展，我觉得最重要的不是“我已经跑了几轮训练”，而是：
**我已经逐渐知道，哪些线不能当主线，哪条线才值得继续押注。**

### 已完成的部分
**1）官方 baseline 已完整跑通**
这部分证明环境没问题，给整个项目提供了地基。

**2）自有视频数据预处理已经走了两轮**
* `head_case`：早期试验线
* `B_set`：正式重拍线

**3）COLMAP 主模型判断已经做出来了**
我现在已经明确知道：
* 不能默认 `sparse/0` 就是对的
* `sparse/1` 可能才是真正健康的主模型
* 原生 PINHOLE clean rebuild 比 undistort 中间线更可信

**4）至少两条训练线真实跑完**
* `run_02_reshoot_baseline_s1`
* `run_05_bset_pinhole_smoke`

这意味着项目已经不再停留在“想法”和“脚本层面”，而是真正形成了可以比较的实验分支。

---

## 11. 还没做完的，不只是训练，而是“把科研结构搭起来”

我现在缺的，已经不只是多跑一轮实验，而是把实验真正组织成一套可以写进论文、写进汇报、写进技术博客的方法论。

目前还差的几件事大概是：

**1）A_set 证据补档**
它现在有结论，但日志链不够硬。

**2）run_05 还是 smoke run**
我已经开始往 full run 推进，比如：

```bash
python train.py \
  -s $DATA \
  -m $RUN \
  --eval \
  --iterations 30000
```
也就是 `run_06_bset_pinhole_full30k` 这种更正式的 baseline run。

**3）缺标准化 failure map**
比如固定保留：
* best view
* worst view
* novel view 最差案例
* 侧脸失败帧
* 耳后失败帧
* 发际线失败帧

**4）缺最小改动对照实验**
后面如果继续改，就不能再多个变量一起动了。
必须一次只改一个，比如：
* 去掉最差帧
* 补一点侧脸视角
* 统一 crop
* 提高训练迭代

只有这样，结果才真正可解释。

---

## 12. 这段过程带给我的最大认识

做到现在，我觉得这个项目最有价值的地方，不是我已经有多强的结果，而是我对这个题目开始有了更清醒的判断。

**第一，官方 baseline 只能证明环境，不代表任务成立**
这是很多人刚开始容易混淆的地方。
demo 能跑，只能说明工具没坏，不等于你的问题已经被解决。

**第二，能训练成功，不等于实验真的成功**
`B_set_gs_s1` 就是一个典型例子。
它日志完整、训练有效，但任务语义已经偏了，所以不能直接当主线。

**第三，真正值得继续做的，是一条可信主线，而不是无穷扩散分支**
现在对我来说，最务实的推进方式已经很清楚了：
**固定 `B_set_pinhole_clean`，把它打磨成当前阶段的唯一主线。**

然后在这条线上做：
* full baseline run
* 失败模式分析
* 最小变量对照

这比到处开新分支更重要。

---

## 结尾

如果让我用一句话来形容我现在这个项目所处的状态，那就是：
**我已经走出了环境踩坑期，进入了主线收敛期，但还没进入最终定型期。**

这不是一个“我已经成功了”的阶段，
但也绝对不是“我什么都没做出来”的阶段。

更准确地说，我现在已经知道：
* 什么输入线是错的
* 什么结果虽然能跑，但不该算主结论
* 什么数据集才最值得继续打磨

而对一个科研题目来说，
**明确主线，有时比多跑十轮实验更重要。**

至少到目前为止，我已经不再是在盲目试错，而是在逐步逼近一个真正可分析、可比较、可继续推进的单目头部静态重建主线。