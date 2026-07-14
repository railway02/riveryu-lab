---
title: "AIGC 视频攻防研究：为什么我们必须重新理解“视频”"
date: 2026-05-18T10:00:00+08:00
draft: false
tags: ["AIGC", "Think","CV"]
summary: "关于方向的思考和整理"
uid: aigc-video-rethinking
topics:
  - aigc-authenticity
  - research-engineering
relations:
  - target: latent-tokenizer-aigc
    type: related
    note: "本文提出在真实传播条件下寻找稳定的时序表征与编解码信号，目标文章进一步在表征空间和视频编解码潜空间中分析这些数字指纹。"
---
## 摘要

如果要做 AIGC 视频检测、Deepfake 检测，甚至进一步做攻防研究，不能把问题简单理解成“训练一个真假分类器”。视频不是一张张图片的堆叠，而是由**物理世界演化、相机成像、编码压缩、平台传播、后处理机制**共同形成的复杂信号系统。

AIGC 视频的“假”，也不一定只藏在某一帧的纹理里。更关键的痕迹可能出现在：时间连续性、二阶运动、光照一致性、几何关系、音画同步、补帧痕迹、压缩鲁棒性、跨生成器泛化等层面。

因此，我现在对这个方向的理解是：

> AIGC 视频攻防，不是单纯做攻击，也不是单纯做防御，而是研究：
> **在真实传播、后处理、补帧、压缩、未知生成器等条件下，视频取证信号是否仍然稳定存在，以及如何构建更鲁棒的检测系统。**

---

## 一、视频不是图片序列，而是一个完整系统

很多初学者会把视频理解成：

```text
video = frame_1 + frame_2 + frame_3 + ... + frame_T
```

这个理解太浅。对 AIGC 视频取证来说，更准确的理解应该是：

```text
视频 = 物理世界/生成模型
    + 相机成像/生成渲染
    + 时间采样
    + 编码压缩
    + 平台传播
    + 后处理
    + 解码显示
```

真实视频通常来自物理世界：

```text
真实世界运动 → 相机采样 → 编码压缩 → 平台传播 → 用户看到的视频
```

AIGC 视频则来自生成模型：

```text
文本/图像/视频条件 → 生成模型 → 后处理/补帧/超分 → 编码压缩 → 平台传播
```

这两个过程的源头完全不同。真实视频是现实世界连续演化的观测结果，AIGC 视频是模型根据数据分布生成出来的视觉序列。因此，检测 AIGC 视频不能只看“单帧像不像”，还要看它是否符合真实视频的时序、物理、编码和传播规律。

GenVideo/DeMamba 这类新工作已经把 AIGC 视频检测从简单二分类推进到更现实的评测：不仅要测同分布检测，还要测**跨生成器泛化**和**退化视频鲁棒性**。其 GenVideo benchmark 包含百万级真实和 AI 生成视频，并设计了 cross-generator video classification 与 degraded video classification 两类任务。([arXiv][1])

---

## 二、做 AIGC 视频攻防，首先要知道视频由什么组成

一个视频文件至少包含这些要素：

```text
1. 容器格式：mp4 / mov / webm / mkv
2. 视频编码：H.264 / H.265 / AV1 / VP9
3. 帧序列：frame_1, frame_2, ..., frame_T
4. 帧率：fps，例如 24 / 30 / 60
5. 分辨率：720p / 1080p / 4K
6. 颜色空间：RGB / YUV
7. 码率：bitrate
8. GOP 结构：I / P / B 帧
9. 音频轨道
10. 元数据
```

这里最容易混淆的是**容器**和**编码**。

比如 `.mp4` 是容器，类似“包装盒”；H.264/H.265 是编码方式，类似“压缩算法”。一个 mp4 文件里面可以装 H.264 编码的视频，也可以装 H.265 编码的视频。

现代视频编码还会利用帧间冗余。I 帧通常是相对完整的图像，P 帧会参考过去帧，B 帧会参考前后帧。也就是说，压缩视频并不是每一帧都独立保存，而是通过运动估计、预测、残差和量化来压缩。这个机制对取证非常重要，因为压缩本身会改变视频的时空统计特征。

对科研实验来说，必须记录这些元信息：

```text
codec
container
fps
resolution
bitrate
pix_fmt
duration
audio/no-audio
color space
degradation type
generator/source
```

否则模型很可能学到的不是 AIGC 痕迹，而是数据集偏差。例如：真实视频都来自某个平台，AI 视频都来自另一个模型；真实视频压缩更重，AI 视频更干净；真实视频有水印，AI 视频没有。这样的检测器看似准确率高，其实是在学 shortcut。

---

## 三、压缩与转码：AIGC 视频检测绕不开的现实问题

视频进入真实传播环境后，几乎一定会被压缩、转码、缩放、裁剪、加字幕或加水印。

常见退化包括：

```text
H.264/H.265 压缩
降低分辨率
改变 fps
裁剪 crop
重新编码 re-encode
加字幕/水印
模糊 blur
降噪 denoise
锐化 sharpen
颜色变化
录屏再上传
```

很多 Deepfake/AIGC 检测器在 clean 数据上表现很好，但一旦视频经过压缩，性能会明显下降。FaceForensics++ 早期就把不同压缩质量纳入 benchmark，说明压缩会影响人脸篡改检测结果。FaceForensics++ 本身不是攻防论文，也不是强方法论文，但它对 deepfake detection 的重要意义在于：它把任务定义、伪造类型、benchmark 和压缩评测标准化了。([arXiv][2])

后来的鲁棒检测研究也在反复强调同一个问题：真实世界退化会破坏很多低层伪影，尤其是高频纹理、边缘痕迹、频域异常、局部融合瑕疵。NTIRE 2026 Robust Deepfake Detection 相关工作也指出，现实中的模糊、压缩、缩放等复合退化会让 clean 数据集上很强的检测器出现显著性能下降。([arXiv][3])

所以，真正有价值的检测研究不应该只报告：

```text
clean accuracy = 98%
```

而应该报告：

```text
clean AUC
compressed AUC
resized AUC
cropped AUC
re-encoded AUC
cross-generator AUC
robustness drop
```

也就是：

```text
Robust Drop = AUC_clean - AUC_degraded
```

这比单纯刷准确率更接近真实科研问题。

---

## 四、补帧机制：AIGC 视频攻防中非常容易被忽略的一环

我现在越来越觉得，补帧是 AIGC 视频攻防中必须理解的机制。

补帧，或者说 Video Frame Interpolation，目标是在已有帧之间合成中间帧，提高视频帧率或流畅度。VFI 领域已经从传统运动补偿发展到 kernel-based、flow-based、hybrid、Transformer、Mamba、GAN、diffusion 等多种方法。AceVFI 综述把 VFI 描述为在现有帧之间合成中间帧，同时维持空间和时间一致性的低层视觉任务。([arXiv][4])

补帧为什么对检测重要？

因为它会改变时序信号。

### 1. 重复帧

最简单的补帧方式是重复：

```text
A B C
→ A A B B C C
```

这种方式会让运动看起来卡顿，帧间差分变小，可能干扰时序检测器。

### 2. 混合帧

混合补帧会把相邻帧做平均：

```text
middle = 0.5 * frame_t + 0.5 * frame_{t+1}
```

它可能带来重影、模糊、边缘残影。

### 3. 运动补偿插帧

运动补偿插帧会估计物体从前一帧到后一帧怎么运动，然后生成中间位置的帧。这和 optical flow 有关。

这类方法可能让视频看起来更流畅，也可能在遮挡边界、快速运动、复杂纹理处产生错误。

### 4. AI 补帧

现代深度学习补帧可能进一步修复视觉质量，但这也带来取证问题：

```text
它可能抹平 AIGC 原本的闪烁；
也可能引入新的补帧伪影；
检测器可能误把“补帧痕迹”当作“AIGC 痕迹”。
```

所以在 AIGC 视频攻防实验里，补帧不应该只是普通数据增强，而应该作为一个独立实验因素：

```text
no interpolation
frame duplication
frame blending
motion-compensated interpolation
learning-based VFI
```

我们要问：

```text
检测器检测的是 AIGC，还是检测是否经过补帧？
补帧会不会降低二阶运动异常？
补帧会不会制造新的伪影？
```

这正是“视频机制”进入 AIGC 攻防研究的地方。

---

## 五、物理一致性：AI 视频的高阶检测线索

真实视频来自现实世界，因此受到物理约束。

这些约束包括：

```text
运动连续性
速度与加速度
光照与阴影
几何投影
遮挡关系
物体永久性
接触与重力
人体结构
音画同步
相机成像规律
```

AIGC 视频可能单帧很逼真，但在这些规律上仍然会出问题。

### 1. 运动连续性

真实世界中，物体通常不会无缘无故瞬移。位置、速度、加速度具有一定连续性。

AI 视频中常见问题包括：

```text
头发细节漂移
手指形状跳变
衣服纹理闪烁
背景纹理不稳定
物体运动突然改变
人物身份轻微漂移
```

这就是为什么现在越来越多研究关注时间特征，而不是只看单帧图像。

D3 提出从二阶动态特征检测 AI 生成视频。它认为现有检测方法对 temporal artifacts 探索不足，因此从牛顿力学和二阶动态分析角度出发，用 second-order temporal discrepancies 检测 AI 视频，并在 GenVideo、VideoPhy、EvalCrafter、VidProM 等数据集上验证。([arXiv][5])

最简单的形式可以写成：

[
d_t = |f_t - f_{t-1}|_2
]

[
a_t = |f_{t+1} - 2f_t + f_{t-1}|_2
]

其中 (f_t) 可以是第 (t) 帧经过 CLIP、DINOv2 或视频编码器得到的特征。

(d_t) 描述一阶变化，(a_t) 描述二阶变化。直觉上，一阶变化看“动了多少”，二阶变化看“运动变化是否自然”。

### 2. 表征轨迹曲率

除了差分，还可以看帧特征在表征空间里的轨迹。

如果每一帧都映射成一个 embedding：

```text
f_1, f_2, ..., f_T
```

那么视频就变成了一个高维空间中的轨迹。真实视频的轨迹通常应具有较自然的连续性，而 AI 视频可能出现异常弯折、跳变或过度平滑。

ReStraV 这类工作就基于 perceptual straightening 思想，用 DINOv2 等视觉表征分析视频帧轨迹的距离和曲率，从而检测 AI 生成视频。

这里我更建议将其作为方法启发：

```text
视频检测不一定必须训练大模型；
也可以先用强视觉表征，再分析帧间轨迹的几何结构。
```

### 3. 光照与几何

人脸 Deepfake 特别容易出现局部光照和几何问题：

```text
脸部光照和脖子不一致
眼睛反光不自然
阴影方向不稳定
转头时五官位置变化不符合 3D 几何
脸像贴在头上一样
```

如果结合人脸关键点、头姿估计和相机模型，可以进一步研究：

```text
头部姿态是否连续？
人脸重投影误差是否异常？
脸部局部区域亮度变化是否和整体运动一致？
```

这和 3DGS、COLMAP、头部重建方向是可以连接起来的。

### 4. 音画同步

Deepfake 和 talking-head 视频里，音画同步是非常重要的线索。

真实说话视频中：

```text
嘴型
下巴运动
发音节奏
脸部肌肉变化
音频节奏
```

应该基本同步。

如果嘴型和声音错位，就可能是伪造或后处理不一致。SyncNet 系列工作就把音频和嘴部视频映射到共享空间，用跨模态同步来判断音画关系。

对 talking-head deepfake，可以记录：

```text
最佳音画偏移
峰值同步置信度
嘴部运动强度与音频能量关系
```

这比只看图像纹理更接近“视频理解”。

---

## 六、AIGC 视频生成模型为什么容易留下痕迹

现代视频生成模型已经很强，但它们仍然面临几个根本难题：

```text
时间一致性
长程记忆
物体永久性
运动自然性
遮挡关系
物理交互
复杂光照
镜头运动
多对象一致性
```

很多视频生成模型可以生成很漂亮的单帧，但视频不是单帧质量竞赛。真实视频要求所有帧在时间上共同成立。

AIGC 视频的典型问题包括：

```text
前几帧人物像 A，后几帧身份漂成 B
背景纹理慢慢变形
物体遮挡后再出现时形状改变
手指数量或结构变化
头发、衣服、边缘区域闪烁
物理接触不合理
影子和光照不一致
```

所以，AIGC 视频检测未来不应该只停留在：

```text
CNN / ViT 判断某帧真假
```

而应该走向：

```text
时序表征
二阶运动
物理一致性
跨模态一致性
鲁棒评测
可解释取证
```

---

## 七、攻防到底怎么理解：防是目标，攻是压力测试

我现在对“攻防”的理解是：

> **防是研究身份，攻是评测工具。**

不应该把研究方向表述成“如何绕过检测器”，而应该表述成：

```text
我们系统研究现有检测器在真实传播退化、补帧、压缩、转码、未知生成器等压力测试下的脆弱性，并提出更鲁棒的检测方法。
```

这叫防御性红队评测。

合理的“攻”包括：

```text
压缩
转码
降分辨率
裁剪
改 fps
补帧
加字幕/水印
模糊
噪声
颜色变化
平台传播模拟
```

不合理的方向是把研究写成“如何逃避检测”。那样不仅伦理上敏感，而且科研叙事也不够建设性。

好的攻防论文结构应该是：

```text
1. 现有检测器在 clean 数据上表现很好；
2. 但在真实传播退化下明显失效；
3. 分析原因：它依赖脆弱的低层伪影；
4. 提出更鲁棒的时序/物理/表征方法；
5. 用相同压力测试证明新方法更稳。
```

也就是说：

```text
发现脆弱性 → 解释脆弱性 → 提出防御 → 再评测
```

这比“我加了一个模块，提高 2% accuracy”更像科研。

---

## 八、一个适合学生团队的研究路线

我认为比较适合我们的路线不是直接训练超大视频模型，而是先做一个轻量、可解释、能形成实验闭环的系统。

### 1. 基础 baseline

```text
视频抽帧
→ CLIP/DINOv2 提取每帧特征
→ 每帧真假分数
→ mean / max / top-k 聚合
→ 视频真假分数
```

这是最基本的 frame-level baseline。

### 2. 时序特征

在帧特征基础上计算：

```text
一阶差分 d_t
二阶差分 a_t
轨迹曲率 κ_t
分布统计：mean / std / max / top-k / percentile
```

视频级特征可以由这些统计量组成，再训练轻量分类器。

### 3. 光流与运动

进一步可以加入：

```text
光流幅值
光流方向变化
光流二阶差分
遮挡边界异常
运动区域稳定性
```

这部分能让方法更接近“运动物理”。

### 4. 人脸/人体场景

如果聚焦 Deepfake 或人物视频，可以加入：

```text
人脸关键点
头姿估计
身份 embedding 稳定性
嘴型-音频同步
脸部局部亮度变化
```

### 5. 鲁棒评测

必须系统测试：

```text
clean
H.264 CRF 18 / 23 / 28 / 35
H.265
resize 720p / 480p / 360p
crop 90% / 80%
fps 30 → 15
VFI 30 → 60
blur
noise
watermark
compound degradation
```

评价指标包括：

```text
AUC
AP
F1
EER
Robust Drop
Cross-generator AUC
Calibration / ECE
Frame-level anomaly localization
```

---

## 九、最小可行实验设计

一个最小但完整的实验闭环可以这样设计。

### Step 1：数据准备

```text
real videos
AI-generated videos
deepfake/talking-head videos
metadata.csv
```

metadata 至少包括：

```text
video_id
label
source/generator
duration
fps
resolution
codec
degradation_type
```

### Step 2：生成退化版本

用 FFmpeg 构造：

```text
原始版本
压缩版本
降分辨率版本
裁剪版本
改 fps 版本
补帧版本
组合退化版本
```

### Step 3：抽帧与特征

每个视频采样 8/16/32 帧，提取 CLIP 或 DINOv2 特征。

### Step 4：计算时序统计

```text
d_t = ||f_t - f_{t-1}||
a_t = ||f_{t+1} - 2f_t + f_{t-1}||
κ_t = trajectory curvature
```

统计：

```text
mean
std
max
95 percentile
top-k mean
```

### Step 5：训练轻量检测器

先不要上复杂大模型，可以从：

```text
logistic regression
MLP
XGBoost
linear classifier
```

开始。

### Step 6：评估鲁棒性

分别报告：

```text
clean
compressed
resized
cropped
fps changed
interpolated
cross-generator
```

画出：

```text
AUC vs CRF
AUC vs resolution
AUC vs fps
real/fake temporal statistics distribution
frame-level anomaly curve
```

这些图比单纯一张准确率表更有说服力。

---

## 十、这个方向真正的科研问题

我认为最值得抓住的问题不是：

```text
能不能训练一个 AIGC 视频检测器？
```

而是：

```text
AIGC 视频经过真实传播和后处理之后，还有哪些稳定、可解释、跨生成器泛化的取证信号？
```

这句话才是方向核心。

它可以拆成几个子问题：

### 1. 检测器是否学到了数据集偏差？

很多检测器可能只是在识别：

```text
分辨率
压缩方式
水印
画幅
生成器风格
数据来源
```

而不是真正识别 AI 生成痕迹。

### 2. 时序特征是否比单帧纹理更鲁棒？

压缩会破坏高频纹理，但某些时序动态、表征轨迹、物理一致性 proxy 可能更稳定。

### 3. 补帧会削弱还是增强检测信号？

补帧可能抹平 AI 视频的时间不稳定，也可能引入补帧伪影。这个问题本身就有研究价值。

### 4. 物理一致性如何被量化？

真实视频满足运动、光照、几何、接触、音画同步等规律，但这些规律很难直接精确建模。因此更现实的做法是使用 physical-consistency proxy，而不是宣称“证明物理定律”。

### 5. 如何构建真实传播场景下的鲁棒检测？

最终目标不是 clean accuracy，而是现实场景可靠性。

---

## 十一、推荐学习路线

为了支撑这个方向，我认为需要补四类知识。

### 1. 视频机制

必须懂：

```text
帧
fps
分辨率
容器
编码
码率
CRF
I/P/B 帧
GOP
颜色空间
音频同步
```

### 2. 视频处理工具

必须会：

```text
ffprobe
ffmpeg
OpenCV
抽帧
转码
压缩
降分辨率
改 fps
补帧
```

### 3. 物理与视觉

重点懂：

```text
运动连续性
速度/加速度
光流
相机投影
几何一致性
光照/阴影
接触/重力
音画同步
```

### 4. 检测与攻防

需要读：

```text
FaceForensics++
Celeb-DF
DFDC
GenVideo/DeMamba
D3
ReStraV
VFI survey
Robust Deepfake Detection challenge
```

FaceForensics++ 是 deepfake detection 的地基，它标准化了人脸篡改检测 benchmark；Celeb-DF 进一步强调高质量 deepfake 比早期数据集更难；GenVideo/DeMamba 把问题推进到 AIGC 视频、跨生成器和退化鲁棒；D3 和 ReStraV 则分别代表二阶时间特征和表征轨迹几何这两条更可解释的新路线。([arXiv][2])

---

## 十二、我的阶段性判断

AIGC 视频检测和 Deepfake 检测正在从“单帧伪影分类”走向“视频系统级取证”。

早期路线关注：

```text
脸部边界
纹理异常
频域伪影
单帧分类
```

现在更值得关注的是：

```text
时序一致性
二阶运动
物理现实一致性
跨模态同步
压缩鲁棒性
补帧影响
跨生成器泛化
检测器校准
可解释定位
```

如果只是做一个普通真假分类器，这个方向很容易变成红海。

但如果把问题定义为：

> **真实传播与后处理条件下的 AIGC 视频鲁棒取证**

它仍然有很大空间。

我认为比较好的研究定位是：

```text
防为主线，攻为评测，鲁棒性为创新。
```

也就是：

```text
不是研究怎么绕过检测器，
而是研究检测器在真实世界压力测试下为什么失效，以及如何让它更可靠。
```

---

## 结语：从“分类器思维”转向“视频系统思维”

这段时间我最大的认知变化是：AIGC 视频攻防不是单纯的深度学习分类问题，而是一个横跨视频机制、物理一致性、生成模型、后处理链条和鲁棒评测的系统性问题。

视频不是简单的帧序列。
AIGC 视频也不是简单的假图像集合。
检测器不能只看 clean accuracy。
攻防不能只理解成攻击和防御二选一。

真正值得做的是：

> 理解真实视频如何形成，理解 AIGC 视频如何生成，理解压缩、补帧、转码如何改变取证信号，然后构建在真实传播场景下仍然可靠、可解释、可泛化的检测系统。

这也是我后续做 AIGC/Deepfake 视频攻防研究最应该坚持的主线。

[1]: https://arxiv.org/abs/2405.19707?utm_source=chatgpt.com "DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark"
[2]: https://arxiv.org/abs/1901.08971?utm_source=chatgpt.com "FaceForensics++: Learning to Detect Manipulated Facial Images"
[3]: https://arxiv.org/abs/2604.25889?utm_source=chatgpt.com "Robust Deepfake Detection: Mitigating Spatial Attention Drift via Calibrated Complementary Ensembles"
[4]: https://arxiv.org/abs/2506.01061?utm_source=chatgpt.com "AceVFI: A Comprehensive Survey of Advances in Video Frame Interpolation"
[5]: https://arxiv.org/abs/2508.00701?utm_source=chatgpt.com "D3: Training-Free AI-Generated Video Detection Using Second-Order Features"
