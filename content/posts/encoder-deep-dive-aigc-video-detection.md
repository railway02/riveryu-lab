---
title: "编码器 Encoder 深度笔记：从特征表示到 AIGC 视频检测"
date: 2026-06-03T10:00:00+08:00
draft: false
tags: ["Deep Learning", "Computer Vision", "AIGC Detection", "Mathematics", "Research"]
summary: "探讨 Encoder 如何从像素层抽象出证据空间，并结合 GRU 与 Normalization 构建鲁棒的视频检测框架。"
---
## 一、核心观点

编码器 Encoder 的本质是：

$$z = E_\theta(x)$$

它把原始输入 (x) 映射成特征表示 (z)。

对于图像：

$$
x \in \mathbb{R}^{3 \times H \times W}
$$

表示一张 RGB 图像。

经过编码器后：

$$
z \in \mathbb{R}^{d}
$$

表示一个 (d) 维特征向量。

最直接的理解：

> 编码器把原始像素、视频帧、文本或音频，转换成模型更容易计算、比较、预测和判断的表示。

更深一层：

> 编码器在学习一个新的特征空间。原始数据在这个空间里会更有结构，更适合下游任务。

对于 AIGC 视频检测，编码器决定检测器能观察到什么信号。检测器后面的 GRU、Transformer、残差模块、分类器都依赖编码器提供的特征。

---

## 二、为什么需要编码器？

一张 (224×224) 的 RGB 图像有：


3×224×224=150528


个像素值。

这些像素包含很多底层信息：

```text
颜色
亮度
边缘
纹理
噪声
压缩痕迹
局部高频细节
```

但模型进行判断时，通常需要更抽象的信息：

```text
这是不是人脸
身份是否稳定
头发边缘是否自然
嘴部运动是否连续
背景纹理是否漂移
频域结构是否异常
语义关系是否合理
```

编码器的作用，就是把底层像素逐渐组织成更高级的表示。

可以理解成一条抽象链：

```text
像素
↓
边缘 / 纹理 / 颜色
↓
局部结构 / 物体部件
↓
身份 / 姿态 / 语义
↓
任务相关特征
```

数学上：

$$
x \rightarrow h_1 \rightarrow h_2 \rightarrow h_3 \rightarrow z
$$

其中：

* (h_1)：偏低层特征，例如边缘、颜色、纹理；
* (h_2)：偏中层特征，例如眼睛、嘴巴、头发、局部结构；
* (h_3)：偏高层特征，例如身份、场景、语义；
* (z)：最终特征表示。

---

## 三、编码器真正关键的问题：保留与丢弃

编码器会把高维输入压到较低维的特征空间。

例如：

$$
x \in \mathbb{R}^{150528}
$$

$$
z \in \mathbb{R}^{512}
$$

这意味着编码器必须做信息选择。

它会保留一部分信息，也会压缩、弱化甚至丢掉一部分信息。

对于普通图像分类任务，编码器更关注：

```text
物体类别
形状
纹理
整体语义
```

对于 AIGC 视频检测，编码器需要关注：

```text
身份漂移
时序闪烁
局部纹理跳变
高频异常
压缩伪影
生成器指纹
物理运动不连续
语义与运动不一致
```

这带来一个重要判断：

> 一个 ImageNet 上很强的编码器，未必天然适合 AIGC 检测。因为它可能弱化了生成检测需要的细微信号。

所以在 AIGC 视频检测里，编码器要满足一个更特殊的目标：

> 对良性变化保持稳定，对生成异常保持敏感。

英文：

```text
Robust to benign transformations, sensitive to generative artifacts.
```

中文：

```text
对压缩、缩放、轻微噪声等良性传输变化保持稳定；对身份漂移、纹理跳变、运动异常等生成痕迹保持敏感。
```

---

## 四、编码器定义了“证据空间”

不同编码器会把图像投影到不同的空间。

### 1. ResNet / CNN 编码器

ResNet、EfficientNet、ConvNeXt 这类 CNN 编码器擅长提取：

```text
边缘
纹理
局部结构
形状
物体类别相关特征
```

优点：

```text
实现简单
训练稳定
适合做 baseline
局部纹理能力较强
```

局限：

```text
可能对极细微频域伪影不够敏感
可能更偏分类语义
跨生成器泛化需要验证
```



---

### 2. CLIP 编码器

CLIP 图像编码器更偏语义空间。

它擅长：

```text
图像语义
图文一致性
场景理解
物体关系
高层概念
```

适合检测：

```text
语义不一致
文本-视频不匹配
动作和场景关系异常
常识错误
```

在的多证据系统里，CLIP 更适合作为语义编码器：

$$
E_{sem}(x_t)
$$

---

### 3. DINO / DINOv2 编码器

DINO / DINOv2 是自监督视觉编码器。

它更关注：

```text
视觉结构
局部区域
物体边界
区域一致性
视觉相似性
```

适合的方向：

```text
跨帧结构稳定性
人脸区域一致性
局部纹理状态
视频状态轨迹
```

DINOv2 很适合作为 AIGC 视频检测的强视觉表征编码器。

---

### 4. ArcFace / 人脸身份编码器

ArcFace 这类模型输出身份特征：

$$
f_t^{id} = E_{id}(x_t)
$$

它适合检测：

```text
身份漂移
脸部特征不稳定
换脸视频中的身份不一致
局部编辑后身份特征变化
```

对于 deepfake、人脸编辑、换脸视频，身份编码器非常重要。

---

### 5. 频域编码器

频域编码器关注：

```text
高频噪声
压缩痕迹
上采样模式
周期性纹理
生成器频率指纹
```

可以通过 FFT、DCT、小波变换或专门 CNN 提取频域特征：

$$
f_t^{freq} = E_{freq}(x_t)
$$

适合检测：

```text
生成器纹理统计异常
局部高频伪影
压缩后信号坍塌
频谱不自然
```

---

### 6. 运动编码器

运动编码器关注相邻帧之间的关系：

$$
f_t^{motion} = E_{motion}(x_{t-1}, x_t)
$$

常见输入包括：

```text
光流
帧差
Warp 残差
时序特征差
局部运动场
```

适合检测：

```text
嘴部运动不连续
头发闪烁
背景漂移
物体边缘抖动
物理运动不自然
```

---

## 五、AIGC 视频检测中的多编码器思想

单一编码器容易只看到某一种信号。

更稳的设计是多编码器：

[
f_t^{id} = E_{id}(x_t)
]

[
f_t^{motion} = E_{motion}(x_{t-1},x_t)
]

[
f_t^{freq} = E_{freq}(x_t)
]

[
f_t^{tex} = E_{tex}(x_t)
]

[
f_t^{sem} = E_{sem}(x_t)
]

分别对应：

```text
身份证据
运动证据
频域证据
纹理证据
语义证据
```

然后构造证据向量：

[
e_t = [e_t^{id}, e_t^{motion}, e_t^{freq}, e_t^{texture}, e_t^{semantic}]
]

再用 Softmax 做证据分配：

[
\alpha_t = Softmax(W e_t)
]

最终异常分数：

[
R_t = \sum_k \alpha_{t,k} e_{t,k}
]

含义是：

> 当前视频更应该相信哪种证据，由模型自己学习。

例如：

```text
换脸视频：身份异常权重更高
文生视频：运动和物理异常权重更高
局部编辑视频：纹理和闪烁异常权重更高
压缩转码视频：频域证据需要谨慎使用
```

这会让检测器更可解释。

---

## 六、编码器和 hidden state 的关系

在深度网络中，每一层都会产生一个内部表示。

这个内部表示就可以称为 hidden state。

对于 Transformer：

[
h_i^l
]

表示第 (l) 层、第 (i) 个 token 的 hidden state。

对于视频检测：

[
f_t = E(x_t)
]

可以看成第 (t) 帧的观测特征。

经过时序模型后：

[
s_t = F_\theta(f_1,\dots,f_t)
]

这里的 (s_t) 就是视频状态 hidden state。

直观理解：

```text
f_t：当前帧被编码后的观测
s_t：模型结合历史后得到的当前视频状态
```

所以编码器负责生成观测，时序模型负责维护状态。

---

## 七、Attention、MLP 与编码器

Transformer 编码器通常由 Attention 和 MLP 堆叠而成。

一层 Transformer 可以写成：

[
x' = x + Attention(Norm(x))
]

[
x_{next} = x' + MLP(Norm(x'))
]

其中：

### Attention 的作用

Attention 负责信息交互。

它让一个 token、patch 或帧特征去关注其他位置的信息。

在图像编码器里，Attention 可以让一个图像 patch 关注其他 patch。

在视频编码器里，Attention 可以让某一帧关注其他帧。

核心思想：

```text
哪些位置对当前判断最重要？
哪些帧最异常？
哪些区域应该被重点观察？
```

### MLP 的作用

MLP 负责对每个位置的特征做非线性加工。

Attention 更像信息路由，MLP 更像特征变换。

在编码器中：

```text
Attention：整合上下文
MLP：加工表示
Norm：稳定尺度
Residual：保留信息通路
```

---

## 八、Pre-Norm 对编码器和视频检测的启发

Pre-Norm 公式：

[
x_{l+1}=x_l+F(Norm(x_l))
]

它的核心思想是：

> 先把输入拉回稳定尺度，再进行复杂变换。

放到 AIGC 视频检测里：

每帧特征：

[
f_t = \Phi(x_t)
]

先归一化：

[
\tilde{f}_t = Norm(f_t)
]

再做时序残差：

[
r_t = |\tilde{f}*t - A*\theta(\tilde{f}_{t-1})|
]

这样可以降低以下因素带来的干扰：

```text
光照差异
分辨率差异
压缩率差异
人脸大小差异
背景复杂度差异
生成器风格差异
```

可以概括成一句方法原则：

```text
Normalize before temporal reasoning.
```

中文表达：

```text
先进行状态尺度归一化，再进行时序状态转移建模。
```

这很适合你的 SITF / Stable Reality State 框架。

---

## 九、编码器和 RNN / GRU 的关系

视频是时间序列。

逐帧编码后得到：

[
f_1, f_2, \dots, f_T
]

为了建模时间状态，可以使用 GRU：

[
s_t = GRU(f_t, s_{t-1})
]

其中：

```text
f_t：当前帧特征
s_{t-1}：过去状态
s_t：当前状态
```

GRU 的门控形式：

[
h_t = z_t \odot h_{t-1} + (1-z_t)\odot \tilde{h}_t
]

含义：

```text
z_t 大：更多保留旧状态
z_t 小：更多写入新状态
```

这对视频检测很重要。

因为 AIGC 视频异常常常体现在：

```text
长期身份状态不稳定
局部纹理状态漂移
运动状态无法自然继承
某几帧突然跳变
```

所以可以设计：

[
f_t = E(x_t)
]

[
s_t = GRU(f_t, s_{t-1})
]

[
r_t = |s_t - A_\theta(s_{t-1})|
]

其中 (r_t) 表示当前状态和预测状态之间的偏差。

这个偏差越大，说明视频状态演化越可疑。

---

## 十、编码器和 Diffusion 的关系

Diffusion 的生成过程可以理解成一个去噪动力系统。

前向加噪：

[
q(x_t|x_{t-1})=
\mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)
]

反向去噪：

[
p_\theta(x_{t-1}|x_t)
]

它的过程是：

```text
清晰图像
↓
逐步加噪
↓
纯噪声

纯噪声
↓
逐步去噪
↓
生成图像
```

对视频生成模型来说，每一帧或每一段视频都经历类似的生成动力过程。

如果这个动力过程在时间上不稳定，就可能出现：

```text
头发纹理闪烁
人脸身份漂移
背景局部跳变
物体边缘抖动
运动轨迹不自然
```

所以 AIGC 视频检测可以借鉴 Diffusion 的稳定性思想：

> 检测生成视频的状态轨迹是否平滑、连续、可预测。

编码器在这里承担观测函数：

[
s_t = E(x_t)
]

然后检测：

[
r_t = |s_t - A_\theta(s_{t-1})|
]

这可以理解成：

```text
当前帧状态是否可以由上一帧状态自然演化而来？
```

---

## 十一、编码器和 VAE 的关系

VAE 的核心结构是：

```text
x
↓
Encoder
↓
z
↓
Decoder
↓
x_hat
```

编码器输出潜变量分布：

[
q_\phi(z|x)
]

通常输出：

[
\mu(x), \log\sigma^2(x)
]

然后采样：

[
z = \mu + \sigma \epsilon
]

其中：

[
\epsilon \sim \mathcal{N}(0,I)
]

VAE 的目标：

[
\mathcal{L}
===========

## \mathbb{E}*{q*\phi(z|x)}[\log p_\theta(x|z)]

D_{KL}(q_\phi(z|x)|p(z))
]

第一项希望重建效果好。

第二项希望 latent 分布接近标准正态：

[
p(z)=\mathcal{N}(0,I)
]

VAE 对编码器的启发：

> 好的 latent 空间应该连续、规整、可采样，同时保留足够重建信息。

放到 AIGC 视频检测：

```text
真实视频的 latent 轨迹应该更连续
生成视频的 latent 轨迹可能存在跳变
局部编辑可能导致 latent 状态异常偏移
```

可以设计：

[
z_t = E(x_t)
]

[
r_t^{latent} = |z_t - A_\theta(z_{t-1})|
]

用来检测 latent 状态演化异常。

---

## 十二、编码器在 AIGC 视频检测中的统一框架

可以把整个检测框架写成：

[
o_t = \Phi(x_t)
]

[
\tilde{o}_t = Norm(o_t)
]

[
s_t = F_\theta(\tilde{o}_1,\dots,\tilde{o}_t)
]

[
\hat{s}*t = A*\theta(s_{t-1})
]

[
r_t = |s_t - \hat{s}_t|
]

[
p_{fake}=C(Pool({s_t,r_t}_{t=1}^{T}))
]

解释：

```text
x_t：第 t 帧
Φ：编码器
o_t：当前帧观测
Norm：状态尺度归一化
Fθ：状态估计器
Aθ：状态转移预测器
r_t：状态残差
C：分类器
p_fake：生成视频概率
```

这对应你的核心思路：

```text
视频检测不只看单帧真假，而要看视频状态是否稳定演化。
```

---

## 十三、编码器的最大风险：shortcut

AIGC 检测中，编码器可能学到数据集偏差。

例如：

```text
fake 视频分辨率固定
fake 视频压缩率更高
fake 视频带水印
real 和 fake 来自不同平台
real 和 fake 的背景分布不同
某个生成器有固定频域指纹
```

这些信号可能让训练集准确率很高，但跨生成器、跨平台、跨压缩条件下失效。

所以评估编码器时，不能只看 accuracy。

还要看：

```text
跨生成器泛化
压缩后是否稳定
resize 后是否稳定
转码后是否稳定
跨数据集是否有效
是否依赖水印或 codec
证据权重是否合理
```

这也正是 SCAR Protocol 的价值：

> 用受控变换测试检测信号是否稳定。

---

## 十四、如何判断一个编码器好不好？

对于 AIGC 视频检测，可以从五个维度评估。

### 1. 判别性

真实视频和生成视频在特征空间是否可分：

[
E(x_{real}) \neq E(x_{fake})
]

可以用 t-SNE / UMAP 可视化。

---

### 2. 稳定性

良性变换前后，特征是否稳定：

[
E(x) \approx E(T(x))
]

其中 (T) 可以是：

```text
压缩
resize
轻微加噪
转码
帧率变化
```

---

### 3. 敏感性

对生成异常是否敏感：

```text
身份漂移
纹理跳变
时序闪烁
频域异常
局部编辑痕迹
```

---

### 4. 可解释性

能否解释模型依赖的证据：

```text
身份
运动
频域
纹理
语义
物理一致性
```

---

### 5. 泛化性

是否能跨生成器有效：

```text
Sora
Runway
Pika
Kling
VACE
Stable Video Diffusion
MiniMax
Seedance
```

训练集上好，跨生成器差，说明编码器可能依赖 shortcut。

---

## 十五、你当前最务实的实现路线

### 第一阶段：跑通 baseline

结构：

```text
视频
↓
抽帧
↓
ResNet18 / ResNet50 编码器
↓
LayerNorm
↓
GRU
↓
Pooling
↓
real / fake 分类
```

公式：

[
f_t = E(x_t)
]

[
\tilde{f}_t = Norm(f_t)
]

[
s_t = GRU(\tilde{f}*t, s*{t-1})
]

[
p_{fake}=C(Pool(s_1,\dots,s_T))
]

目标：

```text
先跑通完整训练和测试流程
得到基础准确率
保存每帧特征和状态
画出状态轨迹
```

---

### 第二阶段：加入时序残差

结构：

```text
帧特征
↓
状态归一化
↓
上一帧预测当前帧
↓
计算残差
↓
残差池化
↓
分类
```

公式：

[
\hat{f}*t = A*\theta(\tilde{f}_{t-1})
]

[
r_t = |\tilde{f}_t-\hat{f}_t|
]

目标：

```text
检测视频状态是否连续
找到异常帧
输出残差曲线
```

---

### 第三阶段：多编码器证据

加入：

```text
身份编码器
DINO 结构编码器
CLIP 语义编码器
频域编码器
运动编码器
```

构造：

[
e_t = [e_t^{id}, e_t^{motion}, e_t^{freq}, e_t^{texture}, e_t^{semantic}]
]

用 Softmax 融合：

[
R_t = \sum_k \alpha_{t,k}e_{t,k}
]

目标：

```text
让检测器可解释
知道每种视频主要由哪类证据支撑判断
```

---

### 第四阶段：SCAR 鲁棒性测试

对视频做受控变换：

```text
H.264 压缩
resize
加噪
转码
抽帧
补帧
局部编辑
```

观察：

```text
fake score 是否稳定
证据权重是否变化
残差曲线是否变化
检测器是否依赖 shortcut
```

目标：

```text
从普通分类器升级为鲁棒性分析框架
```

---

## 十六、最小代码框架

```python
import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self, out_dim=512, freeze_backbone=True):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.proj = nn.Linear(512, out_dim)

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x):
        feat = self.feature_extractor(x)
        feat = feat.flatten(1)
        feat = self.proj(feat)
        return feat


class VideoEncoder(nn.Module):
    def __init__(self, feature_dim=512, freeze_backbone=True):
        super().__init__()

        self.image_encoder = ImageEncoder(
            out_dim=feature_dim,
            freeze_backbone=freeze_backbone
        )

        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, video):
        B, T, C, H, W = video.shape

        frames = video.view(B * T, C, H, W)

        feats = self.image_encoder(frames)

        feats = feats.view(B, T, -1)

        feats = self.norm(feats)

        return feats


class GRUVideoDetector(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, num_classes=2):
        super().__init__()

        self.encoder = VideoEncoder(
            feature_dim=feature_dim,
            freeze_backbone=True
        )

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, video):
        feats = self.encoder(video)

        states, _ = self.gru(feats)

        video_state = states.mean(dim=1)

        logits = self.classifier(video_state)

        return logits, feats, states
```

这段代码对应：

```text
视频帧
↓
图像编码器提取每帧特征
↓
LayerNorm 做状态尺度归一化
↓
GRU 建模时间状态
↓
平均池化得到视频表示
↓
分类器输出 real / fake
```

---

## 十七、总结


 
编码器是整个 AIGC 视频检测系统的观测模块。它将原始视频帧从像素空间映射到特征空间，使后续模型能够在语义、结构、纹理、身份、频域和运动等不同证据空间中分析视频。

在我们的方法中，每一帧先经过视觉编码器得到 $f_t = Φ(x_t)$，随后进行状态尺度归一化，再输入时序模型进行状态估计。我们关注的核心问题不是单帧是否可疑，而是视频状态能否在时间上稳定演化。

因此，编码器的选择决定了检测器能观察到哪些异常信号。ResNet 可以提供基础视觉纹理特征，DINO 更适合结构状态建模，CLIP 更适合语义证据，ArcFace 更适合身份一致性，频域编码器则用于捕捉生成器指纹和高频异常。

最终，我们希望构建一个多编码器、多证据、可解释、可鲁棒评估的视频检测框架。


---

## 十八、最核心结论

编码器可以从四个层次理解：

### 1. 工程层

[
f = E(x)
]

把输入变成特征。

---

### 2. 表示学习层

编码器学习一个更适合任务的特征空间。

---

### 3. 信息选择层

编码器决定保留哪些信息，弱化哪些信息。

---

### 4. 科研方法层

编码器定义了检测器观察 AIGC 视频异常的证据空间。

对于你的课题，最重要的一句话是：

> AIGC 视频检测的关键，不只是训练一个 real/fake 分类器，更重要的是选择和构造能够稳定揭示生成异常的编码空间。

再进一步：

> 编码器决定检测器能看见什么，时序模型决定检测器如何理解这些观测随时间的演化，鲁棒性测试决定这些证据是否真的可靠。
