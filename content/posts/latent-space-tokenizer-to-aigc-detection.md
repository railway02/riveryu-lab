---
title: "从 RAE 到 Cosmos Tokenizer：生成模型潜空间与 AIGC 视频数字指纹"
date: 2026-06-16T10:00:00+08:00
draft: false
tags: ["AutoEncoder", "Tokenizer", "AIGC Detection", "Video Generation", "Deep Learning"]
summary: "梳理 RAE、Causal 3D VAE 及 Cosmos Tokenizer "
math: true
---
## 1. 背景： AutoEncoder / Tokenizer

现代生成模型很少直接在像素空间生成完整图像或视频。更常见的做法是先把图像或视频压缩到一个潜空间，然后让 diffusion、DiT 或 autoregressive transformer 在潜空间里建模，最后再通过 decoder 还原成图像或视频。

传统流程是：

```text
图像 / 视频
↓
AutoEncoder / Tokenizer 编码
↓
latent / token
↓
Diffusion / Transformer 生成
↓
Decoder 解码
↓
图像 / 视频
```

因此，AutoEncoder / Tokenizer 决定了生成模型“看见什么信息”和“在哪个空间里学习生成”。

如果潜空间质量差，即使后面的 DiT 或视频生成模型很强，生成结果也会受到限制。
如果潜空间语义强、信息保留充分、时序结构稳定，生成模型会更容易学习到高质量图像或视频。

这就是 RAE、RAEv2、Causal 3D VAE、Cosmos Tokenizer 这些方法的重要性。

---

## 2. 从 VAE 到 RAE：潜空间从“压缩空间”变成“表征空间”

### 2.1 传统 VAE 的作用

在 Stable Diffusion、DiT 等 latent diffusion 模型中，传统 VAE 通常承担图像压缩器的角色。

基本过程是：

$$
z = E_{\text{VAE}}(x)
$$

$$
\hat{x} = D_{\text{VAE}}(z)
$$

其中：


x：输入图像
z：VAE latent
E：编码器
D：解码器
$x_\text{hat}$：重建图像


VAE 的优势是：

```text
1. 能把高分辨率图像压缩到低维 latent
2. 降低 diffusion 模型的训练和采样成本
3. latent 空间相对连续，便于生成
```

但是传统 VAE 也有明显问题：

```text
1. latent 维度低，信息容量有限
2. encoder 主要靠重建任务训练，语义表征能力弱
3. 重建会损失细节
4. 对 DiT 这类强 backbone 来说，VAE latent 可能成为瓶颈
```

VAE 更像一个“图像压缩器”，它关心的是怎么把图像压小并还原回来，但它未必能提供足够强的语义表示。

---

## 3. RAE：Representation AutoEncoder

RAE 的全称是：

```text
Representation AutoEncoder
```

它的核心思想是：

**用已经训练好的强视觉表征模型作为 encoder，再训练一个 decoder，让生成模型在更强的视觉表征空间里工作。**

RAE 的结构是：

```text
图像 x
↓
冻结的预训练视觉 encoder
例如 DINOv2 / SigLIP / MAE
↓
高维 representation latent z
↓
训练一个 decoder
↓
重建图像 x_hat
```

公式是：

$$
z = E_{\text{rep}}(x)
$$

$$
\hat{x} = D_\theta(z)
$$

其中：

```text
E_rep：冻结的视觉表征 encoder
D_theta：需要训练的 decoder
z：高维语义 latent
x_hat：重建图像
```

RAE 的关键变化是：

```text
传统 VAE：用重建训练出来的 encoder 作为压缩器
RAE：用 DINO / SigLIP / MAE 这种强视觉表征模型作为 encoder
```

所以 RAE 的潜空间不再只是压缩空间，而是语义表征空间。

---

## 4. RAE 的重要性

RAE 的核心判断是：**生成模型的瓶颈不只在 diffusion backbone，也在 autoencoder 的 latent 质量。**

过去很多工作都在改 diffusion backbone，例如从 U-Net 到 DiT，从普通 attention 到更大的 transformer，但底层 latent 仍然常常使用传统 VAE。

RAE 认为，如果生成模型在弱 latent 上学习，即使模型规模很大，也会受到信息瓶颈限制。

RAE 的改进方向是：

```text
1. 用更强的视觉表征 encoder
2. 保留更多语义和结构信息
3. 提供更高维、更丰富的 latent
4. 让 DiT 在 representation latent 上生成
```

可以把 RAE 理解为：**把 latent diffusion 的潜空间从低维压缩空间升级成高维视觉表征空间。**

---

## 5. RAE 的训练流程

### 5.1 第一阶段：训练 decoder

encoder 是冻结的，只训练 decoder。

```text
输入图像 x
↓
冻结 encoder 得到 z
↓
decoder 重建 x_hat
↓
计算重建损失
↓
更新 decoder
```

伪代码：

```python
for x in dataloader:
    with torch.no_grad():
        z = encoder(x)

    x_hat = decoder(z)

    loss = reconstruction_loss(x_hat, x)
    loss += perceptual_loss(x_hat, x)
    loss += adversarial_loss(x_hat, x)

    loss.backward()
    optimizer.step()
```

常见损失包括：

```text
1. L1 / L2 重建损失
2. LPIPS 感知损失
3. GAN loss
```

这里的关键是：**encoder 冻结，decoder 学会从强视觉表征中还原图像。**

---

### 5.2 第二阶段：在 RAE latent 上训练 DiT

训练好 RAE 后，可以用它来构建 diffusion / DiT 的潜空间。

流程是：

```text
图像 x
↓
RAE encoder 得到 latent z
↓
对 z 加噪
↓
DiT 学习去噪 / flow matching
↓
生成 latent
↓
RAE decoder 解码成图像
```

伪代码：

```python
for x in dataloader:
    with torch.no_grad():
        z = encoder(x)

    z = normalize(z)

    t = sample_timestep()
    noise = torch.randn_like(z)

    z_t = add_noise(z, noise, t)
    target = get_training_target(z, noise, t)

    pred = dit(z_t, t, condition)

    loss = mse_loss(pred, target)

    loss.backward()
    optimizer.step()
```

采样时：

```text
随机噪声 latent
↓
DiT 逐步去噪
↓
得到生成 latent
↓
RAE decoder 解码
↓
生成图像
```

---

## 6. RAE 的关键技术点

### 6.1 高维 latent 带来的问题

RAE latent 通常比传统 VAE latent 高维很多。

传统 VAE latent 可能是：

```text
4 × H/8 × W/8
```

RAE latent 可能是：

```text
768 × H' × W'
1024 × H' × W'
```

这带来一个问题：

```text
latent 更强，但 DiT 更难建模。
```

如果直接把原来的 DiT 套到 RAE latent 上，训练可能不稳定，收敛困难，效果甚至变差。

---

### 6.2 DiT width 需要匹配 token 维度

RAE latent 的 token 维度很高，如果 DiT 的 hidden dimension 太小，会形成信息瓶颈。

直观理解：

```text
RAE token 是 768 维
DiT hidden dim 如果只有 384
模型一开始就压缩信息
训练会变困难
```

因此，RAE 需要让 DiT 的宽度和 token 维度匹配。

设计原则：

```text
DiT hidden dimension ≥ RAE token dimension
```

---

### 6.3 噪声调度需要适配高维 latent

传统 diffusion 的噪声调度多为低维 VAE latent 或像素空间设计。

RAE latent 维度更高，通道更多，同样强度的噪声在高维空间中的作用会发生变化。

因此 RAE 需要使用和 latent 维度相关的噪声调度调整。

直观理解：

```text
latent 维度变了
噪声强度和信噪比也要重新调整
```

否则模型可能在训练中遇到信噪比失衡问题。

---

### 6.4 Decoder noise augmentation

RAE decoder 第一阶段主要从干净 latent 重建图像。

但 diffusion 生成出来的 latent 不可能完全干净，会带有一定误差。

如果 decoder 只见过 clean latent，它可能对 diffusion 输出的 noisy latent 很敏感。

所以可以在 decoder 训练阶段加入 latent noise augmentation：

```text
z_clean = encoder(x)
z_noisy = z_clean + noise
x_hat = decoder(z_noisy)
```

这样 decoder 能适应带噪 latent，提高生成阶段的稳定性。

---

### 6.5 DDT Head

RAE latent 维度高，如果把整个 DiT 主体都加宽，计算成本会很高。

因此可以使用一个浅而宽的输入/输出 head 来适配高维 latent。

直观理解：

```text
宽 head 负责接收高维 RAE token
主体 DiT 负责核心建模
输出 head 再映射回 latent 空间
```

这样可以在增加表达能力的同时控制计算量。

---

## 7. RAE 的创新点总结

RAE 的创新可以总结为五点：

```text
1. 用预训练视觉表征 encoder 替代传统 VAE encoder
2. 把 latent 从压缩空间升级为语义表征空间
3. 证明 DINO / SigLIP / MAE 等表征 encoder 也可以支持高质量重建
4. 让 DiT 在高维 semantic latent 上训练
5. 通过宽 head、噪声调度、decoder noise augmentation 等方法解决训练稳定性问题
```

RAE 的价值在于：

```text
它改变了 latent diffusion 的底层表示。
```

以前的问题是：

```text
如何让 diffusion backbone 更强？
```

RAE 提出另一个关键问题：

```text
生成模型到底应该在哪个 latent space 里学习？
```

---

## 8. RAE 的不足

### 8.1 高维 latent 成本更高

RAE latent 信息更多，但计算和显存压力更大。尤其是做视频时，如果每帧都使用高维 RAE latent，时序长度一增加，成本会迅速上升。

### 8.2 对 DiT 结构适配要求高

RAE 不能简单替换 VAE。需要调整 DiT width、噪声调度、head 结构和 decoder 训练方式。

这说明 RAE 的收益依赖完整系统设计。

### 8.3 原始 RAE 对局部空间细节仍有不足

原始 RAE 使用强语义 encoder 的最后层特征。最后层特征语义强，但局部空间细节可能不足。

对于图像生成，这可能影响纹理细节。对于视频生成，这可能导致帧间局部细节不稳定。

### 8.4 原始 RAE 主要面向图像

RAE 最初主要在图像生成场景中验证。如果直接用于视频，还需要考虑：

```text
1. 帧间一致性
2. 运动连续性
3. 视频 latent 的时序压缩
4. decoder 在时间维度上的稳定性
```

因此，单独 RAE 不是视频生成或视频检测的完整答案。

---

# 9. RAEv2：改进版 Representation AutoEncoder

RAEv2 的目标是：

```text
让 RAE 更简单、更稳定、更快收敛、更适合图像和视频任务。
```

RAEv2 主要解决原始 RAE 的几个问题：

```text
1. 只用最后一层特征可能损失局部空间信息
2. 原始 RAE 收敛速度仍然不够理想
3. RAE 和 representation alignment 方法之间的关系没有完全理清
4. 视频任务中局部空间信息不足会导致闪烁和不稳定
```

---

## 10. RAEv2 的核心改进一：多层特征聚合

原始 RAE 通常使用视觉 encoder 的最后一层特征。

但是视觉 encoder 的不同层包含不同信息：

```text
浅层：纹理、边缘、局部空间结构
中层：部件、局部语义
深层：全局语义、类别信息
```

只用最后一层，语义强，但局部细节可能不足。

RAEv2 提出使用最后 K 层特征的聚合：

$$
z = \sum_{k=1}^{K} \alpha_k h_{L-k+1}
$$

其中：

```text
h_l：encoder 第 l 层特征
alpha_k：聚合权重
K：参与聚合的层数
```

最简单的情况可以直接求和或平均：

$$
z = \frac{1}{K} \sum_{k=1}^{K} h_{L-k+1}
$$

这样可以同时保留：

```text
1. 高层语义
2. 中层结构
3. 局部空间信息
```

这是 RAEv2 最重要的改进之一。

---

## 11. RAEv2 的核心改进二：RAE + REPA 互补

REPA 可以理解为一种 representation alignment 方法，它让 diffusion model 的中间特征向外部视觉表征对齐。

原始理解中，可能会觉得：

```text
RAE 已经使用 representation latent，REPA 可能没必要。
```

但 RAEv2 的结论是：

```text
RAE 和 REPA 是互补的。
```

RAE 提供更强的 latent space。
REPA 进一步约束生成模型内部的中间表示。

可以理解为：

```text
RAE：改变模型输入输出所在的 latent space
REPA：约束 DiT 内部特征学习方向
```

二者结合后，生成效果和收敛速度都可以进一步提升。

---

## 12. RAEv2 的核心改进三：Self-guidance

传统 CFG 或 AutoGuidance 往往需要额外 forward 或额外模型。

RAEv2 里可以利用 REPA head 形成 self-guidance。
它相当于在模型内部构造一个较弱预测器，然后用强弱预测差异引导生成。

直观理解：

```text
不再额外训练一个弱模型
也不需要额外跑一次完整模型
而是利用内部 representation head 做引导
```

这使采样更高效。

---

## 13. RAEv2 面向视频

视频生成比图像生成更依赖局部空间信息的稳定性。

如果相邻帧中局部结构不稳定，就会出现：

```text
闪烁
纹理跳动
边缘漂移
物体局部形状变化
窗口数量变化
背景细节不一致
```

原始 RAE 主要使用最后层特征，可能过于偏全局语义。
RAEv2 通过多层特征聚合保留更多局部信息，因此更有利于帧间一致性。

从视频检测角度看，RAEv2 也更适合提取：

```text
1. 局部结构 latent
2. 语义 latent
3. 帧间 representation trajectory
4. 重建残差
```

因此，RAEv2 比原始 RAE 更适合作为 AIGC 视频检测中的 representation fingerprint encoder。

---

## 14. RAE 与 RAEv2 对比

| 维度         | RAE                                      | RAEv2                                   |
| ------------ | ---------------------------------------- | --------------------------------------- |
| Encoder      | 冻结视觉表征 encoder                     | 冻结视觉表征 encoder                    |
| 特征使用     | 多使用最后层特征                         | 聚合最后 K 层特征                       |
| 局部空间信息 | 相对弱                                   | 更强                                    |
| 收敛速度     | 已优于传统方案                           | 进一步加快                              |
| 生成质量     | 强                                       | 更强                                    |
| 视频一致性   | 有潜力但不够充分                         | 更适合视频任务                          |
| 核心思想     | 用 representation latent 替代 VAE latent | 让 representation latent 更完整、更稳定 |

一句话总结：

```text
RAE 解决“用什么 latent 生成”的问题；
RAEv2 解决“怎样让 representation latent 更好用”的问题。
```

---

# 15. Causal 3D VAE：面向视频的时空压缩器

RAE 和 RAEv2 主要从图像表征出发。
但视频生成还有一个更关键的问题：

```text
视频不是独立图像序列，而是时空连续信号。
```

如果逐帧使用图像 VAE 或图像 RAE，会忽略视频中的 temporal redundancy。

视频中相邻帧高度相关：

```text
背景相似
物体连续运动
光照连续变化
纹理逐渐变化
相机运动有轨迹
```

因此，视频 tokenization 应该同时压缩空间和时间。

Causal 3D VAE 的目标就是：

**用 3D 卷积同时编码空间和时间，把视频压缩成时空 latent。**

---

## 16. Causal 3D VAE 的基本结构

输入视频：

$$
X = {x_1, x_2, ..., x_T}
$$

编码为视频 latent：

$$
Z = E_{\text{3D}}(X)
$$

再解码：

$$
\hat{X} = D_{\text{3D}}(Z)
$$

其中：

```text
X：输入视频
Z：时空 latent
E_3D：3D VAE encoder
D_3D：3D VAE decoder
X_hat：重建视频
```

和逐帧图像 VAE 相比：

```text
图像 VAE：每帧独立编码
Causal 3D VAE：整段视频联合编码
```

对比：

```text
逐帧 VAE：
x_1 → z_1
x_2 → z_2
x_3 → z_3

Causal 3D VAE：
{x_1, x_2, x_3} → Z_video
```

---

## 17. 为什么要 Causal？

Causal 表示当前时刻的编码不能依赖未来帧。

例如编码第 t 帧时，只能使用：

```text
x_1, x_2, ..., x_t
```

不能使用：

```text
x_{t+1}, x_{t+2}, ...
```

这样设计有几个好处：

```text
1. 支持流式视频建模
2. 更符合时间因果关系
3. 避免未来信息泄漏
4. 有利于自回归或在线生成场景
```

Causal 3D convolution 的直觉是：

```text
在空间维度上看上下左右邻域
在时间维度上只看当前和过去
```

---

## 18. Causal 3D VAE 的关键技术

### 18.1 时空压缩

Causal 3D VAE 同时做空间压缩和时间压缩。

例如：

```text
空间压缩：H × W → H/8 × W/8
时间压缩：T → T/4
```

最终 latent 形状可能类似：

```text
C × T/4 × H/8 × W/8
```

这比逐帧 VAE 更高效，因为它利用了视频帧间冗余。

---

### 18.2 Scale-agnostic encoder

视频可能有不同长度和分辨率。
Scale-agnostic encoder 的目标是让 encoder 对不同尺度更稳定，避免只适配固定长度或固定分辨率。

---

### 18.3 Spatio-temporal down/up-sampling block

视频压缩不能只在空间上降采样，还要在时间上降采样。

因此需要设计专门的时空下采样和上采样模块：

```text
Downsample：压缩时间和空间，得到紧凑 latent
Upsample：恢复时间和空间，重建视频
```

如果上采样设计不好，视频容易出现：

```text
闪烁
运动断裂
帧间不连续
细节跳变
```

---

### 18.4 Flow regularization loss

视频重建不仅要像素相似，还要运动合理。

Flow regularization 的作用是让重建视频的运动场和真实视频更接近。

可以理解为：

```text
不只要求每一帧重建得像
还要求帧与帧之间的运动关系也像
```

这对视频 tokenizer 很重要，因为视频生成模型最终需要生成连续运动，而不只是生成一张张独立图片。

---

## 19. Causal 3D VAE 的创新点

Causal 3D VAE 的创新可以总结为：

```text
1. 把图像和视频统一到一个 VAE tokenization 框架中
2. 使用 causal 3D convolution 联合处理空间和时间
3. 同时进行空间压缩和时间压缩
4. 设计时空下采样 / 上采样模块
5. 使用 flow regularization 改善运动解码
```

它的核心价值是：

**让视频生成模型获得更适合视频的 latent space。**

---

## 20. Causal 3D VAE 的不足

Causal 3D VAE 也有不足：

```text
1. 计算量高于图像 VAE
2. 时序压缩可能损失细粒度运动
3. 对长视频建模仍然困难
4. decoder 如果不稳定，容易产生时间维度伪影
5. 对真实视频检测任务而言，可能会受到压缩格式、帧率、分辨率影响
```

但对于视频生成和视频检测，它比单帧 VAE 更接近真实视频生成链路。

---

# 21. Cosmos Tokenizer：图像和视频的通用神经 tokenizer

Cosmos Tokenizer 是 NVIDIA 提供的一套 image/video neural tokenizer。

它的目标是：

```text
把图像或视频压缩成连续 latent 或离散 token，
用于 diffusion 模型或 autoregressive transformer。
```

Cosmos Tokenizer 的输出有两类：

```text
Continuous latent：连续潜变量
Discrete token：离散 token id
```

媒体类型也有两类：

```text
Image tokenizer
Video tokenizer
```

所以它形成了四种组合：

```text
1. Continuous Image tokenizer
2. Discrete Image tokenizer
3. Continuous Video tokenizer
4. Discrete Video tokenizer
```

---

## 22. Cosmos Tokenizer 的基本流程

输入视频：

```text
视频 X
↓
Cosmos Encoder
↓
latent / token
↓
Cosmos Decoder
↓
重建视频 X_hat
```

连续 tokenizer：

$$
Z = E(X)
$$

$$
\hat{X} = D(Z)
$$

离散 tokenizer：

$$
q = Q(E(X))
$$

$$
\hat{X} = D(q)
$$

其中：

```text
Z：连续 latent
q：离散 token id
Q：量化器
```

---

## 23. Continuous Tokenizer 与 Discrete Tokenizer 的区别

### 23.1 Continuous Tokenizer

Continuous tokenizer 输出连续 latent。

特点：

```text
1. 更适合 diffusion / flow matching
2. 保留细节更平滑
3. 可以直接做连续空间去噪
```

类似：

```text
视频 → 连续 latent → diffusion 生成 → decoder
```

### 23.2 Discrete Tokenizer

Discrete tokenizer 输出离散 token。

特点：

```text
1. 更适合 autoregressive transformer
2. 可以像语言模型一样预测下一个 token
3. 方便统计 token 频率和 transition pattern
```

类似：

```text
视频 → 离散 token 序列 → autoregressive 生成 → decoder
```

对于 AIGC 视频检测来说，离散 token 特别有意思，因为可以分析：

```text
1. token frequency
2. token entropy
3. token transition
4. token n-gram pattern
5. real/fake token 分布差异
```

这很像语言模型检测里的 token 统计分析。

---

## 24. Cosmos Tokenizer 的优势

Cosmos Tokenizer 的主要优势是：

```text
1. 同时支持图像和视频
2. 同时支持连续和离散 latent
3. 支持高压缩率
4. 重建质量较高
5. 可以服务 diffusion 和 autoregressive 两类生成模型
```

对视频生成来说，它是一个高效视频压缩器。
对视频检测来说，它是一个很好的“数字指纹投影器”。

---

## 25. Cosmos Tokenizer 和 Causal 3D VAE 的关系

Causal 3D VAE 更像一种方法设计：

```text
用 causal 3D convolution 做图像/视频联合 tokenization。
```

Cosmos Tokenizer 更像一套完整工具系统：

```text
提供 image/video、continuous/discrete 多种 tokenizer。
```

二者共同点：

```text
1. 都关注视觉 tokenization
2. 都把图像/视频压缩到 latent/token 空间
3. 都能用于生成模型
4. 都能为 AIGC 视频检测提供潜空间分析入口
```

区别是：

| 维度     | Causal 3D VAE                  | Cosmos Tokenizer                   |
| -------- | ------------------------------ | ---------------------------------- |
| 核心     | Causal 3D convolution 视频 VAE | NVIDIA 图像/视频 tokenizer 套件    |
| 输出     | 主要是连续时空 latent          | 连续 latent 或离散 token           |
| 重点     | 图像/视频联合时空压缩          | 高效、高质量、多模式 tokenization  |
| 适合模型 | 视频 diffusion / tokenizer     | diffusion 和 autoregressive 都适合 |
| 检测价值 | 时空 latent 指纹               | latent 指纹 + token 统计指纹       |

---

# 26. 四者关系总结


```text
VAE：
低维压缩 latent，适合早期 latent diffusion

RAE：
用强视觉表征 encoder 替代 VAE encoder，把 latent 变成语义表征空间

RAEv2：
改进 RAE，用多层特征聚合保留更多局部空间信息，提升收敛和视频一致性

Causal 3D VAE：
面向视频，把图像/视频联合压缩到时空 latent

Cosmos Tokenizer：
更完整的图像/视频 tokenizer 系统，支持连续 latent 和离散 token
```

它们关注的问题不同：

```text
RAE / RAEv2：
生成模型应该用什么图像表征 latent？

Causal 3D VAE / Cosmos Tokenizer：
视频生成模型应该如何压缩时空信息？
```

---

# 27. 对 AIGC 视频检测的启发

传统检测器通常是：

```text
输入视频
↓
CNN / ViT / Transformer
↓
fake / real
```

这种方式容易学到数据集 shortcut，例如：

```text
分辨率
压缩格式
水印
生成器特定纹理
平台下载痕迹
```

更底层的思路是：

```text
输入视频
↓
投影到 latent / token space
↓
分析数字指纹
↓
判断 fake / real
```

也就是说，检测器不只看像素内容，而是看视频在生成模型潜空间里的统计结构。

---

## 28. 基于 RAEv2 的 Representation Fingerprint

用 RAEv2 编码每一帧：

$$
z_t^R = E_R(x_t)
$$

得到 latent 序列：

$$
Z^R = {z_1^R, z_2^R, ..., z_T^R}
$$

可以分析：

```text
1. 每帧 latent 分布
2. channel correlation
3. representation entropy
4. latent velocity
5. latent acceleration
6. 重建残差
7. 帧间 representation consistency
```

latent velocity：

$$
\Delta z_t = z_{t+1} - z_t
$$

latent acceleration：

$$
\Delta^2 z_t = z_{t+1} - 2z_t + z_{t-1}
$$

真实视频的 representation 轨迹通常更符合物理连续性。生成视频可能出现局部跳变、过度平滑、身份漂移或背景 representation 抖动。

---

## 29. 基于 Causal 3D VAE / Cosmos 的 Video Codec Fingerprint

用视频 tokenizer 编码整段视频：

$$
Z^V = E_V(X)
$$

可以分析：

```text
1. 时空 latent 分布
2. temporal latent spectrum
3. token transition pattern
4. token entropy
5. reconstruction residual
6. motion consistency
7. compression artifact
```

如果使用 Cosmos 的 discrete tokenizer，还可以看：

```text
1. token 频率分布
2. token bigram / trigram
3. token transition matrix
4. rare token ratio
5. token entropy over time
```

这就把视频检测变成了类似“生成链路数字取证”的问题。

---

## 30. 检测框架：双潜空间数字指纹

可以设计一个框架：

它包含两个空间：

```text
1. Representation Space
2. Video Codec Space
```

---

### 30.1 Representation Space

使用 RAEv2 / DINO / SigLIP / MAE 这类 encoder。

关注：

```text
语义结构
局部空间结构
身份一致性
物体 representation 轨迹
重建残差
```

适合发现：

```text
身份漂移
物体结构不稳定
语义空间轨迹异常
局部纹理跳变
```

---

### 30.2 Video Codec Space

使用 Causal 3D VAE / Cosmos Tokenizer。

关注：

```text
时空 latent
视频 token
运动压缩结构
时序频谱
decoder residual
token transition
```

适合发现：

```text
生成模型解码痕迹
时序压缩伪影
运动 latent 过平滑
局部运动不自然
插帧和压缩后的残留指纹
```

---

## 31. 双潜空间检测算法

输入视频：

$$
X = {x_1, x_2, ..., x_T}
$$

第一步：RAEv2 编码每一帧：

$$
z_t^R = E_R(x_t)
$$

第二步：视频 tokenizer 编码整段视频：

$$
Z^V = E_V(X)
$$

第三步：提取特征：

$$
F_R = f_R(Z^R)
$$

$$
F_V = f_V(Z^V)
$$

$$
F = [F_R, F_V]
$$

第四步：分类：

$$
s = C(F)
$$

其中：

```text
F_R：representation fingerprint
F_V：video codec fingerprint
C：简单分类器或统计检测器
s：fake score
```

分类器可以很简单：

```text
1. Logistic Regression
2. SVM
3. MLP
4. Mahalanobis distance
5. One-class classifier
```

这比直接训练黑箱视频检测器更可解释。

---

## 32. 适合鲁棒检测

真实传播链路中的视频会经历：

```text
压缩
转码
插帧
裁剪
缩放
滤镜
平台二次处理
```

普通检测器容易被这些后处理破坏。

而潜空间指纹方法关注的是：

```text
生成模型组织 latent 的方式
decoder 还原图像/视频的残差
时序 token 的统计规律
representation 轨迹是否自然
```

这些特征比单纯像素纹理更接近生成机制。

所以它适合研究：

```text
1. 插帧后检测是否仍然有效
2. 压缩后 latent fingerprint 是否保留
3. 不同生成器是否存在共享潜空间异常
4. 检测器是否依赖 generator-specific shortcut
5. 能否从模型特定指纹走向通用生成指纹
```

---

## 33. 可能的实验设计

### 33.1 数据

准备：

```text
真实视频
AIGC 视频
AIGC + 压缩
AIGC + 插帧
AIGC + 插帧 + 压缩
AIGC + 缩放 / 裁剪 / 滤镜
```

### 33.2 编码器

使用：

```text
RAEv2 / DINO / SigLIP
Causal 3D VAE
Cosmos continuous video tokenizer
Cosmos discrete video tokenizer
```

### 33.3 特征

提取：

```text
latent mean / variance
channel correlation
latent spectrum
temporal velocity
temporal acceleration
reconstruction residual
token entropy
token transition
motion consistency
```

### 33.4 指标

评估：

```text
AUC
EER
AP
Robust AUC
cross-generator performance
cross-postprocessing performance
```

### 33.5 关键问题

需要回答：

```text
1. RAEv2 latent 是否比 VAE latent 更适合检测？
2. 视频 tokenizer 是否比逐帧 encoder 更适合检测？
3. 插帧是否会破坏 pixel fingerprint，但保留 latent fingerprint？
4. 压缩是否会破坏 residual fingerprint，但保留 token transition fingerprint？
5. 哪些 fingerprint 是 generator-specific，哪些是 generator-agnostic？
```

---

# 34. 最后总结

RAE、RAEv2、Causal 3D VAE 和 Cosmos Tokenizer 都属于生成模型底层表示空间的关键技术。

RAE 的意义在于：

```text
用强视觉表征 encoder 替代传统 VAE encoder，
让图像生成模型在语义更强的 latent space 中学习。
```

RAEv2 的意义在于：

```text
通过多层特征聚合等改进，让 representation latent 同时保留语义和局部空间信息，
提升重建、收敛速度和视频一致性。
```

Causal 3D VAE 的意义在于：

```text
把视频作为时空连续信号来压缩，
利用 causal 3D convolution 建模时间维度，
更适合视频生成。
```

Cosmos Tokenizer 的意义在于：

```text
提供一套图像/视频通用 tokenizer，
同时支持连续 latent 和离散 token，
可以服务 diffusion 和 autoregressive 两类生成模型。
```

对 AIGC 视频检测来说，这些技术提供了一个新的方向：

```text
不要只在像素空间判断真假，
而要进入 latent / token space，
寻找生成模型留下的数字指纹。
```

更有潜力的研究框架是：

```text
RAEv2 表征指纹
+
Causal 3D VAE / Cosmos 视频编解码指纹
=
双潜空间数字指纹检测
```

这个方向既贴近生成模型机制，又适合做鲁棒检测，可以进一步研究插帧、压缩、转码等真实传播链路下的检测抗压性。
