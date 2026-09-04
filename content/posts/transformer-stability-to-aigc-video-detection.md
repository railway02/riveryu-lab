---
title: "Transformer 数值稳定性、归一化机制与 AIGC 视频检测启发"
date: 2026-05-28T10:00:00+08:00
draft: false
tags: ["Deep Learning", "Transformer", "AIGC Detection", "Mathematics", "Research"]
summary: "跨界迁移到 AIGC 视频检测的课题中"
uid: transformer-stability-aigc
topics:
  - aigc-authenticity
  - deep-learning-foundations
relations:
  - target: encoder-aigc-video-detection
    type: builds-on
    note: "从编码器表征和隐状态出发，进一步讨论归一化、残差尺度与数值稳定性如何支撑时序检测。"
  - target: adversarial-reality
    type: extends
    note: "把图像级 detector-guided residual editor 继续发展为带残差归一化、时序一致性和状态分布约束的视频攻击目标。"
---


## 1. 总问题：为什么深层网络一定要控制数值尺度？

Transformer 的每一层都在做类似这样的操作：

$$ x_{l+1}=x_l+F_l(x_l) $$

其中 $x_l$ 是第 $l$ 层的 hidden state，$F_l$ 可能是 Attention，也可能是 MLP。

问题在于，深度网络不是只做一层矩阵乘法，而是反复做：

$$ x \rightarrow Wx \rightarrow \text{Activation} \rightarrow W'x \rightarrow \cdots $$

如果每一层都让方差稍微变大一点，几十层之后就会指数式放大。

假设每层让激活方差放大 $\alpha$ 倍：

$$ Var(x_{l+1})=\alpha Var(x_l) $$

那么经过 $L$ 层：

$$ Var(x_L)=\alpha^L Var(x_0) $$

如果：

$$ \alpha=1.1,\quad L=100 $$

那么：

$$ 1.1^{100}\approx 13780 $$

这意味着方差已经被放大了一万多倍。

所以深层模型的核心危险不是“某一层算错”，而是：

> 每一层都只偏一点点，堆深之后整体数值分布彻底失控。

这就是归一化存在的根本原因。

---

## 2. 数值失控的三种后果

### 2.1 激活值爆炸

如果 hidden state 变得很大：

$$ |x_l|\gg 1 $$

那么后续矩阵乘法、指数函数、Softmax 都会面临溢出。

在 FP16 中，最大有限值约为：

$$ 65504 $$

如果中间值超过这个范围，就可能出现：

$$ \text{Inf} $$

再参与后续运算，就可能变成：

$$ \text{NaN} $$

FP16 数值范围有限，BF16 范围更大但尾数短，容易出现截断误差和“大数吃小数”的问题。

---

### 2.2 Softmax 饱和

Attention 里有：

$$ P=\text{Softmax}(S) $$

如果 logits $S$ 的差距太大，比如：

$$ S=[20,0,-20] $$

那么：

$$ \text{Softmax}(S)\approx [1,0,0] $$

这时 Softmax 从“软选择”退化成“硬选择”。

Softmax 的导数大致和：

$$ p(1-p) $$

相关。

当：

$$ p\approx 1 $$

或者：

$$ p\approx 0 $$

都有：

$$ p(1-p)\approx 0 $$

于是梯度消失。未缩放的高维点积会让 Softmax 接近 Hardmax，从而导致梯度消失。

---

### 2.3 残差累加导致主干漂移

Transformer 有残差：

$$ x_{l+1}=x_l+F_l(x_l) $$

如果 $F_l(x_l)$ 的尺度没有被控制，那么每一层都往主干里加东西：

$$ x_L=x_0+\sum_{l=1}^{L}F_l(x_l) $$

当层数很深时，主干 $x_L$ 的尺度可能越来越大。

这就是为什么现代大模型不仅要有 Norm，还要配合：

$$ \text{Pre-Norm} $$

$$ \text{残差分支初始化缩放} $$

$$ \text{RMSNorm} $$

$$ \text{无 bias linear} $$

这些设计是一套互相配合的数值稳定系统。

---

# 3. 方差：深度学习里最重要的“尺度语言”

神经网络里说“数值稳定”，本质是在说：

$$ E[x] $$

和：

$$ Var(x) $$

是否可控。

均值控制中心位置，方差控制波动范围。

给一组 hidden state：

$$ x=[x_1,x_2,\dots,x_d] $$

均值：

$$ \mu=\frac{1}{d}\sum_{i=1}^{d}x_i $$

方差：

$$ \sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2 $$

标准差：

$$ \sigma=\sqrt{\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2} $$

归一化的核心就是：

$$ \hat{x}_i=\frac{x_i-\mu}{\sigma} $$

这样得到：

$$ E[\hat{x}]=0 $$

$$ Var(\hat{x})=1 $$

这个操作本质上是在说：

> 我不关心你的绝对数值是多少，我只关心你相对于整体分布的位置。

是 Transformer、AIGC 检测、对抗扰动、时序一致性分析之间的共同数学语言。

---

# 4. LayerNorm：控制一个 Token 内部的特征分布

在语言模型里，一个 token 的 hidden state 是一个 $d$ 维向量：

$$ x_t\in \mathbb{R}^d $$

LayerNorm 对同一个 token 的 hidden size 维度做归一化：

$$ \mu_t=\frac{1}{d}\sum_{i=1}^{d}x_{t,i} $$

$$ \sigma_t^2=\frac{1}{d}\sum_{i=1}^{d}(x_{t,i}-\mu_t)^2 $$

$$ \text{LayerNorm}(x_{t,i})= \gamma_i\frac{x_{t,i}-\mu_t}{\sqrt{\sigma_t^2+\epsilon}} + \beta_i $$

它的含义是：

> 每个 token 自己内部完成一次尺度校准。

这和 BatchNorm 不同。BatchNorm 是跨 batch 统计，而 LayerNorm 是 token 内部统计。

对语言模型来说，BatchNorm 不合适，因为不同样本可能来自完全不同的文本上下文。一个 batch 里可能同时有数学题、诗歌、代码、聊天，如果强行跨样本计算均值和方差，会让样本互相污染。

所以 Transformer 使用 LayerNorm / RMSNorm，本质上是为了保证：

$$ \text{每个 token 的归一化结果不依赖其他请求} $$

也就是训练和推理更一致。

---

# 5. RMSNorm：现代 LLM 的工程化归一化

LayerNorm 做两件事：

$$ x_i-\mu $$

和：

$$ \frac{1}{\sigma} $$

也就是：

1. 平移：减去均值；
2. 缩放：除以标准差。

RMSNorm 认为，最关键的是缩放，不一定需要减均值。

它定义：

$$ \text{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2} $$

然后：

$$ \text{RMSNorm}(x_i) = \gamma_i \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\epsilon}} $$

它不再计算：

$$ \mu $$

也不做：

$$ x_i-\mu $$

只控制向量的整体能量尺度。

重点是：RMSNorm 去掉均值计算后，减少了 reduction 状态、寄存器压力和 element-wise 减法，更适合高性能 CUDA/Triton kernel；而 LayerNorm 即使用 $E[X^2]-(E[X])^2$ 做融合，也仍然需要维护均值相关统计量，且在低精度下可能有灾难性抵消问题。

---

## 5.1 LayerNorm 和 RMSNorm 的本质差异

LayerNorm 控制的是：

$$ \frac{x-\mu}{\sigma} $$

RMSNorm 控制的是：

$$ \frac{x}{\text{RMS}(x)} $$

LayerNorm 去掉了均值方向：

$$ x \rightarrow x-\mu $$

RMSNorm 保留了均值方向，只控制整体长度。

从几何角度看，RMSNorm 更像是在控制向量模长：

$$ |x|_2 $$

因为：

$$ \text{RMS}(x)=\frac{|x|_2}{\sqrt{d}} $$

所以：

$$ \frac{x}{\text{RMS}(x)} = \frac{\sqrt{d}x}{|x|_2} $$

这说明 RMSNorm 本质上是在把向量投影到一个固定半径的超球面附近。

这句话很重要：

> RMSNorm 不强制特征中心为 0，而是强制特征能量稳定。

对于大模型来说，稳定能量比强行中心化可能更关键。

---

# 6. Pre-Norm：为什么现代大模型喜欢把 Norm 放在前面？

Transformer 有两种常见结构。

Post-Norm：

$$ x_{l+1}=\text{Norm}(x_l+F_l(x_l)) $$

Pre-Norm：

$$ x_{l+1}=x_l+F_l(\text{Norm}(x_l)) $$

Post-Norm 的问题是，反向传播时梯度每一层都要穿过 Norm。

如果网络很深，梯度可能被 Norm 的导数反复缩放。

Pre-Norm 的优势是主干路径干净：

$$ x_{l+1}=x_l+\text{something} $$

反向传播时：

$$ \frac{\partial x_{l+1}}{\partial x_l} = I+ \frac{\partial F_l(\text{Norm}(x_l))}{\partial x_l} $$

这里的 $I$ 是关键。

它保证梯度至少有一条接近恒等映射的路径可以传回去。

所以 Pre-Norm 的本质是：

> 牺牲一点前向表达的规整性，换取反向传播的稳定性。

这和 ResNet-v2 的思想非常像：让残差主路尽量保持 identity mapping，不要在主干路径上塞太多操作。

---

## 6.1 Pre-Norm 的隐患：表征坍塌

Pre-Norm 也有问题。

它不断累加：

$$ x_L=x_0+\sum_{l=1}^{L}F_l(\text{Norm}(x_l)) $$

因为每个残差分支输入都被 Norm 控制住，所以：

$$ F_l(\text{Norm}(x_l)) $$

的尺度相对稳定。

但主干 $x_l$ 会不断累加，可能越来越大。

于是深层新增的信息相对于主干来说越来越小。

这可以理解为：

$$ \frac{|F_l(\text{Norm}(x_l))|}{|x_l|} \rightarrow 0 $$

这就是为什么有人说 Pre-Norm 深层可能出现 representation collapse。

但是现代 LLM 仍然普遍使用 Pre-Norm，因为它带来的训练稳定性太重要。

结论是：

> Post-Norm 更规整，但深层训练难；Pre-Norm 更稳定，但深层表达可能变弱。现代 LLM 选择了工程上更可靠的 Pre-Norm。

---

# 7. Attention 缩放：为什么必须除以 $\sqrt{d_k}$？

Attention 的核心公式：

$$ \text{Attention}(Q,K,V) = \text{Softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right)V $$

关键问题是：为什么要除以 $\sqrt{d_k}$？

对一个 query 和 key：

$$ q\cdot k=\sum_{i=1}^{d_k}q_i k_i $$

假设：

$$ E[q_i]=0,\quad Var(q_i)=1 $$

$$ E[k_i]=0,\quad Var(k_i)=1 $$

且各维独立。

那么：

$$ E[q_i k_i]=E[q_i]E[k_i]=0 $$

并且：

$$ Var(q_i k_i)=E[q_i^2k_i^2]-E[q_ik_i]^2 $$

由于独立：

$$ E[q_i^2k_i^2]=E[q_i^2]E[k_i^2] $$

又因为方差为 1、均值为 0：

$$ E[q_i^2]=1,\quad E[k_i^2]=1 $$

所以：

$$ Var(q_i k_i)=1 $$

现在 $d_k$ 个项相加：

$$ q\cdot k=\sum_{i=1}^{d_k}q_i k_i $$

独立变量和的方差等于方差之和：

$$ Var(q\cdot k)=\sum_{i=1}^{d_k}Var(q_i k_i)=d_k $$

所以点积的标准差是：

$$ \sqrt{d_k} $$

### 原理说明
如果不做缩放，$d_k$ 越大，logits 数值波动越剧烈，Softmax 函数也越容易进入饱和状态。

对点积结果除以 $\sqrt{d_k}$ 做缩放处理后，方差推导如下：
$$ \text{Var}\left( \frac{q \cdot k}{\sqrt{d_k}} \right) = \frac{\text{Var}(q \cdot k)}{d_k} = 1 $$

这是该设计精妙的数学核心：**高维向量的点积会天然放大方差，通过 $\sqrt{d_k}$ 缩放，可将方差重新约束为单位方差**。



点积累加后的方差会变成 $d_k$，标准差是 $\sqrt{d_k}$，除以它可以把 logits 的方差重新缩放回 1。

---

# 8. Softmax：概率归一化，也是数值稳定系统的一部分

Softmax：

$$ p_i=\frac{e^{z_i}}{\sum_j e^{z_j}} $$

它做了三件事：

1. 把任意实数变成正数；
2. 归一化成概率；
3. 放大差异。

但 Softmax 有一个巨大风险：

$$ e^{z_i} $$

如果 $z_i$ 很大，指数会爆炸。

所以工程上永远不会直接算原始 Softmax，而是 Safe Softmax：

$$ p_i= \frac{e^{z_i-M}}{\sum_j e^{z_j-M}} $$

其中：

$$ M=\max_j z_j $$

为什么数学上等价？

因为：

$$ \begin{aligned} \frac{e^{z_i-M}}{\sum_j e^{z_j-M}} &= \frac{e^{z_i}e^{-M}}{e^{-M}\sum_j e^{z_j}} \\ &= \frac{e^{z_i}}{\sum_j e^{z_j}} \end{aligned} $$

所以减最大值不会改变结果。

但是工程上，它把最大指数项变成：

$$ e^0=1 $$

其他项都变成：

$$ e^{z_i-M}\leq 1 $$

因此避免了上溢。

这和 Normalization 的思想高度一致：

> 数学上保持相对关系不变，工程上改变绝对尺度以保证稳定。

Normalization、Softmax、Gumbel-Max 等底层优化的共同点，是用数学等价变换、数值稳定重写或经过验证的近似，换取更好的访存局部性、并行度和 kernel 融合空间。

---

# 9. Causal Mask：用数值操作表达结构约束

自回归语言模型要求第 $i$ 个 token 不能看未来 token。

所以 attention score：

$$ S_{ij} $$

如果：

$$ j>i $$

就设为：

$$ -\infty $$

也就是：

$$ S_{ij}= \begin{cases} \frac{q_i k_j^T}{\sqrt{d_k}}, & j\le i \ -\infty, & j>i \end{cases} $$

经过 Softmax：

$$ e^{-\infty}=0 $$

所以未来位置的注意力权重为 0。

这说明一件事：

> 模型里的“结构规则”，最后经常会被转化成 logits 空间里的数值规则。

因果约束不是通过自然语言告诉模型“别看未来”，而是在矩阵上直接把未来位置打成 $-\infty$。

这对你的课题很有启发。

AIGC 视频检测也可以把很多“真实世界约束”转化为数值约束：

真实视频中：

$$ \text{identity}_{t}\approx \text{identity}_{t-1} $$

$$ \text{lighting}_{t}\approx \text{lighting}_{t-1} $$

$$ \text{motion}_{t}\approx A(\text{motion}_{t-1}) $$

如果违反，就让异常分数升高。

也就是说，你不是主观说“它看起来假”，而是把“假”变成：

$$ \text{constraint violation} $$

---

# 10. FlashAttention：大模型工程里最值得学习的思想

朴素 Attention 会计算：

$$ S=QK^T $$

$$ P=\text{Softmax}(S) $$

$$ O=PV $$

问题是，如果序列长度是 $N$，那么 $S$ 和 $P$ 都是：

$$ N\times N $$

长文本下，这两个矩阵会产生巨大的显存读写。

FlashAttention 的核心不是近似 Attention，而是：

> 精确 Attention，但不把中间矩阵 $S$ 和 $P$ 写回 HBM。

它利用 Online Softmax 分块计算。

对一个 softmax 向量，维护：

$$ m=\max(z) $$

$$ l=\sum_j e^{z_j-m} $$

当新 block 来时：

$$ m_{new}=\max(m_{old},m_{block}) $$

旧分母要校正：

$$ l_{new} = l_{old}e^{m_{old}-m_{new}} + \sum_{j\in block}e^{z_j-m_{new}} $$

为什么要乘：

$$ e^{m_{old}-m_{new}} $$

因为以前的累加是基于旧最大值 $m_{old}$：

$$ e^{z_j-m_{old}} $$

现在最大值变了，要改成基于 $m_{new}$：

$$ e^{z_j-m_{new}} $$

两者关系是：

$$ e^{z_j-m_{new}} = e^{z_j-m_{old}}e^{m_{old}-m_{new}} $$

这就是 Online Softmax 的数学核心。

它不是拍脑袋优化，而是严格等价变换。

FlashAttention 把这个思想扩展到：

$$ O=\text{Softmax}(QK^T)V $$

在每个 tile 内同时完成：

$$ QK^T $$

$$ \text{Online Softmax} $$

$$ PV $$

最终避免写出巨大的 $S$ 和 $P$。

这对科研有一个很大的方法论启发：

> 好的研究不是只提出一个新模块，而是找到数学结构里的冗余，把它变成可计算、可优化、可扩展的形式。

---

# 11. 从 Transformer 数值稳定性迁移到 AIGC 视频检测课题

现在进入最重要的部分：这套笔记和我的科研到底有什么关系？

课题不是单纯做 LLM，但这里的数学思想可以直接迁移。

Transformer 的核心问题：

$$ \text{深层特征如何稳定传播？} $$

AIGC 视频检测核心问题：

$$ \text{视频状态如何稳定演化？} $$

这两个问题在数学形式上非常像。

---

## 11.1 Transformer 里要控制 hidden state，视频检测里要控制 temporal state

Transformer 中每层 hidden state：

$$ x_l $$

希望它不要爆炸、不要塌缩、不要漂移。

视频中每一帧的状态：

$$ s_t $$

也希望它符合真实世界的演化规律。

真实视频可以抽象为：

$$ s_t=A(s_{t-1},m_t)+\epsilon_t $$

其中：

* $s_t$：第 $t$ 帧状态；
* $m_t$：运动、姿态、相机变化等条件；
* $A$：真实世界的状态转移规律；
* $\epsilon_t$：小噪声。

如果是真实视频：

$$ |s_t-A(s_{t-1},m_t)| $$

应该比较小，并且稳定。

如果是生成视频：

$$ |s_t-A(s_{t-1},m_t)| $$

可能突然变大，或者呈现不自然的高频波动。

所以你可以把 AIGC 视频检测写成：

$$ r_t=|s_t-\hat{s}_t| $$

其中：

$$ \hat{s}_t=A_\theta(s_{t-1},m_t) $$

最后分类：

$$ p_{fake}=C(\text{Pool}({r_t,s_t}_{t=1}^{T})) $$

这比“检测伪影”高级得多。

你是在检测：

> 生成视频是否破坏了真实视频的状态转移分布。

---

## 11.2 Normalization 对你的启发：检测相对异常，不要迷信绝对值

Transformer 归一化告诉我们：

> 绝对数值不可靠，分布内的相对关系更重要。

这对 AIGC 视频检测非常关键。

不同视频的亮度、分辨率、压缩率、风格都不一样。如果你直接比较像素绝对值：

$$ x_t $$

很容易被无关因素干扰。

更合理的是比较归一化后的特征变化：

$$ \tilde{f}_t=\text{Norm}(f(x_t)) $$

然后看：

$$ \Delta_t=\lVert\tilde{f}_t-\tilde{f}_{t-1}\rVert $$

或者：

$$ r_t=|\tilde{s}_t-\hat{\tilde{s}}_t| $$

这可以减少亮度、风格、压缩等无关尺度的影响。

你可以在论文/PPT 里这样说：

> Inspired by normalization in deep networks, we focus on relative temporal deviations rather than raw feature magnitudes.

中文就是：

> 受深度网络归一化思想启发，我们不直接依赖原始特征幅值，而是关注归一化状态空间中的相对时序偏差。

这句话有研究味。

---

## 11.3 Softmax 对你的启发：检测器其实是在做“异常证据分配”

Softmax 把 logits 变成概率：

$$ p_i=\frac{e^{z_i}}{\sum_j e^{z_j}} $$

它的本质是：

> 在多个候选之间分配注意力或概率质量。

你的 AIGC 视频检测也可以类似处理。

假设你有多种异常证据：

$$ e_t^{id} $$

身份异常；

$$ e_t^{motion} $$

运动异常；

$$ e_t^{freq} $$

频域异常；

$$ e_t^{texture} $$

纹理异常；

$$ e_t^{semantic} $$

语义异常。

可以构造证据向量：

$$ e_t=[e_t^{id},e_t^{motion},e_t^{freq},e_t^{texture},e_t^{semantic}] $$

然后用 Softmax 做证据权重：

$$ \alpha_t=\text{Softmax}(W e_t) $$

最终异常分数：

$$ R_t=\sum_k \alpha_{t,k}e_{t,k} $$

这样你的检测器不是简单平均所有异常，而是学会：

> 当前视频更应该相信哪一种异常证据。

例如：

* 换脸视频：身份异常权重大；
* 文生视频：运动和物理异常权重大；
* 视频编辑/补帧：频域和边缘异常权重大；
* VACE 这类局部编辑：局部纹理和时序闪烁权重大。

这就是把 Softmax 从“分类概率”提升到“证据分配机制”。

---

# 12. 对你的 detector-guided residual editor 的启发

你之前做的图像残差编辑可以写成：

$$ x'=\text{clip}(x+\epsilon\tanh(r),0,1) $$

其中：

$$ r=G_\theta(x) $$

损失函数：

$$ \mathcal{L} = \lambda_{rec}\lVert x'-x\rVert_1 + \lambda_{det}D(x') + \lambda_{tv}\mathcal{L}_{TV}(r) + \lambda_{res}\lVert r\rVert_2^2 $$

这个方向很对，但它有一个危险：

> 如果残差没有稳定约束，模型可能学到全局脏纹理、色偏、摩尔纹，而不是人眼不可见的有效扰动。

这和大模型里的激活爆炸很像。

所以可以引入 residual normalization：

$$ \bar{r}=\frac{r}{\text{RMS}(r)+\epsilon} $$

然后：

$$ x'=\text{clip}(x+\eta\bar{r},0,1) $$

这样残差方向由网络学习，但残差强度被显式控制。

也可以做通道级归一化：

$$ \bar{r}_{c}=\frac{r_c}{\sqrt{\frac{1}{HW}\sum_{h,w}r_{c,h,w}^2+\epsilon}} $$

这样避免某个颜色通道过度偏移。

---

## 12.1 视频版本必须加 temporal residual norm

如果扩展到视频：

$$ x'_t=x_t+\delta_t $$

不能只控制单帧：

$$ |\delta_t| $$

还要控制时间变化：

$$ |\delta_t-\delta_{t-1}| $$

否则就会出现帧间闪烁。

可以设计：

$$ \mathcal{L}_{temp} = \sum_{t=2}^{T} |\delta_t-\delta_{t-1}|_1 $$

更进一步，考虑光流：

$$ \mathcal{L}_{flow} = \sum_{t=2}^{T} |\delta_t-\text{Warp}(\delta_{t-1},Flow_{t-1\rightarrow t})|_1 $$

这比普通 temporal loss 更强，因为它考虑了物体运动。

最终视频对抗编辑目标可以写成：

$$ \min_{\delta} \quad \lambda_{det}D(V+\delta) + \lambda_{rec}\sum_t\lVert x'_t-x_t\rVert_1 + \lambda_{temp}\sum_t\lVert\delta_t-\text{Warp}(\delta_{t-1})\rVert_1 + \lambda_{tv}\mathcal{L}_{TV}(\delta) $$

这个公式很适合放进你的课题 PPT。

---

# 13. AIGC 视频检测可以借鉴“Pre-Norm 思想”

Pre-Norm 的核心是：

$$ x_{l+1}=x_l+F(\text{Norm}(x_l)) $$

它不是直接处理原始 $x_l$，而是先把 $x_l$ 拉回稳定尺度。

你的检测模型也可以这样设计。

假设每帧特征：

$$ f_t=\Phi(x_t) $$

先归一化：

$$ \tilde{f}_t=\text{Norm}(f_t) $$

再计算时序残差：

$$ r_t=\tilde{f}_t-A_\theta(\tilde{f}_{t-1}) $$

这样比直接用：

$$ f_t-A_\theta(f_{t-1}) $$

更稳定。

原因是不同视频的原始特征尺度可能不同：

* 光照不同；
* 分辨率不同；
* 压缩率不同；
* 生成器风格不同；
* 人脸大小不同；
* 背景复杂度不同。

先归一化可以让检测器更关注“异常模式”而不是“尺度差异”。

这可以成为你方法设计里的一个原则：

> Normalize before temporal reasoning.

中文可以写：

> 先进行状态尺度归一化，再进行时序状态转移建模。

---

# 14. AIGC 视频中的“伪影”可以重新定义为分布失稳

传统说法：

> AIGC 视频有闪烁、纹理不稳定、身份漂移、边缘异常。

这比较散。

更高级的说法是：

> AIGC 视频的伪影是生成模型在多层状态空间中的分布失稳。

可以分成四层。

## 14.1 像素层失稳

$$ r_t^{pix}=|x_t-\text{Warp}(x_{t-1})| $$

检测的是局部像素变化是否符合运动。

对应问题：

* 闪烁；
* 边缘抖动；
* 局部纹理跳变；
* 背景不连续。

---

## 14.2 特征层失稳

$$ r_t^{feat}=|\Phi(x_t)-\Phi(\text{Warp}(x_{t-1}))| $$

其中 $\Phi$ 可以是 CLIP、DINO、CNN backbone、人脸识别网络等。

对应问题：

* 语义漂移；
* 物体形状变化；
* 局部结构不稳定。

---

## 14.3 身份层失稳

对人物视频：

$$ r_t^{id}=1-\cos(E_{id}(x_t),E_{id}(x_{t-1})) $$

对应问题：

* 人脸身份漂移；
* 眼睛、嘴巴、人脸轮廓不稳定；
* 发型局部变化导致身份特征漂移。

这和你做 VACE 改发型、head video、FlashAvatar 对比实验都有关系。

---

## 14.4 状态转移层失稳

设：

$$ s_t=F_\theta(x_1,\dots,x_t) $$

预测下一状态：

$$ \hat{s}_{t+1}=A_\theta(s_t,m_t) $$

残差：

$$ r_{t+1}^{state}=|s_{t+1}-\hat{s}_{t+1}| $$

这对应更高级的异常：

* 动作不符合物理；
* 头部转动不连续；
* 表情变化不自然；
* 手部运动不符合动力学；
* 镜头运动和人物运动不一致。

这就是你 SITF / 状态空间检测最应该强调的地方。

---

# 15. 一个更强的课题表达：从“伪影检测”升级到“状态分布稳定性检测”

你可以把课题重新表述为：

> 现有 AIGC 视频检测方法往往关注局部伪影或单帧统计差异，但高质量生成模型正在快速消除这些表层缺陷。因此，我们更关注真实视频在时间、语义、身份、频率和物理状态空间中的稳定演化规律，并检测生成视频对这些规律的系统性破坏。

对应数学框架：

给定视频：

$$ V={x_1,x_2,\dots,x_T} $$

提取多层观测：

$$ o_t= [ \Phi_{sem}(x_t), \Phi_{id}(x_t), \Phi_{freq}(x_t), \Phi_{motion}(x_t) ] $$

归一化：

$$ \tilde{o}_t=\text{Norm}(o_t) $$

状态估计：

$$ s_t=F_\theta(\tilde{o}_1,\dots,\tilde{o}_t) $$

状态预测：

$$ \hat{s}_t=A_\theta(s_{t-1},m_t) $$

残差：

$$ r_t=|s_t-\hat{s}_t| $$

池化：

$$ s_V=\text{Pool}({s_t,r_t}_{t=1}^{T}) $$

分类：

$$ p_{fake}=C(s_V) $$

证据输出：

$$ E_V= {r_t^{sem},r_t^{id},r_t^{freq},r_t^{motion}} $$

这样你的方法不只是输出 real/fake，还能输出解释：

* 哪一帧异常；
* 哪一种状态异常；
* 是身份异常、运动异常、频域异常，还是语义异常；
* 异常是否集中在局部区域。

这会让你的课题更像一个完整系统，而不是简单分类器。

---

# 16. 对抗生成方向：如何让 AIGC 视频骗过检测器？

如果你的课题同时做“攻”和“防”，那么攻击端可以写成：

给定原视频或生成视频：

$$ V={x_t}_{t=1}^{T} $$

学习一个扰动生成器：

$$ \delta_t=G_\theta(x_t,h_t) $$

生成对抗视频：

$$ x'_t=\text{clip}(x_t+\delta_t,0,1) $$

目标：

$$ \min_{\theta} \quad D(V') + \lambda_{vis}\mathcal{L}_{visual}(V,V') + \lambda_{temp}\mathcal{L}_{temporal}(V,V') + \lambda_{freq}\mathcal{L}_{frequency}(V,V') $$

其中 $D(V')$ 是检测器认为它是 fake 的分数。

这和你之前 residual editor 的思路是一致的，但视频版本必须强调：

$$ \text{temporal consistency} $$

否则攻击虽然降低检测分数，但肉眼会看到闪烁。

---

## 16.1 更高级的攻击目标：不只是降低检测分数，而是伪装状态分布

如果防守端检测：

$$ r_t=|s_t-\hat{s}_t| $$

攻击端就不能只优化像素相似度，而要优化：

$$ \min_\delta \sum_t |s'_t-\hat{s}'_t| $$

也就是说，让修改后的视频在检测器的状态空间里也显得稳定。

攻击目标可以写成：

$$ \mathcal{L}_{attack} = \lambda_{det}D(V') + \lambda_{state}\sum_t\lVert s'_t-\hat{s}'_t\rVert + \lambda_{rec}\sum_t\lVert x'_t-x_t\rVert + \lambda_{temp}\sum_t\lVert\delta_t-\text{Warp}(\delta_{t-1})\rVert $$

这比普通 adversarial noise 更像科研：

> 攻击不再是添加噪声，而是把生成视频投影回真实视频状态流形附近。

这句话可以作为你课题里的一个高阶观点。

---

# 17. 应该真正吸收的科研方法论


## 17.1 控制分布

LayerNorm 控制：

$$ E[x], Var(x) $$

RMSNorm 控制：

$$ |x|_2 $$

Softmax 控制：

$$ \sum_i p_i=1 $$

$\sqrt{d_k}$ 控制：

$$ Var(q\cdot k) $$

Causal Mask 控制：

$$ p(j>i)=0 $$

FlashAttention 控制：

$$ \text{HBM IO} $$

所以在看任何模型模块时，都应该问：

> 它到底在控制什么变量？控制的是均值、方差、能量、概率、信息流、显存访问，还是梯度路径？

这比背论文强得多。

---

## 17.2 好方法通常满足三层统一

一个强方法通常要同时说清楚三层。

第一层：数学上为什么成立。

例如：

$$ \text{Softmax}(z)=\text{Softmax}(z-\max z) $$

第二层：算法上解决什么问题。

例如避免指数溢出。

第三层：工程上为什么快。

例如减少 HBM 访问、便于 kernel fusion。

你以后看论文，也可以用这个框架：

| 层次   | 要问的问题                                 |
| ------ | ------------------------------------------ |
| 数学层 | 它有什么等价变换、约束、优化目标？         |
| 算法层 | 它解决了什么训练/泛化/鲁棒性问题？         |
| 工程层 | 它是否更快、更省显存、更稳定、更容易部署？ |

这正是你从“会跑实验”走向“会做科研”的关键。

---

# 18. 最终总结：知识和课题的统一主线

Transformer 的数值稳定性：

> 保留相对关系，控制绝对尺度。

AIGC 视频检测也可以总结：

> 保留真实视频的状态演化规律，检测生成视频对这些规律的破坏。

两者的共同数学思想是：

$$ \text{Normalize} \rightarrow \text{Compare} \rightarrow \text{Detect Residual} \rightarrow \text{Stabilize / Classify} $$

在 Transformer 里：

$$ x \rightarrow \text{Norm}(x) \rightarrow QK^T/\sqrt{d_k} \rightarrow \text{Softmax} \rightarrow \text{stable representation} $$

在 AIGC 视频检测里：

$$ x_t \rightarrow \Phi(x_t) \rightarrow \text{Norm}(\Phi(x_t)) \rightarrow r_t=\lVert s_t-\hat{s}_t\rVert \rightarrow p_{\mathrm{fake}} $$

真正吸收的核心：

> 大模型是靠稳定的尺度、清晰的相对关系和可控的信息流。
> AIGC 视频检测不应该只找表面伪影，而应该建模视频内部的稳定状态、相对变化和异常残差。
中心观点：

> 从 Transformer 数值稳定性出发，我们可以重新理解 AIGC 视频检测：检测器真正要学习的不是某个固定伪影，而是真实视频在时间、语义、身份、频率和物理状态空间中的稳定演化规律。高质量生成视频之所以难检测，是因为它正在逐渐逼近单帧真实分布；未来更强的检测方法必须转向跨帧状态分布、预测残差和多证据一致性建模。

这就是更有研究味的表达。
