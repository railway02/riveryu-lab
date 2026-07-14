---
title: "傅里叶特征在 StyleGAN3 中的作用：从高频拟合到连续坐标场"
date: 2026-06-18T10:00:00+08:00
draft: false
tags: ["Math", "Research", "StyleGAN"]
summary: "傅里叶特征"
uid: stylegan3-fourier-features
topics:
  - generative-representations
  - deep-learning-foundations
relations:
  - target: chordedit-optimal-transport
    type: related
    note: "两文分别在连续空间特征场与潜空间编辑控制场中，用连续性约束减少采样混叠与大步积分失稳。"
---

## 1. 问题背景：StyleGAN3 为什么需要傅里叶特征

在普通坐标 MLP 或 NeRF 中，傅里叶特征的主要作用是**增强网络表达高频细节的能力**。原始坐标 $(x,y)$ 太平滑，MLP 容易优先学习低频结构，导致图像边缘、纹理、细节拟合困难。加入傅里叶特征后，坐标被映射成一组不同频率的正弦/余弦基底，MLP 可以更容易组合出高频变化。

但 StyleGAN3 使用傅里叶特征的动机更深。StyleGAN3 面对的核心问题不是单张图像拟合不清晰，而是生成器内部特征和像素网格发生了绑定，导致纹理在运动时“粘”在屏幕坐标上。这种现象通常叫 **texture sticking**。

直观来说：
* **真实物体运动时**：脸部轮廓、皮肤纹理、头发细节应该一起运动。
* **StyleGAN2 中常见的问题**：脸的整体结构在动，但局部纹理像贴在屏幕坐标上一样。

这说明生成器内部的信息流受到了离散像素网格的影响。StyleGAN3 的目标是让生成器满足更强的空间一致性：

$$G(Tz) \approx T G(z)$$

也就是：
> 先在内部表示中做平移/旋转，再生成图像 **应该接近** 先生成图像，再对图像做平移/旋转。

这就是**等变性 (Equivariance)**。所以，StyleGAN3 中的傅里叶特征不只是为了让网络学到高频，更重要的是构造一个**连续、有限带宽、可解析平移和旋转**的初始坐标场。

---

## 2. StyleGAN3 的核心转变：从离散 feature map 到连续信号

传统卷积网络通常把 feature map 看成离散数组：

$$X \in \mathbb{R}^{C \times H \times W}$$

也就是一堆像素网格上的数值。StyleGAN3 则从信号处理角度重新解释 feature map：

$$X[u]$$



> **离散 feature map = 连续空间信号在采样点上的取值**

只有把 feature map 看成连续信号的采样结果，才有办法严肃讨论：
* 平移 0.25 个像素
* 旋转 3 度
* 亚像素级运动
* 连续空间中的坐标变换

如果 feature map 只是一个普通矩阵，那么“平移半个像素”本身就没有干净定义。但如果 feature map 来自一个连续函数，就可以先在连续空间变换函数，再重新采样。这也是 StyleGAN3 引入**有限带宽、采样率、低通滤波、alias-free layers** 的理论基础。

---

## 3. StyleGAN2 的 learned constant 有什么问题

StyleGAN2 的 synthesis network 一开始使用一个 learned constant input：

`learned constant` $\rightarrow$ `卷积` $\rightarrow$ `上采样` $\rightarrow$ `调制卷积` $\rightarrow$ `输出图像`

这个 learned constant 是一个固定空间网格上的张量。它天然带有**绝对像素位置**。这种设计有一个隐含问题：**生成器从第一层开始就知道“屏幕坐标”在哪里。**

它可以利用这个固定网格位置生成图像细节，比如：
* 某个纹理总是在某个 feature map 坐标附近出现
* 某种高频细节和像素网格对齐
* 局部纹理依赖屏幕位置，而不是依赖物体表面位置

当 latent 改变导致人脸姿态、位置、形状变化时，纹理不一定跟着物体走，于是出现 texture sticking。StyleGAN3 用 Fourier features 替代 learned constant，本质是在换掉生成器的初始空间表示：

* **StyleGAN2**: 固定离散 learned constant
* **StyleGAN3**: 连续傅里叶坐标场的采样

这个变化不是简单换输入格式，而是让生成器一开始就具有连续空间坐标系统。

---

## 4. StyleGAN3 的 SynthesisInput 在生成什么

StyleGAN3 的输入模块可以理解为生成一组二维平面波。对于第 $c$ 个通道：

$$x_c(u) = \sin\left(2\pi(f_c^T u + \phi_c)\right)$$

其中：
* $u=(x,y)$：二维连续坐标；
* $f_c \in \mathbb{R}^2$：第 $c$ 个通道的二维频率向量；
* $\phi_c$：相位；
* $c$：通道编号。

每个通道都是一个方向、频率、相位不同的正弦平面波。整体输入就是：

$$x(u) = [x_1(u), x_2(u), \dots, x_C(u)]$$

这相当于在二维空间中铺了一组随机方向的周期波，然后把这些周期波在当前采样网格上取值，得到初始 feature map。对应代码结构大致是：

```python
freqs = torch.randn([channels, 2])
phases = torch.rand([channels]) - 0.5

grid = build_2d_sampling_grid()

x = grid @ freqs.T
x = x + phases
x = torch.sin(2 * torch.pi * x)
```

输出形状是：`[batch, channels, height, width]`。

所以 SynthesisInput 的本质是：**用随机频率和随机相位定义一个连续二维傅里叶特征场，再在当前分辨率的采样网格上查询这个场。**

---

## 5. StyleGAN3 不是在做标准傅里叶变换

StyleGAN3 并没有对图像做 FFT，也没有密集枚举所有频率。它只是随机采样一批二维频率，构造一组 Fourier features。

标准傅里叶变换的目标是分析一个已有信号：

$$f(x) = \sum_k a_k \cos(2\pi kx) + b_k \sin(2\pi kx)$$

它关心的是：一个信号里面有哪些频率成分？每个频率的系数是多少？

而StyleGAN3 的傅里叶特征是用来给生成器提供一个连续坐标场。它关心的是：

* 如何让生成器的起始特征具有连续空间结构？
* 如何让这个结构可以被解析地平移和旋转？
* 如何避免从第一层就绑定像素网格？

因此，StyleGAN3 使用的是**随机 Fourier features**，而不是标准傅里叶展开。

---

## 6. 频率是怎么采样的：不是固定半径，而是有限带宽圆盘

StyleGAN3 中会随机生成二维频率：

```python
freqs = torch.randn([channels, 2])
radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
freqs /= radii * radii.square().exp().pow(0.25)
freqs *= bandwidth
```

这段代码的目的不是让所有频率大小相等，而是让频率落在一个**二维圆盘**内。也就是说：

* 频率方向是随机的
* 频率大小也是随机的
* 整体被限制在 `bandwidth` 控制的范围内

如果所有频率大小相等，那么频率点会落在一个圆环上。StyleGAN3 更接近在二维频率圆盘中采样，覆盖从低频到较高频的一片区域。这很重要，因为生成器初始坐标场需要具有不同尺度的空间变化。如果只有一个固定频率半径，初始特征会缺少多尺度结构。

---

## 7. bandwidth 和 sampling_rate 的意义

StyleGAN3 中的两个重要参数是 `bandwidth` 和 `sampling_rate`。它们来自信号处理。

### 7.1 bandwidth

`bandwidth` 表示当前连续信号允许的最高频率。

* **bandwidth 越大**：初始 Fourier field 中可以包含更快的空间变化。
* **bandwidth 越小**：初始 Fourier field 更平滑。

### 7.2 sampling_rate

`sampling_rate` 表示当前 feature map 的采样率。根据 Nyquist-Shannon 采样定理，如果一个连续信号的最高频率是 $f_{\max}$，采样率 $s$ 至少要满足：

$$s > 2 f_{\max}$$

否则会出现 aliasing。因此 StyleGAN3 要控制：

$$\text{bandwidth} < \frac{\text{sampling\_rate}}{2}$$

这里的核心思想是：连续信号可以被离散 feature map 表示，前提是它是有限带宽的，并且采样率足够高。这正是 StyleGAN3 alias-free 设计的基础。

---

## 8. 为什么 StyleGAN3 允许对 Fourier input 做平移和旋转

StyleGAN3 的 SynthesisInput 不是生成一个固定不动的 Fourier field。它会根据 latent $w$ 生成一个全局变换，包括平移和旋转。这部分非常关键。

对于一个二维正弦平面波：

$$x(u) = \sin(2\pi(f^T u + \phi))$$

如果对坐标做仿射变换 $u' = A u + t$，那么：

$$x(u') = \sin(2\pi(f^T(Au+t)+\phi))$$

展开：

$$x(u') = \sin(2\pi((A^T f)^T u + f^Tt + \phi))$$

这说明：

* 坐标空间中的**旋转/线性变换**，可以转化为 **frequency vector** 的变化。
* 坐标空间中的**平移**，可以转化为 **phase** 的变化。

具体对应关系是：

$$f' = A^T f$$

$$\phi' = \phi + f^Tt$$

所以，StyleGAN3 不需要真的在离散 feature map 上做旋转或平移。它可以直接改 frequency 和 phase，然后重新采样 Fourier field。这就是它的漂亮之处：**离散图像上的几何变换，被转化成连续傅里叶特征参数上的解析变换。**

---

## 9. 平移为什么只改 phase

先看一维情况：

$$x(u)=\sin(2\pi(fu+\phi))$$

如果把坐标平移 $t$：

$$x(u+t)=\sin(2\pi(f(u+t)+\phi))$$

展开：

$$x(u+t)=\sin(2\pi(fu+ft+\phi))$$

因此平移 $t$ 等价于把相位改成：

$$\phi'=\phi+ft$$

二维情况下：

$$\phi'=\phi+f^Tt$$

这就是代码中根据 translation 更新 phases 的原因。这也解释了为什么 Fourier features 特别适合做连续位移：对普通离散 feature map 来说，亚像素平移需要插值；对正弦波来说，平移可以直接通过相位变化精确实现。

---

## 10. 旋转为什么会改变 frequency

二维平面波：

$$x(u)=\sin(2\pi(f^T u+\phi))$$

如果坐标旋转 $u'=Ru$，那么：

$$f^T Ru = (R^T f)^T u$$

所以旋转坐标等价于旋转频率向量：

$$f' = R^T f$$

因此，StyleGAN3 可以通过改变 frequency vector 的方向来实现 Fourier input field 的旋转。这和直接旋转像素图完全不同：

* **直接旋转离散图像**：需要插值，容易引入模糊和 aliasing。
* **旋转 Fourier field**：直接变换频率向量，再重新采样，数学上更干净。

---

## 11. StyleGAN3 为什么只用 sin，不显式拼接 cos

普通 Fourier feature 常写成：

$$\gamma(x) = [\sin(2\pi Bx), \cos(2\pi Bx)]$$

这样做的一个好处是，两个编码的内积可以通过三角恒等式写成 $\cos(2\pi b^T(x_1-x_2))$，从而得到平移不变的核结构。

StyleGAN3 中只用了 $\sin(2\pi(f^Tu+\phi))$，没有显式使用 cos。这个设计可以从三个角度理解：

### 11.1 随机 phase 已经提供了相位自由度

因为 $\cos(x)=\sin(x+\frac{\pi}{2})$，如果每个通道都有随机相位 $\phi$，那么 $\sin(2\pi(f^Tu+\phi))$ 已经能够表达不同相位的周期信号。也就是说：**sin + random phase 已经在功能上覆盖了很多 sin/cos 成对编码的作用。**

### 11.2 后面还有可训练通道混合

StyleGAN3 的 Fourier input 生成后，并不是原样送进后续网络。它还会经过可训练线性通道混合。这意味着生成器可以学习如何组合不同频率、不同方向、不同相位的 sin 通道。因此，从表达能力角度看：**sin-only + random phase + learned channel mixing 已经足够构造丰富的周期坐标场。**

### 11.3 在随机相位期望下，sin-only 也能恢复类似平移不变结构

如果只看 $\sin(a)\sin(b)$，根据恒等式：

$$\sin(a)\sin(b) = \frac{1}{2}[\cos(a-b)-\cos(a+b)]$$

其中 $\cos(a+b)$ 依赖绝对位置，不具备平移不变性。但如果加入随机相位 $\phi$：

$$\sin(a+\phi)\sin(b+\phi)$$

对均匀随机相位取期望：

$$\mathbb{E}_{\phi}[\sin(a+\phi)\sin(b+\phi)] = \frac{1}{2}\cos(a-b)$$

含有 $(a+b+2\phi)$ 的项在相位平均中被抵消。所以，更准确的说法是：sin-only 本身不严格等价于 sin+cos；sin-only 加随机 phase 后，在大量通道的统计意义上可以获得类似平移不变的结构。StyleGAN3 刚好使用了随机 phases，因此只用 sin 是合理的。

---

## 12. 用户实验中的 grid + 10 应该如何理解

如果把坐标整体平移 $x \rightarrow x+10$，Fourier feature 变成：

$$\sin(2\pi b(x+10)) = \sin(2\pi bx + 20\pi b)$$

这等价于给每个频率通道加了一个固定相位偏移。如果重新训练一个 MLP，loss 几乎不变，这说明：整体平移坐标后，Fourier features 的表达能力没有下降。

但这个实验不能严格证明：同一个已经训练好的模型天然对绝对位置不敏感。因为重新训练时，MLP 可以重新适配新的相位系统。

更严格的测试方式应该是：

1. 用原始 grid 训练好模型；
2. 冻结模型参数；
3. 输入 grid + Δ；
4. 判断输出是否等价于原图的空间平移。

在 StyleGAN3 中，官方关注的也是这种意义上的 equivariance：对内部 Fourier input 做解析平移/旋转，看最终输出是否发生相应的平移/旋转。所以，grid + 10 实验可以作为表达能力实验，但不能作为严格的等变性证明。

---

## 13. Fourier Feature 在 StyleGAN3 中更像一个“连续全局坐标系”

在普通坐标 MLP 中，Fourier feature 是一种输入增强：
`坐标` $\rightarrow$ `多频率编码` $\rightarrow$ `MLP 更容易拟合高频`

在 StyleGAN3 中，Fourier feature 的身份更接近**生成器的连续空间坐标系统**。它告诉生成器：

* 哪里是空间中的不同位置
* 如何在连续坐标中移动
* 如何在连续坐标中旋转
* 如何避免直接绑定离散像素网格

这比“帮助学高频”更关键。可以这样理解：

* **NeRF / 坐标 MLP**：Fourier feature 是函数拟合的输入编码。
* **StyleGAN3**：Fourier feature 是生成器初始空间场的定义方式。

---

## 14. Fourier Feature 只是 StyleGAN3 的第一步，alias-free layers 才是完整方案

如果只把 StyleGAN2 的 learned constant 换成 Fourier features，而后续仍然使用普通离散上采样、非线性、卷积，那么 aliasing 仍然会出现。StyleGAN3 真正的改造包括：

1. 用 Fourier features 替代 learned constant；
2. 去掉逐层随机 noise input，减少像素级随机扰动；
3. 把 feature map 解释为连续有限带宽信号的采样；
4. 每次上采样/下采样都使用低通滤波；
5. 对非线性造成的新频率进行过滤；
6. 使用 filtered leaky ReLU；
7. StyleGAN3-R 中进一步增强旋转等变性。

其中非常重要的一点是：**非线性会产生新频率**。例如输入是低频信号，经过 ReLU / LeakyReLU 后可能产生更高频的分量。如果这些高频超过当前采样率能表示的范围，就会 alias 回低频，污染 feature map。因此，StyleGAN3 要在非线性和采样过程中加入低通滤波，确保信号始终符合有限带宽假设。

完整链条应该是：

`Fourier input 提供连续坐标场` $\rightarrow$ `alias-free synthesis layers 保持连续信号解释` $\rightarrow$ `低通滤波抑制 aliasing` $\rightarrow$ `输出图像获得更好的平移/旋转等变性` $\rightarrow$ `减少 texture sticking`

所以不能把 StyleGAN3 简化成“用了傅里叶特征”。更准确地说：

* 傅里叶特征解决**输入坐标场**问题；
* alias-free layers 解决**后续信号传播**问题。

---

## 15. StyleGAN3-T 和 StyleGAN3-R 的区别

StyleGAN3 主要有两个方向：

* **StyleGAN3-T**：更强调平移等变性
* **StyleGAN3-R**：进一步增强旋转等变性

### 15.1 StyleGAN3-T

StyleGAN3-T 主要解决 translation equivariance，也就是图像平移时内部特征和输出应保持一致。它的核心目标是减少纹理粘连，使细节随着物体运动，而不是粘在屏幕像素上。

### 15.2 StyleGAN3-R

StyleGAN3-R 进一步考虑 rotation equivariance。为了增强旋转等变性，它会更谨慎地处理方向性问题，例如使用更接近径向对称的滤波器，减少普通卷积和采样操作带来的方向偏置。

可以粗略理解为：

* **StyleGAN3-T**：解决“平移时纹理不要粘屏幕”。
* **StyleGAN3-R**：进一步解决“旋转时纹理和结构也要一致运动”。

---

## 16. NTK 分析在这里是否必要

如果研究普通坐标 MLP，NTK 是有解释价值的。因为普通坐标 MLP 的问题是：为什么输入 Fourier features 后，MLP 更容易学习高频？
NTK 可以解释：

* 输入编码改变了网络隐含核函数；
* 核函数的频谱影响模型学习低频/高频的速度；
* Fourier features 调整了 MLP 的频谱偏置。

但如果研究 StyleGAN3，NTK 不是主线。StyleGAN3 的主线是信号处理：

* continuous signal
* bandlimit
* sampling rate
* Nyquist
* aliasing
* low-pass filtering
* equivariance
* phase
* coordinate field

因此，理解 StyleGAN3 时，应把 NTK 放在辅助位置。更合适的理解顺序是：

1. 连续信号与采样定理；
2. 为什么离散 feature map 会产生 aliasing；
3. 为什么 texture 会粘在像素网格上；
4. Fourier features 如何构造连续初始坐标场；
5. affine transform 如何转化为 frequency 和 phase 的变化；
6. alias-free layers 如何保持连续信号假设；
7. 最终如何得到平移/旋转等变性。

---

## 17. 对“傅里叶特征真正意义”的修正

在普通 MLP 拟合连续数据中，可以说：傅里叶特征显式提供多频率坐标编码，帮助 MLP 学习高频细节。

但在 StyleGAN3 中，这个说法不够完整。更准确的表述是：

> 在 StyleGAN3 中，傅里叶特征的核心意义是构造一个有限带宽、连续、可解析平移和旋转的初始特征场。它替代 StyleGAN2 的 learned constant，使生成器从第一层开始就不再强绑定离散像素网格。后续 alias-free synthesis layers 继续维护连续信号假设，从而减少 texture sticking，并提升生成图像在平移和旋转下的一致性。

压缩成一句话：

* 坐标 MLP 用 Fourier Feature 是为了**更容易拟合高频**；
* StyleGAN3 用 Fourier Feature 是为了**让生成器拥有一个不粘像素网格的连续坐标系统**。

---

## 18. 最终总结

StyleGAN3 中傅里叶特征的作用可以分成四层：

* **第一层：输入替代**
它替代 StyleGAN2 的 learned constant，不再使用固定离散网格作为生成器起点。
* **第二层：连续坐标场**
它通过随机二维频率和随机相位构造连续 Fourier field：$x_c(u)=\sin(2\pi(f_c^T u+\phi_c))$，feature map 是这个连续场在离散网格上的采样。
* **第三层：可解析几何变换**
对坐标的平移和旋转可以转化为 $f' = A^T f$ 和 $\phi'=\phi+f^Tt$。因此 StyleGAN3 可以在 Fourier 参数空间中干净地实现输入场的平移和旋转。
* **第四层：alias-free 生成**
傅里叶输入只解决起点问题。后续网络还必须通过低通滤波、采样率控制、filtered nonlinearity 等方式避免 aliasing，才能真正减少 texture sticking。

最终，StyleGAN3 的创新不只是“加入傅里叶特征”，而是把整个生成器重新放到连续信号处理框架中理解和设计。

一句最重要的话：

> **Fourier features give StyleGAN3 a continuous coordinate system; alias-free layers keep that coordinate system from being destroyed.**

