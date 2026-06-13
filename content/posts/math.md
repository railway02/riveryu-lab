---
title: "计算机科研数学学习笔记：系统版"
date: 2026-06-06T10:00:00+08:00
draft: false
tags: ["Math", "Virtual Human", "First Principles", "Computer Vision"]
summary: "数学整理"
---

## 0. 学习内容

计算机科研里的数学，不是按“高数、线代、概率论”这种考试顺序学，而是**围绕科研问题学**。

所有计算机视觉与 AI 研究问题，大致都可以拆成这条链：

$$\text{数据} \rightarrow \text{表示} \rightarrow \text{比较} \rightarrow \text{损失} \rightarrow \text{优化} \rightarrow \text{泛化} \rightarrow \text{解释}$$

对应数学是：

| 科研问题                         | 对应数学分支             |
| -------------------------------- | ------------------------ |
| **数据怎么表示？**               | 线性代数                 |
| **模型怎么学习？**               | 微积分                   |
| **参数怎么更新？**               | 最优化                   |
| **数据和预测有不确定性怎么办？** | 概率论                   |
| **实验结论是否可信？**           | 统计学                   |
| **loss、分布、信息怎么理解？**   | 信息论                   |
| **图像、视频、3D 有什么结构？**  | 信号处理、几何、时序建模 |
| **模型是不是学了 shortcut？**    | 因果推断、不变性、鲁棒性 |

**你的学习路径优先级：**

1. **核心基石**：`线性代数` + `微积分` + `概率统计` + `信息论` + `最优化`
2. **领域进阶（CV/3D/AIGC）**：`信号处理` + `几何` + `时序建模` + `因果鲁棒性`

---

## 1. 线性代数：数据和特征的语言

### 1.1 向量：样本的数字表示

深度学习里，数据最终都会变成向量。一张图片 $224 \times 224 \times 3$ 可以看成：

$$x \in \mathbb{R}^{150528}$$

模型提取特征 $z = f_\theta(x)$，比如：

$$z \in \mathbb{R}^{512}$$

这个 $z$ 就是模型对图片的理解。

> **💡 直觉**：模型把复杂对象压缩成了一个高维空间中的“语义坐标”。

### 1.2 矩阵：把一个向量变成另一个向量

全连接层方程 $y = Wx + b$，在 PyTorch 里是：

```python
nn.Linear(in_features=d, out_features=m)

```

数学上就是：
$x \in \mathbb{R}^{d}$，$W \in \mathbb{R}^{m \times d}$，最终映射为 $y \in \mathbb{R}^{m}$。

> **💡 直觉**：矩阵 $W$ 把输入向量从一个空间变换（旋转、拉伸）到了另一个空间。

### 1.3 张量：高维数组

* **图像 batch**：$X \in \mathbb{R}^{B \times C \times H \times W}$
* **视频 batch**：$X \in \mathbb{R}^{B \times T \times C \times H \times W}$

（其中：$B$ = batch size，$T$ = 帧数，$C$ = 通道数，$H$ = 高度，$W$ = 宽度）

**核心经验：以后看代码，第一件事不是看公式，而是看 shape。**

例如：

```python
q.shape = [B, D]
k_neg.shape = [B, N, D]
q.unsqueeze(2).shape = [B, D, 1]

```

此时执行 `torch.bmm(k_neg, q.unsqueeze(2))`，对应的就是：

$$[B, N, D] \times [B, D, 1] \rightarrow [B, N, 1]$$

再执行 `.squeeze(2)` 就会变成 $[B, N]$。这就是 Batch Matrix Multiplication。

### 1.4 内积：相似度的基础

两个向量 $a, b \in \mathbb{R}^d$ 的内积：

$$a^\top b = \sum_i a_i b_i$$

如果两个向量都归一化（$|a|=1, |b|=1$），那么 $a^\top b = \cos\theta$，这就是**余弦相似度**。CLIP、对比学习、检索系统都在用：

$$\text{sim}(a,b) = \frac{a^\top b}{|a||b|}$$

> **💡 直觉**：两个特征方向越接近，它们的语义越相似。

### 1.5 范数：长度、距离、误差

* **L1 范数**：$|x|_1 = \sum_i |x_i|$ （常见于：重建损失、稀疏约束、Pix2Pix 的 L1 loss）
* **L2 范数**：$|x|_2 = \sqrt{\sum_i x_i^2}$ （常见于：欧氏距离、特征距离、正则化、**重投影误差**）
* **无穷范数**：$|x|_\infty = \max_i |x_i|$ （常见于对抗攻击：$|\delta|_\infty \leq \epsilon$，意思是每个像素最多只能改 $\epsilon$）

### 1.6 Attention 的线性代数本质

Self-Attention 公式：

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

拆解来看：

1. $QK^\top$ 是相似度矩阵。如果 $Q, K \in \mathbb{R}^{N \times d}$，那么 $QK^\top \in \mathbb{R}^{N \times N}$，表示每个 token 和每个 token 的相似度。
2. $\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)$ 把相似度变成权重。
3. 最后乘 $V$ 得到加权信息聚合。

> **💡 一句话**：Attention 就是“基于相似度的加权信息汇聚”。

### 1.7 SVD、PCA、低秩与 LoRA

SVD 分解：

$$A = U\Sigma V^\top$$

* $V^\top$：旋转输入空间
* $\Sigma$：按重要方向缩放
* $U$：旋转到输出空间

大模型微调中的 **LoRA** 形式：$W' = W + \Delta W$，其中 $\Delta W = BA$（$B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$）。如果 $r$ 很小，说明更新是低秩的。

> **💡 直觉**：大模型微调时，不需要改变整个大矩阵，只需要在少数重要方向上调整。

---

## 2. 微积分：模型为什么能学习

> **核心问题**：参数变一点，loss 会怎么变？

### 2.1 导数与梯度：多维变化率

一维导数 $f'(x) = \frac{dy}{dx}$ 表示 $x$ 变化一点，$y$ 大概变化多少。

在多维参数 $\theta = [\theta_1, \theta_2, \dots, \theta_n]$ 下，loss 函数 $\mathcal{L}(\theta)$ 的梯度为：

$$\nabla_\theta\mathcal{L} = \left[ \frac{\partial \mathcal{L}}{\partial \theta_1}, \dots, \frac{\partial \mathcal{L}}{\partial \theta_n} \right]$$

梯度方向是 loss 增长最快的方向，所以**训练要往反方向走**。

代码对应：

```python
optimizer.zero_grad()  # 清空旧梯度
loss.backward()        # 计算梯度
optimizer.step()       # 根据梯度更新参数

```

### 2.2 泰勒展开：梯度下降的根本原因

一阶泰勒展开：

$$\mathcal{L}(\theta+\Delta\theta) \approx \mathcal{L}(\theta) + \nabla_\theta \mathcal{L}^\top \Delta\theta$$

如果选择 $\Delta\theta = -\eta\nabla_\theta\mathcal{L}$，那么：

$$\mathcal{L}(\theta+\Delta\theta) \approx \mathcal{L}(\theta) - \eta |\nabla_\theta\mathcal{L}|^2$$

所以只要学习率 $\eta$ 合适，loss 必定会下降。这就是梯度下降的数学保证：

$$\theta_{t+1} = \theta_t - \eta\nabla_\theta\mathcal{L}$$

### 2.3 链式法则：反向传播的本质

如果 $z = f(y)$ 且 $y = g(x)$，那么 $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$。

神经网络是很多函数复合：$x \rightarrow h_1 \rightarrow h_2 \rightarrow h_3 \rightarrow \hat{y} \rightarrow \mathcal{L}$。要求第一层参数 $W_1$ 对 loss 的影响，就是从后往前连乘：

$$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h_3} \cdot \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1}$$

### 2.4 Jacobian 与 Hessian

* **Jacobian（一阶导数矩阵）**：$f:\mathbb{R}^n \rightarrow \mathbb{R}^m$，导数是矩阵 $J_f(x) = \frac{\partial f}{\partial x} \in \mathbb{R}^{m \times n}$。比如生成器 $G$ 的 Jacobian 表示 latent vector 每一维变化，会如何影响输出图像的每一个像素。
* **Hessian（二阶导数矩阵）**：$H = \nabla_\theta^2\mathcal{L}$，表示 loss 曲面的弯曲程度。它帮助我们理解 local minima、鞍点（saddle point）、sharp/flat minimum 以及模型的泛化能力。

---

## 3. 概率论：不确定性的语言

机器学习不是在处理绝对确定的世界，而是在处理不确定性。

### 3.1 期望与方差

期望是加权平均：

$$\mathbb{E}_{x\sim p(x)}[f(x)] = \int p(x)f(x)dx$$

代码里通常用 batch 平均近似：$\mathbb{E}[f(x)] \approx \frac{1}{B} \sum_{i=1}^B f(x_i)$。
论文里写 $\mathbb{E}_{x\sim p_{data}}[\ell(x)]$，代码里大概率就是 `loss = loss_per_sample.mean()`。

实验中方差很重要：$Var(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$。一个方法平均提高 0.2%，但标准差 1.0%，不能说明真的更好。科研实验结果最好报 $\text{mean} \pm \text{std}$。

### 3.2 条件概率与贝叶斯

生成任务的核心都是条件概率：

* 分类：$p(\text{class} | \text{image})$
* Pix2Pix：$p(\text{target image} | \text{input image})$
* 文本生图：$p(\text{image} | \text{text})$

贝叶斯公式贯穿始终：

$$p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)}$$

> **💡 科研贝叶斯思维**：看到一个好结果，你应该问：这结果来自方法本身的概率有多大？来自数据泄漏的概率多大？来自 random seed 的概率多大？来自 shortcut 的概率多大？

---

## 4. 统计学：实验结论是否可信

> **核心问题**：你看到的提升是真的，还是偶然？

### 4.1 经验风险与数据泄漏

我们真正关心的是真实风险 $R(f) = \mathbb{E}[\ell(f(x), y)]$，但训练时只能最小化经验风险 $\hat{R}(f) = \frac{1}{n}\sum \ell(f(x_i), y_i)$。

在 AIGC 视频检测等任务中，**数据泄漏**是致命的。常见的泄漏陷阱：

* 同一个视频的不同帧同时进 train 和 test
* 同一个 prompt 的生成结果同时进 train 和 test
* 原视频和压缩版本跨 train/test

正确做法：必须按 identity、prompt、generator 或 scene 严格 split。

### 4.2 置信区间与消融实验

对比两个方法时，如果 $90.1 \pm 0.8$ 和 $90.4 \pm 0.9$，不能轻易断言后者显著更好。

**消融实验 (Ablation Study)** 本质是控制变量法。提出模块 M，应比较：`baseline` -> `baseline + M` -> `baseline + N` -> `baseline + M + N`。

---

## 5. 信息论：loss 和表征学习的钥匙

### 5.1 熵与交叉熵

* **熵**：平均不确定性 $H(X) = \mathbb{E}_{x\sim p}[-\log p(x)]$。分类输出 $[0.5, 0.5]$ 熵大，$[0.99, 0.01]$ 熵小。
* **交叉熵**：$H(p,q) = -\sum_x p(x)\log q(x)$。分类任务中真实标签是 one-hot 的，所以交叉熵的本质就是：**让模型给真实类别更高的概率**。
* PyTorch 中的 `F.cross_entropy` 内部等价于 `log_softmax` + `nll_loss`。



### 5.2 KL 散度和 JS 散度

**KL 散度**：用 $q$ 去近似 $p$ 会额外损失多少信息：

$$KL(p||q) = \sum_x p(x)\log\frac{p(x)}{q(x)}$$

注意，KL 散度是有方向的（$KL(p||q) \neq KL(q||p)$）：

* **Forward KL** ($p_{data} || p_{model}$)：倾向于覆盖所有模式（mode-covering）。
* **Reverse KL** ($p_{model} || p_{data}$)：倾向于保守，可能导致 mode-collapse（mode-seeking）。

**JS 散度**是对称的，原始 GAN 最优判别器下，相当于最小化真实分布和生成分布之间的 JS 散度。

### 5.3 InfoNCE 与对比学习

对比学习的核心 loss：

$$\mathcal{L} = -\log \frac{\exp(q^\top k^+/\tau)}{\exp(q^\top k^+/\tau) + \sum_i \exp(q^\top k_i^-/\tau)}$$

> **💡 直觉**：本质是分类——给定 query，拉近正样本（$k^+$），推远负样本（$k^-$）。让同一语义对象的不同视图之间互信息更高。

---

## 6. 最优化与数值稳定性

论文里的目标函数通常长这样：

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_1\mathcal{L}_{\text{reg}} + \lambda_2\mathcal{L}_{\text{contrast}} + \lambda_3\mathcal{L}_{\text{recon}}$$

读论文时必须问：每一项在惩罚什么？去掉会怎样？$\lambda$ 怎么选？

### 6.1 常见优化器直觉

* **SGD**：引入 batch 噪声，有时能帮助找到更平坦的解。
* **Momentum**：梯度方向持续一致就加速，来回震荡就抵消。
* **Adam**：经常梯度大的参数步子变小，经常梯度小的参数步子相对变大。

### 6.2 代码的生死线：数值稳定

* **Softmax 溢出**：若 $x_i=1000$，$e^{1000}$ 会溢出。稳定写法是分子分母同除以最大值的指数 $e^m$。
* **Log-sum-exp**：用于对比损失、CRF 中，稳定写法为 $m + \log\sum_i e^{x_i-m}$。
* **float16 问题**：容易 nan 或 inf，需要 mixed precision、loss scaling 或梯度裁剪。

---

## 7. 信号、几何与因果：通向高水平研究

### 7.1 傅里叶变换与频域信号

傅里叶变换把信号分解成频率：

* **低频**：整体颜色、光照、大轮廓
* **高频**：纹理、噪声、压缩痕迹

在 AIGC 检测中，频域极其重要，因为生成图像或视频往往会有异常高频、周期性纹理或频谱分布异常。另外，采样率不足导致的混叠（aliasing）在视频中表现为边缘闪烁、纹理跳变。

### 7.2 几何：3D 视觉的核心

3D 点 $X = [X,Y,Z,1]^\top$ 投影到图像平面的基本方程：

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K[R|t] \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

其中 $K$ 是内参，$R,t$ 是外参。真实观测点与预测投影点之间的**重投影误差**（Reprojection Error）是 SfM、COLMAP 和 3DGS 优化的绝对核心。

### 7.3 因果与鲁棒性

机器学习极易学到 shortcut。比如 AIGC 检测器可能仅仅学到了“fake 视频分辨率更低”或“有特定压缩伪影”。
这也是为什么设计评测协议（如 SCAR protocol）需要做**干预实验 (Intervention)**：改变 codec、分辨率、码率。如果一压缩就失效，说明模型没有学到真正的“生成机制导致的不一致性”，而是学了混杂变量（confounder）。

---

## 8. 总结：读论文公式的 7 步法

以后看到天书一样的公式，按这个框架拆：

1. **找变量**：谁是输入？特征？参数？标签？
2. **看 shape**：维度是怎么变换的？
3. **看操作**：是矩阵乘法、范数、期望还是 KL？
4. **看比较对象**：是在对比预测和标签，还是正负样本？
5. **看 loss 惩罚什么**：每个 loss 都在说“我不希望某件事发生”。
6. **看优化方向**：是 min 还是 max，还是对抗的 min-max？
7. **翻译成人话**：用自己的直觉语言把公式的物理意义讲出来。

> **🌟 最终信仰**
> 线性代数让数据能表示，微积分让模型能学习，概率统计让结论可信，信息论让 loss 有意义，优化让训练可执行，信号与几何让你真正理解视觉世界。