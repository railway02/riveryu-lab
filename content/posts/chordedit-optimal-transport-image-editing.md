---
title: "ChordEdit：一步图像编辑为什么需要低能量传输？"
date: 2026-06-11T22:00:00+08:00
draft: false
tags: ["Computer Vision", "Image Editing", "Diffusion", "Optimal Transport", "Paper Reading"]
summary: "为什么一步图像编辑会因为高能量漂移场而失稳，如何用动态最优传输、局部二次估计和时间平滑构造稳定的 Chord Control Field"
math: true
uid: chordedit-optimal-transport
topics:
  - generative-representations
  - deep-learning-foundations
relations:
  - target: latent-tokenizer-aigc
    type: related
    note: "两文分别从低能量编辑控制场与视频潜表示轨迹讨论生成过程的连续性和稳定性。"
  - target: adversarial-reality
    type: applies-to
    note: "把低能量、时空平滑的控制场思路迁移为反取证扰动约束与检测轨迹线索。"
----------

## 导语

一步文本到图像模型已经可以非常快地生成图像，例如 SD-Turbo、SwiftBrush-v2、InstaFlow 等模型都在尝试把原本多步的 diffusion 或 flow 过程压缩到一步或少数几步。

但是，生成快不代表编辑也稳定。

图像编辑比图像生成更难。生成可以从噪声直接到图像，而编辑需要在改变目标语义的同时，尽量保持原图中不该改变的部分。例如把 “dog” 编辑成 “wolf”，模型应该改变动物语义，但背景、姿态、光照、构图都应该尽量保留。

ChordEdit 这篇论文关注的正是这个问题：

**如何在 training-free、inversion-free 的条件下，让一步图像编辑既快又稳定？**

它的核心回答是：

**不要直接使用 source prompt 和 target prompt 的漂移差，而要把编辑看成一个低能量传输问题，再通过时间平滑构造稳定的 Chord Control Field。**

---

## 1. 问题背景：一步编辑的困难

传统 training-free 图像编辑方法中，一个常见思路是直接比较两个 prompt 下的模型输出。

设：

* 源文本为 $c_{src}$
* 目标文本为 $c_{tar}$
* 当前图像状态为 $x$
* 时间为 $t$

模型在不同 prompt 下会给出不同的漂移方向：

$$ v(x,t,c_{src}) $$

$$ v(x,t,c_{tar}) $$

于是最直接的编辑方向可以写成：

$$ \Delta v(x,t)=v(x,t,c_{tar})-v(x,t,c_{src}) $$

这个式子的解释是：

**目标 prompt 下模型想让图像往哪里走，减去源 prompt 下模型想让图像往哪里走，剩下的就是从源语义改到目标语义所需要的方向。**

这个想法很自然，但在一步模型中会出现严重问题。

在多步 diffusion 编辑中，每一步都很小。即使某一步方向有误差，后面还有很多步可以逐渐修正。

但是一步编辑中，只有一次大步更新：

$$ x_{new}=x_{old}+h\Delta v $$

当步长 $h$ 很大，而 $Δv$ 又不稳定时，结果就容易崩坏。

常见失败包括：

1. 目标物体形状扭曲；
2. 背景被错误修改；
3. 非编辑区域结构崩坏；
4. 局部纹理出现伪影；
5. 语义编辑和身份保持无法兼顾。

ChordEdit 的基本判断是：

**一步编辑失败的根本原因，不只是 prompt 不准，而是 naive drift difference 产生了高能量、不平滑、不适合大步长积分的控制场。**

---

## 2. 图像编辑的动力系统视角

ChordEdit 首先把文本到图像模型看成一个条件概率流。

图像状态记为：

$$ x_t $$

这里的 $x_t$ 通常不是 RGB 图像，而是 VAE latent 空间中的高维向量。

文本条件记为：

$$ c $$

模型给出的速度场或漂移场记为：

$$ v(x_t,t,c) $$

于是生成或编辑过程可以写成一个常微分方程：

$$ \frac{dx_t}{dt}=v(x_t,t,c) $$

这个公式可以这样理解：

**图像状态如何随时间变化，由当前状态、当前时间和文本条件共同决定。**

如果把图像 latent 空间想象成一张高维地图，那么：

* `x_t` 是当前位置；
* `c` 是导航目标；
* `v(x_t,t,c)` 是模型给出的移动方向；
* 图像编辑就是在这张高维地图里移动图像状态。

---

## 3. 普通漂移差会变成高能量场的原因

普通编辑使用：

$$ \Delta v=v_{tar}-v_{src} $$

其中：

$$ v_{tar}=v(x,t,c_{tar}) $$

$$ v_{src}=v(x,t,c_{src}) $$

问题在于，两个大向量相减，不一定得到一个平滑的小向量。

在一步模型中，文本条件到向量场的映射往往非常敏感。source prompt 和 target prompt 对应的两个方向可能差异很大。直接相减后，得到的残差场可能具有三个危险特征。

### 3.1 幅度大

如果：

$$ |\Delta v| $$

很大，说明编辑方向本身太猛烈。

一步更新时，方向太大就容易把图像 latent 推到错误区域。

### 3.2 时间上变化剧烈

如果：

$$ |\partial_t \Delta v| $$

很大，说明不同时间点上的编辑方向变化很快。

一步模型只取一个大步，无法细致跟踪中间变化，所以容易偏航。

### 3.3 空间上变化剧烈

如果：

$$ |\nabla_x \Delta v| $$

很大，说明图像状态稍微变化，编辑方向就大幅变化。

这会导致相邻区域被推向不同方向，进而破坏结构、边缘和背景。

因此，一步编辑要成功，关键不是简单增强语义，而是让控制场满足：

1. 能量低；
2. 时间平滑；
3. 空间平滑；
4. 适合大步长积分。

---

## 4. Euler 积分视角：一步编辑需要控制场更稳定

对一个动力系统：

$$ \frac{dx}{dt}=u(x,t) $$

最简单的数值解法是 Euler 方法：

$$ x_{k+1}=x_k+h u(x_k,t_k) $$

其中 $h$ 是步长。

多步编辑中，$h$ 很小。
一步编辑中，$h$ 很大。

Euler 方法的误差与向量场的变化程度有关。论文中用下面的量描述稳定性风险：

$$ \mathcal{C}(u)=\lVert\partial_t u\rVert_\infty+\lVert\nabla_x u\rVert_\infty+\lVert u\rVert_\infty $$

这个式子包含三部分含义：

第一，$u$ 本身不能太大。

$$ |u|_\infty $$

第二，$u$ 随时间不能变化太快。

$$ |\partial_tu|_\infty $$

第三，$u$ 对图像状态不能太敏感。

$$ |\nabla_xu|_\infty $$

如果这三个量都很大，那么一次 Euler 大步更新就很危险。

所以 ChordEdit 的核心目标是：

**构造一个比 naive drift 更平滑、更低能量、更适合一步 Euler 更新的控制场。**

---

## 5. 最优传输视角：编辑是分布之间的低能量搬运

ChordEdit 没有只停留在单张图像的层面，而是把编辑理解为分布之间的搬运。

设源 prompt 对应的图像分布是：

$$ \rho_1=p_1(\cdot|c_{src}) $$

目标 prompt 对应的图像分布是：

$$ \rho_0=p_0(\cdot|c_{tar}) $$

图像编辑可以理解为：

**把源图像分布搬运到目标图像分布。**

这个搬运过程由一个速度场驱动：

$$ u_t(x) $$

动态最优传输的目标是：

$$ \min_{\rho,u} \int_0^1\int \frac{1}{2}|u_t(x)|^2\rho_t(x),dx,dt $$

它的约束是连续性方程：

$$ \partial_t\rho_t(x)+\nabla\cdot(\rho_t(x)u_t(x))=0 $$

这个优化问题的解释是：

**在所有能把源分布搬到目标分布的方法中，选择总动能最小的一种。**

其中：

$$ \frac{1}{2}|u_t(x)|^2 $$

表示局部动能。

因此，最优传输天然偏好：

1. 低能量路径；
2. 平滑移动；
3. 少破坏结构；
4. 避免突然跳跃；
5. 避免不必要的大幅度修改。

这就是 ChordEdit 的理论出发点。

---

## 6. 连续性方程的直观含义

连续性方程是：

$$ \partial_t\rho_t(x)+\nabla\cdot(\rho_t(x)u_t(x))=0 $$

这个式子可以用水流来理解。

设：

* $ρ_t(x)$ 是某个位置的水密度；
* $u_t(x)$ 是水流速度；
* $∂_tρ_t(x)$ 是这个位置水量随时间的变化；
* $∇·(ρ_tu_t)$ 是水流流入或流出的程度。

连续性方程表达的是：

**密度的变化只能来自流动，不能凭空产生或消失。**

放在图像编辑中，它意味着：

图像分布应该被连续地搬运到目标分布，而不是突然跳到一个不自然的区域。

这也是 ChordEdit 追求低能量控制场的原因。

---

## 7. 理想控制场不可得，所以要构造可观测代理场

最优传输中的理想控制场是：

$$ u_t(x) $$

但真实情况下，我们无法直接得到它。

模型不会直接输出最优传输意义下的编辑方向。因此论文构造一个可观测代理场：

$$ R(x_\tau,t) $$

它的定义是：

$$ R(x_\tau,t) = \mathbb{E}_{z\sim K_t(\cdot|x_\tau)} \left[ B_t \left( Q(z,t,c_{tar})-Q(z,t,c_{src}) \right) \right] $$

这个公式可以拆成六步。

第一，固定源图像作为锚点：

$$ x_\tau $$

第二，根据时间 $t$ 对源图像加噪，得到 noisy state：

$$ z\sim K_t(\cdot|x_\tau) $$

第三，分别用目标 prompt 和源 prompt 查询模型：

$$ Q(z,t,c_{tar}) $$

$$ Q(z,t,c_{src}) $$

第四，做差得到 prompt 差异信号：

$$ Q(z,t,c_{tar})-Q(z,t,c_{src}) $$

第五，用线性映射 $B_t$ 把不同模型输出统一到 velocity 或 drift 空间：

$$ B_t(\cdot) $$

第六，对加噪随机性取平均：

$$ \mathbb{E}[\cdot] $$

最终得到：

$$ R(x_\tau,t) $$

它可以理解为：

**通过查询模型得到的、带噪声的编辑方向。**

---

## 8. 从最优传输到统计估计

论文进一步假设：

$$ R(x_\tau,t)=u_t(x_\tau)+\epsilon_t $$

其中：

$$ \mathbb{E}[\epsilon_t]=0 $$

这表示：

可观测方向 $R$ 等于真实理想方向 $u$ 加上一些零均值噪声。

于是问题从：

**如何求解完整最优传输场？**

转化为：

**如何从带噪观测 R 中估计稳定的控制场 u？**

因为完整的动态最优传输在高维 latent 空间中非常难解。ChordEdit 没有强行求全局解，而是在源图像锚点附近做局部估计。

这是一种非常实用的科研思路：

**用理论确定目标结构，再用可计算的局部近似落地。**

---

## 9. 局部二次优化：Chord Control Field 的由来

论文在短时间窗口：

$$ [t-\delta,t] $$

内估计一个局部常量方向：

$$ u\in \mathbb{R}^d $$

构造目标函数：

$$ \Phi_t(u;x_\tau) = t|u-\hat{u}_{t-\delta}(x_\tau)|^2 + \int_{t-\delta}^{t} |u-R(x_\tau,\xi)|^2d\xi $$

这个目标函数有两项。

第一项：

$$ t|u-\hat{u}_{t-\delta}(x_\tau)|^2 $$

表示新的方向不要离上一个稳定估计太远,提供稳定性。

第二项：

$$ \int_{t-\delta}^{t} |u-R(x_\tau,\xi)|^2d\xi $$

表示新的方向要贴近当前窗口内观测到的编辑信号,提供语义编辑能力。

因此，这个优化目标在平衡：

1. 稳定性；
2. 编辑语义；
3. 局部低能量。

---

## 10. 对二次目标求解

因为 $Φ_t$ 是关于 $u$ 的二次函数，所以可以通过求导等于 0 得到最优解。

最终得到：

$$ u_t^\star(x_\tau) = \frac{t}{t+\delta}\hat{u}_{t-\delta}(x_\tau) + \frac{1}{t+\delta} \int_{t-\delta}^{t} R(x_\tau,\xi)d\xi $$

接着使用一阶近似：

$$ \hat{u}_{t-\delta}(x_\tau)\approx R(x_\tau,t-\delta) $$

$$ \int_{t-\delta}^{t}R(x_\tau,\xi)d\xi \approx \delta R(x_\tau,t) $$

代入后得到 Chord Control Field：

$$ \hat{u}_t(x_\tau) = \frac{ tR(x_\tau,t-\delta)+\delta R(x_\tau,t) }{ t+\delta } $$

这个公式是整篇论文的核心。

它本质上是一个加权平均：

* 用 $R(t-δ)$ 提供更稳定的历史方向；
* 用 $R(t)$ 提供当前目标语义；
* 用 $t$ 和 $δ$ 控制两者的权重。

---

## 11. Chord Control Field 的直觉

核心公式：

$$ \hat{u}_t(x_\tau) = \frac{ tR(x_\tau,t-\delta)+\delta R(x_\tau,t) }{ t+\delta } $$

可以理解为：

**最终编辑方向 = 稍早时刻的编辑方向与当前编辑方向的加权平均。**

如果：

$$ t=0.90,\quad \delta=0.15 $$

那么：

$$ \hat{u}_t = \frac{0.90R(t-0.15)+0.15R(t)}{1.05} $$

也就是：

$$ \hat{u}_t \approx 0.857R(t-0.15)+0.143R(t) $$

这表示它更相信较早的稳定方向，同时保留当前方向中的目标语义。

所以 ChordEdit 的关键操作可以概括为：

**不要直接使用暴躁的当前编辑场，而要使用短时间窗口内的平滑编辑场。**

---

## 12. 时间平滑能降低能量的原因

论文把 Chord Control Field 看成一种时间卷积平滑：

$$ \hat{u}=K_\delta * R $$

其中平滑核满足：

$$ K_\delta\ge0 $$

$$ \int K_\delta=1 $$

也就是说，$\hat{u}$ 是 $R$ 的加权平均。

根据 Jensen 不等式：

$$ \left| \int K_\delta(s)R(t-s)ds \right|^2 \le \int K_\delta(s)|R(t-s)|^2ds $$

因此可以得到：

$$ \int|\hat{u}|^2 \le \int|R|^2 $$

这意味着：

**平滑后的控制场总能量不会超过原始观测场。**

直观理解是：平均会削弱尖峰和抖动。

如果 naive field 在某些区域突然很大，时间平滑会压低这些高能量尖峰，使控制场更适合一步更新。

---

## 13. 平滑能改善数值稳定性

除了降低平均能量，平滑还会控制最大幅度和变化率。

论文中给出类似下面的关系：

$$ \lVert\hat{u}\rVert_\infty \le \lVert R\rVert_\infty $$

$$ \lVert\partial_t\hat{u}\rVert_\infty \le \lVert\partial_tR\rVert_\infty $$

$$ \lVert\nabla_x\hat{u}\rVert_\infty \le \lVert\nabla_xR\rVert_\infty $$

这三行分别表示：

1. 平滑后的方向最大幅度不增大；
2. 平滑后的方向时间变化不增大；
3. 平滑后的方向空间敏感性不增大。

于是 Euler 稳定性指标：

$$ \mathcal{C}(u)=\lVert\partial_t u\rVert_\infty+\lVert\nabla_x u\rVert_\infty+\lVert u\rVert_\infty $$

也会降低：

$$ C(\hat{u})\le C(R) $$

这就是 ChordEdit 能够支持一步大步长更新的数学原因。

---

## 14. 算法流程

ChordEdit 的核心算法非常简洁。

输入：

* 源图像 `x_src`
* 源文本 `c_src`
* 目标文本 `c_tar`
* 时间 `t`
* 平滑窗口 `δ`
* 编辑强度 `λ`
* refinement 时间 `t_c`

第一步，计算 Chord Control Field：

$$ \hat{u} = \frac{ tR(x_{in},t-\delta)+\delta R(x_{in},t) }{ t+\delta } $$

第二步，一步更新：

$$ x^{pred}=x_{in}+\lambda \hat{u} $$

第三步，可选 proximal refinement：

$$ x_{tar}=\text{prox}(x^{pred},t_c,c_{tar}) $$

其中：

* Chord transport 负责稳定结构；
* proximal refinement 负责增强目标语义。

---

## 15. Proximal refinement 的作用

经过 Chord transport 后，得到：

$$ x^{pred} $$

这个结果通常结构保持较好，但目标语义可能还不够强。

因此论文加入一个可选的 proximal refinement：

$$ \text{prox}(x^{pred},t_c,c_{tar}) $$

它的作用是用目标 prompt 再增强一次语义。

可以把整个方法理解成两阶段：

第一阶段：低能量搬运。
目标是稳，尽量保持背景、身份、结构。

第二阶段：语义增强。
目标是让结果更符合目标 prompt。



如果一个步骤同时追求强语义和强结构保持，往往容易冲突。ChordEdit 先保证路径稳定，再补充语义强度，因此整体更可控。

---

## 16. 关键参数理解

### 16.1 时间位置 `t`

`t` 决定在什么噪声时间查询模型。

较大的 `t` 往往带来更强的语义编辑信号，但如果过于接近数值不稳定区域，线性转换 `B_t` 可能放大误差。

### 16.2 平滑窗口 `δ`

`δ` 是 ChordEdit 中非常关键的参数。

当：

$$ \delta=0 $$

方法退化为 naive baseline。

当：

$$ \delta>0 $$

方法引入时间平滑，降低能量和方差。

`δ` 太小，编辑更激进，但不稳定。
`δ` 太大，方向更平滑，但语义可能变弱。

所以 `δ` 本质上控制稳定性和编辑强度的权衡。

### 16.3 编辑强度 `λ`

一步更新是：

$$ x^{pred}=x_{in}+\lambda \hat{u} $$

`λ` 越大，编辑越强。
但 `λ` 太大，仍然可能破坏图像结构。

### 16.4 refinement 时间 `t_c`

`t_c` 控制 proximal refinement 的强度。

它主要影响目标语义增强程度。

---

## 17. ChordEdit 与 naive drift 的区别

### Naive drift

Naive 方法直接使用：

$$ R(x_\tau,t) $$

或者：

$$ v(x,t,c_{tar})-v(x,t,c_{src}) $$

它的问题是：

* 能量高；
* 方差大；
* 时间不平滑；
* 空间不平滑；
* 一步更新容易崩。

### ChordEdit

ChordEdit 使用：

$$ \hat{u}_t(x_\tau) = \frac{ tR(x_\tau,t-\delta)+\delta R(x_\tau,t) }{ t+\delta } $$

它的优点是：

* 降低能量；
* 降低方差；
* 抑制高频尖峰；
* 改善 Euler 一步稳定性；
* 更好保持背景和结构。

本质区别是：

**Naive drift 是直接差分，ChordEdit 是低能量估计。**

---

## 18. 实验结果


### 18.1 一步 naive drift 会产生高能量场

当编辑步数减少到一步时，naive field 的能量明显升高，背景一致性下降。

这说明 naive drift 不适合一步大步长积分。

### 18.2 Chord Control Field 能降低能量

引入 `δ=0.15` 后，控制场能量降低，PSNR 更稳定，非编辑区域保持更好。

这验证了时间平滑和低能量传输思想。

### 18.3 ChordEdit 在语义和结构之间有更好权衡

ChordEdit 不只追求背景保持，也能保持较好的 CLIP 语义对齐。

这说明低能量控制场并没有简单牺牲编辑能力，而是在语义和稳定性之间取得了更合理的平衡。

---

## 19. 值得学习的地方

ChordEdit 的贡献不在于训练了一个新网络，也不在于堆了复杂模块。

它真正值得学习的是建模方式。

普通思路是：

$$ \text{editing}=\text{prompt drift difference} $$

ChordEdit 的思路是：

$$ \text{editing}=\text{low-energy transport field estimation} $$

当直接差分在一步极限下失效时，作者没有继续暴力调参数，而是问了一个更本质的问题：

**什么样的编辑方向适合一步大步长积分？**

答案是：**低能量、低方差、时间平滑、空间平滑的控制场。**

于是论文用动态最优传输提供低能量原则，用局部二次优化构造闭式估计，用 Jensen 不等式解释能量收缩，用 Euler 稳定性解释为什么一步更新更可靠。

这是一条完整的数学链路。

---

## 20. 对 AIGC 检测和对抗的启发

ChordEdit 对 AIGC 检测和对抗研究有很强启发。

### 20.1 对抗攻击不应只追求误判

很多对抗攻击只关心让检测器输出错误标签，但如果扰动能量过高，就容易留下异常痕迹。

可以考虑构造低能量攻击：

$$ x_{adv}=x+\Delta x $$

其中 `Δx` 不只要让检测器误判，还要满足：

* 低能量；
* 局部平滑；
* 时间一致；
* 结构保持；
* 不破坏自然图像统计。

这和 ChordEdit 的低能量控制思想一致。

### 20.2 检测器可以关注传输轨迹异常

真实图像或真实视频的变化可能更接近自然低能量路径。

生成或编辑内容可能存在：

* 局部高能量编辑痕迹；
* 时间方向不连续；
* identity 状态突变；
* 背景区域异常漂移；
* latent trajectory 不平滑。

这些都可以成为检测特征。

### 20.3 视频 deepfake 可以扩展到时空控制场

对视频而言，编辑不只发生在单帧，还发生在时间维度上。

可以考虑构造时空控制场：

$$ u(x,t,\tau) $$

其中 `τ` 表示视频帧时间。

一个好的视频编辑或伪造过程应该满足：

* 帧内低能量；
* 帧间平滑；
* 身份一致；
* 动作连续；
* 背景稳定。

反过来，检测器也可以从这些维度寻找异常。

---

## 21. 最重要的公式主线

整篇论文可以压缩成三行公式。

第一，模型观测到的编辑方向是带噪的：

$$ R=u+\epsilon $$

第二，用 Chord Control Field 做平滑估计：

$$ \hat{u}_t = \frac{ tR(t-\delta)+\delta R(t) }{ t+\delta } $$

第三，用平滑后的方向做一步编辑：

$$ x^{pred}=x+\lambda\hat{u} $$

这三行就是：

**带噪观测 → 平滑估计 → 一步更新。**

---

## 22. 总结

ChordEdit 解决的是一步图像编辑中的稳定性问题。

它指出，直接使用 source prompt 和 target prompt 的漂移差，会在一步模型中产生高能量、不平滑、不稳定的控制场，进而导致物体扭曲和背景破坏。

为了解决这个问题，论文把图像编辑重新建模为源分布到目标分布之间的低能量传输问题，并构造了 Chord Control Field：

$$ \hat{u}_t(x_\tau) = \frac{ tR(x_\tau,t-\delta)+\delta R(x_\tau,t) }{ t+\delta } $$

这个公式的本质是短时间窗口内的加权平均。它能降低编辑场能量、压制高频震荡、减少方差，并改善一步 Euler 更新的稳定性。

这篇论文最值得学习是它的科研范式：

**当一个直接方法在极限场景下失效时，不要只调参数，要找到失效背后的数学原因，再构造更稳定的数学对象。**

在 ChordEdit 中，这个更稳定的对象就是：**低能量传输控制场。**
