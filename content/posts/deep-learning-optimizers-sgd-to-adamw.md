---
title: "深度学习优化算法系统笔记：从梯度下降到 AdamW"
date: 2026-06-09T10:00:00+08:00
draft: false
tags: ["Deep Learning", "Mathematics", "Optimization", "PyTorch"]
summary: "SGD、Momentum、RMSProp 到 AdamW"
---

# 深度学习优化算法系统笔记：从梯度下降到 AdamW

## 0. 优化器到底在做什么？

神经网络训练的目标是最小化损失函数：

$$ \min_{\omega} L(\omega) $$

其中：

- $\omega$：模型参数，可以是一个参数向量，也可以代表神经网络中的全部权重矩阵；
- $L(\omega)$：当前参数下的损失函数；
- 优化器的任务：不断更新 $\omega$，让 $L(\omega)$ 变小。

最基础的更新公式是：

$$ \omega_{t+1}=\omega_t-\eta g_t $$

其中：

$$ g_t=\nabla_\omega L(\omega_t) $$

含义是：

> 第 $t$ 次迭代时，在当前参数 $\omega_t$ 处，计算 loss 对参数 $\omega$ 的梯度。

符号说明：

| 符号           | 含义                |
| -------------- | ------------------- |
| $\omega_t$     | 第 $t$ 步的全部参数 |
| $g_t$          | 第 $t$ 步的梯度     |
| $\eta$         | 学习率              |
| $-g_t$         | loss 下降最快的方向 |
| $\omega_{t+1}$ | 更新后的参数        |

需要特别注意：

$$ \omega_t $$

表示第 $t$ 步时的参数整体，不表示第 $t$ 个参数。

如果模型有 $n$ 个参数：

$$ \omega_t=[\omega_{t,1},\omega_{t,2},\cdots,\omega_{t,n}] $$

那么梯度也是一个向量：

$$ g_t=[g_{t,1},g_{t,2},\cdots,g_{t,n}] $$

其中：

$$ g_{t,i}=\frac{\partial L}{\partial \omega_{t,i}} $$

---

## 1. 普通梯度下降：只看当前梯度

普通梯度下降公式：

$$ \omega_{t+1}=\omega_t-\eta g_t $$

它的逻辑是：

1. 当前参数是 $\omega_t$；
2. 计算当前梯度 $g_t$；
3. 梯度方向是 loss 上升最快方向；
4. 沿着负梯度方向更新；
5. 更新步长由学习率 $\eta$ 控制。

### 1.1 单参数情况

如果只有一个参数 $\omega$：

$$ \omega_{t+1} = \omega_t - \eta \frac{\partial L}{\partial \omega_t} $$

如果：

$$ \frac{\partial L}{\partial \omega_t}>0 $$

说明增大 $\omega$ 会让 loss 增大，所以应该减小 $\omega$。

如果：

$$ \frac{\partial L}{\partial \omega_t}<0 $$

说明增大 $\omega$ 会让 loss 减小，所以应该增大 $\omega$。

### 1.2 多参数情况

如果参数为：

$$ \omega=[\omega_1,\omega_2] $$

则梯度为：

$$ g_t= \left[ \frac{\partial L}{\partial \omega_{t,1}}, \frac{\partial L}{\partial \omega_{t,2}} \right] $$

对应更新为：

$$ \omega_{t+1,1} = \omega_{t,1} - \eta \frac{\partial L}{\partial \omega_{t,1}} $$

$$ \omega_{t+1,2} = \omega_{t,2} - \eta \frac{\partial L}{\partial \omega_{t,2}} $$

也就是每个参数都沿着自己的负梯度方向走一步。

---

## 2. 普通梯度下降的问题

### 2.1 所有参数共用同一个学习率

普通梯度下降中：

$$ \omega_{t+1,i} = \omega_{t,i} - \eta g_{t,i} $$

所有参数都使用同一个学习率 $\eta$。

但真实训练中，不同参数的梯度尺度可能差别很大：

- 有些参数梯度很大，需要小步更新；
- 有些参数梯度很小，需要保留较大更新；
- 一个统一学习率很难同时适配所有参数。

### 2.2 只看当前梯度，容易震荡

如果 loss 曲面像狭长山谷，梯度可能表现为：

```text
水平方向：长期一致
竖直方向：上下来回震荡
```

普通梯度下降每次只看当前梯度，会导致参数路径在山谷两侧来回摆动，收敛速度变慢。

### 2.3 学习率很难调

学习率过大：

```text
可能越过最优点，导致 loss 爆炸或不收敛。
```

学习率过小：

```text
训练速度慢，长期停留在高 loss 区域。
```

后续优化算法基本都围绕这些问题改进：

```text
1. 方向如何更稳定？
2. 每个参数的步长如何自适应？
3. 历史梯度如何影响当前更新？
4. 训练初期和后期如何更稳定？
```

---

## 3. Momentum：利用历史方向减少震荡、加速稳定方向

Momentum 的核心思想是：

> 当前梯度可能抖动，但历史梯度方向可以提供更稳定的运动趋势。

普通梯度下降：

$$ \omega_{t+1} = \omega_t - \eta g_t $$

Momentum 引入速度变量 $v_t$：

$$ v_t = \beta v_{t-1} + g_t $$

然后更新参数：

$$ \omega_{t+1} = \omega_t - \eta v_t $$

其中：

| 符号    | 含义                    |
| ------- | ----------------------- |
| $v_t$   | 动量 / 历史梯度方向累计 |
| $\beta$ | 动量系数，常用 $0.9$    |
| $g_t$   | 当前梯度                |
| $\eta$  | 学习率                  |

有些资料会写成：

$$ v_t = \beta v_{t-1} + (1-\beta)g_t $$

这是指数滑动平均写法。两种写法思想一致，区别在于是否把 $(1-\beta)$ 放进动量定义里。

### 3.1 Momentum 的展开形式

从：

$$ v_t = \beta v_{t-1} + g_t $$

不断展开：

$$ v_t = g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \beta^3 g_{t-3} +\cdots $$

说明 $v_t$ 是历史梯度的加权和。越新的梯度权重大，越旧的梯度权重小。

### 3.2 为什么稳定方向会加速？

假设某个方向的梯度长期一致：

$$ g_t=a $$

那么：

$$ v_t = a+\beta a+\beta^2 a+\cdots $$

当 $t$ 足够大时：

$$ v_t \approx \frac{a}{1-\beta} $$

如果：

$$ \beta=0.9 $$

则：

$$ \frac{1}{1-0.9}=10 $$

这说明长期一致的方向会被放大，大约加速到 10 倍。

### 3.3 为什么震荡方向会被削弱？

如果某个方向的梯度反复变化：

$$ g_1=b,\quad g_2=-b,\quad g_3=b,\quad g_4=-b $$

则动量累积中会出现：

$$ b-\beta b+\beta^2 b-\beta^3 b+\cdots $$

正负方向会互相抵消。

所以 Momentum 的效果是：

```text
长期一致的方向：加速
反复震荡的方向：抵消
```

---

## 4. AdaGrad：让每个参数拥有自己的学习率

AdaGrad 解决的问题是：

> 不同参数的梯度尺度不同，因此不同参数应该拥有不同的有效学习率。

AdaGrad 维护历史梯度平方和：

$$ G_t = G_{t-1} + g_t\odot g_t $$

然后更新：

$$ \omega_{t+1} = \omega_t - \frac{\eta}{\sqrt{G_t}+\epsilon} \odot g_t $$

其中：

| 符号       | 含义             |
| ---------- | ---------------- |
| $G_t$      | 历史梯度平方累计 |
| $\odot$    | 逐元素乘法       |
| $\epsilon$ | 防止除零的小常数 |

### 4.1 单个参数形式

对第 $i$ 个参数：

$$ G_{t,i} = G_{t-1,i} + g_{t,i}^2 $$

$$ \omega_{t+1,i} = \omega_{t,i} - \frac{\eta}{\sqrt{G_{t,i}}+\epsilon} g_{t,i} $$

定义有效学习率：

$$ \eta_{t,i} = \frac{\eta}{\sqrt{G_{t,i}}+\epsilon} $$

于是：

$$ \omega_{t+1,i} = \omega_{t,i} - \eta_{t,i}g_{t,i} $$

所以 AdaGrad 本质上是在给每个参数分配自己的动态学习率。

### 4.2 AdaGrad 的直觉

如果某个参数历史梯度很大：

$$ G_{t,i}\text{ 大} $$

则：

$$ \eta_{t,i}\text{ 小} $$

这个参数后续更新会变谨慎。

如果某个参数历史梯度很小：

$$ G_{t,i}\text{ 小} $$

则：

$$ \eta_{t,i}\text{ 相对较大} $$

这个参数仍然可以继续学习。

### 4.3 AdaGrad 为什么使用平方梯度？

梯度有正有负。如果直接累加：

$$ g_1+g_2+\cdots $$

可能正负抵消。例如：

$$ 10+(-10)=0 $$

这并不说明该方向不重要，只说明梯度方向发生过变化。

使用平方：

$$ g_t^2 $$

可以记录梯度大小，不受正负号影响。

### 4.4 AdaGrad 的缺点

AdaGrad 的累计项：

$$ G_t = G_{t-1} + g_t^2 $$

只会越来越大。因此有效学习率：

$$ \frac{\eta}{\sqrt{G_t}+\epsilon} $$

会越来越小。

训练后期可能出现：

```text
有效学习率过小
参数几乎不再更新
模型还未充分收敛，训练已经走不动
```

这就是 RMSProp 要解决的问题。

---

## 5. RMSProp：只记近期梯度平方，避免 AdaGrad 后期步长过小

RMSProp 可以看作 AdaGrad 的改进。

AdaGrad 使用完整历史累计：

$$ G_t = G_{t-1} + g_t^2 $$

RMSProp 改成指数滑动平均：

$$ s_t = \rho s_{t-1} + (1-\rho)g_t\odot g_t $$

然后更新：

$$ \omega_{t+1} = \omega_t - \frac{\eta}{\sqrt{s_t}+\epsilon} \odot g_t $$

其中：

| 符号    | 含义                           |
| ------- | ------------------------------ |
| $s_t$   | 梯度平方的指数滑动平均         |
| $\rho$  | 衰减系数，常用 $0.9$ 或 $0.99$ |
| $g_t^2$ | 当前梯度平方                   |

### 5.1 RMSProp 的展开形式

从：

$$ s_t = \rho s_{t-1} + (1-\rho)g_t^2 $$

展开得到：

$$ s_t = (1-\rho)g_t^2 + \rho(1-\rho)g_{t-1}^2 + \rho^2(1-\rho)g_{t-2}^2 +\cdots $$

所以 RMSProp 更重视近期梯度平方，久远梯度的影响会指数衰减。

### 5.2 RMSProp 与 AdaGrad 的核心区别

AdaGrad：

$$ G_t = g_1^2+g_2^2+\cdots+g_t^2 $$

RMSProp：

$$ s_t = (1-\rho) \sum_{\tau=1}^{t} \rho^{t-\tau} g_\tau^2 $$

区别可以概括为：

```text
AdaGrad：所有历史梯度平方都记住
RMSProp：近期梯度平方更重要，久远历史逐渐遗忘
```

### 5.3 RMSProp 的优点

RMSProp 避免了 AdaGrad 分母无限增长的问题。如果最近梯度变小，$s_t$ 也会下降，有效学习率有机会恢复。因此 RMSProp 更适合非平稳的深度学习训练过程。

### 5.4 RMSProp 的问题

RMSProp 的有效学习率是：

$$ \eta_t = \frac{\eta}{\sqrt{s_t}+\epsilon} $$

如果 $s_t$ 下降，分母变小，有效学习率会上升。这在一些情况下是合理的，但也可能造成训练不稳定。

---

## 6. Adam：Momentum + RMSProp + 偏差修正

Adam 全称为：

$$ \text{Adaptive Moment Estimation} $$

它结合了两个核心思想：

```text
Momentum：用历史梯度平均稳定方向
RMSProp：用历史梯度平方平均调整步长
```

Adam 维护两个变量。

一阶动量：

$$ m_t = \beta_1m_{t-1} + (1-\beta_1)g_t $$

二阶动量：

$$ v_t = \beta_2v_{t-1} + (1-\beta_2)g_t\odot g_t $$

偏差修正：

$$ \hat{m}_t = \frac{m_t}{1-\beta_1^t} $$

$$ \hat{v}_t = \frac{v_t}{1-\beta_2^t} $$

参数更新：

$$ \omega_{t+1} = \omega_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} $$

---

## 7. Adam 每一项的含义

### 7.1 当前梯度 $g_t$

$$ g_t = \nabla_\omega L(\omega_t) $$

表示当前 batch 下 loss 对参数的瞬时梯度。

### 7.2 一阶动量 $m_t$

$$ m_t = \beta_1m_{t-1} + (1-\beta_1)g_t $$

它是梯度的指数滑动平均。作用是平滑当前梯度、减少方向抖动、让长期一致的方向更稳定。

如果：

$$ \beta_1=0.9 $$

则：

$$ m_t = 0.9m_{t-1} + 0.1g_t $$

### 7.3 二阶动量 $v_t$

$$ v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2 $$

它是梯度平方的指数滑动平均。作用是估计每个参数最近的梯度尺度，梯度大的方向缩小步长，梯度小的方向保留更新。

如果：

$$ \beta_2=0.999 $$

则：

$$ v_t = 0.999v_{t-1} + 0.001g_t^2 $$

### 7.4 偏差修正 $\hat{m}_t$、$\hat{v}_t$

Adam 初始化时通常令：

$$ m_0=0,\quad v_0=0 $$

这会导致训练初期 $m_t$ 和 $v_t$ 被 0 明显拉低。

所以需要偏差修正：

$$ \hat{m}_t = \frac{m_t}{1-\beta_1^t} $$

$$ \hat{v}_t = \frac{v_t}{1-\beta_2^t} $$

偏差修正的本质：

```text
指数滑动平均从 0 开始，训练初期会被 0 拉低。
理论上第 t 步会少乘一个 1 - β^t。
所以除以 1 - β^t，可以把估计值放大回合理尺度。
```

---

## 8. Adam 偏差修正的数学推导

以一阶动量为例：

$$ m_t = \beta_1m_{t-1} + (1-\beta_1)g_t $$

并且：

$$ m_0=0 $$

递推展开：

$$ m_t = (1-\beta_1) \sum_{\tau=1}^{t} \beta_1^{t-\tau}g_\tau $$

假设每一步梯度的期望恒定为：

$$ E[g_\tau]=\mu $$

两边取期望：

$$ E[m_t] = E\left[ (1-\beta_1) \sum_{\tau=1}^{t} \beta_1^{t-\tau}g_\tau \right] $$

利用期望的线性性质：

$$ E[m_t] = (1-\beta_1) \sum_{\tau=1}^{t} \beta_1^{t-\tau}E[g_\tau] $$

代入：

$$ E[g_\tau]=\mu $$

得到：

$$ E[m_t] = (1-\beta_1) \sum_{\tau=1}^{t} \beta_1^{t-\tau}\mu $$

把 $\mu$ 提出来：

$$ E[m_t] = (1-\beta_1)\mu \sum_{\tau=1}^{t} \beta_1^{t-\tau} $$

观察求和项：

$$ \sum_{\tau=1}^{t} \beta_1^{t-\tau} = 1+\beta_1+\beta_1^2+\cdots+\beta_1^{t-1} $$

这是等比数列：

$$ 1+\beta_1+\beta_1^2+\cdots+\beta_1^{t-1} = \frac{1-\beta_1^t}{1-\beta_1} $$

代回：

$$ E[m_t] = (1-\beta_1)\mu \frac{1-\beta_1^t}{1-\beta_1} $$

约掉 $(1-\beta_1)$：

$$ E[m_t] = (1-\beta_1^t)\mu $$

这说明 $m_t$ 的期望比真实均值 $\mu$ 少了一个系数：

$$ 1-\beta_1^t $$

所以修正为：

$$ \hat{m}_t = \frac{m_t}{1-\beta_1^t} $$

于是：

$$ E[\hat{m}_t] = \mu $$

二阶动量同理：

$$ v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2 $$

所以：

$$ \hat{v}_t = \frac{v_t}{1-\beta_2^t} $$

---

## 9. 用数字理解偏差修正

设：

$$ \beta_1=0.9 $$

且每一步梯度都等于：

$$ g_t=10 $$

真实平均梯度为：

$$ \mu=10 $$

第 1 步：

$$ m_1 = 0.1\times 10 = 1 $$

明显比 10 小。

偏差修正：

$$ \hat{m}_1 = \frac{m_1}{1-0.9^1} = \frac{1}{0.1} = 10 $$

第 2 步：

$$ m_2 = 0.9\times1 + 0.1\times10 = 1.9 $$

偏差修正：

$$ \hat{m}_2 = \frac{1.9}{1-0.9^2} = \frac{1.9}{0.19} = 10 $$

这说明偏差修正能够把前期被 0 压小的估计恢复到合理尺度。

---

## 10. 为什么 $v_t$ 的偏差更严重？

Adam 常用：

$$ \beta_2=0.999 $$

第 1 步：

$$ 1-\beta_2^1 = 0.001 $$

说明未修正的 $v_1$ 只有真实二阶矩的千分之一。

第 100 步：

$$ 1-0.999^{100} \approx 0.095 $$

也只有真实二阶矩的约 $9.5\%$。

因此二阶动量如果不做修正，训练初期会严重偏小。

Adam 同时修正 $m_t$ 和 $v_t$，是为了让前期的方向估计和尺度估计都更加合理。

---

## 11. Adam 的直觉总结

Adam 更新公式：

$$ \omega_{t+1} = \omega_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} $$

可以理解为：

```text
分子 \hat{m}_t：
告诉模型应该往哪个方向走。

分母 \sqrt{\hat{v}_t}：
告诉模型这个方向最近波动大不大，步子应该多大。

偏差修正：
解决训练初期 m_t 和 v_t 从 0 开始导致估计偏小的问题。
```

一句话概括：

```text
Adam 用一阶动量稳定方向，用二阶动量自适应步长，用偏差修正解决冷启动偏差。
```

---

## 12. AdamW：现代深度学习更常用的 Adam 变体

Adam 的原始 weight decay 与自适应学习率耦合在一起，可能影响泛化。

AdamW 的核心改进是：

> 把 weight decay 从 Adam 的自适应梯度更新中解耦出来。

AdamW 可以简化理解为：

$$ \omega_{t+1} = \omega_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} - \eta\lambda\omega_t $$

其中：

- $\lambda$：weight decay 系数；
- 前一项：Adam 的自适应梯度更新；
- 后一项：独立的权重衰减。

### 12.1 AdamW 的优势

AdamW 在现代深度学习中非常常用，尤其适合：

```text
Transformer
ViT
CLIP
多模态模型
扩散模型
AIGC 检测模型
预训练模型微调
```

实践中常用组合：

```text
AdamW + warmup + cosine decay
```

---

## 13. AMSGrad：防止 Adam 有效学习率突然变大

Adam 和 RMSProp 都有一个潜在问题：

$$ v_t $$

是指数滑动平均，可能变小。

如果 $v_t$ 变小，那么：

$$ \sqrt{v_t} $$

变小，于是有效学习率：

$$ \frac{\eta}{\sqrt{v_t}+\epsilon} $$

会变大。

AMSGrad 的解决方法是维护历史最大二阶动量：

$$ \tilde{v}_t = \max(\tilde{v}_{t-1},v_t) $$

其中 max 是逐元素最大。

然后更新：

$$ \omega_{t+1} = \omega_t - \eta \frac{\hat{m}_t}{\sqrt{\tilde{v}_t}+\epsilon} $$

这样 $\tilde{v}_t$ 不会下降，分母不会突然变小，有效学习率不会突然变大。

---

## 14. 各优化器之间的主线关系

```text
SGD
│
├── Momentum
│   └── 解决梯度方向震荡问题
│
├── AdaGrad
│   └── 解决每个参数学习率相同的问题
│
├── RMSProp
│   └── 解决 AdaGrad 历史累计过大、后期学习率过小的问题
│
├── Adam
│   └── Momentum + RMSProp + 偏差修正
│
├── AdamW
│   └── Adam + 解耦 weight decay
│
└── AMSGrad
    └── Adam + 防止二阶动量下降导致有效学习率突然增大
```

---

## 15. 公式总表

| 优化器   | 核心变量     | 关键公式                              | 核心思想                     |
| -------- | ------------ | ------------------------------------- | ---------------------------- |
| SGD      | 无           | $\omega_{t+1}=\omega_t-\eta g_t$      | 沿当前负梯度更新             |
| Momentum | $v_t$        | $v_t=\beta v_{t-1}+g_t$               | 累积历史方向，减少震荡       |
| AdaGrad  | $G_t$        | $G_t=G_{t-1}+g_t^2$                   | 历史梯度越大，后续学习率越小 |
| RMSProp  | $s_t$        | $s_t=\rho s_{t-1}+(1-\rho)g_t^2$      | 用近期梯度平方调节学习率     |
| Adam     | $m_t,v_t$    | 一阶动量 + 二阶动量 + 偏差修正        | 稳定方向 + 自适应步长        |
| AdamW    | $m_t,v_t$    | Adam + 解耦权重衰减                   | 更适合现代深度学习           |
| AMSGrad  | $\tilde v_t$ | $\tilde v_t=\max(\tilde v_{t-1},v_t)$ | 防止有效学习率突然变大       |

---

## 16. 从几何角度理解优化器

所有优化器都可以抽象为：

$$ \omega_{t+1} = \omega_t - \eta P_t g_t $$

其中 $P_t$ 可以理解为“梯度预处理器”。

不同优化器的本质区别就在于如何构造 $P_t$。

### 16.1 SGD

$$ P_t=I $$

直接使用当前梯度。

### 16.2 Momentum

Momentum 用历史梯度方向共同决定当前更新方向，主要改变方向稳定性。

### 16.3 AdaGrad / RMSProp / Adam

它们使用类似：

$$ P_t = \frac{1}{\sqrt{\text{历史梯度平方}}+\epsilon} $$

的形式，主要改变每个参数的有效学习率。

### 16.4 AdamW

AdamW 在 Adam 的基础上额外加入解耦权重衰减，使正则化更加清晰。

---

## 17. 优化器选择建议

### 17.1 通用深度学习任务

优先使用：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)
```

适合：

```text
Transformer
ViT
CLIP
多模态模型
AIGC 检测
扩散模型
预训练模型微调
```

### 17.2 CNN 图像分类任务

可以比较：

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True
)
```

SGD + Momentum 有时泛化更好，但学习率和训练轮数更难调。

### 17.3 微调预训练 backbone

推荐使用分组学习率：

```python
optimizer = torch.optim.AdamW([
    {"params": backbone.parameters(), "lr": 1e-5},
    {"params": classifier.parameters(), "lr": 1e-4},
], weight_decay=0.01)
```

原因：

```text
backbone 已有预训练知识，学习率小一点；
classifier 是新加的分类头，学习率大一点。
```

### 17.4 优化器要和 scheduler 一起看

常见现代组合：

```text
AdamW + warmup + cosine decay
```

含义：

```text
warmup：
训练初期学习率从小逐渐升高，防止一开始更新过猛。

cosine decay：
训练中后期学习率逐渐下降，让模型稳定收敛。
```

---

## 18. PyTorch 代码对应

### 18.1 SGD

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
```

### 18.2 SGD + Momentum

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)
```

### 18.3 AdaGrad

```python
optimizer = torch.optim.Adagrad(
    model.parameters(),
    lr=0.01
)
```

### 18.4 RMSProp

```python
optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=0.001,
    alpha=0.99,
    eps=1e-8
)
```

其中 `alpha` 对应公式中的 $\rho$。

### 18.5 Adam

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

### 18.6 AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
```

### 18.7 AdamW + AMSGrad

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    amsgrad=True
)
```

---

## 19. 一句话总结每个优化器

### SGD

```text
只看当前梯度，简单直接，但容易震荡，学习率难调。
```

### Momentum

```text
把历史梯度方向累积成动量，稳定方向加速，震荡方向抵消。
```

### AdaGrad

```text
每个参数根据历史梯度平方调整学习率，历史梯度越大，后续学习率越小。
```

### RMSProp

```text
用近期梯度平方的滑动平均调整学习率，避免 AdaGrad 后期学习率过小。
```

### Adam

```text
用 Momentum 管方向，用 RMSProp 管步长，再用偏差修正解决训练初期估计偏小。
```

### AdamW

```text
在 Adam 基础上解耦 weight decay，是现代深度学习最常用的默认优化器之一。
```

### AMSGrad

```text
限制二阶动量不下降，防止 Adam 的有效学习率突然增大。
```

---

## 20. 最终主线记忆

优化器的发展可以记成一句话：

```text
SGD 只看当前梯度；
Momentum 加入历史方向；
AdaGrad 给每个参数自适应学习率；
RMSProp 让 AdaGrad 忘掉太久远的历史；
Adam 把 Momentum 和 RMSProp 结合；
AdamW 让 weight decay 更合理；
AMSGrad 让 Adam 的有效学习率更稳定。
```

最重要的理解是：

```text
优化器不是魔法，它本质上在决定四件事：

1. 往哪个方向走；
2. 每个方向走多大；
3. 历史梯度如何影响现在；
4. 如何避免训练初期和后期的不稳定。
```

实际深度学习中，最稳妥的默认组合是：

```text
AdamW + warmup + cosine decay
```

经典 CNN 分类任务中，SGD + Momentum 仍然值得作为强基线。

严谨实验不应该只比较优化器名字，还应该同时比较：

```text
学习率
weight decay
scheduler
batch size
训练轮数
数据增强
冻结策略
分组学习率
```
