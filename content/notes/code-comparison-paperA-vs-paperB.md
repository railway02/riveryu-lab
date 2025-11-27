+++
date = '2025-11-27T19:40:47+08:00'
draft = true
math = true
title = 'e4e 与 HairCLIPv2 的反演解码机制和代码对比研究'
+++

---
# e4e 与 HairCLIPv2 的反演解码机制和代码对比研究

## 1. 引言与研究背景

**StyleGAN 与图像反演概述：**
近年来，以 StyleGAN 系列为代表的生成对抗网络（GAN）在高保真图像生成方面取得了突破性进展。StyleGAN 引入了中间潜码空间 $W$（及其扩展 $W^+$），使生成器能够在不同层次上控制图像属性，并展现出一定的属性解耦能力。为了对真实图像进行编辑，首先需要将图像“反演”到生成器的潜码空间，即找到一个 $w \in W^+$，使预训练 StyleGAN 生成器 $G(w)$ 尽可能重构输入图像。高质量的反演需要在重建精度和编辑潜力之间取得平衡。早期的方法（如 pSp）直接训练编码器输出 $W^+$ 代码，追求重建质量；而 e4e（*encoder for editing*）则强调编辑友好性，允许适度牺牲重建以换取潜码落在更可编辑的空间区域。

**图像编辑背景：**
有了反演的潜码，我们即可在潜空间施加操作实现人脸属性编辑。例如改变年龄、表情、发型等，都可通过对潜码施加方向移动或替换某些分量实现。然而，对于局部且精细的属性（如头发）编辑，仅依赖潜码操作可能导致无关区域受影响，因为 $W^+$ 空间的语义解耦有限。HairCLIP 是首个将 CLIP 引入发型编辑的工作，支持以文本或参考图像驱动的发型/发色编辑，但它不支持手绘草图或遮罩等精细交互，且可能无法完美保持身份和背景。HairCLIPv2 在此背景下提出，旨在统一多种交互方式（文本、遮罩、草图、参考图像等）进行头发编辑，并在保持身份和背景等无关属性方面优于 HairCLIP。其核心思想是将所有头发编辑转化为“发型迁移”问题，通过代理特征混合在特定层的特征空间融合来自不同来源的发型/发色信息，从而实现编辑效果。

本文将对比 e4e 与 HairCLIPv2 的反演解码机制和代码实现细节。首先介绍 e4e 方法及其代码结构，然后分析 HairCLIPv2 对生成器的改动和特征混合机制，最后总结两者在解码一致性上的机制差异，并分享当前阶段的理解与实验情况。

---

## 2. e4e 方法概述与代码结构分析

### 2.1 e4e 方法动机

e4e（Encoder for Editing）由 Tov 等人在 2021 年提出，目标是设计一个更适合编辑的 StyleGAN 编码器。作者发现直接优化重建会导致潜码落入难以编辑的区域，因此 e4e 在训练中引入了失真–可编辑性权衡，让编码器输出略偏离精确重建所需的潜码，以确保后续编辑操作平滑有效。换言之，e4e 比 Pixel2Style2Pixel (pSp) 更注重潜码的语义对齐和编辑友好性。

### 2.2 模型结构与关键文件

e4e 基于 pSp 框架实现，其代码结构主要包含 encoder 和 decoder 两部分。以官方实现的 `psp.py` 为例：

* `pSp` 类继承自 `nn.Module`，在初始化时构建编码器和解码器两个子模块；
* 编码器通常采用预训练人脸识别网络（如 IR-SE50）作为骨干，通过渐进式样式层提取不同尺度的特征并映射到多层风格向量（即 $W^+$ 空间）；
* 代码中根据 `opts.encoder_type` 切换不同编码器实现，例如 `GradualStyleEncoder` 和 `Encoder4Editing`，后者即对应 e4e 定制的编码器架构。

解码器则直接使用预训练的 StyleGAN2 生成器。在构造 `pSp` 时，代码通过：

```python
self.decoder = Generator(opts.stylegan_size, 512, 8, channel_multiplier=2)
```

实例化生成器。随后在 `load_weights()` 中加载预训练权重：

* 若提供了 e4e 训练的 checkpoint，则分别加载编码器和解码器权重；
* 否则，用预训练的 IR-SE50 初始化编码器，用预训练 StyleGAN2 的 `g_ema` 权重初始化生成器。

因此，e4e 的解码器本质上就是一个冻结的 StyleGAN2 模型，使编码器学习到的潜码能够还原人脸图像。

### 2.3 反演与解码流程

e4e 推理时，输入一张人脸图像 $I_{in}$，编码器 $E$ 提取其多尺度特征输出 $W^+$ 潜码（一般是 $18 \times 512$ 维，对应 StyleGAN2 不同层的风格参数）。`pSp.forward` 函数代码如下：

```python
def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):
        if input_code:
            codes = x
            #已经是 W+ latent
        else:
            codes = self.encoder(x)
            #如果不是，就反演：E(x) -> W+
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if codes.ndim == 2:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
            #加上平均latent，使其更接近“正常人脸”分布
        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code
        
        #进行解码，用Stylegan2 Generator 处理 codes
        images, result_latent = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images
```

解释：

* `input_code=False`：输入为图像，需要经过编码器；
* `input_code=True`：输入已经是潜码，跳过编码器；
* 编码器输出的 $W^+$ 通常加上平均潜码 `latent_avg`，使其更接近“正常”分布；
* 最后调用 StyleGAN 生成器 `decoder(...)` 生成图像。

在 e4e 场景下，`decoder([codes], input_is_latent=True)` 直接将编码器算出的风格码送入生成器。StyleGAN2 的 PyTorch 实现（`stylegan2/model.py`）会根据 `input_is_latent` 判断：

* 若为 `True`：跳过映射网络，将输入视为 $W$ 空间向量；
* 若为 `False`：先通过 8 层全连接映射网络，将高斯噪声 $z$ 映射到 $W$。

下面是 StyleGAN2 生成器前向的关键片段（简化示意非完整代码）：

```python
# StyleGAN2 Generator.forward (简化示意)

if not input_is_latent:
    styles = [self.style(s) for s in styles]  # 映射网络：z -> w

# 整理成 (batch, n_latent, 512) 的 latent
if len(styles) < 2:
    inject_index = self.n_latent
    latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)  # 复制同一风格到 18 层
else:
    # 两种风格混合的情况，略
    ...

# 第 0 层：4×4 常数输入 + 第一层卷积
out = self.input(latent)  # 常数映射为特征图
out = self.conv1(out, latent[:, 0], noise=noise[0])
skip = self.to_rgb1(out, latent[:, 1])  # 初始 RGB

# 后续每两个风格码对应一组 conv1 + conv2
i = 1
for conv1, conv2, noise1, noise2, to_rgb in ...:
    out = conv1(out, latent[:, i],   noise=noise1)
    out = conv2(out, latent[:, i+1], noise=noise2)
    skip = to_rgb(out, latent[:, i+2], skip)  # 累积 RGB
    i += 2

image = skip
return image, latent
```

在 e4e 中，我们通过编码器得到的 $W^+$ 向量本身就是 18 组风格码，所以整个生成流程相当于**直接利用预训练生成器解码**。e4e 使用 L2 损失、LPIPS 感知损失等监督训练编码器，使 $G(E(I_{in}))$ 尽可能重构输入，同时保持潜码的可编辑性。

实际效果：e4e 重建的图像视觉上与原图接近，但细节可能略有模糊或身份有轻微变化，这是为增强潜码可编辑性付出的代价。

### 2.4 代码结构小结

* `psp.py`：整体模型（编码器+解码器）与前向逻辑；
* `models/encoders/`：`GradualStyleEncoder`、`Encoder4Editing` 等具体编码器；
* `models/stylegan2/model.py`：StyleGAN2 生成器/判别器。

e4e 完全复用预训练 StyleGAN2 的架构和权重，在反演阶段**不修改解码器**，只学习编码器。这为后续像 HairCLIP/HairCLIPv2 这样的工作提供了基础。

---

## 3. HairCLIPv2 方法与代码实现

### 3.1 任务目标与整体框架

HairCLIPv2 (Wei et al., ICCV 2023) 针对“头发编辑”提出了一个统一框架，支持多种输入形式（文本描述、参考图像、用户绘制的发型/发色遮罩或草图等）来编辑人像的发型和发色。相比初代 HairCLIP：

* HairCLIP：主要是文本/图像驱动的**全局**发型/发色编辑；
* HairCLIPv2：兼顾

  * 文本等高层语义控制；
  * mask/草图等精细局部控制；
  * 更好地保持身份、表情和背景。

核心思想：**将所有头发编辑转化为“发型迁移”**。无论输入控制是什么，最终都生成一个“代理发型”特征（proxy），再将这个代理特征与源图像的头发特征在中间特征空间中混合，实现“把代理头发移植到源人脸上”。

在实现层面：

* 仍然使用预训练 StyleGAN2 作为解码器；
* 借助 e4e/pSp 获取源图像的初始 $W^+$ 潜码；
* 对生成器的 `forward` 做了扩展，支持**分段解码**和**中间特征插入**；
* 在 `feature_blending.py` 中实现 FS 特征混合（$F$ + hairmask）。

### 3.2 生成器的分层解码修改（model.py）

在 HairCLIPv2 中，`models/stylegan2/haircilpv2_model.py` 对 StyleGAN2 的 `Generator` 做了如下改动：

* 新增参数：

  * `start_layer`：从第几层开始解码；
  * `end_layer`：解码到第几层提前返回；
  * `layer_in`：当从中间层开始时，外部提供的特征；
* 前向 `forward` 加入了一些 `if/elif` 分支来支持“只跑前几层”或“从中间层开始接着跑”。

简化后的前向逻辑大致如下：

```python
def forward(
            self,
            styles,
            return_latents=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
            layer_in=None, #从中间层输入的 feature
            skip=None,#中间的 RGB skip
            start_layer=0,#从第几层开始跑
            end_layer=8,#在第几层结束并返回
            return_rgb=False,

    ):

        ......

        out = self.input(latent)# 初始常数输入
        
        if start_layer == 0:
            out = self.conv1(out, latent[:, 0], noise=noise[0])  # 0th layer
            skip = self.to_rgb1(out, latent[:, 1])
        if end_layer == 0:
            return out, skip
        i = 1
        current_layer = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            #FS 空间特征替换再解码”
            if current_layer < start_layer:
                pass #没有到达起始层，就pass，不做
            elif current_layer == start_layer: 
                #用传进来的 layer_in 作为输入，从中间层“接上”
                out = conv1(layer_in, latent[:, i], noise=noise1)
                out = conv2(out, latent[:, i + 1], noise=noise2)
                skip = to_rgb(out, latent[:, i + 2], skip)
            elif current_layer > end_layer:
                return out, skip # 跑到 end_layer 就直接返回
            else:
                out = conv1(out, latent[:, i], noise=noise1)
                out = conv2(out, latent[:, i + 1], noise=noise2)
                skip = to_rgb(out, latent[:, i + 2], skip)
            current_layer += 1
            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None
```

通过这些参数，生成器支持两种“模式”：

1. **0 → k 层**：只运行前半段，拿中间特征（例如 F7/F14）；
2. **k → N 层**：从中间层开始，给定 `layer_in`，接着往后生成完整图像。

这就把原来的“从头到尾一次解码”变成了“前半段 + 后半段”的可组合结构。

### 3.3 新增工具函数（三个 helper）

在 `haircilpv2_model.py` 中，围绕这个“可拆分式 Generator”，定义了几个常用的 helper 函数（函数名可能略有出入，这里是含义）：

1. `generate_im_from_w_space(code)`

   * 输入：$W^+$ 潜码（如 e4e 得到的）；
   * 功能：调用生成器从第 0 层跑到最后，生成完整图像；
   * 作用：相当于封装了“用 StyleGAN2 解码”的流程，是 e4e 解码方式的直接复用。

2. `generate_initial_intermediate(code)`

   * 输入：一个潜码；
   * 功能：只跑前几层（如 0–3 层），返回中间特征（例如 F7，对应约 32×32）；
   * 作用：用于获取“源图像 / 代理图像”在某一层的特征表示，为后续 FS 特征混合做准备。

3. `update_on_FS(code, initial_intermediate, initial_F, initial_S)`

   * 输入：

     * `code`：新的潜码（比如经过 CLIP 优化后的潜码）；
     * `initial_intermediate`：源图像在 0–3 层的原始特征 $F_{\text{src}}$；
     * `initial_F`：已经混合（blending）过代理发型后的目标特征 $F_{\text{blend}}$；
     * `initial_S`：源图像的原始 $W^+$ 潜码（风格码）。

   * 步骤：

     1. 用 `code` 通过 0–3 层生成特征 `intermediate`；
     2. 计算差分：
        $$
        \text{difference} = initial\_F - initial\_intermediate
        $$
     3. 叠加到新特征上：
        $$
        new_intermediate = intermediate + difference
        $$
     4. 以 `new_intermediate` 为中间输入、`initial_S` 为后续风格，从第 4 层继续生成图像。

   * 作用：在 FS（特征空间）里用“差分”的方式，把已经计算好的“发型变化”迁移到新潜码对应的特征上，同时后半段仍然使用源潜码来保证脸和背景不变。

### 3.4 feature_blending.py：FS + hairmask 的中间特征编辑

真正进行特征混合的是 `feature_blending.py`，其中主要逻辑是：

1. 准备几个潜码：

   * $w_{\text{src}}$：源图像潜码；
   * $w_{\text{bald}}$：秃头代理潜码（去头发的“光头”人脸）；
   * $w_{\text{global}}$：目标发型代理潜码（由文本/参考图像得到）。

2. 用生成器（前半段）计算这些潜码的中间特征（例如第 3 层）：

   * $F_{\text{src}}$、$F_{\text{bald}}$、$F_{\text{global}}$。

3. 使用预训练人脸分割网络 `seg`，对源图像和代理图像进行语义分割，得到：

   * 源图像的“头发+耳朵 mask”：$M_{\text{bald}}$；
   * 代理发型图像的“头发 mask”：$M_{\text{hair}}$。

   将这些 mask 下采样到与特征 $F$ 相同的分辨率（如 32×32）。

4. 在特征空间做两步混合：

```python
def hairstyle_feature_blending(generator, seg,
                               src_latent, src_feature,
                               visual_mask,
                               latent_bald, latent_global=None,
                               latent_local=None, local_blending_mask=None):

    #  用 Generator 前几层算各种 F7
    bald_feature,   _ = generator([latent_bald],   ..., start_layer=0, end_layer=3)
    global_feature, _ = generator([latent_global], ..., start_layer=0, end_layer=3)
    # 这里的 bald/global_feature 就是光头 proxy、文本 proxy 的 F7

    #  根据分割图算出耳朵+头发区域的二维 mask -> 下采样到 32×32
    bald_blending_mask_down = F.interpolate(..., size=(32,32))
    global_hair_mask_down   = F.interpolate(..., size=(32,32))

    #  在 32×32 的 F7 空间线性混合
    src_feature = bald_feature   * bald_blending_mask_down + src_feature * (1 - bald_blending_mask_down)
    src_feature = global_feature * global_hair_mask_down   + src_feature * (1 - global_hair_mask_down)

    #  再用（可能有的）局部 mask 做局部混合
    if latent_local is not None:
        local_feature,_ = generator([latent_local], ..., start_layer=0, end_layer=3)
        local_blending_mask_down = ... -> size(32,32)
        src_feature = local_feature * local_blending_mask_down + src_feature * (1-local_blending_mask_down)

    #  把混好的 src_feature 丢回同一个 Generator，从中间层继续跑到最后
    img_gen_blend, _ = generator([src_latent], input_is_latent=True,
                                 randomize_noise=False,
                                 start_layer=4, end_layer=8,
                                 layer_in=src_feature)
    return src_feature, img_gen_blend
```

5. 得到混合后的特征 `src_feature`（也就是 `initial_F`），再配合 `update_on_FS` / 生成器后半段进行解码，即可生成“源人脸 + 目标发型”的最终图像。

对于发色编辑，思路类似，只不过在更高分辨率的中间层（如 64×64 或 128×128，对应 F14）进行特征混合，从而只修改颜色相关特征。

**小结：**
HairCLIPv2 的实现是“**把 StyleGAN2 解码器劈成两半，在中间插一块 FS 特征编辑模块**”，并且用 head mask 精确控制只在头发区域进行修改。

---

## 4. 代码对比

### 4.1 输入与潜码处理

* **e4e：**

  * `pSp.forward(x, input_code=False)`：输入图像 → 编码器 → $W^+$；
  * `pSp.forward(codes, input_code=True)`：输入已是潜码 → 跳过编码器；
  * 通过 `input_is_latent` 告诉 StyleGAN2：现在输入的是 $W^+$，不要再走映射网络。

* **HairCLIPv2：**

  * 常用入口如 `generate_im_from_w_space(code)`；
  * 核心工作集中在“从潜码出发做特征提取和混合”。

### 4.2 前向解码结构

* **e4e：**
  直接调用标准 StyleGAN2：从常数输入开始，完整跑完所有层，输出图像，不能在中间插入特征。

* **HairCLIPv2：**
  在 Generator 中加入了：

  * `start_layer` / `end_layer`：控制前后半段；
  * `layer_in`：给中间层喂外部特征。

  于是生成器结构可以理解成：

* e4e：
  `Encoder -> Generator(0 ~ N 全部层)`

* HairCLIPv2：
  `Generator(0 ~ k 层) -> 特征混合 (FS + mask) -> Generator(k+1 ~ N 层)`

### 4.3 中间层处理与融合

* **e4e：**

  * 不显式返回/编辑中间特征；
  * 只在潜码级别提供 `latent_mask` 这种简单操作。

* **HairCLIPv2：**

  * 显式提供中间特征的获取（`generate_initial_intermediate`）；
  * 在 `feature_blending.py` 中用 mask 做**特征空间插值/替换**，只动头发区域；
  * 再用 `update_on_FS` 把这些编辑过的特征和新的潜码、原始潜码结合起来，从中间层接着解码。

### 4.4 模块职责对比

* **e4e 模块：**

  * `psp.py`：整体模型，管理 encoder / decoder；
  * `models/encoders/`：各种编码器；
  * `models/stylegan2/model.py`：StyleGAN2 生成器与判别器。

* **HairCLIPv2 模块：**

  * `haircilpv2_model.py`：扩展后的 StyleGAN2 生成器（支持 start_layer / end_layer / layer_in）；
  * `feature_blending.py`：特征空间的头发/发色特征混合（FS + hairmask）；
  * 其他脚本：生成代理潜码（基于 CLIP/文本/草图等）、调用上述模块完成发型/发色编辑。

---

## 5. 解码一致性的机制理解

### 5.1 “几乎完全一致的解码”含义

所谓“几乎完全一致的解码”，是指在使用同一个 StyleGAN 解码器的情况下：

* 若不做编辑，重建图像与输入图像几乎像素级一致；
* 若只做头发编辑，则：

  * 头发区域按照目标发型/发色改变；
  * 非头发区域（脸、背景）几乎和原图一模一样。

传统做法（如单纯在 $W^+$ 上编辑）很难达到这一点，因为：

* 反演本身有误差；
* 在 $W^+$ 空间的操作往往是全局性的，容易牵连背景、脸部一起发生改变。

### 5.2 使用原始潜码保持非目标区稳定

HairCLIPv2 在后半段生成时仍然用源图像的原始潜码 `initial_S`：

* 低层（0–3 层）：通过 FS 特征混合改变头发形状/结构；
* 高层（4 之后）：使用原来的 $W^+$（`initial_S`），保证脸部细节、背景、光照等不变。

这样，即使 CLIP 优化出的潜码 `code` 在语义上发生了很大变化，最后参与高层生成的还是旧的 `initial_S`，因此：

* 新的发型被“挂”在原来的脸和背景上；
* 身份与环境几乎不变。

### 5.3 特征差分补偿重建误差

由于 e4e 编码器本身不能做到完美重建，源特征 $F_{\text{src}}$ 和混合后特征 $F_{\text{blend}}$ 之间存在差异。HairCLIPv2 使用以下差分：

$$
difference = initial\_F - initial\_intermediate
$$

来纠正新潜码生成的中间特征：

$$
new\_intermediate = intermediate + difference
$$

这个差分可以理解为“在第 3 层特征空间中，从**源原始特征**到**混合目标特征**所需要的变化量”。把这个变化量加到新潜码对应的特征上，相当于在 FS 空间中“迁移编辑效果”，同时考虑到原始重建误差，从而保证最终结果更贴近原图。

### 5.4 背景损失与架构设计

在训练时，HairCLIPv2 还可以加入背景保持损失、身份损失等，使得输出图像在非头发区域接近输入。但作者指出，仅靠潜码空间的损失不足以完全防止背景改变，真正起决定作用的是：

* 中间层 FS 特征混合；
* 后半段使用原始潜码解码。

可以总结为：

> 解码一致性 = “架构上用源潜码 + 局部特征混合” 为主，损失约束为辅。

---

## 6. 总结与展望

### 6.1 总结

通过本次对比研究，可以总结出：

* **e4e：**

  * 提供了一条“图像 → 编码器 → $W^+ \to$ StyleGAN 解码”的标准反演–解码路径；
  * 强调潜码的可编辑性，牺牲少量重建精度；
  * 解码器完全使用预训练 StyleGAN2，不做结构改动。

* **HairCLIPv2：**

  * 针对头发编辑任务，在沿用 e4e 反演结果和 StyleGAN2 解码器的前提下，改造了生成器前向：

    * 引入 `start_layer` / `end_layer` / `layer_in`，把生成过程拆成前半段 + 后半段；
    * 在中间层（FS）做局部特征混合，只修改头发区域；
    * 后半段使用原始潜码，保证非头发区域几乎完全一致。
  * 代码上采用模块化思路：在原有 GAN 上通过小改动 + 分割网络 + 特征混合模块，就实现了“只换头发、不动其他”的高质量编辑。

从范式角度看，这是一个很值得借鉴的套路：**不必推倒重来训练一个新 GAN，而是在预训练 GAN 解码器中插入可控模块（中间特征编辑），利用原模型的先验与表达能力**。

### 6.2 展望

后续可以进一步探索的方向包括：

1. **训练与优化细节：**

   * 更系统地理解 HairCLIPv2 如何获取和优化代理潜码；
   * 在多模态条件下（文本+草图）如何权衡不同条件的影响；
   * 分析特征混合比例是否需要额外调节，而不仅仅依赖 mask。

2. **向 3D 人脸编辑扩展：**

   * 参考 EG3D 等 3D-aware GAN，将“中间特征 + 局部混合”的思想迁移到三维 tri-plane 或体渲染表示上；
   * 思考是否能定义“3D 版 FS 表示”，只在 3D 头发区域进行特征替换，实现真正的 3D 发型编辑；
   * 设计类似 e4e 的 “Encoder for Editing 3D (E3D)”，实现三维人头模型的可编辑反演。

3. **工程与泛化能力：**

   * 提高代理生成的效率（例如用一个小网络预测发型特征，而不是反复优化潜码）；
   * 增强对极端发型（如爆炸头、脏辫、复杂盘发等）的泛化能力。

总的来说，通过这一阶段的学习，我对 StyleGAN 反演和图像编辑有了比较系统的认识，也体会到“论文思想 + 代码细节”结合的重要性。未来会在导师和小组同学的指导下，继续把这些 2D 经验迁移到 3D 人脸/人体编辑方向上，逐步形成自己的研究路线。
