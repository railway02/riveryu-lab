---
title: "Research"
description: "Current research questions, completed experiments, and directions under exploration."
hidemeta: true
---

这里整理我目前关注的研究问题，并区分已经完成的实验与写作、正在形成的方法，以及仍需验证的想法。

## AIGC Authenticity & Robustness

**研究问题：** 如何减少检测器对水印、编码格式和生成器身份等捷径的依赖，并从稳定的时空信号中识别生成视频。

**当前关注点**

- **已经做过：** 围绕 AIGC 视频检测的研究方向、编码器表示、Transformer 数值稳定性与对抗性反取证完成了分析和学习记录。这些内容是问题拆解与技术准备，不代表已经完成一个鲁棒的视频检测系统。
- **正在探索：** 如何把检测依据从生成器特有痕迹转向跨帧一致性、潜在表示和更稳定的时空证据，并评估压缩与后处理对这些证据的影响。

**Related Projects / Selected Writing**

- [AIGC 视频攻防研究：为什么我们必须重新理解“视频”](/posts/aigc-dectector/)
- [编码器 Encoder 笔记：从特征表示到 AIGC 视频检测](/posts/encoder-deep-dive-aigc-video-detection/)
- [Transformer 数值稳定性、归一化机制与 AIGC 视频检测启发](/posts/transformer-stability-to-aigc-video-detection/)
- [当 AI 学会“完美伪装”：深度解析《Adversarial Reality》与反取证实战](/notes/agaigc/)

## Latent Representations & Generative Dynamics

**研究问题：** 生成模型的潜空间保留了什么结构，数据如何在潜空间中变化和传输。

**当前关注点**

- **已经做过：** 梳理了生成模型 Tokenizer 与潜空间、一步图像编辑中的最优传输、StyleGAN3 的连续信号表示，以及优化算法的数学基础。这些成果目前主要是技术分析和学习笔记。
- **正在探索：** 潜在表示中的结构如何与生成动态、编辑轨迹和数字指纹相联系，以及哪些变化规律能够跨模型稳定存在。

**Related Projects / Selected Writing**

- [从 RAE 到 Cosmos Tokenizer：生成模型潜空间与 AIGC 视频数字指纹](/posts/latent-space-tokenizer-to-aigc-detection/)
- [ChordEdit：一步图像编辑为什么需要低能量传输？](/posts/chordedit-optimal-transport-image-editing/)
- [傅里叶特征在 StyleGAN3 中的作用：从高频拟合到连续坐标场](/posts/styleganv3/)
- [深度学习优化算法系统笔记：从梯度下降到 AdamW](/posts/deep-learning-optimizers-sgd-to-adamw/)

## 3D Vision & Human Reconstruction

**研究问题：** 如何从有限二维观测恢复可编辑、可驱动、身份一致的三维人物表示。

**当前关注点**

- **已经做过：** 跑通官方 3DGS Baseline，搭建从视频预处理、COLMAP 到训练与评估的数据链路，并在自采数据上完成多轮实验与失败分析。这些工作形成了可复盘的工程和实验基础，但还不是一个已经解决的全头重建方法。
- **正在探索：** 如何从静态近似走向包含身份、表情和动态变化的建模，并理解几何先验、Gaussian 表示与可驱动 Avatar 之间的关系。

**Related Projects / Selected Writing**

- [从官方 3DGS Baseline 到单目转头视频静态头部重建](/posts/3dgs-ex/)
- [科研日志 #1：3DGS 数据与训练流水线](/posts/3dgs-pipeline/)
- [科研日志 #2：3DGS 单目头部重建实验复盘](/posts/research-log-2-experiment-records/)
- [从第一性原理全景解构 3D 虚拟人](/notes/first-principles-3d-virtual-human/)
- [从单目视频到 3D 头部 Avatar：GPHM 学习笔记](/notes/gphm-paper-reading/)
