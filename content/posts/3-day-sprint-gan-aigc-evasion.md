---
title: "实战记录：3 天构建 GAN-like AIGC 图像修改器（Bypass 伪造检测）"
date: 2026-04-18T10:00:00+08:00
draft: false
tags: ["AI Security", "GAN", "MVP", "DevLog", "Engineering"]
summary: "停止盲目刷 Benchmark。用 3 天的敏捷研发 Sprint，将开源 AIGC 检测器转换为打分器，快速跑通一个对抗图像修改器的最小闭环（MVP）。"
---
# 实战记录：3 天构建 GAN-like AIGC 图像修改器（Bypass 伪造检测）

## 导语
很多时候，我们在跑通一篇开源论文（比如各种 AIGC Detector）后，容易陷入“无休止刷 Benchmark”的陷阱。本文记录了一次为期 3 天的敏捷研发 Pivot（方向锚定）：停止盲目复现，将前期跑通的 UFD (Universal Fake Detector) 转化为 GAN 里的冻结判别端（Scorer），快速构建一个能降低 AIGC 检测分数的“图像修改器原型”。

这次 Sprint 的核心目标不是发 Paper，而是快速交付一个最小闭环（MVP），验证技术路线的可行性。

---

## 🎯 核心思路与 3 天 Sprint 目标

前期跑通 UFD 的工作并没有白费，它的角色现在非常明确：作为我们生成网络中的冻结打分器（Frozen Discriminator）。

接下来的核心不再是死磕完整的数据集，而是要在 3 天内跑通以下流程：

- 输入一张 Fake 图  
- 生成器（Generator）输出一张修改后的 Fake 图  
- 约束条件：
  - 修改后的图像与原图肉眼极度相似  
  - 在 UFD 下的“AIGC 伪造分数”显著下降  

这足以证明我们已经打通了 **Adversarial Modification** 的主线任务。

---

