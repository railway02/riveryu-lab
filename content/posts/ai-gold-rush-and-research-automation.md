---
title: "AI 时代的“淘金热”：GPU 是铲子，但真正的机会是生产系统"
date: 2026-05-10T10:00:00+08:00
draft: false
tags: ["AI", "Methodology", "Engineering", "Research WorkFlow", "Agent"]
summary: "从 AI 淘金热到底层工具链，关于过去与未来"
---

**现在的 AI 时代，和当年的淘金热到底像在哪里？又有什么根本不同？**

一开始，我觉得显卡很像淘金热里的铲子。

当年淘金热中，很多人冲向矿区，幻想自己能挖到黄金。但最后真正长期赚钱的，不一定是挖金子的人，而是卖铲子、卖牛仔裤、做运输、建铁路、开旅馆、做金融的人。

今天 AI 时代也很像这样。

很多人都在追逐 AI 应用、AI 创业、AI 自媒体、AI Agent、AI 产品。但如果往底层看，会发现整个 AI 世界都在围绕一些基础设施运转：GPU、算力、数据中心、云服务、模型框架、开发工具、数据管线、Agent 工作流。

---

## 一、GPU 像淘金热里的铲子

淘金热里，普通人要挖金子，需要铲子、筛盘、牛仔裤、帐篷、马车、食物和运输。
AI 时代里，训练大模型、跑推理、做图像生成、做视频生成、做 3D 重建、做自动驾驶、做机器人，也都离不开 GPU。

GPU 在今天更像是：

* 铲子；
* 蒸汽机；
* 铁路；
* 发电站；
* 工厂设备；
* 甚至是新时代的战略资源。

尤其是大模型时代，GPU 的需求会自我放大。

模型越强，使用的人越多；
使用的人越多，公司越愿意投入；
公司越投入，就会训练更大的模型；
模型越大，又需要更多 GPU。

这形成了一个正反馈循环。

所以今天 NVIDIA 的地位，更像 AI 时代的基础设施公司，甚至某种意义上是“AI 工业革命的军火商”。

但是，如果只把 GPU 看成铲子，还不够。因为 AI 时代的“铲子”不只有硬件。

---

## 二、AI 时代的“铲子”其实有三层

### 第一层：硬件铲子

这包括：

* GPU；
* HBM 高带宽显存；
* NVLink；
* 数据中心；
* 电力；
* 液冷；
* 云服务器。

这些是 AI 时代最底层的基础设施。没有这些，模型训练和大规模推理都无法进行。

### 第二层：软件铲子

这一层正在变得越来越重要。

比如：

* AI IDE；
* Codex；
* Claude Code；
* Cursor；
* Agent 框架；
* 自动评测系统；
* 模型部署工具；
* 数据处理 pipeline；
* 实验管理系统。

这些工具不是直接训练大模型，但它们会改变人的工作方式。

以前一个人写代码、调环境、看报错、整理实验，可能要花很多时间。
现在 AI 可以帮你读代码、写脚本、分析报错、生成测试、总结结果。

这类工具，本质上也是铲子。

### 第三层：行业铲子

这是我觉得未来更重要的一层。

大模型最终不会只停留在聊天窗口里，而会进入具体行业，变成行业工作流的一部分。

比如：

* AI 科研系统；
* AI 法律系统；
* AI 医疗辅助系统；
* AI 设计系统；
* AI 教育系统；
* AI 编程系统；
* AI 视频生产系统；
* AI 3D 内容生成系统。

真正有长期价值的，不是简单做一个“套壳聊天机器人”，而是把 AI 接进真实问题、真实流程、真实行业里。

---

## 三、淘金热和 AI 时代最大的区别

淘金热和 AI 时代确实很像，但也有一个根本区别：

> 淘金热挖的是有限资源，AI 时代挖的是生产力。

黄金是有限的。你挖走一块，别人就少一块。

但 AI 能力不是这样。

一个模型可以被很多人调用；
一个工具可以嵌入很多流程；
一个 Agent 可以帮助不同的人完成不同任务；
一个自动化系统可以不断复用、不断改进。

所以淘金热更像是抢资源，AI 时代更像是重构生产方式。

淘金热的结局是：**少数人发财，多数人陪跑，但基础设施和城市留下来了。**

AI 时代的结局大概率是：**大量泡沫应用消失，少数基础设施公司和真正解决问题的系统留下来，但所有行业的工作方式都会被改变。**

---

## 四、我在科研复现实验里感受到的 AI 时代痛点

我最近做科研、跑论文实验时，有一个很深的感受：很多时候，跑通一篇论文并不是最难在算法，而是难在环境配置。

一般流程是：

```text
下载代码
配置环境
准备数据
下载权重
编译 CUDA 扩展
运行 demo
跑训练
看结果
分析实验
```

但最痛苦的往往是前面几步。

比如做 3DGS、NeRF、Gaussian Avatar、FlashAvatar、GPHM、动态头部重建这些方向，经常会遇到：

* Python 版本不匹配；
* PyTorch 版本不匹配；
* CUDA 版本不匹配；
* gcc/g++ 版本问题；
* C++/CUDA 扩展编译失败；
* submodule 没拉下来；
* requirements.txt 太旧；
* 某些依赖已经失效；
* 数据格式不符合要求；
* COLMAP 相机模型不兼容；
* 预训练权重找不到；
* demo 数据缺失。

最折磨人的地方是：

> 花了很多时间解决环境问题，但感觉自己没有真正提升科研能力。

---

## 五、正确方向：建立论文复现 Agent 流水线

理想流程：

```text
论文仓库
  ↓
AI 读取 README / environment.yml / requirements.txt / setup.py / Dockerfile
  ↓
AI 判断 Python / PyTorch / CUDA / GCC 版本
  ↓
AI 生成 REPRO_PLAN.md
  ↓
AI 生成 install_env.sh
  ↓
AI 生成 check_env.py
  ↓
AI 生成 smoke_test.sh
  ↓
执行安装
  ↓
收集日志
  ↓
AI 分析报错
  ↓
AI 修改脚本
  ↓
跑通最小 demo
  ↓
冻结环境并记录 RUNBOOK.md
```

> 让 AI 负责读仓库、生成环境方案、写脚本、分析报错、迭代修复；
> 我负责判断论文是否值得跑、实验结果是否可信、方法是否对我的研究有价值。

---

## 六、AGENTS.md：给 AI 写一份长期工作协议

为了避免每次都重复告诉 AI：

“不要污染 base 环境”
“先读 README”
“不要直接 pip install”
“先跑 smoke test”
“日志要保存”
“用 micromamba”
“缓存放到 /root/autodl-tmp”

这时候就需要`AGENTS.md`。它的作用类似于给 AI 的长期规则文档。

比如规定：

```text
不要盲目安装依赖。
先检查仓库，再生成计划。
每篇论文都必须生成：
REPRO_PLAN.md
install_env.sh
check_env.py
smoke_test.sh
RUNBOOK.md

不要跑完整训练，先跑 smoke test。
失败后必须基于日志分析，不要玄学乱试。
```

因为如果没有规则，AI 很容易变成“高级乱试工具”。
但如果有了规则，它就会更像一个科研工程助手。

---

## 八、Skills、CLI、插件

### 第一阶段：用 CLI + AGENTS.md

CLI 负责实际操作项目目录。AGENTS.md 负责固定行为规范。

目标是先让 AI 能稳定完成：

* 仓库体检；
* 环境方案；
* 安装脚本；
* 环境检查；
* smoke test；
* 日志分析。

### 第二阶段：沉淀成 Skill

等跑通 1-2 篇论文之后，就可以把流程封装成一个 Skill。

Skill 本质上是可复用的工作流。

包括：

* 复现规则；
* 常见环境坑；
* 3DGS 项目检查清单；
* AutoDL 目录规范；
* 日志收集脚本；
* smoke test 模板。

这样以后每次复现论文，就不用重新设计流程。

### 第三阶段：再考虑插件 / MCP / 自动化平台

插件、MCP、GitHub Actions、实验数据库这些东西更高级，但不是第一优先级。

它们适合后期做：

* 自动拉 GitHub issue；
* 自动下载 HuggingFace 权重；
* 自动读取论文库；
* 自动同步实验结果；
* 自动生成报告；
* 自动创建 dashboard。

---

## 九、AutoDL 使用 CLI 的真正难点

AutoDL 里主要有三类网络需求：

```text
1. GitHub 代码下载
2. HuggingFace 模型权重下载
3. OpenAI / Claude / Codex CLI 通信
```

AutoDL 的学术加速通常对 GitHub、HuggingFace 有帮助，但不一定能解决 OpenAI / Claude 这类 AI CLI 的通信问题。

所以更稳的方案有两个。

---

## 十、方案 A：AutoDL 直接跑 CLI，通过本地 Clash 做 SSH 反向代理

这个方案的结构是：

```text
AutoDL 上的 Codex CLI
        ↓
访问 AutoDL 的 127.0.0.1:7897
        ↓
SSH 反向隧道
        ↓
本地 Windows / WSL2 的 Clash Verge
        ↓
OpenAI / Claude / GitHub / HuggingFace
```

也就是说，把本地 Clash 的代理能力，通过 SSH 反向隧道转发给 AutoDL 使用。

本地执行：

```bash
ssh -N -R 7897:127.0.0.1:7897 root@你的AutoDL地址 -p 你的SSH端口
```

然后 AutoDL 里设置：

```bash
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export all_proxy=socks5h://127.0.0.1:7897
export ALL_PROXY=socks5h://127.0.0.1:7897
export NO_PROXY=localhost,127.0.0.1
```

再测试：

```bash
curl -I https://api.openai.com
curl -I https://github.com
```

如果能通，就可以在 AutoDL 上运行 CLI。

这个方案的优点是：
AI Agent 可以直接在 AutoDL 里读仓库、改脚本、执行命令、看 GPU 环境。

缺点是：
代理链路稍微复杂，本地电脑和 Clash 不能断。

---

## 十一、方案 B：本地 WSL2 跑 CLI，AutoDL 只负责执行脚本

这个方案更稳。

结构是：

```text
本地 WSL2 / Windows
运行 Codex / Claude Code
使用本地 Clash 网络
        ↓
生成 install_env.sh / check_env.py / smoke_test.sh
        ↓
rsync / scp 同步到 AutoDL
        ↓
AutoDL 执行脚本并保存日志
        ↓
把日志拿回本地给 AI 分析
```

优点是：

* 本地网络最稳定；
* 登录 OpenAI / Claude 更方便；
* 不需要把 API key 放到 AutoDL；
* 不怕 AutoDL 代理断。

缺点是：

* AI 不能直接实时感知 AutoDL 的环境；
* 需要通过 ssh 执行命令、同步日志。

---

## 十二、训练的能力

### 1. 判断一篇论文是否值得跑

不是所有论文都值得复现。

有些论文只需要读方法；
有些只需要看代码；
有些只需要跑 demo；
只有和我主线强相关的，才值得深度复现。

### 2. 建立实验工厂，而不是手工作坊

```text
paper_factory/
├── repos/
├── data/
├── weights/
├── runs/
├── env_logs/
└── scripts/
```

每篇论文都进入这个体系，而不是到处乱放。

### 3. 先 smoke test，再完整训练

不要一开始就跑几万 iteration。

先跑最小 demo，验证：

* 环境能不能启动；
* GPU 能不能调用；
* 数据能不能读取；
* loss 是否正常；
* 输出是否生成；
* 结果是否明显崩坏。

### 4. 用日志驱动调试

失败后不要玄学乱试，而是保存日志：

```bash
bash install_env.sh 2>&1 | tee install.log
bash smoke_test.sh 2>&1 | tee smoke_test.log
```

然后让 AI 基于日志判断根因。

### 5. 成功后冻结环境

一旦跑通，立刻保存：

```bash
pip freeze > pip_freeze.txt
conda env export > environment_export.yml
git rev-parse HEAD > git_commit.txt
nvidia-smi > nvidia_smi.txt
```
---

## 十三、最终结论：学会造铲子和组织矿场

AI 时代确实像淘金热。GPU 是铲子，云是矿场，数据中心是铁路，Agent 是新工人，模型是新的生产力引擎。

但对我个人来说，最重要的不是冲进去盲目“挖金子”。

真正重要的是：

> 把 AI 变成自己的科研生产系统。

在科研里，这意味着：

* AI 帮我读论文；
* AI 帮我读代码；
* AI 帮我生成环境方案；
* AI 帮我写安装脚本；
* AI 帮我分析报错；
* AI 帮我跑 smoke test；
* AI 帮我总结实验结果；
* 我负责判断方向、设计实验、理解算法、提出改进。

这就是我理解的 AI 时代个人机会。

建立一个属于自己的：

> **AI 驱动科研工作流。**

淘金热里，真正留下来的是铁路、城市、工具和组织能力。AI 时代里，真正留下来的也会是基础设施、工作流、自动化系统和解决真实问题的能力。

