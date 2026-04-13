---
title: "FlashAvatar 工程实践：从 Conda 到 Micromamba 的一次环境突围"
date: 2026-04-12T10:00:00+08:00
draft: false
tags: ["3DGS", "Environment Setup", "Micromamba", "Computer Vision", "FlashAvatar"]
summary: "记录我在 AutoDL 上配置 FlashAvatar 环境的完整过程：为什么放弃 Conda、如何用 Micromamba 提速，以及如何逐步解决子模块、PyTorch3D 与自定义算子的安装问题。"

---

对于做 3D 人脸重建、动态头像建模或 Gaussian Splatting 工程复现的人来说，真正劝退人的，往往不是论文本身，而是环境。

大项目、深依赖、CUDA 版本约束、子模块编译、国内云服务器网络波动……这些东西一旦叠在一起，`conda env create` 就很容易从“自动化部署”变成“玄学抽卡”。

最近我在 AutoDL 上复现 USTC3DV 团队的 [FlashAvatar](https://github.com/USTC3DV/FlashAvatar-code) 时，就完整经历了一遍这个过程。前前后后踩了不少坑之后，我把这套环境搭建流程重新拆了一遍，最后收敛成了一条更稳、更快、也更适合国内算力平台的工程路径。

这篇文章不是“官方步骤翻译”，而是一份真正从踩坑里长出来的配置记录。

---

## 1. 为什么 FlashAvatar 的环境容易把人卡住？

FlashAvatar 不是那种只要一个 `requirements.txt` 就能跑起来的轻量项目。它本质上是一个牵涉到：

- PyTorch / CUDA 版本耦合
- Git 子模块递归拉取
- 自定义 C++ / CUDA 扩展编译
- PyTorch3D 兼容性
- 多个底层视觉与渲染依赖

的综合型工程。

官方给了 `environment.yml`，这当然是好事，但问题在于：**“给了环境文件”不等于“环境就能顺利建出来”。**

在 AutoDL 这类国内云服务器上，常见阻塞点主要有两个。

### 1.1 Git 子模块不完整

FlashAvatar 依赖了多个子模块，比如 `diff-gaussian-rasterization`、`simple-knn` 等底层组件。如果只是普通 `git clone`，没有把子模块完整拉下来，后面一到编译阶段就会开始报错，而且很多报错表面看像是编译问题，实际上源头是代码根本没拉全。

这种坑最烦，因为它不是“立即报错”，而是会在后面绕一圈再来打你。

### 1.2 Conda 依赖求解太慢，甚至卡死

另一个更常见的问题，是 Conda 在复杂环境下的依赖解析。像 FlashAvatar 这种项目，一旦把 PyTorch、CUDA、PyTorch3D、视觉库和编译工具链全部卷进来，传统 Conda 的求解过程就很容易卡在：

```bash
Solving environment...
```

然后就是漫长等待。运气好是几十分钟，运气不好就是直接锁死、爆内存，或者因为网络抖动中途失败。

所以这次我真正想解决的，不只是“装环境”，而是两个更本质的问题：

1. 怎么让依赖解析更稳定  
2. 怎么让安装过程更适配国内算力平台

而答案最后落在了 Micromamba 上。

---

## 2. 为什么我最后放弃 Conda，转向 Micromamba？

如果把 Conda、Mamba、Micromamba 放在一起看，它们其实是一条技术演进线。

### 2.1 Conda：能用，但越来越重

Conda 最大的优点是生态成熟，大家都认识它，很多开源项目也默认给它写配置文件。

但它的问题也很明显：

- 依赖解析慢
- 历史包袱重
- 在复杂环境下容易卡死
- 对网络和镜像状态敏感

对于科研环境里这种“版本高度耦合 + 编译多”的场景，它经常不是不能用，而是用起来成本太高。

### 2.2 Mamba：快了很多，但还是依附 Conda

Mamba 的出现，本质上是在 Conda 生态里换了一个更快的求解器。它用 C++ 重写核心逻辑，引入 `libsolv`，在环境解析速度上相比 Conda 提升非常明显。

但 Mamba 还是有一个现实问题：它通常还是构建在已有 Conda 体系上的。也就是说，你往往得先有一个 Conda，再用它装 Mamba。

这在本地开发机上问题不大，但在云服务器或临时容器里，就显得有点重。

### 2.3 Micromamba：更适合云平台的极简方案

Micromamba 是我这次真正觉得“工程上顺手”的点。

它的优势非常直接：

- 单文件二进制，安装轻
- 不依赖 base 环境
- 兼容 Conda 生态和 `.condarc`
- 底层依然使用 `libsolv`
- 很适合 AutoDL、容器、远程服务器这种场景

它不是“新概念”，而是把 Mamba 的速度优势用一种更干净的方式落了地。

对于像 FlashAvatar 这种依赖复杂、环境构建容易翻车的项目来说，Micromamba 最大的价值不是“更优雅”，而是**更少出事**。

---

## 3. Conda、Mamba、Micromamba 到底差在哪？

为了方便理解，我把这三者放在一张表里对比一下。

| 特性                   | Conda       | Mamba        | Micromamba       |
| :--------------------- | :---------- | :----------- | :--------------- |
| 底层实现               | Python 为主 | C++ + Python | C++ 单文件二进制 |
| 依赖求解速度           | 较慢        | 很快         | 很快             |
| 是否依赖已有 Base 环境 | 是          | 通常是       | 否               |
| 适合云服务器/临时容器  | 一般        | 较好         | 很适合           |
| 对复杂科研环境的容错性 | 一般        | 较强         | 较强             |

我的实际感受很简单：

- 本地开发机长期维护环境：Conda 还能忍
- 已有 Conda 体系，想加速：Mamba 合适
- AutoDL / 云容器 / 新环境快速起盘：Micromamba 最顺手

---



## 4. 实战流程：我在 AutoDL 上是怎么把环境拉起来的？

下面这部分，就是我最后收敛出来的一套更稳的安装路径。

---

## 5. 第一步：先把源码和子模块拉完整

这一阶段最重要的事只有一件：**确保仓库不是“半残版”。**

在 AutoDL 上，我通常会先开网络加速，把 GitHub 仓库和子模块完整拉下来。

```bash
# 开启网络加速（AutoDL 环境常用）
source /etc/network_turbo

# 克隆仓库
git clone https://github.com/USTC3DV/FlashAvatar-code.git
cd FlashAvatar-code

# 避免部分网络环境下的大文件 / 连接问题
git config --global http.version HTTP/1.1

# 递归拉取所有子模块
git submodule update --init --recursive
```

这一步不要省。很多后续编译错误，根本原因都不是“你不会编”，而是“你缺文件”。

如果这一步没做干净，后面的问题会越来越隐蔽。

---

## 6. 第二步：关闭代理干扰，切换清华 Conda 镜像

这里有个很容易忽略，但实际非常关键的点：

**拉 GitHub 代码时，网络加速是有用的；但装 Conda 包时，代理反而可能导致 SSL 握手失败。**

所以我后面的处理方式是：**Git 阶段开代理，建环境阶段关代理。**

```bash
# 清理代理环境变量，避免 Conda / Micromamba 访问镜像源时 SSL 出错
unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
```

然后配置清华镜像：

```bash
cat > ~/.condarc <<'EOF'
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  nvidia: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch3d: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
```

到这里，网络环境才算真正进入“适合建环境”的状态。

---

## 7. 第三步：用 Micromamba 创建环境

接下来才是核心步骤：正式建环境。

```bash
micromamba env create -f environment.yml
micromamba activate FlashAvatar
```

这一步的最大体验差异在于：**它终于不像 Conda 那样让人怀疑人生了。**

尤其是在依赖复杂的工程里，Micromamba 的解析速度和稳定性都明显更适合这种“先把环境立住再说”的场景。

不过要注意一件事：`environment.yml` 能把大部分东西装好，但不代表所有关键组件都能一步到位。FlashAvatar 这类项目里，真正难搞的部分，往往在后面的扩展库和底层渲染模块。

---

## 8. 第四步：把 PyTorch3D 和底层依赖拆开处理

这是整个流程里最容易翻车的阶段之一。

很多教程上来就是一句：

```bash
pip install pytorch3d
```

但在云服务器、CUDA 版本复杂、PyTorch 版本锁定的环境里，这种做法很容易直接失败。

我最后的经验是：**不要把它当成一个“一键安装包”，而要把它拆成几个层次来处理。**

### 8.1 先安装基础依赖

先把常用的数据结构和 I/O 相关依赖装好：

```bash
micromamba install -c fvcore -c iopath -c conda-forge fvcore iopath -y
```

### 8.2 再补底层 GPU 相关库

有些依赖本质上是在为 CUDA 扩展和图形模块兜底，比如 `nvidiacub`：

```bash
micromamba install -c bottler nvidiacub -y
```

### 8.3 最后安装 PyTorch3D

```bash
micromamba install pytorch3d -c pytorch3d -y
```

这样做的意义在于：把失败点拆开，而不是把所有问题都赌在同一条命令上。

如果哪一步挂了，你至少知道问题出在哪一层，而不是面对一个巨大的安装日志发呆。

---

## 9. 第五步：手动编译项目自带的高斯光栅化模块

FlashAvatar 的关键能力之一，离不开它依赖的定制化 CUDA / C++ 模块，比如：

- `diff-gaussian-rasterization`
- `simple-knn`

这些部分通常不适合“放任自动安装”。更稳的做法是明确指定源，并把编译日志开详细。

```bash
pip install ./submodules/diff-gaussian-rasterization             ./submodules/simple-knn             scipy loguru opencv-python lpips             -i https://pypi.tuna.tsinghua.edu.cn/simple -v
```

这里的几个关键点是：

- 用国内 PyPI 源减少下载失败
- `-v` 打开详细日志，方便定位编译问题
- 显式安装子模块，而不是等它在某个间接依赖里悄悄失败

如果你后面遇到的是 NVCC、CUDA_HOME、编译器版本之类的问题，这一步的详细日志会非常有价值。

---

## 10. 第六步：别让 CPU 成为整个流程里最弱的一环

很多人环境配完就直接开跑，但对 3D 任务来说，真正的训练并不是“GPU 一把梭”。

数据加载、图像预处理、部分几何操作和日志处理，其实都会吃掉不少 CPU 资源。如果线程数没开，GPU 很容易进入“看起来在跑，其实一直在等数据”的状态。

所以我一般会手动设置线程数：

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

如果想长期生效，可以写进 `~/.bashrc`：

```bash
echo 'export OMP_NUM_THREADS=8' >> ~/.bashrc
echo 'export MKL_NUM_THREADS=8' >> ~/.bashrc
source ~/.bashrc
```

这个数值不是固定答案，还是要结合你机器的 CPU 核数来调。但核心原则很简单：**不要默认让系统替你决定一切。**

---

## 11. 这一套流程背后的本质，其实不是“装软件”

回头看这次 FlashAvatar 配环境的过程，我越来越觉得，环境配置这件事，本质上不是“照着 README 敲命令”，而是在做一件更底层的事：

**你在协调一整套系统。**

你要同时处理：

- Git 仓库完整性
- 国内外网络路径差异
- 包管理器求解效率
- CUDA 与 PyTorch 的兼容性
- C++ / CUDA 扩展编译链
- CPU 与 GPU 的资源配比

所以它看起来像“安装依赖”，实际上更像一次小型系统工程。

也正因为如此，我现在越来越不相信“一键安装”这件事。对于复杂科研项目来说，真正可靠的方式往往不是“赌官方脚本一次成功”，而是把依赖树拆开，一层层确认，一层层落地。

Micromamba 之所以好用，也不是因为它神奇，而是因为它帮你把最容易卡死的那一层先处理掉了。

---

## 12. 我对这次配置的最终结论

如果你也准备在 AutoDL 或类似云平台上复现 FlashAvatar，我现在更推荐这样理解整个流程：

1. **先保证仓库和子模块完整**
2. **把代理和镜像问题分阶段处理**
3. **用 Micromamba 替代 Conda 做环境解析**
4. **把 PyTorch3D 和底层库拆开装**
5. **手动编译关键 CUDA 模块**
6. **最后再去优化线程与训练吞吐**

这样做的好处不是“命令更高级”，而是它更符合工程现实。

你不是在和一个 Python 环境打交道，而是在和一整套 3D 视觉工程栈打交道。只要接受这一点，很多原本看起来随机的报错，其实都会变得可以理解、可以拆解、也可以解决。

---

## 13. 收尾

环境配置本身不会产出论文结果，也不会直接提高指标，但它决定了你有没有资格进入真正的实验阶段。

很多科研推进慢，不是因为不会做方法，而是长期被困在环境、编译和依赖这些“看起来不高级”的基础环节里。

但恰恰是这些基础环节，最能区分“只是看过项目”与“真正把项目跑起来了”的人。

这次从 Conda 切到 Micromamba，对我来说不是一次单纯的工具替换，更像是一次工程思维上的校正：  
**少迷信一键成功，多把系统拆开看。**

希望这份记录，能帮后来者少掉几次坑，也更快进入 FlashAvatar 真正有价值的部分：数据、建模、训练与结果分析。