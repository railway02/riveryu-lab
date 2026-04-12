---
title: "FlashAvatar  工程实践"
date: 2026-04-12T10:00:00+08:00
draft: false
tags: ["3DGS", "Environment Setup", "Micromamba", "Computer Vision"]
summary: "配置环境，由conda转为micromamba"
---

对于从事 3D 人脸重建或高斯溅射（Gaussian Splatting）研究的开发者来说，环境配置往往是劝退的第一步。庞大的子模块、严苛的 CUDA 版本要求，以及国内服务器的网络环境，常常让 `conda env create` 沦为一个跑不完的进度条。

最近在跑 USTC3DV 团队的 [FlashAvatar](https://github.com/USTC3DV/FlashAvatar-code) 时，我也经历了这番折磨。经过多次试错与底层依赖的拆解，我总结出了一套在 AutoDL 平台上极速、稳定构建该环境的工程流。

---

### 1. 痛点分析：为什么你的环境总是配不出来？

官方提供的 `environment.yml` 虽然完整，但在国内云服务器直接执行，通常会遇到两个致命死结：

1.  **GitHub 子模块拉取失败：** 项目依赖了 `diff-gaussian-rasterization` 等定制化的 3D 引擎，常规 clone 会漏掉核心代码，导致后续编译疯狂报错。
2.  **Conda 解析死锁：** 依赖树过于复杂，基础的 conda 求解器很容易卡死在 "Solving environment" 阶段，耗费数小时甚至因为内存溢出而崩溃。

面对第二点，传统的 Conda 已经无能为力。这就引出了本次实战的核心破局点：**更换包管理工具**。

### 2. 技术选型：为什么放弃 Conda，拥抱 Micromamba？

在 Python 环境管理界，Conda、Mamba 和 Micromamba 就像是“祖孙三代”。它们共享同一个生态（都去 Conda-forge 等仓库拉取包，共用 `.condarc` 配置），但在底层架构上却有天壤之别：

* **一代目 Conda（大而全的原住民）：** 纯 Python 编写。功能全但极其缓慢，面对包含大量底层 C++ 库和 3D 算子的复杂环境时，其依赖求解算法经常陷入死胡同。
* **二代目 Mamba（性能外挂）：** 核心用 C++ 重写，引入了原本为 Linux 打造的 `libsolv` 求解器和并发下载机制，速度飙升了几十倍。但痛点是“叠床架屋”——你通常得先装一个臃肿的 Conda 基础环境，才能在里面装 Mamba。
* **三代目 Micromamba（终极形态）：** C++ 静态链接的单文件（Single binary）。它把 Mamba 的核心能力打包成了一个不到 20MB 的可执行文件。不依赖 Base 环境，解压即用，极度纯粹。

| 特性               | Conda               | Mamba                   | Micromamba                     |
| :----------------- | :------------------ | :---------------------- | :----------------------------- |
| **底层语言**       | Python              | C++ & Python            | C++ (纯静态链接)               |
| **求解速度**       | 极慢（极易卡死）    | 极快（libsolv）         | **极快（libsolv）**            |
| **依赖 Base 环境** | 是（本身就是 Base） | 是（需基于 Conda 安装） | **否（独立单文件，即插即用）** |

技术的演进往往是从“能用”到“好用”，再到“无感”的做减法。在云算力平台上，抛弃沉重的 Conda，使用体积小巧、解析极快的 Micromamba，能最大程度规避网络波动和解析死锁。

---

### 3. 核心操作步骤

理清了思路，接下来就是行云流水的实战操作。

#### 步骤一：网络隔离与完整源码拉取
首先，必须在开启网络加速的状态下，确保所有 Git Submodules 被完整拉取。


# 开启网络加速 (AutoDL 专属)
source /etc/network_turbo

# 克隆代码并强制同步所有子模块
git clone [https://github.com/USTC3DV/FlashAvatar-code.git](https://github.com/USTC3DV/FlashAvatar-code.git)
cd FlashAvatar-code

# 解决大文件 clone 报错
git config --global http.version HTTP/1.1
git submodule update --init --recursive


### 步骤二：使用 Micromamba 极速建站
在创建环境时，必须关闭网络加速代理，让机器直连清华镜像源，否则极易出现 SSL 握手错误。


# 清除网络代理的环境变量，防止镜像源握手失败
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY

# 配置清华源 .condarc
cat > ~/.condarc <<'EOF'
show_channel_urls: true
default_channels:
  - [https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main](https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main)
  - [https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r](https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r)
custom_channels:
  conda-forge: [https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud)
  pytorch: [https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud)
  nvidia: [https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud)
  pytorch3d: [https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud)
EOF

# 使用 Micromamba 极速创建并激活环境 (解析过程通常只需几秒)
micromamba env create -f environment.yml
micromamba activate FlashAvatar
步骤三：精准剥离，攻克 PyTorch3D 与底层算子
很多教程喜欢一键 pip install pytorch3d，这在算力云平台上几乎必败。正确的做法是：先处理 C++ 基础依赖，再调用系统底层的 NVCC 进行编译。

Bash
# 1. 单独安装基础数据结构和 I/O 库
micromamba install -c fvcore -c iopath -c conda-forge fvcore iopath -y

# 2. 安装调用 NVIDIA GPU 的底层核心库
micromamba install -c bottler nvidiacub -y

# 3. 最后再正式编译安装核心 3D 引擎
micromamba install pytorch3d -c pytorch3d -y
此外，对于项目自带的定制化高斯光栅化引擎，切换到国内 PyPI 源，开启详细输出模式 (-v) 手动编译：

Bash
pip install ./submodules/diff-gaussian-rasterization \
            ./submodules/simple-knn \
            scipy loguru opencv-python lpips \
            -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple) -v
4. 性能压榨：不要让 CPU 成为瓶颈
环境跑通后，不要急于启动训练。3D 数据的 I/O 和预处理非常消耗 CPU 资源。在服务器上，我们需要显式释放多线程算力，避免默认单线程拉低 GPU 利用率：

Bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 写入配置文件，使其永久生效
echo 'export OMP_NUM_THREADS=8' >> ~/.bashrc
echo 'export MKL_NUM_THREADS=8' >> ~/.bashrc
source ~/.bashrc
总结
看似枯燥的环境配置，其实是对 Linux 系统、编译链和网络协议的一次综合调度。抛弃对“一键安装”的幻想，学会拆解依赖树，并选择像 Micromamba 这样锐利的工具，才是避免陷入“依赖地狱”的唯一解。

希望这份踩坑记录能帮你省下宝贵的科研时间，早日跑出精美的渲染结果。