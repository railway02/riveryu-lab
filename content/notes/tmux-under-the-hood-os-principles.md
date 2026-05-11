---
title: "揭开 tmux 的底牌：从操作系统原理看远程终端复用"
date: 2026-05-10T22:00:00+08:00
draft: false
tags: ["Linux", "tmux", "Operating System", "Engineering", "Server"]
summary: "程序不是跑在窗口里，而是跑在进程里。从 PTY 伪终端、SIGHUP 信号到进程树模型，带你彻底搞懂 tmux 为什么能保护你的深度学习训练任务。"
---

## 0. 理解 tmux

**tmux 是一个运行在服务器上的“长期终端工作台”。**


普通 SSH：

```text
本地电脑 SSH 连接
    ↓
远程 bash
    ↓
python train.py
```

SSH 断了，bash 和 Python 任务可能受影响。

 tmux：

```text
本地电脑 SSH 连接
    ↓
tmux client
    ↓
tmux server
    ↓
bash
    ↓
python train.py
```

SSH 断了，只是 `tmux client` 断开，`tmux server` 还在，里面的 bash 和 Python 任务继续运行。

---

# 1. tmux 解决的核心问题

科研实验、深度学习训练、远程服务器开发经常有长时间任务：

```bash
python train.py
```

普通 SSH 连接存在几个风险：

```text
网络断开
VS Code Remote SSH 掉线
电脑关机
终端窗口误关
本地代理不稳定
```

这些都会导致远程终端断开。

问题是：

> 程序看起来是在“服务器上跑”，但它实际上依赖当前 SSH 终端环境。

tmux 的作用是把：

```text
网络连接
终端会话
实验进程
```

三者拆开。

拆开后：

```text
网络连接可以断
终端会话留在服务器
实验进程继续运行
```

---

# 2. tmux 计算机知识

tmux 主要涉及两大类知识：

```text
操作系统知识：核心
计算机网络知识：使用场景
```

## 2.1 操作系统知识

tmux 背后强相关的操作系统知识包括：

```text
进程 process
父子进程 fork/exec
会话 session
进程组 process group
控制终端 controlling terminal
信号 signal
SIGHUP / SIGINT
TTY / PTY
文件描述符 stdin/stdout/stderr
I/O 多路复用 select/poll/epoll
Unix Domain Socket
```

## 2.2 计算机网络知识

tmux 常用于 SSH 远程服务器场景，涉及：

```text
SSH 协议
TCP 长连接
网络断连
远程登录
客户端/服务器模型
VS Code Remote SSH
NAT / 防火墙 / 心跳超时
```

准确说：

> tmux 本身主要是操作系统工具，但它解决的是远程网络连接不稳定带来的实验中断问题。

---

# 3. 普通 SSH 终端中，程序是怎么跑的？

SSH 到服务器后执行：

```bash
python train.py
```

进程关系大概是：

```text
sshd
└── bash
    └── python train.py
```

其中：

```text
sshd   负责远程登录连接
bash   负责解释你的命令
python 负责运行实验程序
```

从代码角度看，bash 执行命令大概是：

```c
pid_t pid = fork();

if (pid == 0) {
    execlp("python", "python", "train.py", NULL);
} else {
    waitpid(pid, NULL, 0);
}
```

也就是：

```text
bash fork 出子进程
子进程 exec 成 python
python train.py 开始运行
```

---

# 4. 为什么 SSH 断了，程序可能会停？

普通 SSH 结构可以理解为：

```text
你的电脑键盘/屏幕
        ↓
SSH 客户端
        ↓ TCP 网络连接
SSH 服务端 sshd
        ↓
伪终端 PTY
        ↓
bash
        ↓
python train.py
```

当网络断开：

```text
TCP 连接断开
    ↓
SSH 连接断开
    ↓
远程终端关闭
    ↓
bash 收到 SIGHUP
    ↓
bash 退出
    ↓
子进程 python 可能被影响
```

这里最关键的是：

```text
SIGHUP = hang up signal = 终端挂断信号
```

早期 Unix 时代，用户通过物理终端连接主机。终端线断了，系统就给相关进程发送 `SIGHUP`。

现代 SSH 继承了这个机制。

所以普通 SSH 中：

```text
SSH 断线
≈ 控制终端消失
≈ bash 收到 SIGHUP
≈ 相关进程可能退出
```

---

# 5. 操作系统中的几个关键概念

## 5.1 进程 process

每个运行中的程序都是进程。

例如：

```bash
python train.py
```

就是一个 Python 进程。

可以用下面命令查看：

```bash
ps -o pid,ppid,sid,pgid,tty,stat,cmd
```

字段含义：

```text
PID   进程 ID
PPID  父进程 ID
SID   会话 ID
PGID  进程组 ID
TTY   控制终端
STAT  进程状态
CMD   命令
```

普通 SSH 中可能看到：

```text
PID    PPID   SID    PGID   TTY      STAT   CMD
1200   1190   1200   1200   pts/0    Ss     -bash
1300   1200   1200   1300   pts/0    R+     python train.py
```

重点：

```text
bash 和 python 都绑定在 pts/0 这个终端上
```

---

## 5.2 会话 session

Linux 本身就有 session 概念，不是 tmux 独有。

SSH 登录后，系统会创建一个登录会话：

```text
SSH 登录会话
└── bash
    └── python train.py
```

一个 session 通常绑定一个控制终端。

---

## 5.3 进程组 process group

进程组用于作业控制。

例如你在终端里运行：

```bash
python train.py
```

它会成为当前前台进程组。

你按：

```text
Ctrl+C
```

终端会把 `SIGINT` 发给前台进程组。

所以：

```text
Ctrl+C
    ↓
前台进程组
    ↓
python train.py
```

这就是为什么 Ctrl+C 能停掉当前程序。

---

## 5.4 控制终端 controlling terminal

可以用：

```bash
tty
```

查看当前终端：

```text
/dev/pts/0
```

普通 SSH 中：

```text
bash 和 python 的控制终端 = SSH 创建的 /dev/pts/0
```

问题是：

> 如果 SSH 连接断开，这个控制终端可能关闭，进程可能收到 SIGHUP。

tmux 的核心目标，就是让程序不要直接依赖 SSH 的控制终端。

---

# 6. tmux 改变了什么？

tmux 的核心改变是：

> 让你的程序不再直接依赖 SSH 终端，而是依赖 tmux 创建的伪终端。

不用 tmux：

```text
SSH 终端
└── bash
    └── python train.py
```

使用 tmux：

```text
tmux server
└── bash
    └── python train.py
```

SSH 只负责连接 tmux：

```text
你的电脑
    ↓ SSH
tmux client
    ↓
tmux server
    ↓
bash
    ↓
python train.py
```

所以 SSH 断了以后：

```text
tmux client 消失
tmux server 还在
bash 还在
python 还在
```

---

# 7. tmux 的 server-client 架构

tmux 分为两部分：

```text
tmux server
tmux client
```

## 7.1 tmux server

负责真正管理：

```text
session
window
pane
PTY
shell
进程
历史输出
终端状态
```

## 7.2 tmux client

负责：

```text
接收你的键盘输入
显示 tmux server 的画面
把输入转发给 server
```

结构：

```text
SSH 连接
└── tmux client
        ↓
   tmux server
        ↓
   pane 对应的 PTY
        ↓
       bash
        ↓
  python train.py
```

当你执行：

```bash
tmux new -s mt
```

tmux 会创建一个 server，并创建一个名为 `mt` 的 session。

当你按：

```text
Ctrl+b d
```

只是断开 client：

```text
tmux client 退出
tmux server 继续运行
bash/python 继续运行
```

这就是 detach。

---

# 8. tmux 的核心技术：PTY 伪终端

## 8.1 什么是 PTY？

PTY 是 pseudo terminal，伪终端。

它通常有两端：

```text
PTY master
PTY slave
```

结构：

```text
程序 A  ↔  PTY master  ↔  PTY slave  ↔  程序 B
```

在 tmux 中：

```text
tmux server 控制 PTY master
bash/python 连接 PTY slave
```

从 bash 的角度看：

```text
我连接在一个正常终端上
```

但实际上这个终端是 tmux 创建出来的。

---

## 8.2 tmux pane 与 PTY 的关系

tmux 中每个 pane 背后通常都有一个 PTY。

```text
tmux server
└── session
    └── window
        └── pane
            └── PTY
                └── bash
                    └── python train.py
```

所以 pane 不是单纯的视觉分屏，它背后是真正独立的终端环境。

---

# 9. 用 Python 理解 PTY

下面是一个极简版“伪 tmux pane”。

它创建一个 PTY，并在里面启动 bash。

```python
# mini_pty.py
import os
import pty
import subprocess
import select
import sys
import tty
import termios

# 创建一对伪终端：master 和 slave
master_fd, slave_fd = pty.openpty()

# 在 slave 端启动 bash
proc = subprocess.Popen(
    ["/bin/bash"],
    stdin=slave_fd,
    stdout=slave_fd,
    stderr=slave_fd,
    close_fds=True
)

os.close(slave_fd)

print("启动了一个 bash，PID =", proc.pid)
print("现在你输入的内容会被转发给这个 bash")
print("输入 exit 可以退出")

# 当前终端切换为 raw 模式，方便逐字符转发
old_settings = termios.tcgetattr(sys.stdin)
tty.setraw(sys.stdin.fileno())

try:
    while True:
        rlist, _, _ = select.select([sys.stdin, master_fd], [], [])

        # 用户键盘输入 → PTY master → bash
        if sys.stdin in rlist:
            data = os.read(sys.stdin.fileno(), 1024)
            if not data:
                break
            os.write(master_fd, data)

        # bash 输出 → PTY master → 用户屏幕
        if master_fd in rlist:
            data = os.read(master_fd, 1024)
            if not data:
                break
            os.write(sys.stdout.fileno(), data)

finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
```

运行：

```bash
python mini_pty.py
```

你会进入一个由 Python 托管的 bash。

这个程序的结构是：

```text
你的键盘输入
    ↓
Python 程序
    ↓
PTY master
    ↓
PTY slave
    ↓
bash

bash 输出
    ↓
PTY slave
    ↓
PTY master
    ↓
Python 程序
    ↓
你的屏幕
```

这就是 tmux 的核心机制。

真实 tmux 更复杂，但最底层的思路类似。

---

# 10. tmux 如何处理输入和输出？

普通终端：

```text
键盘输入 → bash/python
bash/python 输出 → 屏幕
```

tmux 中：

```text
键盘输入
    ↓
tmux client
    ↓
tmux server
    ↓
PTY master
    ↓
PTY slave
    ↓
bash/python
```

输出反过来：

```text
bash/python 输出
    ↓
PTY slave
    ↓
PTY master
    ↓
tmux server
    ↓
tmux client
    ↓
屏幕
```

tmux 接管了：

```text
stdin
stdout
stderr
终端尺寸
颜色控制
光标位置
滚动缓冲区
窗口状态
分屏布局
```

---

# 11. detach 和 attach 的本质

## 11.1 detach

你按：

```text
Ctrl+b d
```

表面上是退出 tmux。

本质上是：

```text
断开 tmux client
保留 tmux server
```

detach 前：

```text
SSH
└── tmux client
        ↓
   tmux server
        ↓
      bash
        ↓
   python train.py
```

detach 后：

```text
tmux server
    ↓
  bash
    ↓
python train.py
```

所以 detach 不会杀掉程序。

---

## 11.2 attach

你执行：

```bash
tmux attach -t mt
```

本质是：

```text
新建一个 tmux client
连接旧的 tmux server
显示旧 session 的当前状态
```

attach 不是恢复程序。

更准确地说：

> 程序从来没死，只是你重新连接回去了。

---

# 12. tmux server 和 client 怎么通信？

tmux server 和 client 通常通过 Unix Domain Socket 通信。

路径类似：

```text
/tmp/tmux-0/default
```

完整结构：

```text
你的电脑
    ↓ SSH / TCP
服务器上的 tmux client
    ↓ Unix Domain Socket
服务器上的 tmux server
    ↓ PTY
bash / python
```

注意区分：

```text
SSH 是本地电脑到服务器之间的网络通信
tmux client/server 是服务器内部通信
```

这就是网络知识和操作系统知识的结合点。

---

# 13. tmux 的事件循环

tmux 要同时处理很多东西：

```text
用户键盘输入
pane 0 输出
pane 1 输出
pane 2 输出
client 连接
client 断开
窗口大小变化
```

所以 tmux 需要事件循环。

极简伪代码：

```python
while True:
    readable_fds = select.select(all_fds, [], [])[0]

    for fd in readable_fds:
        if fd == client_fd:
            data = read_from_client()
            send_to_active_pane(data)

        elif fd in pane_master_fds:
            output = read_from_pane(fd)
            save_to_buffer(output)
            render_to_client(output)
```

本质是：

```text
从 client 读输入 → 发给对应 pane
从 pane 读输出 → 保存并显示给 client
```

---

# 14. 为什么 tmux 可以保存历史输出？

因为程序输出不是直接显示到你的本地屏幕，而是先进入 tmux server。

```text
python 输出日志
    ↓
PTY
    ↓
tmux server
    ↓
tmux buffer
    ↓
tmux client
    ↓
屏幕
```

tmux server 会维护滚动缓冲区：

```text
scrollback buffer
```

所以你 detach 后，程序继续输出，tmux server 仍然能接收一部分历史内容。

attach 回来后，你可以看到当前画面，也可以进入滚动模式查看历史。

---

# 15. 为什么 tmux 可以分屏？

因为 tmux 本质上是一个终端窗口管理器。

假设你的屏幕是：

```text
120 列 × 40 行
```

tmux 可以切成：

```text
左边 60 列 × 40 行
右边 60 列 × 40 行
```

每个 pane 背后有独立的 PTY：

```text
pane 0 → PTY 0 → bash 0
pane 1 → PTY 1 → bash 1
pane 2 → PTY 2 → bash 2
```

tmux server 负责：

```text
把键盘输入发给当前 pane
把不同 pane 的输出画到不同屏幕区域
处理边框
处理状态栏
处理窗口切换
```

---

# 16. Ctrl+C 在 tmux 中为什么仍然有效？

你在 tmux 里按：

```text
Ctrl+C
```

路径是：

```text
键盘 Ctrl+C
    ↓
tmux client
    ↓
tmux server
    ↓
PTY master
    ↓
PTY slave
    ↓
前台进程组
    ↓
python train.py
```

所以 Python 仍然能收到正常的 `SIGINT`。

tmux 并不是粗暴地“截获一切”，而是正确模拟了一个终端环境。

---

# 17. tmux 与 nohup 的区别

## 17.1 nohup

```bash
nohup python train.py > train.log 2>&1 &
```

含义：

```text
忽略 SIGHUP
输出写入 train.log
进程放到后台
```

它保护的是一个命令。

## 17.2 tmux

tmux 保护的是整个终端会话：

```text
当前目录
环境变量
conda/micromamba 环境
shell 状态
多个窗口
多个 pane
实时输出
交互式操作
```

对比：

| 工具      | 本质                | 适合场景             |
| --------- | ------------------- | -------------------- |
| `&`       | 放到 shell 后台     | 临时后台任务         |
| `nohup`   | 单命令忽略 SIGHUP   | 简单长任务           |
| `disown`  | 从 shell 作业表移除 | 避免 shell 退出影响  |
| `screen`  | 老牌终端复用器      | 类似 tmux            |
| `tmux`    | 完整终端会话管理    | 科研实验、远程服务器 |
| `systemd` | 系统服务管理        | 长期服务             |

对科研实验来说，tmux 更合适。

---

# 18. tmux 的层级结构

tmux 的核心层级：

```text
tmux server
└── session
    ├── window
    │   ├── pane
    │   │   └── PTY
    │   │       └── bash
    │   │           └── python train.py
    │   └── pane
    │       └── PTY
    │           └── watch nvidia-smi
    └── window
        └── pane
            └── PTY
                └── vim
```

## 18.1 session

session 是一组工作空间。

例如：

```bash
tmux new -s flash
tmux new -s 3dgs
tmux new -s monogs
```

可以分别用于不同项目。

## 18.2 window

window 类似浏览器标签页。

例如：

```text
flash session
├── window 0: preprocess
├── window 1: train
└── window 2: monitor
```

## 18.3 pane

pane 是分屏区域。

例如：

```text
window 0
├── pane 0: python train.py
└── pane 1: watch nvidia-smi
```

---

# 19. 用伪代码理解 tmux 数据结构

一个极简 tmux 可以这样抽象：

```python
class Pane:
    def __init__(self):
        self.master_fd = None
        self.slave_fd = None
        self.process = None
        self.buffer = []

class Window:
    def __init__(self, name):
        self.name = name
        self.panes = []
        self.active_pane = 0

class Session:
    def __init__(self, name):
        self.name = name
        self.windows = []
        self.active_window = 0

class TmuxServer:
    def __init__(self):
        self.sessions = {}

    def new_session(self, name):
        session = Session(name)
        window = Window("0")
        pane = Pane()

        window.panes.append(pane)
        session.windows.append(window)
        self.sessions[name] = session
```

真实 tmux 比这个复杂很多，但核心结构类似。

---

# 20. 用伪代码理解 detach/attach

## 20.1 detach

```python
def detach_client(client):
    clients.remove(client)
    client.close()
```

注意：

```text
只移除 client
不删除 session
不关闭 pane
不杀 bash
不杀 python
```

## 20.2 attach

```python
def attach_client(client, session_name):
    session = sessions[session_name]
    client.session = session

    active_pane = session.current_pane()
    client.send(active_pane.buffer)
```

attach 的本质是：

```text
新的 client 连接已有 session
重新显示已有 pane 的状态
```

---

# 21. 真实 tmux 源码可以按这些模块理解

tmux 是 C 写的。

可以按模块理解：

```text
server.c        后台 server 主逻辑
client.c        客户端连接逻辑
session.c       session 管理
window.c        window 管理
window-pane.c   pane 管理
tty.c           终端输入输出
screen.c        屏幕缓冲区
input.c         解析终端控制序列
cmd-*.c         各种 tmux 命令
```

执行：

```bash
tmux new -s mt
```

大概流程：

```text
解析 new-session 命令
创建 session
创建 window
创建 pane
创建 PTY
在 PTY slave 端启动 shell
server 进入事件循环
client 连接 server 显示画面
```

执行：

```text
Ctrl+b d
```

大概流程：

```text
client 发送 detach-client 命令
server 解除 client 和 session 的绑定
client 退出
server 保留 session/window/pane
```

执行：

```bash
tmux attach -t mt
```

大概流程：

```text
新 client 连接 server socket
请求 attach 到 mt session
server 把 session 当前画面发给 client
后续继续转发输入输出
```

---

# 22. 观察 tmux 原理的实验命令

## 22.1 查看当前终端

普通 SSH 中：

```bash
tty
```

可能输出：

```text
/dev/pts/0
```

进入 tmux 后再执行：

```bash
tty
```

可能输出：

```text
/dev/pts/2
```

说明 tmux 创建了新的伪终端。

---

## 22.2 查看是否在 tmux 中

```bash
echo $TMUX
```

在 tmux 中可能输出：

```text
/tmp/tmux-0/default,12345,0
```

不在 tmux 中一般为空。

---

## 22.3 查看进程树

```bash
ps -ef --forest
```

或者：

```bash
pstree -p
```

你可能看到：

```text
tmux(12345)
 └─bash(12360)
    └─python(12400)
```

说明 Python 是挂在 tmux 下面的。

---

## 22.4 查看进程关系

```bash
ps -o pid,ppid,sid,pgid,tty,stat,cmd
```

可以观察：

```text
PID
PPID
SID
PGID
TTY
```

这能帮助理解：

```text
进程
父子进程
会话
进程组
控制终端
```

---

# 23. 推荐科研服务器使用流程

以后只要任务超过 5 分钟，都建议用 tmux。

## 23.1 新建会话

```bash
tmux new -s experiment_name
```

例如：

```bash
tmux new -s flash
tmux new -s 3dgs
tmux new -s monogs
```

## 23.2 在 tmux 中配置环境

例如：

```bash
export PATH="$HOME/.local/bin:$PATH"
eval "$(micromamba shell hook -s bash)"

micromamba activate /root/autodl-tmp/envs/mt39

cd /root/autodl-tmp/flashavatar_tools/metrical-tracker
```

## 23.3 运行实验

```bash
python train.py
```

或者：

```bash
python tracker.py
```

## 23.4 暂时离开

```text
Ctrl+b
然后按 d
```

注意不是三个键同时按。

## 23.5 查看已有会话

```bash
tmux ls
```

## 23.6 回到会话

```bash
tmux attach -t flash
```

简写：

```bash
tmux a -t flash
```

## 23.7 杀掉会话

确认任务不需要后：

```bash
tmux kill-session -t flash
```

---

# 24. 常用快捷键

tmux 默认前缀键是：

```text
Ctrl+b
```

常用操作：

```text
Ctrl+b d       detach，退出但不停止任务
Ctrl+b c       新建 window
Ctrl+b n       下一个 window
Ctrl+b p       上一个 window
Ctrl+b "       上下分屏
Ctrl+b %       左右分屏
Ctrl+b x       关闭当前 pane
Ctrl+b [       进入滚动模式
q              退出滚动模式
```

常用命令：

```bash
tmux new -s name          # 新建会话
tmux ls                   # 查看会话
tmux a -t name             # 进入会话
tmux kill-session -t name  # 删除会话
```

---

# 25. tmux 的边界：它不能解决什么？

tmux 能解决：

```text
SSH 断开
VS Code Remote SSH 掉线
本地终端关闭
网络不稳定
```

tmux 不能解决：

```text
服务器关机
AutoDL 实例被释放
程序自己崩溃
CUDA OOM
磁盘满了
环境配置错误
训练代码报错
```

所以：

> tmux 防的是连接断开，不是防服务器宕机，也不是防程序错误。

---

# 26. 最重要的心智模型

普通 SSH：

```text
SSH 连接控制终端
终端控制 bash
bash 控制 python
```

所以 SSH 断开，链条容易断。

tmux：

```text
tmux server 控制伪终端
伪终端控制 bash
bash 控制 python

SSH 连接只控制 tmux client
```

所以 SSH 断开，只是 client 没了：

```text
tmux client 消失
```

真正任务还在：

```text
tmux server
└── bash
    └── python train.py
```

---

# 27. 最终总结

tmux 的本质是：

> 一个运行在服务器上的用户态终端复用器。

它通过：

```text
server-client 架构
PTY 伪终端
Unix Domain Socket
I/O 转发
终端缓冲区
session/window/pane 管理
```

实现：

```text
会话保持
断线重连
多窗口
分屏
历史输出保存
长期任务托管
```

从计算机专业角度看，tmux 是一个非常好的综合案例。

它把这些知识串起来：

```text
操作系统：
进程、父子进程、会话、进程组、信号、控制终端、PTY、文件描述符、I/O 多路复用

计算机网络：
SSH、TCP、远程连接、网络断开、客户端/服务器模型
```

一句话总结：

> 程序不是跑在“窗口”里，而是跑在操作系统的进程模型里；窗口只是你观察和控制程序的入口。tmux 的价值，就是把这个入口和真正运行的程序解耦。
