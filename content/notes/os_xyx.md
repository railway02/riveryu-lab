---
title: "操作系统学习笔记：从状态机视角的底层解构"
date: 2026-06-01T10:00:00+08:00
draft: false
tags: ["OS", "System Programming", "C", "Learning Notes"]
summary: "从状态机的视角出发，全面梳理并发、虚拟化与持久化三大核心机制，并记录 OS 实验中的踩坑与顿悟。"
---

**导语：**

我们每天都在敲代码、跑模型、配环境，但底层的硬件资源究竟是如何被调度的？程序在操作系统眼中到底长什么样？为什么一个简单的 `count++` 在多线程环境下会出错？为什么看起来顺序执行的代码，在现代 CPU 上可能并不真的按源代码顺序发生？

这篇笔记记录我系统学习操作系统核心原理的过程。它不是单纯背概念，而是试图从一个更底层的视角理解操作系统：**程序是状态机，操作系统是状态机的管理者，并发、虚拟化和持久化都是围绕“状态”展开的设计。**

---

## 一、核心心智模型：程序即状态机

> **核心思想：任何程序都可以抽象为一个状态机。C 语言程序的执行，本质上是状态在指令驱动下不断迁移。**

在操作系统眼中，程序不是一段“高级语言文本”，而是一台正在运行的状态机。

### 1. 状态是什么？

程序的状态主要包括：

- CPU 寄存器的值；
- 程序计数器 PC；
- 栈上的局部变量；
- 堆上的动态对象；
- 全局变量；
- 打开的文件描述符；
- 进程的地址空间；
- 线程的执行上下文。

也就是说，一个程序“现在运行到哪里了”“变量是多少”“栈里有什么”“打开了哪些文件”，这些共同构成了程序当前的状态。

### 2. 状态迁移是什么？

CPU 每执行一条指令，程序状态就发生一次迁移。

例如 C 语言里的：

```c
x = x + 1;
```

在底层可能对应：

```asm
load x
add 1
store x
```

**高级语言中的“一句代码”，在机器层面往往不是一个原子操作。**

### 3. 系统调用是什么？

普通程序不能随便访问硬件，也不能直接操作磁盘、网卡、进程表等核心资源。

当程序需要操作系统帮忙时，就会发起系统调用，例如：

```c
read();
write();
fork();
exec();
wait();
open();
close();
```

从状态机视角看，系统调用就是：

> 用户程序这个状态机向操作系统发出请求，操作系统以内核权限修改系统状态，然后把结果返回给用户程序。

比如：

```c
write(1, "hello\n", 6);
```



---

## 二、并发：多个状态机同时运行时的混乱与秩序

并发是操作系统里最容易出错、也最能体现底层理解能力的部分。

单线程程序只有一个状态机，执行路径相对清晰。多线程/多进程程序则是多个状态机交错运行。如果它们共享数据，就可能出现非常隐蔽的问题。

### 1. 为什么需要同步？

操作系统中有一类进程叫 cooperating processes，也就是会互相影响、共享数据或通过消息协作的进程。

它们可能共享：

- 同一段地址空间；
- 共享内存；
- 文件；
- 管道；
- 内核对象；
- 某些全局数据结构。

一旦多个进程或线程同时访问共享数据，就可能发生 data inconsistency，也就是数据不一致。

经典例子是 producer-consumer，也就是生产者-消费者问题。

假设有一个固定大小的 buffer：

- producer 负责生产数据并放入 buffer；
- consumer 负责从 buffer 取出数据并消费；
- `count` 表示当前 buffer 中有多少个元素。

直觉上，producer 放入一个元素：

```c
count++;
```

consumer 取出一个元素：

```c
count--;
```

如果初始 `count = 5`，一个线程加一，一个线程减一，最后似乎应该还是 5。

但问题在于，`count++` 并不是一步完成的。它可能被拆成：

```c
register = count;
register = register + 1;
count = register;
```

`count--` 也类似。

一种可能的执行顺序是：

```text
初始 count = 5

Producer 读取 count，得到 5
Consumer 读取 count，也得到 5

Producer 计算 5 + 1 = 6
Consumer 计算 5 - 1 = 4

Producer 写回 count = 6
Consumer 写回 count = 4
```

最后结果变成了 4。

可是从逻辑上讲，一次生产和一次消费之后，`count` 应该仍然是 5。

这就是 race condition。

### 2. Race Condition：结果取决于谁先跑

Race condition 的本质是：

> 多个进程/线程同时访问并修改共享数据，程序最终结果取决于它们具体的执行交错顺序。

这个问题非常危险，因为它往往不是每次都出现。

同一段代码，可能今天运行正常，明天运行错误；单核上正常，多核上出错；加了打印语句正常，去掉打印语句又崩。

这类问题难调试的根本原因是：**bug 不只由代码决定，还由调度时机决定。**

在内核里，这个问题更加严重。

例如两个进程同时调用 `fork()`，内核需要给它们的新子进程分配 PID。如果内核维护了一个 `next_available_pid`，但没有保护好这个变量，就可能出现两个子进程拿到同一个 PID 的严重错误。



---

## 三、临界区问题：给共享状态加秩序

### 1. Critical Section 是什么？

临界区 critical section 是指：

> 一段会访问或修改共享数据的代码。

例如：

```c
buffer[in] = next_produced;
in = (in + 1) % BUFFER_SIZE;
count++;
```

这段代码修改了共享 buffer、`in` 和 `count`，因此属于临界区。

一个典型进程结构如下：

```c
while (true) {
    entry section;       // 申请进入临界区
    critical section;    // 操作共享数据
    exit section;        // 退出临界区
    remainder section;   // 其他代码
}
```

同步机制要解决的问题就是：

> 如何让多个进程安全地进入和退出临界区？

### 2. 临界区问题的三个要求

一个正确的 critical-section solution 必须满足三个条件。

#### Mutual Exclusion：互斥

同一时间最多只能有一个进程进入临界区。

这是最基本要求。

如果两个线程能同时修改共享变量，那么同步就失败了。

#### Progress：前进

如果当前没有进程在临界区，并且有进程想进入临界区，那么系统必须在有限时间内选出一个进程进入。

也就是说，系统不能所有人都在等，没人推进。

#### Bounded Waiting：有限等待

一个进程提出进入临界区请求之后，不能无限期等待。

更具体地说，在它提出请求后，其他进程只能有限次地先于它进入临界区。

这保证了不会有某个进程永远抢不到资源，也就是避免 starvation。

---

## 四、Peterson 算法：纯软件互斥的思想实验

Peterson's Solution 是一个经典的双进程互斥算法。

它使用两个共享变量：

```c
int turn;
boolean flag[2];
```

含义是：

```text
flag[i]：进程 i 是否想进入临界区
turn：发生竞争时，应该让谁先进入
```

进程 `Pi` 的逻辑如下：

```c
while (true) {
    flag[i] = true;
    turn = 1 - i;

    while (flag[1 - i] && turn == 1 - i)
        ;

    /* critical section */

    flag[i] = false;

    /* remainder section */
}
```

它的直觉可以这样理解：

- `flag[i] = true`：我想进临界区；
- `turn = 1 - i`：如果对方也想进，我让对方先；
- `while (...)`：如果对方确实想进，并且现在轮到对方，那我等待。

如果两个进程同时想进入临界区，二者都会把自己的 `flag` 设置为 true，同时争夺 `turn`。

最终 `turn` 只能是 0 或 1。谁被 `turn` 指向，谁拥有优先权；另一个进程等待。

Peterson 算法的价值在于，它展示了一个纯软件互斥协议如何同时考虑：

- 我是否想进入；
- 对方是否想进入；
- 冲突时谁让步。



---

## 五、现代 CPU 上的新问题：重排与内存模型

Peterson 算法在理论模型中成立，但在现代计算机上不一定可靠。

原因是现代 CPU 和编译器为了性能，会进行 instruction reordering 或 memory reordering。

例如源代码写的是：

```c
flag[i] = true;
turn = 1 - i;
```

编译器或处理器可能认为，在单线程语义下交换顺序不影响结果，于是重排成：

```c
turn = 1 - i;
flag[i] = true;
```

在单线程程序中，这可能没有区别。

但在多线程程序中，另一个 CPU 核心正在观察这些共享变量，顺序变化就可能破坏同步协议。

这说明一个非常重要的事实：

> 多线程程序不能只用单线程直觉理解。

### 1. Memory Barrier：内存屏障

为了解决重排问题，系统需要 memory barrier。

Memory barrier 的作用是：

> 保证屏障之前的内存访问先于屏障之后的内存访问完成，并让这些修改对其他处理器可见。

简单说，它是在告诉 CPU 和编译器：

```text
这里不能乱排。
前面的读写必须先完成。
后面的读写不能提前。
```

### 2. Memory Model：强一致与弱一致

不同处理器架构对内存可见性的保证不同。

一般可以分为：

- strongly ordered：一个处理器上的内存修改会立刻对其他处理器可见；
- weakly ordered：一个处理器上的内存修改不一定马上对其他处理器可见。

现代并发编程里，锁、原子变量、内存屏障，本质上都是在和这些底层内存模型打交道。

---

## 六、硬件原子指令：同步机制的地基

纯软件方案在现代硬件上会遇到很多问题，因此实际系统通常依赖硬件原子指令。

所谓 atomic，意思是：

> 一个操作要么完整执行，要么完全不执行，执行过程中不能被其他线程打断。

### 1. test_and_set

`test_and_set()` 可以抽象为：

```c
bool test_and_set(bool *target) {
    bool rv = *target;
    *target = true;
    return rv;
}
```

它做了两件事：

1. 读出 `target` 的旧值；
2. 把 `target` 设置为 true。

关键在于，读和写是一个不可分割的原子操作。

可以用它实现简单自旋锁：

```c
while (true) {
    while (test_and_set(&lock))
        ;

    /* critical section */

    lock = false;

    /* remainder section */
}
```

如果 `lock` 原来是 false，某个线程执行 `test_and_set` 后会获得锁，并把 `lock` 改成 true。

其他线程再来时，发现 `lock` 是 true，只能等待。

### 2. compare_and_swap：CAS

CAS 是另一个非常重要的原子指令。

```c
int compare_and_swap(int *value, int expected, int new_value) {
    int temp = *value;

    if (*value == expected)
        *value = new_value;

    return temp;
}
```

它的含义是：

> 如果 value 当前值等于 expected，就把它改成 new_value；否则不修改。

整个过程是 atomic 的。

用 CAS 可以实现锁：

```c
while (true) {
    while (compare_and_swap(&lock, 0, 1) != 0)
        ;

    /* critical section */

    lock = 0;

    /* remainder section */
}
```

意思是：

- 如果 `lock == 0`，说明锁空闲；
- CAS 把它改成 1；
- 修改成功的线程进入临界区；
- 其他线程发现 `lock != 0`，继续等待。

### 3. Atomic Variable：原子变量不是万能药

CAS 还可以实现原子变量。

例如把不安全的：

```c
count++;
```

改成：

```c
increment(&count);
```

一种 CAS 风格实现是：

```c
void increment(atomic_int *v) {
    int temp;

    do {
        temp = *v;
    } while (temp != compare_and_swap(v, temp, temp + 1));
}
```

它的思想是：

1. 先读出旧值；
2. 尝试把旧值改成旧值 + 1；
3. 如果期间别人没有改过，就成功；
4. 如果别人改过，就重新尝试。

这是一种典型的乐观并发控制。

但是原子变量只能保护单个变量的一次更新，不能解决所有同步问题。

例如 producer-consumer 不只涉及 `count`，还涉及：

- buffer；
- in；
- out；
- count；
- buffer 满和空的条件。

这些是多个状态之间的一致性问题，仅靠一个 atomic count 不够。

---

## 七、锁：从自旋到阻塞

### 1. Mutex Lock：互斥锁

Mutex 是最常见的同步工具。

使用方式一般是：

```c
acquire();

/* critical section */

release();
```

含义很直接：

- 进入临界区前先拿锁；
- 离开临界区后释放锁；
- 没拿到锁的线程等待。

最朴素的实现是：

```c
void acquire() {
    while (!available)
        ;
    available = false;
}

void release() {
    available = true;
}
```

但这段代码本身不安全，因为检查 `available` 和修改 `available` 不是原子的。

所以真实的 mutex 底层必须依赖 test_and_set、CAS 或其他硬件同步指令。

### 2. Spinlock：自旋锁

如果线程拿不到锁，就一直循环等待：

```c
while (lock_is_busy)
    ;
```

这就是 busy waiting，也叫 spinlock。

自旋锁的特点是：

- 不让出 CPU；
- 不进入睡眠；
- 一直检查锁是否释放。

它的优点是：如果锁很快释放，可以避免上下文切换开销。

它的缺点是：如果锁持有时间较长，会浪费大量 CPU。

所以：

```text
短临界区：自旋锁可能更合适
长临界区：阻塞式 mutex 更合适
```

### 3. Blocking Mutex：阻塞式锁

阻塞式锁在拿不到锁时，不是一直空转，而是把线程挂起，放入等待队列。

当锁释放时，操作系统再唤醒等待队列里的线程。

这避免了忙等浪费 CPU，但代价是需要上下文切换。

所以同步工具的选择，本质上是性能权衡：

```text
忙等浪费 CPU，但避免切换成本；
阻塞节省 CPU，但增加调度成本。
```

---

## 八、Semaphore：不只是互斥，更是顺序控制

Semaphore 是比 mutex 更通用的同步工具。

一个 semaphore 是一个整数变量，只能通过两个原子操作访问：

```c
wait(S);
signal(S);
```

传统上也叫：

```text
P 操作
V 操作
```

### 1. wait 和 signal 的含义

```c
void wait(S) {
    while (S <= 0)
        ;
    S--;
}
```

含义是：

> 我要申请一个资源。如果资源数量大于 0，就占用一个；如果没有资源，就等待。

```c
void signal(S) {
    S++;
}
```

含义是：

> 我释放一个资源。

### 2. Binary Semaphore 和 Counting Semaphore

Binary semaphore 的值只能是 0 或 1，类似 mutex。

Counting semaphore 的值可以大于 1，适合表示多个同类资源。

例如有 5 台打印机：

```c
semaphore printer = 5;
```

每个进程使用前：

```c
wait(printer);
```

使用后：

```c
signal(printer);
```

这样最多允许 5 个进程同时使用打印机。

### 3. 用 semaphore 解决 producer-consumer

生产者-消费者问题一般用三个信号量：

```c
semaphore mutex = 1;   // 保护 buffer 的互斥访问
semaphore empty = n;   // 空槽数量
semaphore full = 0;    // 已有数据数量
```

Producer：

```c
while (true) {
    produce item;

    wait(empty);
    wait(mutex);

    add item to buffer;

    signal(mutex);
    signal(full);
}
```

Consumer：

```c
while (true) {
    wait(full);
    wait(mutex);

    remove item from buffer;

    signal(mutex);
    signal(empty);

    consume item;
}
```

这里有一个非常重要的区分：

```text
mutex 解决互斥问题；
empty/full 解决同步顺序问题。
```

也就是说：

- 互斥：同一时间只能有一个线程操作 buffer；
- 同步：buffer 空时不能消费，buffer 满时不能生产。

很多人学信号量混乱，就是因为没有分清“互斥”和“同步顺序”。

### 4. 带等待队列的 semaphore

朴素的 semaphore 会 busy waiting：

```c
while (S <= 0)
    ;
```

这会浪费 CPU。

真实操作系统通常会给 semaphore 加等待队列：

```c
typedef struct {
    int value;
    struct list_head *waiting_queue;
} semaphore;
```

`wait()`：

```c
wait(semaphore *S) {
    S->value--;

    if (S->value < 0) {
        add this process to S->waiting_queue;
        block();
    }
}
```

`signal()`：

```c
signal(semaphore *S) {
    S->value++;

    if (S->value <= 0) {
        remove a process P from S->waiting_queue;
        wakeup(P);
    }
}
```

这样，拿不到资源的进程会进入阻塞态，不再浪费 CPU。

---

## 九、死锁与优先级反转：同步工具的副作用

同步机制可以保护共享数据，但如果使用不当，也会制造新的问题。

### 1. Deadlock：死锁

死锁是指多个进程互相等待对方释放资源，导致谁也无法继续执行。

经典例子是：

```text
线程 A 拿着锁 1，等待锁 2；
线程 B 拿着锁 2，等待锁 1。
```

最后二者都无法前进。

死锁通常需要四个必要条件：

1. Mutual Exclusion：资源互斥使用；
2. Hold and Wait：持有一个资源的同时等待另一个资源；
3. No Preemption：资源不能被强制抢占；
4. Circular Wait：存在环形等待关系。

破坏其中任意一个条件，都可以避免死锁。

在实际编程中，最常见的做法是：

> 规定多个锁的获取顺序，避免循环等待。

### 2. Priority Inversion：优先级反转

优先级反转发生在优先级调度系统中。

假设有三个进程：

```text
H：高优先级
M：中优先级
L：低优先级
```

现在：

1. L 拿到了某个锁；
2. H 需要这个锁，因此被阻塞；
3. M 不需要这个锁，而且优先级高于 L；
4. CPU 开始运行 M；
5. L 无法运行，也就无法释放锁；
6. H 虽然优先级最高，却一直等不到锁。

结果就是：

> 中优先级的 M 间接阻塞了高优先级的 H。

这就是 priority inversion。

解决方法是 priority inheritance，优先级继承。

如果低优先级进程 L 持有高优先级进程 H 需要的锁，那么 L 临时继承 H 的优先级。

这样 L 可以尽快运行并释放锁，之后 H 继续执行，L 再恢复原本优先级。

---

## 十、虚拟化：操作系统制造的两种幻觉

操作系统通过虚拟化，让程序产生两种幻觉：

```text
我独占了 CPU；
我拥有一大片连续内存。
```

这两个幻觉极大简化了程序设计。

### 1. CPU 虚拟化：进程与上下文切换

一个 CPU 核心同一时刻只能执行一个线程。

但我们平时感觉很多程序在“同时运行”，这是因为操作系统不断进行上下文切换。

上下文切换时，操作系统会保存当前进程的状态，例如：

- 程序计数器；
- 寄存器；
- 栈指针；
- 页表信息；
- 调度相关信息。

然后恢复另一个进程的状态。

从状态机视角看：

> 上下文切换就是操作系统暂停一个状态机，保存现场，再恢复另一个状态机。

### 2. 进程 API：fork、exec、wait

Unix 风格的进程 API 非常经典。

`fork()` 创建一个几乎一模一样的子进程。

`exec()` 用新程序替换当前进程的地址空间。

`wait()` 等待子进程结束并回收资源。

它们组合起来，就能实现 shell 运行命令的基本逻辑：

```c
pid = fork();

if (pid == 0) {
    exec("program");
} else {
    wait();
}
```

这套设计非常优雅：创建进程和加载新程序被拆成了两个动作，因此 shell 可以在 `fork()` 和 `exec()` 之间做重定向、管道等操作。

### 3. 调度算法

操作系统需要决定下一个运行谁。

常见调度算法包括：

- FIFO：先来先服务；
- SJF：最短作业优先；
- RR：时间片轮转；
- MLFQ：多级反馈队列。

调度算法本质上是在不同目标之间权衡：

- 响应时间；
- 周转时间；
- 吞吐量；
- 公平性；
- 交互体验。

---

## 十一、内存虚拟化：地址空间与分页

操作系统还会让每个进程以为自己拥有一片完整、连续、私有的内存空间。

这就是 address space。

一个典型进程地址空间包括：

- code/text 段；
- data 段；
- heap；
- stack；
- mmap 区域；
- kernel 映射区域。

程序使用的是虚拟地址，真正访问内存时，需要通过页表翻译成物理地址。

### 1. 分页机制

分页把虚拟地址空间和物理内存都切成固定大小的页。

虚拟页通过页表映射到物理页框。

这样带来几个好处：

- 每个进程可以拥有独立地址空间；
- 物理内存可以不连续；
- 可以进行权限控制；
- 可以支持共享内存；
- 可以支持按需加载和 swap。

### 2. TLB：地址翻译缓存

如果每次访存都查页表，速度会很慢。

所以 CPU 提供 TLB，也就是 Translation Lookaside Buffer，用来缓存最近的虚拟地址到物理地址的翻译结果。

TLB 命中时，地址翻译很快。

TLB 未命中时，才需要查页表。

### 3. Page Fault：缺页中断

当程序访问的虚拟页还没有对应物理页时，会触发 page fault。

这时操作系统介入，可能会：

- 从磁盘加载页面；
- 分配新的物理页；
- 触发 copy-on-write；
- 或者发现非法访问，杀死进程。

缺页中断说明，内存虚拟化不是简单的地址映射，而是一套动态管理机制。

---

## 十二、持久化：让状态穿越断电

内存中的状态断电就会消失。

持久化解决的问题是：

> 如何把数据安全、可靠地保存到慢速、复杂、可能出错的存储设备上？

### 1. 存储设备

HDD 的性能受寻道时间和旋转延迟影响。

SSD 没有机械结构，但有闪存擦写限制、写放大和垃圾回收问题。

这说明文件系统不能只考虑抽象接口，还要考虑底层设备特性。

### 2. 文件系统

文件系统把磁盘抽象成：

- 文件；
- 目录；
- inode；
- 数据块；
- 元数据；
- 空闲空间管理结构。

inode 保存文件元数据，例如：

- 文件大小；
- 权限；
- 所有者；
- 时间戳；
- 数据块位置。

目录本质上是文件名到 inode 的映射。

### 3. 崩溃一致性

文件系统最难的问题之一是 crash consistency。

例如写文件时，系统可能刚更新 inode，还没写完数据块就突然断电。

这可能导致文件系统状态不一致。

Journaling 文件系统通过先写日志，再真正修改文件系统结构，降低崩溃导致损坏的风险。

