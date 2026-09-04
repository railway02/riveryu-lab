---
title: "FGSM 对抗攻击代码深度笔记：从 LeNet、梯度、图像扰动到模型误判"
date: 2026-06-07T10:00:00+08:00
draft: false
tags: [ "Computer Vision", "FGSM", "Pytorch"]
summary: "对抗笔记"
---

学习了[Pytorch官方文档的教程](https://docs.pytorch.org/tutorials/beginner/fgsm_tutorial.html)，并且进行了扩展补充与思考

这段代码展示了一个经典流程：

```text
MNIST 原始图片
    ↓
LeNet 正常分类
    ↓
计算 loss 对输入图片的梯度
    ↓
按照梯度方向修改图片像素
    ↓
得到对抗样本
    ↓
模型重新分类
    ↓
观察模型是否被误导
```

---

# 一、代码作用

目标是：

**测试一个已经训练好的 LeNet 模型，在不同强度的 FGSM 攻击下，对 MNIST 手写数字的分类准确率会下降多少。**

代码中有一个攻击强度列表：

```python
epsilons = [0, .05, .1, .15, .2, .25, .3]
```

每个 `epsilon` 表示允许给图片加多大的扰动。

当：

```python
epsilon = 0
```

表示不攻击。

当：

```python
epsilon = 0.3
```

表示攻击较强，图像像素会被明显改变，模型更容易误判。

最终代码会输出类似：

```text
Epsilon: 0     Test Accuracy = ...
Epsilon: 0.05  Test Accuracy = ...
Epsilon: 0.1   Test Accuracy = ...
...
```

通常结果会表现为：

```text
epsilon 越大，模型准确率越低
```

这说明模型对某些特定方向的扰动非常敏感。

---

# 二、图片本身是否变化

答案是：

**发生了变化，但不是修改原始数据集，而是在内存中生成了一张新的对抗图片。**

原图是：

```python
data
```
反归一化后是：

```python
data_denorm
```

攻击后生成的新图是：

```python
perturbed_data
```

核心代码：

```python
perturbed_image = image + epsilon * sign_data_grad
```

这一步会改变图片像素值。

比如某个像素原来是：

```text
0.40
```

梯度符号是：

```text
+1
```

如果：

```text
epsilon = 0.1
```

那么攻击后这个像素变成：

```text
0.40 + 0.1 = 0.50
```

如果梯度符号是：

```text
-1
```

则变成：

```text
0.40 - 0.1 = 0.30
```

所以图片确实变化了。

但是原始 MNIST 文件没有被覆盖，因为代码没有保存回硬盘。

因此可以理解为：

```text
原始图片文件：没变
内存中的对抗样本：变了
模型看到的输入：变了
```

---

# 三、FGSM 的核心思想

FGSM 全称是：

```text
Fast Gradient Sign Method
```

中文可以理解为：

```text
快速梯度符号攻击
```

它的核心公式是：

$$ x_{adv}=clip(x+\epsilon \cdot sign(\nabla_x J(\theta,x,y)),0,1) $$

逐个解释：

| 符号            | 含义                               |
| --------------- | ---------------------------------- |
| $x$             | 原始图片                           |
| $x_{adv}$       | 对抗样本图片                       |
| $\epsilon$      | 扰动强度                           |
| $J(\theta,x,y)$ | 模型在输入 (x)、标签 (y) 上的 loss |
| $\nabla_x J$    | loss 对输入图片的梯度              |
| $sign(\cdot)$   | 取梯度符号，只保留方向             |
| $clip$          | 把像素限制在合法范围 `[0,1]`       |


> 找到每个像素往哪个方向变化最容易让模型 loss 变大，然后把所有像素都朝这个方向轻轻推一下。

---

# 四、不是随机加噪声

FGSM 不是随机噪声。随机噪声可能是：

```python
image + random_noise
```

但是 FGSM 是：

```python
image + epsilon * sign(gradient)
```

它利用了模型自己的梯度信息。FGSM 知道：

```text
哪些像素应该变亮
哪些像素应该变暗
怎样改最容易让模型犯错
```

所以 FGSM 是一种有方向、有目的的攻击。

这就是为什么有时候人眼看起来变化很小，但模型已经错了。

---

# 五、从数学上理解 FGSM 

模型的 loss 是：

$$ J(\theta,x,y) $$

其中：

* $\theta$：模型参数；
* $x$：输入图片；
* $y$：真实标签。

攻击者的目标是：

```text
在尽量不明显改变图片的情况下，让 loss 变大
```

也就是：

$$ \max_{\delta} J(\theta, x+\delta, y) $$

同时限制扰动不能太大：

$$ ||\delta||_{\infty} \leq \epsilon $$

其中 $L_\infty$ 约束表示：

> 每一个像素最多只能改变 $\epsilon$。

为了简化问题，对 loss 做一阶泰勒展开：

$$ J(x+\delta) \approx J(x) + \delta^T \nabla_x J(x) $$

其中 $J(x)$ 是固定的，所以要最大化：

$$ \delta^T \nabla_x J(x) $$

在约束：

$$ ||\delta||_{\infty} \leq \epsilon $$

最优选择是：

$$ \delta = \epsilon \cdot sign(\nabla_x J(x)) $$

所以得到 FGSM：

$$ x_{adv}=x+\epsilon \cdot sign(\nabla_x J(x)) $$

代码中对应：

```python
sign_data_grad = data_grad.sign()
perturbed_image = image + epsilon * sign_data_grad
```

---

# 六、 $sign$ 的作用

代码中：

```python
sign_data_grad = data_grad.sign()
```

这里没有直接用：

```python
image + epsilon * data_grad
```

而是用了：

```python
image + epsilon * data_grad.sign()
```

原因是 FGSM 使用的是 $L_\infty$ 范数约束，每个像素最多改变 $\epsilon$。

如果直接用原始梯度，某些像素可能变化很大，某些像素变化很小，不容易保证统一的最大扰动限制。

而用 `sign()` 之后，每个像素的扰动只可能是：

```text
+epsilon
-epsilon
0
```


```text
梯度为正：像素加 epsilon
梯度为负：像素减 epsilon
梯度为零：像素不变
```

这样正好满足：

$$ ||\delta||_\infty \leq \epsilon $$

---

# 七、LeNet 模型结构笔记

代码定义了一个 LeNet 风格的 CNN：

```python
class Net(nn.Module):
```

它由这些层组成：

```text
Conv2d
ReLU
Conv2d
ReLU
MaxPool
Dropout
Flatten
Linear
ReLU
Dropout
Linear
LogSoftmax
```

整体结构可以画成：

```text
输入图片 [1, 1, 28, 28]
        ↓
conv1: 1 → 32, kernel=3
        ↓
[1, 32, 26, 26]
        ↓
ReLU
        ↓
conv2: 32 → 64, kernel=3
        ↓
[1, 64, 24, 24]
        ↓
ReLU
        ↓
MaxPool2d(2)
        ↓
[1, 64, 12, 12]
        ↓
Dropout
        ↓
Flatten
        ↓
[1, 9216]
        ↓
fc1: 9216 → 128
        ↓
ReLU
        ↓
Dropout
        ↓
fc2: 128 → 10
        ↓
log_softmax
        ↓
输出 10 类 log 概率
```

---

# 八、为什么 `fc1` 输入是 9216？

这来自前面卷积和池化后的特征图尺寸。

输入 MNIST 图片是：

```text
[1, 1, 28, 28]
```

第一层卷积：

```python
self.conv1 = nn.Conv2d(1, 32, 3, 1)
```

没有 padding，kernel size 是 3，所以：

$$ 28 - 3 + 1 = 26 $$

得到：

```text
[1, 32, 26, 26]
```

第二层卷积：

```python
self.conv2 = nn.Conv2d(32, 64, 3, 1)
```

尺寸：

$$ 26 - 3 + 1 = 24 $$

得到：

```text
[1, 64, 24, 24]
```

最大池化：

```python
F.max_pool2d(x, 2)
```

尺寸减半：

```text
[1, 64, 12, 12]
```

展平之后：

$$ 64 \times 12 \times 12 = 9216 $$

所以：

```python
self.fc1 = nn.Linear(9216, 128)
```

是合理的。

---

# 九、`forward` 函数逐步解释

```python
def forward(self, x):
```

`forward` 定义输入如何流过模型。

---

## 1. 第一层卷积

```python
x = self.conv1(x)
```

从原始灰度图提取低级特征，比如边缘、笔画方向、局部纹理。

---

## 2. ReLU 激活

```python
x = F.relu(x)
```

ReLU 公式：

$$ ReLU(x)=max(0,x) $$

作用是引入非线性。

如果没有 ReLU，多个线性层叠加起来本质仍然是线性变换，模型表达能力会弱很多。

---

## 3. 第二层卷积

```python
x = self.conv2(x)
```

进一步提取更复杂的局部模式，比如数字的弯曲结构、交叉结构、环形结构。

---

## 4. 最大池化

```python
x = F.max_pool2d(x, 2)
```

最大池化保留局部最强响应。

对于 MNIST 来说，一个数字稍微平移一点，模型仍然应该识别为同一个数字。池化可以带来一定平移鲁棒性。

---

## 5. Dropout

```python
x = self.dropout1(x)
```

训练时随机丢掉部分特征，防止模型过拟合。

测试时因为执行了：

```python
model.eval()
```

所以 Dropout 不再随机丢弃。

---

## 6. 展平

```python
x = torch.flatten(x, 1)
```

从第 1 维开始展平。

原来：

```text
[batch, channel, height, width]
```

变成：

```text
[batch, feature]
```

对于单张图片：

```text
[1, 64, 12, 12] → [1, 9216]
```

这里的 `1` 很重要，表示保留 batch 维度。

---

## 7. 全连接层

```python
x = self.fc1(x)
x = F.relu(x)
```

将局部特征整合成全局分类特征。

卷积层关注局部，线性层开始做整体判断。

---

## 8. 输出类别分数

```python
x = self.fc2(x)
```

输出 10 个数字，每个数字对应一个类别。

比如输出 shape 是：

```text
[1, 10]
```

对应：

```text
数字 0 的分数
数字 1 的分数
...
数字 9 的分数
```

---

## 9. `log_softmax`

```python
output = F.log_softmax(x, dim=1)
```

它把原始分数转换成 log probability。

为什么要用它？

因为后面 loss 用的是：

```python
F.nll_loss(output, target)
```

这一套是配套的：

```text
log_softmax + nll_loss
```

等价于：

```text
cross_entropy
```

也就是说，下面两种写法本质上等价：

```python
output = F.log_softmax(x, dim=1)
loss = F.nll_loss(output, target)
```

和：

```python
loss = F.cross_entropy(logits, target)
```

区别是第一种手动拆开，第二种 PyTorch 封装在一起。

---

# 十、数据预处理：Normalize 的作用

代码中：

```python
transforms.Normalize((0.1307,), (0.3081,))
```

MNIST 的图片被标准化为：

$$ x_{norm} = \frac{x - 0.1307}{0.3081} $$

这里：

```text
0.1307 是 MNIST 的均值
0.3081 是 MNIST 的标准差
```

标准化的作用：

1. 让输入分布更稳定；
2. 让模型训练更容易；
3. 避免不同输入尺度导致优化困难。

原图像素范围是：

```text
[0, 1]
```

标准化后不再局限于 `[0,1]`，可能有负数，也可能大于 1。

例如：

$$ x=0 $$

标准化后：

$$ \frac{0-0.1307}{0.3081} \approx -0.424 $$

所以模型实际吃到的不是原始图，而是标准化后的图。

---

# 十一、攻击前要 `denorm`

代码中：

```python
data_denorm = denorm(data)
```

因为 `data` 是标准化后的图片。

但是 FGSM 的 `epsilon` 通常定义在原始像素空间 `[0,1]` 上。

所以流程是：

```text
标准化图片 data
    ↓ denorm
原始尺度图片 data_denorm，范围约为 [0,1]
    ↓ FGSM
对抗图片 perturbed_data，范围 [0,1]
    ↓ Normalize
标准化后的对抗图片 perturbed_data_normalized
    ↓ model
重新分类
```

如果直接对标准化后的 `data` 做：

```python
data + epsilon * sign(gradient)
```

那么 `epsilon` 的含义就变了，不再是原始像素空间里的扰动大小。

---

# 十二、`denorm` 函数详解

```python
def denorm(batch, mean=[0.1307], std=[0.3081]):
```

这个函数把标准化后的 tensor 转回原始尺度。

标准化是：

$$ x_{norm} = \frac{x - mean}{std} $$

反归一化是：

$$ x = x_{norm} \cdot std + mean $$

代码：

```python
return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
```

这里的：

```python
view(1, -1, 1, 1)
```

是为了适配图像 batch 的 shape：

```text
[B, C, H, W]
```

对于 MNIST：

```text
B = batch size
C = 1
H = 28
W = 28
```

所以：

```text
batch shape: [B, 1, 28, 28]
mean shape:  [1, 1, 1, 1]
std shape:   [1, 1, 1, 1]
```

如果是 RGB 图片，mean/std 有 3 个值，变成：

```text
mean shape: [1, 3, 1, 1]
std shape:  [1, 3, 1, 1]
```

可以对每个通道分别反归一化。

---

# 十三、`test` 函数是整段核心

```python
def test(model, device, test_loader, epsilon):
```

它的作用是：

> 在某个 epsilon 攻击强度下，测试模型被 FGSM 攻击后的准确率，并保存一些对抗样本用于可视化。

---

## 1. 初始化统计变量

```python
correct = 0
adv_examples = []
```

`correct` 表示攻击后仍然分类正确的数量。

`adv_examples` 用来保存若干攻击样本。

---

## 2. 遍历测试集

```python
for data, target in test_loader:
```

因为：

```python
batch_size=1
```

所以每次处理一张图片。

`data` 是图片：

```text
[1, 1, 28, 28]
```

`target` 是标签：

```text
[1]
```

比如：

```text
target = 7
```

表示这张图真实类别是 7。

---

## 3. 把数据放到设备上

```python
data, target = data.to(device), target.to(device)
```

如果模型在 GPU 上，数据也必须在 GPU 上。

常见错误是：

```text
model 在 cuda
data 在 cpu
```

这样会报错。

---

## 4. 对输入开启梯度

```python
data.requires_grad = True
```

这是对抗攻击和普通测试最大的区别。

普通测试只需要：

```python
output = model(data)
```

不需要梯度。

但是 FGSM 要知道：

```text
输入图片的每个像素怎么变化，会让 loss 变大
```

所以必须计算：

$$ \nabla_x J(\theta,x,y) $$

也就是 loss 对输入 (x) 的梯度。

这就是：

```python
data.requires_grad = True
```

的作用。

---

## 5. 第一次前向传播

```python
output = model(data)
```

得到模型对原图的预测结果。

---

## 6. 取初始预测类别

```python
init_pred = output.max(1, keepdim=True)[1]
```

`output` shape 是：

```text
[1, 10]
```

`output.max(1)` 表示在类别维度上取最大值。

它返回：

```text
最大值本身
最大值对应的位置
```

例如：

```python
output.max(1, keepdim=True)
```

可能返回：

```text
values:  tensor([[-0.01]])
indices: tensor([[7]])
```

其中 `indices=7` 表示模型预测为数字 7。

代码中：

```python
[1]
```

取的是第二个返回值，也就是类别编号。

更直观的写法是：

```python
init_pred = output.argmax(dim=1, keepdim=True)
```

---

## 7. 原本预测错就跳过

```python
if init_pred.item() != target.item():
    continue
```

这句表示：

```text
如果模型原本就认错了，就不攻击这张图
```

原因是 FGSM 测试通常关心：

> 原本能被正确分类的样本，在攻击后是否会变错。

如果原来就错了，再攻击就没有太大分析价值。

不过要注意：

最后准确率分母仍然是：

```python
len(test_loader)
```

所以原本就错的样本不会进入 `correct`，最终还是算作错误。

---

## 8. 计算 loss

```python
loss = F.nll_loss(output, target)
```

因为模型输出是 log probability：

```python
F.log_softmax(x, dim=1)
```

所以这里用：

```python
F.nll_loss
```

目标是让真实类别的 log probability 尽可能高。

如果模型对真实类别越不自信，loss 越大。

---

## 9. 清空旧梯度

```python
model.zero_grad()
```

PyTorch 默认梯度会累积。

如果不清空，上一次 backward 的梯度会叠加到这一次。

训练中常见写法是：

```python
optimizer.zero_grad()
```

这里没有优化器，不更新模型参数，只是算梯度，所以使用：

```python
model.zero_grad()
```

---

## 10. 反向传播

```python
loss.backward()
```

这一步计算梯度。

因为：

```python
data.requires_grad = True
```

所以 backward 后会得到：

```python
data.grad
```

它表示：

$$ \frac{\partial loss}{\partial data} $$

也就是每个像素对 loss 的影响方向。

---

## 11. 获取输入梯度

```python
data_grad = data.grad.data
```

这就是 FGSM 需要的梯度。

更现代、更安全的写法可以是：

```python
data_grad = data.grad.detach()
```

不过原代码也可以运行。

---

## 12. 生成对抗样本

```python
data_denorm = denorm(data)
perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)
```

先把标准化图片变回 `[0,1]`，然后执行 FGSM。

核心攻击：

```python
perturbed_image = image + epsilon * sign_data_grad
```

再裁剪：

```python
perturbed_image = torch.clamp(perturbed_image, 0, 1)
```

确保图片像素合法。

---

## 13. 攻击后重新归一化

```python
perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
```

这是为了让对抗样本符合模型输入格式。

模型训练时吃的是 Normalize 后的数据，所以测试也必须 Normalize。

---

## 14. 重新分类对抗样本

```python
output = model(perturbed_data_normalized)
```

这一步看模型面对攻击后的图片会输出什么。

---

## 15. 判断是否攻击成功

```python
final_pred = output.max(1, keepdim=True)[1]
```

取攻击后的预测类别。

如果：

```python
final_pred.item() == target.item()
```

说明攻击失败，模型仍然认对。

如果：

```python
final_pred.item() != target.item()
```

说明攻击成功，模型被误导。

例如：

```text
原本预测：7
攻击后预测：2
```

这就是典型对抗样本。

---

# 十四、可视化部分

最后这部分代码：

```python
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        ...
        plt.imshow(ex, cmap="gray")
```

作用是把不同 epsilon 下的对抗样本画出来。

每一行对应一个 epsilon。

标题：

```python
plt.title(f"{orig} -> {adv}")
```

表示：

```text
攻击前预测类别 -> 攻击后预测类别
```

比如：

```text
7 -> 2
```

说明模型原来预测为 7，攻击后预测为 2。

左边的 ylabel：

```python
plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
```

表示当前这一行的攻击强度。

你通常会看到：

```text
epsilon = 0      图片干净，预测正确
epsilon = 0.05   图片略变，可能仍然正确
epsilon = 0.1    有些开始误判
epsilon = 0.2    噪声明显，误判更多
epsilon = 0.3    图片明显变脏，模型大量错误
```

---

# 十五、关键语法集中总结

## 1. `class Net(nn.Module)`

定义一个 PyTorch 神经网络模型。

所有自定义模型通常都继承：

```python
nn.Module
```

---

## 2. `super(Net, self).__init__()`

调用父类初始化函数。

现代写法：

```python
super().__init__()
```

---

## 3. `nn.Conv2d(1, 32, 3, 1)`

二维卷积层。

含义：

```text
输入通道 1
输出通道 32
卷积核大小 3
步长 1
```

---

## 4. `F.relu(x)`

ReLU 激活函数。

[
ReLU(x)=max(0,x)
]

---

## 5. `F.max_pool2d(x, 2)`

二维最大池化，窗口大小为 2。

一般会让图片高宽减半。

---

## 6. `torch.flatten(x, 1)`

从第 1 维开始展平。

保留 batch 维度。

```text
[1, 64, 12, 12] → [1, 9216]
```

---

## 7. `F.log_softmax(x, dim=1)`

沿类别维度计算 log probability。

`dim=1` 是因为 `x` 的 shape 是：

```text
[batch, class]
```

---

## 8. `datasets.MNIST(..., train=False)`

加载 MNIST 测试集。

---

## 9. `transforms.ToTensor()`

把图片转成 tensor，并把像素从 `[0,255]` 转成 `[0,1]`。

---

## 10. `transforms.Normalize`

标准化输入：

$$ x_{norm} = \frac{x - mean}{std} $$

---

## 11. `DataLoader`

按 batch 读取数据。

```python
batch_size=1
```

表示每次读取一张图。

---

## 12. `.to(device)`

把模型或数据移动到 CPU/GPU。

---

## 13. `model.eval()`

让模型进入测试模式。

影响 Dropout 和 BatchNorm。

---

## 14. `data.requires_grad = True`

允许对输入图片求梯度。

对抗攻击必须有这句。

---

## 15. `loss.backward()`

反向传播，计算梯度。

---

## 16. `data.grad`

得到 loss 对输入图片的梯度。

---

## 17. `.sign()`

取符号：

```text
正数 → 1
负数 → -1
零 → 0
```

---

## 18. `torch.clamp(x, 0, 1)`

把像素限制在 `[0,1]` 范围。

---

## 19. `.item()`

把单元素 tensor 转成 Python 数字。

---

## 20. `.squeeze()`

去掉长度为 1 的维度。

```text
[1, 1, 28, 28] → [28, 28]
```

---

## 21. `.detach().cpu().numpy()`

常用于可视化。

含义：

```text
脱离计算图 → 移到 CPU → 转成 numpy
```

---

# 十六、背后的意义

这段代码展示了深度学习模型的一个根本问题：

> 模型学到的决策边界和人类感知并不完全一致。

人类看 MNIST 数字时，更关注整体结构：

```text
笔画形状
数字轮廓
上下左右结构
```

但神经网络可能依赖高维空间中的某些微妙方向。

FGSM 利用梯度找到这些方向，然后轻微改变输入，使图片跨过模型的决策边界。

所以会出现：

```text
人眼看起来还是 7
模型却认为是 2
```

这说明模型的鲁棒性不足。

---

# 十七、高维空间中小扰动会很危险

MNIST 是 $28 \times 28$ 的图像，也就是 784 维输入。

每个像素只改变一点点，比如：

```text
0.05
```

单独看一个像素，变化很小。

但是 784 个像素一起沿着同一个攻击目标变化，总体影响就会被放大。

这就是高维空间中的累积效应。

可以粗略理解为：

```text
单个像素变化很小
但所有像素一起朝着让模型犯错的方向变化
最终模型输出被明显改变
```

这也是对抗样本存在的重要原因之一。

---

# 十八、FGSM 是白盒攻击

这段代码里的 FGSM 属于：

```text
White-box Attack
```

白盒攻击表示攻击者知道模型结构和参数。

因为代码中攻击者可以访问：

```python
model
loss
data.grad
```

也就是说，攻击者可以直接计算：

$$ \nabla_x J(\theta,x,y) $$

如果攻击者不知道模型结构和参数，那就是黑盒攻击。

黑盒攻击通常要用：

```text
查询模型输出
迁移攻击
替代模型
估计梯度
```

等方式实现。

---

# 十九、FGSM 是 untargeted attack

这段代码实现的是：

```text
Untargeted Attack
```

也就是非定向攻击。

它只要求模型预测错，不关心错成哪一类。

例如真实标签是 7：

```text
7 → 2 可以
7 → 3 可以
7 → 9 也可以
```

只要不是 7，就算攻击成功。

如果是定向攻击，目标会变成：

```text
我要让模型必须把 7 识别成 2
```

定向攻击的目标通常是让目标类别的 loss 变小，或者让真实类别的得分下降、目标类别得分上升。

---

# 二十、FGSM 和 PGD 的关系

FGSM 可以理解为一步攻击。

公式：

$$ x_{adv}=x+\epsilon sign(\nabla_x J) $$

PGD 可以理解为多步 FGSM。

它每次走一小步：

$$ x^{t+1}=Proj_{B_\epsilon(x)}(x^t+\alpha sign(\nabla_x J)) $$

其中：

* $\alpha$：每一步步长；
* $Proj$：投影回允许扰动范围内；
* $B_\epsilon(x)$：以原图为中心、半径为 (\epsilon) 的扰动空间。

简单理解：

```text
FGSM：一步到位
PGD：小步多次攻击，每次都重新计算梯度
```

所以 PGD 通常比 FGSM 更强，但计算成本也更高。

---

# 二十一、梯度空间和图像空间

代码中有一个容易忽视的问题：

```python
data_grad = data.grad.data
data_denorm = denorm(data)
perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)
```

这里的 `data_grad` 是 loss 对标准化输入 `data` 的梯度。

而 `data_denorm` 是反归一化后的原图。

严格来说，如果完全严谨，梯度和扰动空间需要一致。

不过对于 `sign` 来说，标准化是线性变换：

$$ x_{norm} = \frac{x - mean}{std} $$

当 `std` 是正数时，梯度符号方向通常不会因为正比例缩放而改变。

所以这段代码在 MNIST 单通道情况下通常是可接受的。

但如果做更复杂图像攻击，尤其是 RGB、多种预处理、复杂归一化时，要非常注意：

```text
攻击是在 normalized 空间做？
还是在 pixel space 做？
epsilon 的单位到底是什么？
```

---

# 二十二、改进代码

如果想把统计写得更清楚，可以增加一个计数器：

```python
orig_correct = 0
adv_correct = 0
```

因为目前代码跳过了原本预测错的样本，但最终分母仍然是：

```python
len(test_loader)
```

如果想统计：

```text
原本预测正确的样本中，有多少攻击后仍然正确
```

可以这样写：

```python
orig_correct += 1
...
if final_pred.item() == target.item():
    adv_correct += 1
```

最后：

```python
robust_acc = adv_correct / orig_correct
```

这个指标叫：

```text
robust accuracy
```

比普通 accuracy 更能反映攻击效果。

---

# 二十三、普通准确率和鲁棒准确率的区别

这段代码打印的是：

```python
final_acc = correct / float(len(test_loader))
```

这可以理解为：

```text
整个测试集上攻击后的准确率
```

但如果你要研究对抗鲁棒性，通常更关心：

```text
原本能分类正确的样本中，攻击后还有多少保持正确
```

也就是：

$$ RobustAcc = \frac{\text{原本正确且攻击后仍正确的样本数}}{\text{原本正确的样本数}} $$

这可以避免把模型原本就错的样本混入攻击效果分析。

---

# 二十四、 `model.eval()` 的重要性

如果不写：

```python
model.eval()
```

Dropout 会继续随机丢弃神经元。

那么同一张图片每次预测可能都不一样。

这会导致两个问题：

1. 原始预测不稳定；
2. 攻击效果不稳定。

对抗攻击实验中，模型应该处于固定状态。

因此：

```python
model.eval()
```

是必须的。

但是注意：

```python
model.eval()
```

不会关闭梯度。

所以即使 eval 模式下，仍然可以：

```python
loss.backward()
```

计算输入梯度。

如果你写了：

```python
with torch.no_grad():
```

那就不能做 FGSM 了，因为它会关闭梯度追踪。

---

# 二十五、不要在 FGSM 中使用 `torch.no_grad()`

普通测试经常这样写：

```python
with torch.no_grad():
    output = model(data)
```

这样可以节省显存，加快推理。

但是 FGSM 需要梯度，所以不能这样包住攻击过程。

因为 FGSM 要计算：

```python
data.grad
```

如果用了 `torch.no_grad()`，则不会有计算图，也不会有输入梯度。

所以对抗攻击代码和普通测试代码最大的区别之一就是：

```text
普通测试：不需要梯度
对抗攻击：必须需要输入梯度
```

---

# 二十六、为什么 `epsilon=0` 也要测试？

`epsilon=0` 表示不加扰动。

它相当于 baseline。

通过它可以知道模型在干净测试集上的准确率。

然后和其他 epsilon 对比：

```text
epsilon = 0      clean accuracy
epsilon = 0.05   weak attack accuracy
epsilon = 0.1    medium attack accuracy
epsilon = 0.3    strong attack accuracy
```

这样可以画出一条鲁棒性曲线：

```text
x 轴：epsilon
y 轴：accuracy
```

正常情况下曲线向下。

下降越快，说明模型越脆弱。

---

# 二十七、核心的三句话

如果只记三句话，应该记：

```python
data.requires_grad = True
```

表示我要对输入图片求梯度。

```python
loss.backward()
```

表示反向传播，计算 loss 对输入图片的梯度。

```python
perturbed_image = image + epsilon * data_grad.sign()
```

表示沿着让 loss 增大的方向修改图片。

这三句就是 FGSM 的灵魂。

---

# 二十八、和深度伪造检测/鲁棒性研究的联系

检测器本质是分类器：

```text
输入：图片或视频
输出：real / fake
```

如果检测器是：

```text
fake probability = 0.91
```

攻击目标就可能是让它变成：

```text
fake probability = 0.10
```

也就是让 fake 被判成 real。

在 deepfake 检测中，攻击方式可能不只是改像素，还包括：

```text
压缩
转码
resize
加噪
模糊
颜色扰动
频域扰动
时间扰动
帧率变化
视频插帧
后处理增强
```

FGSM 是最基础的像素空间白盒攻击。

> 模型的判断可以被输入空间中很小但有方向的变化破坏。

---

# 二十九、FGSM 对科研的启发

## 1. 高准确率不等于鲁棒

模型在 clean data 上准确率高，不代表面对扰动仍然稳定。

所以实验不能只报告：

```text
clean accuracy
```

还应该报告：

```text
robust accuracy
attack success rate
accuracy under transformations
```

---

## 2. 模型可能依赖脆弱特征

MNIST 模型可能依赖人类不敏感的像素方向。

Deepfake 检测器也可能依赖：

```text
压缩痕迹
水印
分辨率差异
生成器特定纹理
数据集偏差
```

这类特征一旦被后处理破坏，检测器就可能失效。

---

## 3. 梯度暴露了模型的弱点

FGSM 通过梯度找到模型最脆弱的方向。

所以梯度不仅用于训练，也可以用于攻击、解释和诊断模型。

---

## 4. 对抗训练可以提升鲁棒性

如果在训练时加入对抗样本：

```text
原图 + 对抗图一起训练
```

模型可能会变得更鲁棒。

这叫：

```text
Adversarial Training
```

经典目标是：

$$ \min_\theta \mathbb{E}_{(x,y)} \left[ \max_{\delta \in S} J(\theta, x+\delta, y) \right] $$

外层是训练模型，内层是寻找最强攻击。

---

# 三十、完整流程总结

这段代码可以浓缩为：

```text
1. 定义 epsilon 列表，表示攻击强度。
2. 定义 LeNet 模型结构。
3. 加载 MNIST 测试集，并做 ToTensor + Normalize。
4. 选择 CPU/GPU。
5. 初始化模型并加载预训练权重。
6. 设置 model.eval()。
7. 对每张测试图片：
   - 开启 requires_grad；
   - 前向传播；
   - 如果原本预测错，跳过；
   - 计算 loss；
   - backward 得到输入梯度；
   - 取梯度 sign；
   - 对原图加 epsilon * sign；
   - clamp 到 [0,1]；
   - 重新 Normalize；
   - 再次送入模型；
   - 统计攻击后是否仍然正确。
8. 对每个 epsilon 重复测试。
9. 画出不同 epsilon 下的对抗样本。
```

---

# 三十一、最终理解版

> 模型先正常看一张 MNIST 图片，比如它认为这是数字 7。然后我们反过来问模型：如果想让你的 loss 变大，图片的每个像素应该往哪个方向变化？模型通过梯度告诉我们方向。于是我们沿着这个方向轻轻改动图片，得到一张人眼可能仍然认为是 7 的新图片。但模型再次看到它时，可能会认为它是 2。这就是 FGSM 对抗攻击。
