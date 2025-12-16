## 梯度下降

**情形**：需要使用一条已知斜率为 $\hat{w}$ 的一条直线 $y = \hat{w}x +b$ 来拟合回归三个点，这里的目的是为了找到一个b让直线的拟合程度最好。一条直线的拟合程度好不好可以用真实数据点与预测点之间的距离来测量，因此可以把这个距离误差写成一个最小二乘方式
$$
L = \frac{1}{2}\sum |e_{i}|^{2} \ \ \ \ \ 损失函数
$$
然后将损失函数带入方程即可如图所示，损失函数 $L$ 转换成为一个有关 变量 $b$ 的函数。

![image-20251202172659424](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251202172659424.png)

​	为了求出b，我们可以绘制出二次函数的图像。假设我们给定一个随机的b值，可以通过求出当前的斜率再乘于一个常数$ \epsilon $ 然后然让b更新为 b减去这个$\epsilon \times b$ 这样就可以更新b的值，让b距离极值越来越近，从而达到最优值。这就是<b style="color:red">梯度下降算法</b>

![image-20251203000217625](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203000217625.png)

> **梯度**：<u>多变量函数在某一点的变化率方向，指向函数值增加最快的方向</u>

​	以一个更一般的例子来说明这个算法，更多的数据点以及非线性函数，其中需要优化的参数是$ \theta $， 第 $i$ 个样本点的损失函数可以写成 $L(f(x_i,\theta),y_i)$。首先可以求出损失函数 $L$ 关于参数 $\theta$ 的梯度值也就是求偏导然后求均值，参数 $\theta$ 沿着梯度的负方向移动就可以让损失函数更小，其中的常数 $\epsilon$ 也被称为学习率，用来控制梯度下降的步长。

![image-20251203000810332](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203000810332.png)

​	按照上述的算法来计算，则需要首先计算出所有样本的损失函数梯度(<u>每一个点都有自己的梯度</u>)，然后求出均值来更新参数这就会产生如下问题：

1. 样本数量非常多：内存开销大。需要保存每一个样本的梯度。
2. 收敛速度慢：每一次更新都需要重新计算全部的样本点。

因此为了解决上述存在的问题，只需要每次<u>**随机从n个样本中选择m个样本且每次都不重复**</u>，这样就可以解决存在的问题。这种改进方法被称为<b style="color:red">随机梯度下降</b>。

![image-20251203001401908](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203001401908.png)

​	随机梯度下降并非每次都是有效的，深度学习网络训练往往是一个非凸的优化过程，在参数空间里面分布着各种山脊和山谷，假设每次更新的时候，参数更新在山谷两侧来回震荡，难以收敛到最佳的位置。

​	为了解决上述问题，想到在参数进行运动的时候添加阻尼，让移动更加平滑从而达到山谷。因此我们在进行参数更新的时候不仅需要计算新的梯度方向还要保留部分上一次梯度运动的方向，将两个方向的向量合并成为本次更新方向。将保留的历史梯度称为动量，这就是<b style="color:red">动量随机梯度下降</b>。

![image-20251203101729951](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203101729951.png)

​	使用数学公式来表达动量随机梯度下降如下所示，使用 $v$ 表示动量，$a$ 表示保留动量的程度，参数 $\theta$ 更新为 $\theta + v$
$$
g= \frac{1}{m} \nabla_{\theta}\sum_{i=1}^{m}L(f(x_i,\theta),y_i)\\
v \leftarrow av - \epsilon g  \ \ \ a控制动量\\
\theta \leftarrow \theta +v
$$
​	在讨论到学习率，为了让参数更新能够变得更快，一般会设置比较大的学习率，随着训练过程为了找到最优的数值，就不能盲目追求速度，需要降低学习率找到最优值。因此需要设定一个初始值，然后每隔一段时间就降低学习率。这种办法通过人为控制是非常粗糙的。

![image-20251203102552391](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203102552391.png)

​	为了让学习率能够自主下降，引入一个新的参数 $r$ , $r$ 就是梯度大小随时间的积累量，可以观察到将参数 $r$ 放在学习率的分母上如果：

+ 梯度波动很大，学习率迅速下降
+ 梯度波动很小，学习率下降变慢

这样就实现了自动调整学习率，分母中的另一个参数 $\delta$ 是一个小量，用来防止分母为0从而稳定计算。这就是<b style="color:red">AdaGrad算法</b>。 

​	AdaGrad算法使得参数r的变化只和梯度有关，可能让学习率过早的变小而不好控制，因此提出<b style="color:red">RMSProp算法</b>，在 r 更新的公式中加入了可以手动调节的 $\rho$ 来控制优化过程

![image-20251203103848267](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203103848267.png)

​	在改进随机梯度下降的过程中引入了 **动量和自动调节的学习率** ，<b style="color:red">Adam算法</b>将二者同时引入。在Adam算法中定义到：

+ $s$ ：自适应动量。使用参数 $\rho_1$ 来控制
+ $r$ ：自调整参数。使用参数 $\rho_2$ 来控制 
+ 修正两个参数，让两个参数在训练之初比较大帮助算法快速收敛
+ 最终参数的学习率就是 $\theta \leftarrow \theta - \frac{\epsilon \hat{s}}{\sqrt{\hat{r}+\delta}}g$

![image-20251203104445490](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203104445490.png)

> **梯度下降算法总结**
>
> + 梯度下降算法，深度学习核心之一
> + 随机梯度下降算法
> + 动量
> + 自适应学习 AdaGrad 和 RMSProp 算法
> + 动量 & 自适应学习 Adam算法





## 反向传播

**情形**：以一个线性拟合的例子，x 经过一个线性方程得到 y，其中 y = wx + b 使用最小二乘法作为优化的损失函数，其中 $y_{gt}$ 是真实值。在这个例子中假设：

+ x 为 1.5
+ w 为 0.8
+ b 为 0.2
+ y 预测等于 0.8 x 1.5 + 0.2 = 1.4
+ 损失函数L 为 0.18
+ 真实y 为 0.8

![image-20251203105010014](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203105010014.png)

根据梯度下降算法，我们需要计算损失函数 L 对于参数 w 和 b 的梯度值(偏导数)，然后按照梯度的反方向更新两个参数即可。
$$
\frac{\partial{L}}{\partial{w}}\ \  \ \ w\leftarrow w - \epsilon\frac{\partial{L}}{\partial{w}} \\
\frac{\partial{L}}{\partial{b}}\ \  \ \ b\leftarrow b - \epsilon\frac{\partial{L}}{\partial{b}} \\
$$
​	为了更容易的计算偏导数，可以先求出偏导 $\frac{\partial L}{\partial y}$ 然后再计算 偏导 $\frac{\partial y}{\partial w}$ ，两个偏导相乘，这就是求导的链式法则。同理对 b 的计算也是如此。

![image-20251203112524862](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203112524862.png)

​	这种沿着黄色箭头从后先前计算参数梯度值的方法就是<b style="color:red">反向传播算法</b>。现在增加难度，假设现在存在两次线性变换，则如图所示可以反向传播计算。黄色部分就是上次计算的值不需要重复计算。所以可以理解为<u>反向传播 算法就是神经网络中加速计算参数梯度值的方法</u>。

![image-20251203112859402](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203112859402.png)

​	但是在计算机中，计算是一个更加模块化的过程，计算机将计算处理为每一个过程都更加统一化，计算的格式是相当的只有计算的变量不同。这种由单元运算和变量构成的计算流程图也被称为**计算图**。不过下述显示的是计算图的正向传播过程。![image-20251203113113610](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203113113610.png)

现在通过计算图反向求出损失函数对于每一个参数的梯度表达式。

1. 第一步计算损失函数L对于参数$y_2$的偏导。

2. 第二步计算损失函数L对于参数$u_2$的偏导。此时使用链式法则并且可知

   + $L对y_2$的偏导在第一步已经计算
   +  $y_2对u_2$的偏导其实就是1

   由此简便了计算。

![image-20251203114028709](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203114028709.png)

同样的道理可以计算出L对于所有参数的计算表达式

+ 式子中黄色的量：通过之前步骤得到的。
+ 式子中绿色的量：向前传播中求得的量。

![image-20251203114415543](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203114415543.png)

​	在深度学习的框架里，计算图中的元素的定义和使用也是非常便捷的比如乘法运算在torch框架中可以写成如下所示。其中前向传播函数为$forward$ ，反向传播函数为$backward$，参数 grad_z就是损失函数L对z的偏导。在深度学习框架中，所有的单元运算都有定义的向前传播和反向传播函数，这样就可以使用反向传播算法来更新数以亿计的网络参数了。

![image-20251203114604329](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203114604329.png)

> **反向传播算法总结**
>
> + 反向传播算法，深度学习核心之一。
> + 可以利用反向传播加速计算参数梯度值，然后使用梯度下降快速更新参数。
> + 使用计算图的方式来进行模块化的运算。





## 激活函数

**情形**：给定一个线性变换，可以将参数 $x$ 的值映射到一条直线上，输出结果就是函数 $y_1 = w_1 x + b_1$ 如果 $y_1$ 再经过一个线性变换得到 $y_2$ 那么 参数 $x$ 和 $y_2$ 也是一条直线的关系。

![image-20251203203057125](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203203057125.png)

​	也就是说无论使用多少线性变换或者叠加为神经网络最终都只能解决线性问题。

![image-20251203203115976](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203203115976.png)

为了解决非线性问题，需要使用非线性函数 $f$ 当作从输入到输出的激活过程因此被称为<b style="color:red">激活函数</b>。因此激活函数性质如下：

+ 激活函数是一个非线性函数。
+ 激活函数 $f$ 连续可导：根据反向传播的性质。
+ 激活函数 $f$ 一定可以映射所有函数。定义域为 $R$。
+ 激活函数 $f$ 随 y 的增大而增大，随y的减小而减小。只为增加非线性不需要改变输入的响应状态.

第一个常见的激活函数<b style="color:red">sigmoid函数</b>。公式如图所示。

![image-20251203203814763](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203203814763.png)

我们可以对sigmoid求导之后的函数进行研究，最大值是 $0.25$ 。

+ 当y的值非常大\小 的时候 sigmoid函数的导数为 0 ，这种在正负无穷梯度为0的函数被称为**饱和函数**
+ 无论如何取值，导数的最大值是0.25。这就意味着每一层反向传播时，梯度会被动缩小大约$\frac{1}{4}$，如果网络层数很多，或者出现极端的输出就会导致前几层的梯度几乎为0，参数不会被更新，这就是**梯度消失**。
+ 函数的取值是始终大于 0 的，被称为**非零均值函数**。

![image-20251203204257353](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203204257353.png)

如果考虑一个函数的输入是参数 $x_1,x_2$，它们是上一层sigmoid函数的输出所以都是大于 0 的数，进行反向传播后得到 $w_1,w_2$的梯度，其中黄色标注的式子始终大于0，因此这两个参数的梯度正负完全取决于损失函数对 $o$ 的偏导，这就意味着 $w_1,w_2$的梯度符号始终一致，被强制同时正向或反向更新，从而导致收敛速度变慢。

![image-20251203204732749](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203204732749.png)

为了解决上述问题，讨论第二个常见激活函数<b style="color:red">tanh函数</b>。

![image-20251203204843957](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203204843957.png)

+ 是零均值函数，性能比sigmoid函数略优。
+ 依旧是饱和函数，存在梯度消失。

为了解决梯度消失问题，讨论第三个常见激活函数<b style="color:red">ReLU函数</b>。

![image-20251203205132529](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203205132529.png)

ReLU函数在训练中可以动态控制神经元状态，要么激活大于0，要么等于0被抑制，把这种性质称为稀疏性。稀疏性在实际应用中发挥着至关重要的作用：

+ 输入参数发生小幅度变动，只有少部分神经元需要改变状态，这就使得信息的耦合程度降低。
+ 动态 开启/关闭 神经元可以支持不同输入维度和中间层维度的特征学习

![image-20251203205405531](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203205405531.png)

+ 依旧是非零均值函数，可以使用归一化解决。
+ 函数输出没有上界，梯度累积超过上限，导致**梯度爆炸**。这需要参数初始化和重新设计网络结构来解决。
+ 部分神经元始终不被激活，称为**神经元坏死**现象，导致网络表达能力下降。

为了解决上述问题先后出现 <b style="color:red">Leaky ReLU函数</b> 和 <b style="color:red">Paramtric ReLU函数</b>。

Leaky ReLU函数通过保留一点点输出来防止神经元坏死

![image-20251203210416199](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203210416199.png)

Parametric ReLU函数通过训练过程来调整参数 $a$ 来控制稀疏性或抑制神经元

![image-20251203210420133](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203210420133.png)

> **激活函数总结**
>
> + 激活函数，增加非线性表达
> + 饱和函数：sigmoid、tanh
> + 非饱和函数：ReLU
> + 为解决神经元坏死：Leaky relu和parametric relu





## 参数初始化

情形：给定一个神经元，存在三个输入 $x_1,x_2,x_3$ 线性方程输出的 $y$ 值就等于 $w_1 x_1 + w_2 x_2+ w_3 x_3(此处忽略讨论b)$，那么开始训练神经网络之前该如何选择w的值呢？

![image-20251203211826208](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203211826208.png)

如果让 $w_i$全部为0，那么如果当前神经网络的神经元不仅一个神经元，而是由两个神经元构成，里面参数$w_1和w_2$向量的值都是零，根据反向传播：

1. 所有的参数不仅初始化的值一致
2. 训练过程的变化也是一致的

从而导致该层的两个神经元的状态会始终保持一致，无法学习和表达更复杂的特征，称为<b style="color:red">对称现象</b>。

![image-20251203212136217](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251203212136217.png)

为了防止对称现象的出现，需要再参数初始化的时候增加一些随机性，例如在均值为0，方差为1的正态分布中采样w的值，假设$x_1,x_2,x_3$的值是1，y就变成了$w_1 + w_2 + w_3$，因为3个参数都是独立同分布，所以y的方差就是这三个参数 $x$的方差之和等于3，所以y的标准差就是 $\sqrt{3}$ ，假设有n个参数，那么标准差很大，梯度爆炸。如果使用tanh函数又会因为y无穷大小导致梯度消失。

​	为了让神经网络训练，需要让y的方差落在一个可控的范围内。例如让y的方差等于1，这样求出参数w的分布方差就是$\frac{1}{n}$，假设不仅考虑输入的维度还考虑下一层神经元的数量，那么平均后的方差就是 $\frac{2}{n_{in} + n_{out}}$。<b style="color:red">Xavier初始化方法</b>如图所示。

![image-20251204235628314](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251204235628314.png)



>**参数初始化**
>
>+ 参数初始化方法可以减缓梯度爆炸和梯度消失问题
>+ tanh一般使用Xavier初始化方法
>+ ReLU及其变种一般使用Kaiming初始化方法







## 大模型BasicKnowledge

+ 主要任务：不停预测下一个词语。
+ 由于会将模型的上一次输出加入到下一次的输入，因此称呼这种模型为自回归模型
+ 训练阶段不必执行自回归操作，只需要给模型一批数据并告诉模型结果是什么，每一个预测词语都会计算出一个损失值，所有词语的平均损失用来反向传播更新参数。
+ Transformer通过<u>注意力机制</u>以及<u>位置编码</u>，即实现了对<u>序列中关键信息</u>的精准捕捉，又保留了词语在句子中的<u>位置关系</u>，从而突破了传统模型在<u>长距离依赖</u>处理和<u>并行计算</u>上的局限，为高效理解上下文语义奠定了基础。



### <b style="color:blue">1. 准备训练数据</b>

​	海量的文本数据，比如网页，文档，PDF等等，首先需要对这些文本进行分词，每一个词语就是一个token，对所有的词元去重就得到一个庞大的词典，对词元进行编号，就可以得到一个词元映射到ID的匹配表，将方向对调就可以得到一个ID映射到词元的匹配表。

![image-20251211142630128](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251211142630128.png)

假设模型每次允许<u>最多输入3个Token</u>，那么：

+ 从 n 开始到 n + 2 的3个词语就可以作为一条<b style="color:blue">输入数据</b>；
+ 从 n + 1 开始到 n + 3 的3个词语就可以作为一条<b style="color:blue">目标数据</b>；
+ <b style="color:green">根据输入数据 $\to$ 预测 3个词语 $\to$ 和目标数据进行比较，计算损失</b>；
+ 在预测的时候虽然也会输出 3 个Token，但是只有最后一个Token被使用，它会被当成最后一个Token拼接到原来输入的后面；

1. 这里模型限制的长度也被称为 <b style="color:red">上下文长度</b>
2. 上述**目标数据**是从**输入数据**的<u>后一个词元</u>开始，此处也不一定是从后一个词元，可以后n个词元，被称为<b style="color:red">移动步幅</b>
3. 训练的时候也不一定每次只放一个数据对，这被称为<b style="color:red">批样本数</b>。

> **BPE分词法**
>
> ​	英文的普通分词可以按照空格来划分词语，但是不管训练数据有多大都很难覆盖到全部词汇，意味着在预测阶段可能遇到没见过的词汇，被称为<b style="color:red">未登录词</b>，需要一些技巧来处理这些词，比如：
>
> + 统一使用一个默认向量来表示未登录词
>
> ​	BPE分词法则可以将单词拆分为更细的元素，比如词根或者字母，遇到未登录词的时候就将词汇进一步拆解为更多的子元素，这些词根或字母往往带有丰富的信息量（如 前缀im表示否定），这种分词法使得模型可以处理任何没见过的词汇。



### <b style="color:blue">2. 模型训练</b>

+ 模型的主要架构

![image-20251211160329874](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251211160329874.png)

**如何进行词嵌入**

​	为了将词汇转换为可以计算的数值向量，我们需要词汇表，还有一个随机初始化的向量矩阵，向量矩阵的行数等于词汇表的数量，列数表示词元的维度。

![image-20251211160804280](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251211160804280.png)

​	根据 $词汇表(获取ID)\ \to\ 向量矩阵(获取向量) \ \to \ 词嵌入向量$ ，获取到词元向量之后还需要加上位置向量才能作为最终的嵌入向量。

**位置向量如何获得**

位置向量的获取可以有以下两种方式：

1. <b style="color:red">训练法</b>：和计算词元向量的方式一致，设定一个位置向量矩阵，矩阵的行数等于上下文长度，也就是上下文的每一个位置都对应一个随机初始化的向量，列数必须等于词元嵌入的维度，后续将位置向量也当作参数一起更新。（如GPT2）
2. <b style="color:red">公式法</b>：使用公式计算出每一个位置的向量表示，如Transformer论文中公式如下图所示。
   ![image-20251211162534837](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251211162534837.png)

此外，位置嵌入还分为如下两种方式：

1. <b style="color:red">绝对位置</b>
2. <b style="color:red">相对位置</b>：现在使用更多

![image-20251211163350364](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251211163350364.png)

**如何找到词元向量**

​	事实上并非使用词汇-ID表（查找操作）来找到词元向量的，上文只是为了好解释。为了后续的反向传播能够实现，具体做法：

1. 根据词元ID将词元转化为one-hot编码。
2. 使用one-hot编码乘于词元矩阵就获得词元向量

![image-20251211164015305](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251211164015305.png)

**如何计算注意力**

​	将涵盖3个词向量的一句话输入到Transformer层，输出依旧是3个向量。此时输出的向量是包含了其他词语的注意力信息，所谓注意到其他向量的信息其实就是**新向量是根据其他向量计算得来**。

![image-20251211164546052](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251211164546052.png)

​	那么注意力得分如何计算呢？以第二个Token为例，分别与所有的Token进行向量点积运算 (在向量中点积计算结果越大，说明两个向量的相似程度越高，也代表这个Token注意力权重越高) 一般加权求和需要先归一化，让全部权重加起来等于1（通常使用Softmax公式进行归一化处理）。

![image-20251211170051265](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251211170051265.png)

​	但是，实际计算注意力的时候，不是分别与其他所有的Token计算注意力，不能让模型提前看到后面的答案，所以在进行归一化之前，要将当前Token之后的点积结果全部改为负无穷大。（<b style="color:red">掩码机制</b>）

![image-20251211170632369](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251211170632369.png)

![image-20251211170704195](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251211170704195.png)

​	可以看到一个Token会作为三个任务对象，因此就会被拆分为著名的$W_K、W_Q、W_V$矩阵。

![image-20251211171954003](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251211171954003.png)

![image-20251211173636471](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251211173636471.png)

具体的对应转换如图所示。

![image-20251211184633772](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251211184633772.png)

具体的计算过程如图所示。

![image-20251211184856598](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251211184856598.png)

其中Q矩阵和K矩阵进行计算的时候，矩阵中的某些数值的标准差会变大（方差同理），从而导致归一化之后注意力分布极不均匀甚至出现大部分注意力权重为0，因此还需要 除以$\sqrt{k_{dim}}$

![image-20251211185201253](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251211185201253.png)

### <b style="color:blue">3. 嵌入过程（文本说明）</b>

<h4 style="color:purple">1.输入嵌入</h4>

用户输入的文本首先被分词为一系列**词元**，每个词元通过嵌入层映射为一个**d维向量**。假设输入有n个词元，则嵌入结果是一个矩阵：
$$
X\in R^{n \times d}
$$
其中**每一行**代表一个词元的嵌入 embedding 向量



<h4 style="color:purple">2.生成Q、K、V</h4>

通过可学习的权重矩阵$W_Q,W_K,W_V \ \in R^{d\times d_k}$（在标准Transformer中通常 $d_k = d$，或者多头注意力中 $d_k = d/h$），计算：
$$
Q =XW_Q \in R^{n \times d_k}，K =XW_K \in R^{n \times d_k}，V =XW_V \in R^{n \times d_v}，
$$
（通常 $d_k= d_v = d$ 或按头划分）



<h4 style="color:purple">3.计算注意力分数</h4>

计算缩放点积注意力：
$$
A = QK^T \in R^{n\times n}
$$

+ **A 的 第 $i$ 行第 $j$ 列**（既$A_{ij}$）表示 **第 $i$ 个词元对第 $j$ 个词元的 “原始注意力分数”** （未归一化）。
+ 这个分数衡量的是：**在生成第 $i$ 个位置的输出时，模型应 “关注” 第 $j$ 个位置的信息程度**。

> 注意：有时会除以 $\sqrt{d_k}$ 以稳定梯度（即 $\frac{QK^T}{\sqrt{d_k}}$）。



<h4 style="color:purple">4.Softmax 归一化</h4>

对 **每一行** （即每个查询位置）应用softmax：
$$
\text{AttentionWeights}=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)\in\mathbb{R}^{n\times n}
$$

+ 归一化后，**每一行的元素和为 1**，可视为 **第 i 个词元对所有词元的注意力权重分布**。



<h4 style="color:purple">5.加权聚合 Value</h4>

最终输出：
$$
\mathrm{Output}=\text{AttentionWeights}\cdot V\in\mathbb{R}^{n\times d_v}
$$

+ 这个矩阵的第 $i$ 行是 **所有Value 向量按第 i 行注意力权重加权求和的结果**，即融合了上下文信息的新表示。

- **每一行$o_i$（第 i 行）**：表示 **第 i 个输入词元在融合了整个上下文信息后的新的表示（上下文感知的嵌入）**。





### <b style="color:blue">4. Transformer</b>

 <h4 style="color:purple">1.多头注意力</h4>

![image-20251214152334591](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251214152334591.png)

例子：针对 “小猫在沙发上<u>**玩耍**</u>” 这句话，对于 “**玩耍**” 这个词，模型会派出多个注意力镜头去扫描上下文。

+ **第一个镜头**：重点捕捉小猫，给 “小猫” 这个词更高的注意力权重，从而确定玩耍的主体。
+ **第二个镜头**：重点捕捉沙发，给 “沙发” 这个词更高的注意力权重，从而确定发生的位置。
+ **第三个镜头**：扫描整一个句子，给所有词平均的注意力权重，从而感受这个动作发生的整体氛围
+ **最终**：大模型会将 3 个镜头捕捉到的不同信息巧妙的拼接融合起来，于是玩耍这个词就带有丰富的上下文信息。（让模型能够从不同角度同时理解一句话，捕捉更丰富更准确的含义，从而真正的理解语义）



>**多头注意力背后的逻辑**
>
>​	为了能同时捕捉一段文本中可能存在的多种依赖关系（例如动作的执行者、发生地点、修饰方式等），Transformer设计了多头注意力机制，每个头都负责学习并提取文本中某一方面的关联信息。



 <h4 style="color:purple">2.如何实现多头注意</h4>

![image-20251214153135003](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251214153135003.png)

​	假设需要实现 3 头注意力，需要将输入矩阵兵分 3 路 同时计算3个注意力，3条路都是独立的：

+ 每一路都有各自的Q、K、V参数
+ 生成自各自的Q、K、V向量
+ 计算各自的注意力
+ 最后得到各自的输出

随后将**<u>所有的输出拼接在一起</u>**，就得到了多头注意力的输出。

​	为了解决输出维度由于拼接导致的翻倍，可以将每一个注意力头的维度缩小 设定为 $单注意头的输出维度 =\frac{总输出维度}{注意力头数}$，这样的好处：

+ 捕捉到多个维度的注意力
+ 避免输出维度变大



 <h4 style="color:purple">3.FFN前馈神经网络</h4>

![image-20251214153736705](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251214153736705.png)

​	获得多头注意力结果之后需要将这个结果输入到一个小型前馈神经网络中(FFN)，FFN层由两个紧密的全连接层组成：

+ 第一层：将输入向量投影到一个更高的维度空间（维度需要手动设定）
+ 第二层：将输入向量投影回原始的维度（最终输出的维度等于刚进FFN时的维度）

> 整个Transformer快的输入和输出维度必须一致，这样后续才能进行多层的Transformer操作。

<hr>

具体的数学操作就是

1. 输入到FFN的是一个 $X$矩阵。
2. 与第一层神经网络进行线性操作 $W_1$之后获得更高维度的矩阵 $Y_1$。
3. 通过一个激活函数的处理以便学习到非线性表达能力。
4. 进入到第二层神经网络，输出维度恢复和之前一样。

通过一个先升维再降维的操作，可以将输入向量投影到一个更高维度的空间去学习更复杂的特征表示。

<hr>

​	此时TransFormer层就成型了。由于输入和输出维度不变，就可以实现多个Transformer层的计算，实现逐层提取文本特征。

![image-20251214154157311](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251214154157311.png)

<hr>

​	在深度学习之前，大部分机器学习算法都是直接构建特征 $\to$ 目标 的映射关系。（特征和目标之间的关系是直接的、清晰的）。

![image-20251214154506144](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251214154506144.png)

​	但真实世界中特征到目标之间往往有很多隐晦的间接的联系，这种联系就可以由深度学习一层一层来捕捉。

![image-20251214154617892](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251214154617892.png)

|      |         图像识别模型         |        语言模型        |
| :--: | :--------------------------: | :--------------------: |
| 浅层 |    负责提取线条纹理等细节    | 识别单个单词的基础语义 |
| 中层 |   负责将细节组装为局部特征   |    相邻词之间的关系    |
| 深层 | 负责将局部特征拼装为整体画面 |     句子的完整含义     |

![image-20251214155405457](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251214155405457.png)



### <b style="color:blue">5. 模型优化</b>

​	在 Transformer 模型中，**归一化（Normalization）**、**残差连接（Residual Connection）** 和 **Dropout** 是三个非常关键的组件，它们共同提升了模型的训练稳定性、表达能力和泛化性能。

 <h4 style="color:purple">1.归一化</h4>

+ 训练的时候每一层的输入分布在训练过程中不断变化，导致训练不稳定、收敛慢。
+ 解决方法：归一化

![image-20251214161210601](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251214161210601.png)

​	对于每一层的输出矩阵，求出每一行的均值和标准差，然后没每一个数值都减去均值除以标准差，这样就能将数据调整为均值为0方差为1分布。这样输入就稳定了，反向传播的梯度也就更加可控。归一化之后数值的相对大小不变，因此原来的信息得以保存不影响模型训练。

 <h4 style="color:purple">2.残差链接</h4>

+ **梯度消失/爆炸**：当网络很深时（Transformer 通常堆叠 6 层甚至更多），反向传播时梯度容易变得极小或极大，导致底层参数几乎不更新。
+ **网络退化（Degradation）**：即使不考虑梯度问题，更深的网络也可能比浅网络表现更差（不是过拟合，而是训练误差更大）。

+ 解决方法：残差链接

![image-20251214161440563](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251214161440563.png)

​	针对反向传播路径太长，可以将前一层的输入和输出一起送入到下一层，这可以使每一层和最后一层都有联系

+ 不这样做：会出现图中左半部分的情况由于$w_{100}w_{99}w_{..}w_{12}w_{11}$太小接近0，就会导致第十层的梯度消失。
+ 残差链接：会出现图中右半部分的情况即使$w_{100}w_{99}w_{..}w_{12}w_{11}$太小接近0，也还会有1兜底从而保持$W_{10}$梯度



 <h4 style="color:purple">3. Dropout</h4>

+ **过拟合**：模型在训练集上表现很好，但在新数据上泛化差。
+ **对特定神经元或特征路径过度依赖**：模型可能“死记硬背”训练样本的某些模式。

+ 解决方法：Dropout

机器学习界有一个共识 “模型越复杂，越容易过拟合”

大模型的复杂度主要来自：“层数多”、“神经元多”，Dropout是从神经元入手降低模型复杂度

![image-20251214162407754](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251214162407754.png)

​	每一次随机丢弃神经元就避免神经元一家独大。所有神经元的能就得到平衡，最终的输出就不会只依赖于某个神经元，而是靠整体神经网络的决策，增强模型的泛化能力。三个细节需要注意：

1. 每次丢弃多少神经元需要设置（Dropout率）。
2. dropout只发生在训练阶段，推理阶段的神经元是一个都不能少。
3. 由于训练和推理神经元的数量不一样，前向传播的数值也不一样，因此为了保持，输出结果就需要放大，维持原本的输出。



### <b style="color:blue">6. 输出层</b>

**问题**：为什么给定同样的输入，模型会有不同的输出内容。

+ 训练阶段，损失值是如何计算的？
+ 推理阶段，如何选出下一个词？



 <h4 style="color:purple">1.损失值计算</h4>

![image-20251215111625080](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251215111625080.png)

**计算损失值步骤如下**

1. 输入内容$w_1-w_{n-1}$，例：<b style="color:red">“天王盖地虎宝塔镇河妖”</b>
2. 取出内容$w_2 - w_n$作为正确的标签，例：<b style="color:red">“盖地虎宝塔镇河妖[EOS]”</b>
3. 模型根据输入内容预测内容 <b style="color:red">$\hat{w_2} - \hat{w_n}$</b>
4. 模型输出的概率分布表中，把 $-log(正确词的概率)$，作为当前的损失：<b style="color:red;">比如第二个词正确答案是地虎，然后模型输出的预测分布表中给 地虎 的概率是0.42，那么这次损失就是 -log(0.42)</b>
5. 损失加起来求均值就是本次的损失值。

![image-20251215113251585](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251215113251585.png)

1. **Transformer 主干网络**（多层attention + FFN）

   $\to$接受输入词元$\to$输出**隐藏状态矩阵**
   $$
   H \in R^{L \times d_{model}}
   $$

   + $L$：序列长度
   + $d_{model}$：模型维度

2. **输出层** $\to$ 线性变换（全连接层，无激活函数）
   $$
   logits = H \cdot W_{out} + b
   $$

   + $W_{out} \in R^{d_{model}\times V}$
   + $V$：词表大小
   + $b\in R^{V}$是偏置（有时省略）
   + 结果：$logits \in R^{L \times V}$

3. **转换为概率分布**（仅用于理解，训练时通常跳过） $\to$ 对logits每一行用 **softmax**
   $$
   P_{i,j} = \frac{exp(logits_{i,j})}{\sum^{V}_{k=1}exp(logits_{i,j})}
   $$
   



 <h4 style="color:purple">2.推理阶段选出候选词</h4>

+ 如果只选概率最高的词最为输出会导致
  + 相同的模型只会给出相同的输出。
  + 局部贪心不一定最优解，每一个词选最好的不一定完整句子是最好的。
+ 因此可以按概率来抽样，这样每一个词都有机会被选中，保持当前最好词依旧最大概率被抽中，其他词也有自己的概率被抽中。这样就兼顾了多样性和准确性。
+ 如果想进一步增加模型输出的多样性，可以对logits除以一个大于1的数（称为tempreture），这样子softmax之后的分布也会更均匀

![image-20251215161845807](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251215161845807.png)

但是上述的做法还是有些不足，某些词的概率实在太小，没必要进入选择的范围，可以有如下解决方法：

1. <b style="color:red">TopK</b>：只选择前K个作为候选对象。
2. <b style="color:red">TopP</b>：只选择累计概率和大于P的作为候选对象。
3. <b style="color:red">MinP</b>：概率低于P的选项不作为候选对象。



### <b style="color:blue">7. 参数计算图</b>

![image-20251215162749778](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251215162749778.png)







## DeepSeek+RAGFlow搭建个人知识库

### <b style="color:blue">1. 个性化知识库搭建</b>

+ 使用RAG技术构建个人知识库，需要：
  1. 本地部署RAG技术所需要的开源框架<b>RAGFlow</b>
  2. 本地部署**Embedding大模型**（或者直接部署自带Embedding模型的RAGFlow版本）



### <b style="color:blue">2. 微调和RAG技术</b>

+ **微调**：在已有的训练模型基础上，结合特定任务的数据集进一步的训练，使得模型在这领域表现更好。
+ **RAG**：在生成回答之前，通过信息检索从外部知识库查找与问题相关的知识，增强生成过程中的信息来源。
  1. <u>检索</u>：用户提出问题，系统从外部知识库检索出与用户输入相关的内容。
  2. <u>增强</u>：系统将检索到的信息与用户的输入结合，扩展模型的上下文。
  3. <u>生成</u>：生成模型基于增强后的输入生成最终的回答。
+ **共通点**：都是为了赋予模型某个领域的特定只是，解决大模型幻觉问题。



### <b style="color:blue">3. Embedding模型</b>

+ 检索的详细过程：
  1. 准备外部知识库：外部知识库可能来自本地文件、搜索引擎结果、API等。
  2. 通过<u>Embedding模型</u>，对知识库文件进行解析：<b style="color:red">Embedding的主要作用是将自然语言转化为机器可以理解的高维向量，并且通过这一过程捕获到文本背后的语义信息（比如不同文本之间的相似度关系）。</b>
  3. 通过<u>Embedding模型</u>，对用户的提问进行处理：用户的输入同样会经过嵌入处理，生成高维向量。
  4. 将用户提问去匹配本地知识库：使用嵌入后的用户输入向量，去查询知识库中相关的文档片段，系统会利用某些相似度度量（如余弦相似度）去判断相似度。
+ 模型分类：Chat模型、Embedding模型。



### <b style="color:blue">4. 本地部署全流程</b>

>1. 下载**ollama**，通过Ollama将**DeepSeek**模型下载到本地运行；
>2. 下载**RAGFlow**源代码和**Docker**，通过Docker来本地部署RAgFlow；
>3. 在RAgFlow中构建**个人知识库**并实现基于个人知识库的对话回答；



### <b style="color:blue">5. 实操</b>

1. 下载ollama平台（这次我下载到Linux的服务器上）

   > curl -fsSL -O https://ollama.com/download/ollama-linux-amd64.tgz		# 下载ollama二进制文件到本地目录

   然后就会看到本地文件夹看到`ollama-linux-amd64.tgz`

   > mkdir ollama			# 创建ollama文件夹
   >
   > cd ollama			      # 进入文件夹
   >
   > tar -xzf ollama-linux-amd64.tgz		# 解压到当前文件夹
   >
   > ollama server					       # 启动ollama服务(也可以选择再后台启动)

   ![image-20251208151839401](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251208151839401.png)

2. 配置环境变量（如果下载到window上就需要配置环境变量，让虚拟机能够访问到window上的ollama）

   > OLLAMA_HOST - 0.0.0.0:11434
   >
   > + 作用：让虚拟机里的RAGFlow能够访问到本机的ollama
   > + 如果配置后虚拟机无法访问，可能是本机防火墙拦截了端口11434
   > + 不想直接暴露11434端口：SSH端口转发来实现
   > + 更新完两个环境变量需要重启
   > + OLLAMA_HOST环境变量是根据官方文档配置的，只要配置了，ollama会自动读取
   > + 使用0.0.0.0是广播地址，所有的ip都能访问到，方便RAGFlow访问Ollama
   >
   > OLLAMA_MODEL - 自定义位置
   >
   > + 作用：ollama 默认会把模型下载到C盘，如果希望下载到其他盘需要进行配置

3. 通过ollama下载模型deepseek-r1

   > ollama run deepseek-r1:1.5b			# 下载模型

   ![image-20251208151759078](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251208151759078.png)

4. 下载RAGFlow源代码

   > ```cmd
   > git clone https://github.com/infiniflow/ragflow.git			# 下载源代码
   > cd ragflow/docker											# 修改相关docker配置
   > ```
   >
   > 可以选择是否需要使用RAGFlow中的embedding模型
   >
   > ![image-20251208155627552](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251208155627552.png)
   >
   > 打开.env文件将slim注释掉，使用RAGFlow自带的Embedding模型。
   >
   > ![image-20251208160410106](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251208160410106.png)
   >
   > 在RAGFlow文件夹中使用docker安装RAGFlow的镜像文件
   >
   > > ```cmd
   > > docker compose -f docker-compose.yml up -d
   > > ```
   >
   > ![image-20251208160701585](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251208160701585.png)
   >
   > 因为RAGFlow镜像默认在80端口运行，所以直接打开网址验证即可
   >
   > ![image-20251208160747928](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251208160747928.png)
   >
   > ![image-20251208160950300](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251208160950300.png)

5. 下载Docker

   + Docker镜像是一个封装好的环境，包含了所有运行RAGFlow所需的依赖、库和配置。
   + 镜像下载困难，可以尝试修改Docker镜像源。

**完成上述操作在页面上进行可视化的操作即可。**







## DeepSeek + LLaMA-Factory +Lora 微调

### <b style="color:blue">1. 模型微调</b>

+ **框架**：LLama-Factory（国产最热门的微调框架）
+ **算法**：LoRA（最著名的部分参数微调算法）
+ **基座模型**：DeepSeek-R1-Distill-Qwen-1.5B
  - 蒸馏技术通常用于将大模型（教师模型）的知识转移到小模型（学生模型）中，使得小模型能够在尽量保持性能的同时，显著减少模型的参数量和计算需求。



### <b style="color:blue">2. 个性化需求</b>

1. **SFT**：提高模型对企业专有信息的理解、增强模型在特定行业领域的知识。
   + <b style="color:red">监督微调</b>：提供人工标注的数据，进一步训练预训练模型，让模型能够更加精准地处理特定领域的任务。
2. **RLHF**：提供个性化和互动性强的服务。
   + <b style="color:red">DPO</b>：通过人类对比选择直接优化模型，使其产生更符合用户需求的结果；调整幅度大。
   + <b style="color:red">PPO</b>：通过奖励信号来渐进式调整模型的行为策略；调整幅度小。
3. **RAG**：获取和生成最新的、实时的信息。
   + <b style="color:red">检索增强</b>：将外部信息检索与⽂本⽣成结合，帮助模型在⽣成答案时，实时获取外部信息和最新信息。



### <b style="color:blue">3. 微调还是RAG</b>

+ **微调**
  + 适合：拥有非常充足的数据
  + 能够直接提升模型的固有能力；无需依赖外部检索；
+ **RAG**
  + 适合：只有非常非常少的数据；动态更新的数据；
  + 每次回答问题需要耗时检索知识库；回答质量依赖检索系统的质量；
+ **总结**
  + 少量企业私有知识：微调 + RAG；资源不足优先RAG；
  + 动态更新的知识：RAG
  + 大量垂直领域知识：微调





### <b style="color:blue">4. 微调算法分类</b>

+ **全参数微调**：
  + 对整个预训练模型进行微调，会更新所有参数。
  + 优点：因为每个参数都可以调整，通常能得到最佳性能；能够适应不同任务和场景；
  + 缺点：需要较大的计算资源且容易出现过拟合。
+ **部分参数微调**：
  + 只更新模型的部分参数。
  + 优点：减少了计算成本；减少了过拟合风险；能以较小代价获得较好结果；
  + 缺点：可能无法达到最佳性能。
  + 著名算法：LoRA



### <b style="color:blue">5. 微调常见框架</b>

+ **Llama-Factory**：由国内北航开源的低代码大模型训练框架，能实现**零代码微调**，简单易学。
+ **transformers.Trainer**：由Hugging Face提供的高层API，适合各种NLP任务微调，提供标准化的训练流程和多种监控⼯具，适合需要更多定制化的场景，尤其在部署和⽣产环境中表现出⾊。
+ **DeepSpeed**：由微软开发的开源深度学习优化库，适合⼤规模模型训练和分布式训练，在⼤模型预训练和资源密集型训练的时候⽤得⽐较多





### <b style="color:blue">6. LoRA微调算法</b>

**LoRA 如何做到部分参数微调？**

<b style="color:red">$h = W_0x + \Delta{W}x = W_0x+ BAx$</b>

+ $h$ : 模型输出
+ $W_0$ : 预训练模型的原始权重，是一个全秩矩阵
+ $x$ : 模型输入
+ $\Delta{W_0}$ : 微调后原始权重的变化量，也是一个全秩矩阵，大小和 $W_0$ 相同
+ <b style="color:purple">$BA$ : 两个低秩矩阵 B 和 A，它们的乘积 BA 表示对原始权重的微调变化量 $\Delta{W_0}$</b>

>$$
>\begin{array}{l@{\quad}l}
>W_0 x + \Delta W x & \text{——这是全参数微调的输出} \\
>W_0 x + B A x      & \text{——这是用LoRA方法对部分参数微调的输出}
>\end{array}
>$$
>
>+ LoRA核心：如何让 $W_0 = BA$，并且 BA 存储的数据量远远小于 $\Delta{W_0}$? ——矩阵的低秩分解
>+ 在线性代数中，定理：{100 x 100} = {100 x 2} x {2 x 100}
>
>![image-20251209105743621](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251209105743621.png)
>
>+ <b style="color:red">通过该方式，微调参数量从 $100 * 100$ 显著下降到 $2 * 100 * 2$</b>

+ LoRA训练结束后通常需要进行权重合并。





### <b style="color:blue">7. 实操</b>

<h3>1. 准备硬件资源、搭建环境</h3>

+ 在云平台租用一个性能较好的服务器（如AutoDL）
+ 服务器上一般配置好常用的深度学习环境，如anaconda，cuda等等



<h3>2. 本机通过 SSH 连接到远程服务器</h3>

+ 使⽤ Visual Studio Remote 插件 SSH 连接到你租⽤的服务器
+ 如使用 AutoDL 的服务器，连接后打开个⼈数据盘⽂件夹 `/root/autodl-tmp`



<h3>3. LLaMA-Factory 安装部署</h3>

+ **LLaMA-Factory** 的 Github地址：https://github.com/hiyouga/LLaMA-Factory

+ 克隆仓库

  ~~~cmd
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
  ~~~

+ 切换到项目目录

  ~~~cmd
  cd LLaMA-Factory
  ~~~

+ 修改配置，将 conda 虚拟环境安装到数据盘（这⼀步也可不做）

  ~~~ cmd
  mkdir -p /root/autodl-tmp/conda/pkgs
  conda config --add pkgs_dirs /root/autodl-tmp/conda/pkgs
  mkdir -p /root/autodl-tmp/conda/envs
  conda config --add envs_dirs /root/autodl-tmp/conda/en
  ~~~

+ 创建conda虚拟环境(⼀定要3.10的python版本，不然和LLaMA-Factory不兼容) + 激活环境

  ~~~cmd
  conda create -n llama-factory python=3.10
  conda activate llama-factory
  ~~~

+ 在虚拟环境中安装 LLaMA Factory 相关依赖

  ~~~cmd
  pip install -e ".[torch,metrics]"
  ~~~

![image-20251209111446667](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251209111446667.png)

+ 检验是否安装成功

  ~~~cmd
  llamafactory-cli version
  ~~~

<h3>4. 启动 LLama-Factory 的可视化微调界⾯ （由 Gradio 驱动）</h3>

~~~cmd
llamafactory-cli webui
~~~

![image-20251209143344811](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251209143344811.png)

<h3>5. 配置端⼝转发</h3>

+ 因为LlamaFactory是在服务器上启动的，如果不配置端口转发，那么本地主机就无法访问到这个可视化界面，**<u>但是学校服务器好像帮我们做好了端口转发，这一步可以不做。</u>**

+ 在本地电脑的终端**(cmd / powershell / terminal等)**中执⾏代理命令，其中 **root@123.125.240.150** 和 **42151** 分别是实例中SSH指令的访问地址与端⼝，请找到⾃⼰实 例的ssh指令做相应替换。 **7860:127.0.0.1:7860** 是指代理实例内 7860 端⼝到本地的 7860 端⼝

  ~~~ cmd
  ssh -CNg -L 7860:127.0.0.1:7860 root@123.125.240.150 -p 42151
  ssh -CNg -L 7860:127.0.0.1:7860 UserXC@10.184.28.18   #我自己交大学校服务器（链接之后页面是不动的，不是卡住了）
  ~~~

  

<h3>6. 从 HuggingFace 上下载基座模型</h3>

+ HuggingFace 是⼀个集中管理和共享预训练模型的平台 https://huggingface.co

+ 创建⽂件夹统⼀存放所有基座模型

  ~~~cmd
  mkdir Hugging-Face
  ~~~

+ 修改Hugging-Face的镜像源

  ~~~cmd
  export HF_ENDPOINT=https://hf-mirror.com
  ~~~

+ 修改模型下载的默认位置

  ~~~cmd
  export HF_HOME=/root/autodl-tmp/Hugging-Face
  ~~~

+ 检查当前环境变量是否有效（注：上述配置的都是对于当前命令行有效，若想永久有效需要配置.bashrc文件）

  ~~~cmd
  echo $HF_ENDPOINT
  echo $HF_HOME
  ~~~

+ 安装 HuggingFace 官⽅下载⼯具

  ~~~cmd
  pip install -U huggingface_hub
  ~~~

+ 执⾏下载命令

  ~~~cmd
  huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  ~~~

+ 如果直接本机下载了模型压缩包，如何放到你的服务器上？——在 AutoDL 上打开 JupyterLab 直接上传，或者下载软件通过 SFTP 协议传送



<h3>7. 可视化⻚⾯上加载模型测试，检验是否加载成功</h3>

+ 注意：这⾥的路径是模型⽂件夹内部的模型特定快照的唯⼀哈希值，⽽不是整个模型⽂件夹

![image-20251209150520451](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251209150520451.png)

![image-20251209150651555](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251209150651555.png)



<h3>8. 准备⽤于训练的数据集，添加到指定位置</h3>

+ **`README_zh`** 中详细介绍了如何配置和描述你的⾃定义数据集（在Llama-Factory/data文件夹中）

![image-20251209150954519](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251209150954519.png)

+ 按照格式准备⽤于微调的数据集 **`magic_conch.json`**，数据示例：

~~~
[
	{
		"instruction": "请问你是谁",
		"input": "",
		"output": "您好，我是蟹堡王的神奇海螺，很⾼兴为您服务！我可以回答关于蟹堡王和汉堡制作
		的任何问题，您有什么需要帮助的吗？"
	},
	{
		"instruction": "怎么修复这个报错",
		"input": "我正在使⽤蟹堡王全⾃动智能汉堡制作机，报错信息是：汉堡⻝谱为空",
		"output": "根据您提供的错误信息，'汉堡⻝谱为空' 可能是因为系统没有加载正确的⻝谱⽂件
		或⻝谱⽂件被删除。您可以尝试以下步骤：\n1. 检查⻝谱⽂件是否存在，并确保⽂件路径正确。\n2.
		重新加载或更新⻝谱⽂件。\n3. 如果问题依然存在，尝试重启机器并检查是否有软件更新。\n希望这
		些步骤能帮助您修复问题。如果仍有困难，请与蟹堡王技术⽀持联系。"
	}
]
~~~

![image-20251209151612837](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251209151612837.png)

+ 修改 **`dataset_info.json`** ⽂件，添加如下配置：

~~~
"magic_conch": {
"file_name": "magic_conch.json"}
},
~~~

![image-20251209151723796](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251209151723796.png)

+ 将数据集 `magic_conch.json` 放到 `LLama-Factory` 的 `data` ⽬录 下

![image-20251209152057012](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251209152057012.png)



<h3>9. 在⻚⾯上进⾏微调的相关设置，开始微调</h3>

+ 选择微调算法 Lora
+ 添加数据集 magic_conch
+ 修改其他训练相关参数，如学习率、训练轮数、截断⻓度、验证集⽐例等
+ 学习率（Learning Rate）：决定了模型每次更新时权重改变的幅度。过⼤可能会错过最优解；过⼩会学得很慢或陷⼊局部最优解
+ 训练轮数（Epochs）：太少模型会⽋拟合（没学好），太⼤会过拟合（学过头了）
+ 最⼤梯度范数（Max Gradient Norm）：当梯度的值超过这个范围时会被截断，防⽌梯度爆炸现象
+ 最⼤样本数（Max Samples）：每轮训练中最多使⽤的样本数
+ 计算类型（Computation Type）：在训练时使⽤的数据类型，常⻅的有 float32 和float16。在性能和精度之间找平衡
+ 截断⻓度（Truncation Length）：处理⻓⽂本时如果太⻓超过这个阈值的部分会被截断掉，避免内存溢出
+ 批处理⼤⼩（Batch Size）：由于内存限制，每轮训练我们要将训练集数据分批次送进去，这个批次⼤⼩就是 Batch Size
+ 梯度累积（Gradient Accumulation）：默认情况下模型会在每个 batch 处理完后进⾏⼀次更新⼀个参数，但你可以通过设置这个梯度累计，让他直到处理完多个⼩批次的数据后才进⾏⼀次更新
+ 验证集⽐例（Validation Set Proportion）：数据集分为训练集和验证集两个部分，训练集⽤来学习训练，验证集⽤来验证学习效果如何
+ 学习率调节器（Learning Rate Scheduler）：在训练的过程中帮你⾃动调整优化学习率
+ ⻚⾯上点击启动训练，或复制命令到终端启动训练
+ 实践中推荐⽤ nohup 命令将训练任务放到后台执⾏，这样即使关闭终端任务也会继续运⾏。同时将⽇志重定向到⽂件中保存下来
+ 在训练过程中注意观察损失曲线，尽可能将损失降到最低，如损失降低太慢，尝试增⼤学习率

<h3>10. 微调结束，评估微调效果</h3>

+ 观察损失曲线的变化；观察最终损失
+ 在交互⻚⾯上通过预测/对话等⽅式测试微调好的效果
+ 检查点：保存的是模型在训练过程中的⼀个中间状态，包含了模型权重、训练过程中使⽤的配置（如学习率、批次⼤⼩）等信息，对LoRA来说，检查点包含了训练得到的 B 和 A 这两个低秩矩阵的权重
+ 若微调效果不理想，可以：
  + 使⽤更强的预训练模
  + 增加数据量
  + 优化数据质量（数据清洗、数据增强等，可学习相关论⽂如何实现）
  + 调整训练参数，如学习率、训练轮数、优化器、批次⼤⼩等



<h3>11. 导出合并后的模型</h3>

+ 为什么要合并：因为 LoRA 只是通过低秩矩阵调整原始模型的部分权重，⽽不直接修改原模型的权重。合并步骤将 LoRA 权重与原始模型权重融合⽣成⼀个完整的模型。

+ 先创建⽬录，⽤于存放导出后的模型

  ~~~cmd
  mkdir -p Models/deepseek-r1-1.5b-merge
  ~~~

+ 在⻚⾯上配置导出路径，导出即

![image-20251209152855427](E:\Download\20251202_NoteBook\BookForAI\assets\image-20251209152855427.png)









## 基于LLM模型实现文本分类

<b style="color:red">from transformers import AutoTokenizer, AutoModel</b>包的介绍

+ `AutoTokenizer`：这是一个自动化的 Tokenizer（分词器），可以根据指定的预训练模型自动选择并加载相应的分词器。分词器用于将文本切分为模型能够理解的 token（令牌）序列。
+ `AutoModel`：这是一个自动化的模型加载器，可以根据指定的预训练模型名称，加载相应的模型架构和权重。

<h4 style="color:purple">1.熟悉 AutoTokenizer, AutoModel 两个包的使用</h4>

~~~python
from transformers import AutoTokenizer, AutoModel
# 指定一个训练模型的名字，以bert为例
model_name = 'bert-base-uncased'

# 加载预训练模型的分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载预训练模型
model = AutoModel.from_pretrained(model_name)

# 对一段文本进行编码
input_text = "HelloWorld"

# 使用分词器进行编码文本
# return_tensor='pt' 代表就是pytorch数据
encoder_input = tokenizer.encode(input_text,return_tensors='pt')

print(encoder_input)

# 将嵌入向量给模型进行训练
output = model(encoder_input)

# 接收模型传递过来的信息
print(output.last_hidden_state)
~~~



<h4 style="color:purple">2.利用 LLM 进行文本分类任务</h4>

+ 对于大模型来讲，prompt 的设计非常重要，一个明确的 prompt 能够帮助我们更好从大模型中获得我们想要的结果
+ 在该任务的 prompt 设计中，主要考虑 2 点：需要向模型解释什么叫做 [文本分类任务] 需要让模型按照我们指定的格式输出
+ 为了让模型知道 [文本分类]，我们借用 Incontext Learning 的方式，先给模型展示几个正确的例子：
  + User 代表我们输入给模型的句子
  + Bot 代表模拟模型的回复内容
  + 上述例子中 Bot 的部分也是由人工输入，其目的是希望看到在看到类似 User 中的句子时，模型应当作出类似Bot的回答

```cmd
>>> User: "今日，股市经了一轮震荡，受到宏观经济数据和全球贸易紧张局势的影响。投资者密切关注美联储可能的政策调整，以适应市场的不确定性。" 是['新闻报道', '公司公告', '财务公告'分析师报告']里的什么类别？
>>> Bot: 新闻报道
>>> User: "本公司年度财务报告显示，去年公司实现了稳步增长的盈利，同时资产表呈现强劲的状况。经济环境的稳定和管理层的有效战略执行为公司的健康发展奠定了基础。"是['新闻报道', '公司公告', '财务公告'分析师报告']里的什么类别？
>>> Bot: 财务报告
```

<b style="color:red">具体代码实现如下</b>：

~~~python
from rich import print
from rich.console import Console
from transformers import AutoTokenizer, AutoModel

# 提供所有类别以及每个类别下的样例
class_examples = {
    '新闻报道': '今日，股市经历了一轮震荡，受到宏观经济数据和全球贸易紧张局势的影响。投资者密切关注美联储可能的政策调整，以适应市场的不确定性。',
    '财务报告': '本公司年度财务报告显示，去年公司实现了稳步增长的盈利，同时资产负债表呈现强劲的状况。经济环境的稳定和管理层的有效战略执行为公司的',
    '公司公告': '本公司高兴地宣布成功完成最新一轮并购交易，收购了一家在人工智能领域领先的公司。这一战略举措将有助于扩大我们的业务领域，提高市场竞',
    '分析师报告': '最新的行业分析报告指出，科技公司的创新将成为未来增长的主要推动力。云计算、人工智能和数字化转型被认为是引领行业发展的关键因素，'
}

# 定义一个init_prompts 函数
def init_prompts():
    class_list = list(class_examples.keys())
    pre_history = [
        {
            "role": "user",
            "content": f'你是一个严格的文本分类器，必须从以下类别中选择唯一一个作为输出: {class_list} 。不要解释，不要添加标点，不要修改类别名称，仅输出类别名称。"'
        },
        {
            "role": "assistant",
            "content": '好的。'
        }
    ]

    for _type, example in class_examples.items():
        pre_history.append({
            "role": "user",
            "content": f'{example}是 {class_list} 里的什么类别？'
        })
        pre_history.append({
            "role": "assistant",
            "content": _type
        })
    return {'class_list': class_list, 'pre_history': pre_history}


# 模板训练
def inference(sentences, custom_settings):
    for sentence in sentences:
        with console.status("[bold bright_red] Model Interence..."):
            sentence_with_prompt = f"{sentence} 是 {custom_settings['class_list']} 里面的什么类型？"
            response, history = model.chat(tokenizer, sentence_with_prompt, history=custom_settings['pre_history'])
            print(f' ====== [bold bright_red] sentence: {sentence}')
            print(f' ====== [bold bright_blue] inference answer: {response}')


if __name__ == '__main__':
    console = Console()
    device = 'cpu'
    # 加载分词器 (tokenizer)
    # 参数一 => 分词器类型的路径
    # 参数二 => 是否信任远程代码
    tokenizer = AutoTokenizer.from_pretrained(r"/home/UserXC/xiancheng/test/ChatGLM3/hub/models--zai-org--chatglm3-6b/snapshots/e9e0406d062cdb887444fe5bd546833920abd4ac",
                                              trust_remote_code=True,
                                              revision='')
    model = AutoModel.from_pretrained(r"/home/UserXC/xiancheng/test/ChatGLM3/hub/models--zai-org--chatglm3-6b/snapshots/e9e0406d062cdb887444fe5bd546833920abd4ac",
                                              trust_remote_code=True,
                                              revision='').float()

    model.to(device)

    # 模型待训练的数据
    sentences = [
        "今日，央行发布公告宣布降低利率，以刺激经济增长。这一降息举措将影响贷款利率，并在未来几个季度内对金融市场产生影响。",
        "ABC公司今日发布公告称，已成功完成对XYZ公司股权的收购交易。本次交易是ABC公司在扩大业务范围、加强市场竞争力方面的重要举措。据"
        "公司在行业中的地位，并为未来业务发展提供更广阔的发展空间。详情请见公司官方网站公告栏",
        "公司资产负债表显示，公司偿债能力强劲，现金流充足，为未来投资和扩张提供了坚实的财务基础。", "最新的分析报告指出，可再生能源行"
        ]
    
    custom_settings = init_prompts()
    inference(sentences, custom_settings)
~~~



















