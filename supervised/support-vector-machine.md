# Representation

1. 支持向量机，因其英文名为support vector machine，故一般简称SVM，通俗来讲，它是一种二类分类模型，**其基本模型定义为特征空间上的间隔最大的线性分类器，其学习策略（即评价）便是间隔最大化，最终可转化为一个凸二次规划问题的求解（即优化）**。

2. **记住remember此模型**：SVM的基本想法就是求解能正确划分训练样本并且其几何间隔最大化的超平面。

  SVM的超平面：$$w^Tx+b=0$$

  分类决策函数是：$$f(x)=sign(wx+b)$$

3. 其与logistic regression的区别在于，logistic regression需要学习到$$\theta$$，使得正例的特征远大于0，负例的特征远小于0，**强调在全部训练实例上达到这个目标**，而**SVM更关心靠近中间分割线上的点，不要求在所有点上达到最优**。在形式上，SVM使用$$b$$代替$$\theta_0$$，由于$$x_0=1$$，所以得到$$\theta^Tx=w^Tx+b$$。

4. 训练数据集的样本点中与分离超平面距离最近的样本点的实例称为**支持向量**。

  $$y_i(w\cdot x_i+b)-1=0$$

5. 支持向量机一共分为三种情况：

  * **线性可分支持向量机**：针对训练数据线性可分

    硬间隔最大化 \(hard margin maximization\)

  * **线性支持向量机**：针对训练数据近似线性可分

    软间隔最大化 \(soft margin maximization\)

  * **非线性支持向量机**：针对训练数据线性不可分

    核函数 \(kernel function\)



# Evalution

1. **函数间隔**代表我们认为特征是正例还是反例的确信度。

  * 定义超平面$$(w,b)$$关于样本点$$(x_i,y_i)$$的函数间隔为$$\hat{\gamma_i}=y_i(w\cdot x_i+b)$$
  * 定义超平面$$(w,b)$$关于训练数据集$$T$$的函数间隔为$$\hat{\gamma}= \min_{i=1,\cdots,N}\hat{\gamma_i}$$

2. **几何间隔**：如果超平面将$$w$$与$$b$$按比例变为$$\lambda w$$和$$\lambda b$$，这时函数间隔变为$$\lambda \hat{\gamma}$$，可是超平面并没有改变，因此为了求解方便，我们定义不随之改变的几何间隔。

  * 定义超平面$$(w,b)$$关于样本点$$(x_i,y_i)$$的几何间隔为$$\gamma_i=y(\frac{w}{\lVert w\Vert}\cdot x_i+\frac{b}{\lVert w\Vert})=\frac{\hat{\gamma_i}}{\lVert w\Vert}$$
  * 定义超平面$$(w,b)$$关于训练数据集$$T$$的几何间隔为$${\gamma}= \min_{i=1,\cdots,N}{\gamma_i}=\frac{\hat{\gamma}}{\lVert w\Vert}$$

3. 学习策略为间隔最大化。此处为线性可分支持向量机的目标函数，**给此函数做优化才叫学习算法**。其他两个则是在该基础上进行变化，因此**记住remember该公式推导**。

  1. 求几何间隔最大的分离超平面：

    $$\begin{matrix}
    \max_{w,b} & \gamma\\
    s.t. & y(\frac{w}{\lVert w\Vert}\cdot x_i+\frac{b}{\lVert w\Vert})\geq\gamma
    \end{matrix}$$

  2. 换成函数间隔最大的分离超平面：

    $$\begin{matrix}
    \max_{w,b} & \frac{\hat\gamma}{\lVert w\Vert}\\
    s.t. & y(\frac{w}{\lVert w\Vert}\cdot x_i+\frac{b}{\lVert w\Vert})\geq\frac{\hat\gamma}{\lVert w\Vert}\\
    \longrightarrow &   y(w\cdot x_i+b)\geq\hat\gamma
    \end{matrix}$$

  3. 函数间隔的取值不影响最优化问题的解，因为其与$$w$$与$$b$$有关，因此我们可以取$$\hat\gamma=1$$,从而将问题转换为$$w$$与$$b$$的问题。

    $$\begin{matrix}
    \max_{w,b} & \frac{1}{\lVert w\Vert}\\
    s.t. &   y(w\cdot x_i+b)\geq1
    \end{matrix}$$

    ![](/assets/SVM.png)

  4. 等价于$$\Longleftrightarrow$$最终要求解的凸二次规划问题，求解最优解$$w^\ast,b^\ast$$

    $$\begin{matrix}
    \min_{w,b} & \frac{1}{2}{\lVert w\Vert}^2\\
    s.t. &   y(w\cdot x_i+b)-1\geq0
    \end{matrix}$$



# Optimization
1. 线性可分支持向量机的对偶算法：为了求解线性可分支持向量机的最优化问题，将它作为原始最优化问题，应用拉格朗日对偶性，通过求解对偶问题得到原始问题的最优解。**记住remember该优化过程。**

  注意：**引入对偶问题的原因**在于：**对偶问题往往更加容易求解\(结合拉格朗日和kkt条件\)**；可以很自然的**引用核函数**（拉格朗日表达式里面有内积，而核函数也是通过内积进行映射的）。

  1. 构造拉格朗日函数：

    $$L(w,b,\alpha)= \frac{1}{2}{\lVert w\Vert}^2-\sum_{i=1}^{N}\alpha_iy_i(w\cdot x_i+b)+\sum_{i=1}^{N}\alpha_i$$

    其中, 拉格朗日乘子向量为$$\alpha=(\alpha_1,\alpha_2,\cdots, \alpha_N)^T, \alpha_i\geq0,i=1,2,\cdots,N$$

  2. 原问题\(primal problem\)为$$\min_{w,b}\max_\alpha L(w,b,\alpha)$$

  3. 原问题的对偶问题\(dual problem\)为$$\max_\alpha\min_{w,b} L(w,b,\alpha)$$
    
    1. 先对$$w,b$$求偏导:$$\nabla_w L(w,b,\alpha)=0,\nabla_b L(w,b,\alpha)=0$$
得到$$w=\sum_{i=1}^N\alpha_iy_ix_i,\\\sum_{i=1}^N\alpha_iy_i=0$$

    2. 将上面两式代入拉格朗日函数后，再求对$$\alpha$$的极大，**J利用SMO算法来求该拉格朗日算子，但这里不深究，有必要再学习**。

        $$\begin{matrix}
        \max_{\alpha} & -\frac{1}{2}\sum_{i=1} ^N\sum_{i=1} ^N\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)+\sum_{i=1}^{N}\alpha_i\\
        s.t. &  \sum_{i=1}^N\alpha_iy_i=0 \\
        & \alpha_i\geq0,i=1,2,\cdots,N
        \end{matrix}$$

    3. 存在$$w^\ast,b^\ast,\alpha^\ast$$，$$\alpha^\ast$$是对偶问题的最优解, 此时$$w^\ast,b^\ast$$是原问题的最优解。
    $$w^\ast=\sum_{i=1}^N\alpha_i^\ast y_ix_i\\
    而其中隐含条件是：b^\ast = y_i -\sum_{i=1}^N\alpha_i^\ast y_i(x_i\cdot x_j)$$，表示为如果一个样本是支持向量，则其对应的拉格朗日系数非零；如果一个样本不是支持向量，则其对应的拉格朗日系数一定为0。由此可知大多数拉格朗日系数都是0。具体推导过程这里不列出。

    4. 最大分离超平面：$$w^\ast\cdot x+b^\ast = 0$$，即：
    $$\sum_{i=1}^N\alpha_i^\ast y_i(x_i\cdot x)+b^\ast = 0$$

    5. 分类决策函数为$$f(x)=sign(w^\ast x+b^\ast )=sign(\sum_{i=1}^N\alpha_i^\ast y_i(x_i\cdot x)+b^\ast)$$
      说明分类决策函数只依赖于输入$$x$$和训练样本输入$$x_i$$的内积，此外由KKT条件可知，因为$$c_i(x^\ast)$$若为0，就表示该样本是在边界上，就表示是支持向量，所以支持向量的$$\alpha_i^\ast$$大于0，其他的$$\alpha_i^\ast$$等于0（设想为了满足KKT，只能这项为0，才能保证乘积为0），**不用计算内积**。

2. 若训练数据是线性不可分（不存在任何线性分类器可以将其正确分类），或者数据存在噪声。
 线性不可分意味着某些样本点不能满足函数间隔大于等于1的约束条件。引入松弛变量使得错分样本得到惩罚。 其他推导过程和线性可分差不多，暂时**记住remember下面公式**。其中C值称为**惩罚参数**，大于0，其值大时对误分类的惩罚增大，其值小时对误分类的惩罚减小，即表示使得$$\frac{1}{2}{\lVert w\Vert}^2$$尽量小即间隔尽量大，同时使误分类点的个数尽量小，$$C$$是调和两者的系数。

  $$y_i(w^{T}x_{i})-1+\xi_{i} \ge 0 , \xi_{i} \ge 0$$

  ![](/assets/soft margin maximization.png)

  $$\min_{w,b,\xi}\quad \frac{1}{2}{\lVert w\Vert}^2 + C\sum_{i}\xi_{i}\\
  s.t.\quad y_i(w\cdot x_{i})-1+\xi_{i} \ge 0,\quad i=1,2,...,N \\
  s.t. \quad  \xi_{i} \ge 0$$

  其等价于最优化问题：$$\sum_{i=1}^{N}[1-y_i(w\cdot x_i+b)]_+ + \lambda{\lVert w\Vert}^2$$

3. 非线性分类问题是指通过利用非线性模型才能很好地进行分类的问题。我们无法用直线（线性模型）将正负例正确分开，但可以用一条椭圆曲线（非线性模型）将他们正确分开。**核函数的作用**：首先使用变换将原空间的数据映射到新空间；然后在新空间用线性分类学习方法从雪莲数据中学习分类模型。
  ![](/assets/non linear.jpg)

  1. 假设存在一个从输入空间$$\chi$$到特征空间$$\mathcal{H}$$的映射$$\phi(x)$$，使得对所有$$x,z\in \mathcal{H}$$，函数$$K(x,z)$$满足条件：$$K(x,z)=\phi(x)\cdot\phi(x)$$，则称$$K(x,z)$$为核函数，$$\phi(x)$$为映射函数。
  2. 只需将线性支持向量机（即是线性不可分的情况，但是我这里公式写的蛮烦了就没有写出来，可以参照线性可分支持向量机的$$max$$加一个负号换成$$min$$即可，形式上是一样的）对偶形式中的内积换成核函数即可。**记住remember核函数的优化**。

    $$\begin{matrix}
      \min_{\alpha} & \frac{1}{2}\sum_{i=1} ^N\sum_{i=1} ^N\alpha_i\alpha_jy_iy_jK(x_i, x_j)-\sum_{i=1}^{N}\alpha_i\\
      s.t. &  \sum_{i=1}^N\alpha_iy_i=0 \\
       & 0 \leqslant\alpha_i\leqslant C,i=1,2,\cdots,N
      \end{matrix}$$

    得到最终的分类决策函数：
    $$f(x)=sign(\sum_{i=1}^N\alpha_i^\ast y_iK(x\cdot x_j)+b^\ast)$$

  3. 常用核函数

    1. 多项式核函数$$K(x,z)=(x*z+1)^p$$
    2. 高斯核函数$$K(x,z)= \exp(- \frac{\lVert x-z\Vert^2}{2\sigma^2})$$
    3. 字符串核函数：用于字符串处理中。

4. **拉格朗日对偶性**：在约束最优化问题中，我们经常使用拉格朗日对偶性(Lagrange Duality)将原始问题转换为其对偶问题，通过解对偶问题而得到原始问题的解。称此约束最优化问题为原始最优化问题或原始问题。
    
    1. 原始问题：假设$$f(x),c_i(x),h_j(x)$$是定义在$$R^n$$上的连续可微函数，则约束最优化问题表述如下

        $$\begin{equation} \min_{x\in R^n}{\;f(x)} \end{equation}$$
        $$\begin{align} s.t. \quad &c_i(x)\leq 0, \quad i=1,2,\cdots,k\ &h_j(x) = 0, \quad j=1,2,\cdots,l \end{align}$$

    2. 广义拉格朗日函数：

        $$L(x,\alpha,\beta) = f(x) + \sum_{i=1}^{k}{\alpha_i ci(x)} + \sum_{j=1}^{l}{\beta_j h_j(x)}$$

        这里，$$x = (x_1,x_2,\cdots,x_n)^T \in R^n，\alpha_i, \beta_i$$是拉格朗日乘子, $$\alpha_i \geq 0$$。

        从而我们考虑$$x$$的函数：$$\theta_p(x) = \max_{\alpha,\beta:\alpha_i \geq 0}{L(x,\alpha,\beta)}$$，这里的$$P$$表示原始问题。
        
        由约束条件可知，下面公式右侧的**广义拉格朗日函数的极小极大问题与原始问题是等价**的：

        $$\min_{x}{\theta_p(X)} = \min_{x}{\max_{\alpha,\beta:\alpha_i \geq 0}{L(x,\alpha,\beta)}}$$

        定义原始问题的最优值为：$$p^\ast = \min_{x}{\theta_p(x)}$$

    3. 对偶问题，下面公式右侧称为**广义拉格朗日函数的极大极小问题**

        $$\max_{\alpha,\beta:\alpha_i \geq 0}{\theta_D(\alpha,\beta)} = \max_{\alpha,\beta:\alpha_i \geq 0}{\min_{x}{L(x,\alpha,\beta)}}$$

        定义对偶问题的最优值为：$$d^\ast = \max_{\alpha,\beta:\alpha_i \geq 0}{\theta_D(\alpha,\beta)}$$

    4. 原始问题与对偶问题的关系：**若满足Karush-Kuhn-Tucker(KKT)条件，则原始问题和对偶问题的最优值相等**。

        $$\begin{gather}
        \nabla_x L(x^\ast, \alpha^\ast, \beta^\ast) = 0 \\

        \nabla_\alpha L(x^\ast, \alpha^\ast, \beta^\ast) = 0\\
        \nabla_\beta L(x^\ast, \alpha^\ast, \beta^\ast) = 0\\
        \alpha_i^\ast c_i(x^\ast) = 0, \quad i = 1,2,\cdots,k\\
        c_i(x^\ast) \leq 0, \quad i = 1,2,\cdots,k\\
        \alpha_i^\ast \geq 0, \quad i = 1,2,\cdots,k\\
        h_j(x^\ast) = 0, \quad j = 1,2,\cdots,l
        \end{gather}$$

# Code

```python
#Import Library
from sklearn import svm
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.svc() # there is various option associated with it, this is simple for classification. You can refer the link(http://scikit-learn.org/stable/modules/svm.html), for more detail.
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(x_test)
```

在sklearn中，svm.svc\(\)不需要设置参数，直接使用即可。**这个svc函数对应线性支持向量机，即第二种情况加入惩罚参数**；初始化时通过选项kernel指定用什么核。具体可参加sklearn的官方文档。因为很多网上教程都是照搬翻译官方文档的。

# Reference

* [支持向量机通俗导论（理解SVM的三层境界）](http://blog.csdn.net/v_july_v/article/details/7624837)
* [机器学习常见算法个人总结（面试用）](http://kubicode.me/2015/08/16/Machine%20Learning/Algorithm-Summary-for-Interview/)
* [支持向量机SVM](https://clyyuanzi.gitbooks.io/julymlnotes/content/svm.html)
* [C-SVM模型](https://json0071.gitbooks.io/svm/content/c-svm.html)
* [支持向量机-Scikit-learn 使用手册中文版](https://xacecask2.gitbooks.io/scikit-learn-user-guide-chinese-version/content/sec1.4.html)
* [sklearn中SVC的讲解](http://scikit-learn.org/stable/modules/svm.html#svm-mathematical-formulation)
* [拉格朗日对偶性](http://changxu.wang/2015/04/23/la-ge-lang-ri-dui-ou-xing/)
