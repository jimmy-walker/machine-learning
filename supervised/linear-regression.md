# Representation

先简单将$$y$$表示为$$x$$的线性函数：

$$h(x) = \sum_{i=0}^{n}\theta _ix_i=\theta^Tx$$

* 其中$$\theta$$称为参数parameters，也叫做权重weights，参数决定了$$X$$到$$Y$$的射映空间。
* 而用$$x_0=1$$来表示截距项interceptterm。

线性回归主要有两种类型：简单线性回归和多元线性回归。简单线性回归只有一个自变量。而多元线性回归有多个自变量。

# Evalution

回归学习最常用的损失函数（loss function）是平方损失函数，在此情况下，回归问题可以用著名的最小二乘法来解决。
**记住此代价函数**（损失函数(loss function)或代价函数(cost function)往往是相同的）：
$$J(\theta)=\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2$$

注意：**广义的最小二乘准则，是一种对于偏差程度的评估准则，本质上是一种evaluation rule或者说objective funcion**，这里的「最小二乘法」应叫做「最小二乘法则」或者「最小二乘准则」，英文可呼为LSE\(**least square error**\)。可以理解为是通过平方损失函数建立模型优化目标函数的一种思路。

# Optimization

1. 通常我们所说的**狭义的最小二乘，指的是在线性回归下采用最小二乘准则（或者说叫做最小平方），进行线性拟合参数求解的、矩阵形式的公式方法**。所以，这里的「最小二乘法」应叫做「最小二乘算法」或者「最小二乘方法」，百度百科「最小二乘法」词条中对应的英文为**The least square method**。

  **所以狭义的最小二乘法是一种数学优化技术**。它通过最小化误差的平方和寻找数据的最佳函数匹配。

2. 由上述分析可知，广义的最小二乘法是通过平方损失函数建立模型优化目标函数的一种思路，此时求解最优模型过程便具体化为最优化目标函数的过程了。**在具体求解过程中最小二乘有两种情形：线性和非线性。**线性最小二乘的解是closed-form即 $$\theta=(X^TX)^{-1}X^T\overrightarrow y$$，即狭义的最小二乘法或正规方程。而非线性最小二乘没有closed-form，通常用迭代法求解。而梯度下降法便对应最优化目标函数的一种迭代优化算法，具体求解的是使得目标函数能达到最优或者近似最优的参数集。

3. 非线性最小二乘方法是以误差的平方和最小为准则来估计非线性静态模型参数的一种参数估计方法。设非线性系统的模型为$$y=f(x，\theta)$$，式中$$y$$是系统的输出，$$x$$是输入，$$\theta$$是参数（它们可以是向量）。这里的非线性是指对参数$$\theta$$的非线性模型。

4. **正规方程**：对于那些不可逆的矩阵（通常是因为特征之间不独立，如同时包含英尺为单位的尺寸和米
  为单位的尺寸两个特征，也有可能是特征数量大于训练集的数量），正规方程方法是不能用的。
  根据之前的定义将$$y$$表示为$$x$$的线性函数$$h(x) = \sum_{i=0}^{n}\theta _ix_i=\theta^Tx$$，则cost function为：

  $$J(\theta)=\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2=\frac12(X\theta-\overrightarrow y)^T(X\theta-\overrightarrow y)$$

  令其对每个参数求导（J求偏导）并令导数为0，即：

  $$\nabla_{\theta}J(\theta)=0$$

  解之，$$\nabla_{\theta}J(\theta) = X^TX\theta-X^T\overrightarrow y=0$$
  得到正规方程，$$X^TX\theta=X^T\overrightarrow y$$
  求解参数，**可以忽略上述证明只记住此公式**，$$\theta=(X^TX)^{-1}X^T\overrightarrow y$$

5. **梯度下降**：如果打算采用多项式回归模型的话，在进行梯度下降算法前，很有必要先进性特征缩放，使其规约到一个相近的范围内，否则梯度下降算法会迭代非常多次才能收敛。解决方法是尝试所有特征的尺度都尽量缩放到-1到1之间，公式为：$$x_n = \frac{x_n - \mu_n}{s_n}$$，其中的数值分别是平均值和标准差。

  ![](/assets/normalization.PNG)

  梯度指向增长最快的方向。在单变量实值中，梯度等于导数。

  由于负梯度方向是使函数值下降最快的方向，在迭代的每一步，以负梯度方向更新$$x$$的值，从而达到减少函数值的目的。
  给定一个$$\theta$$的一初值，然后不断改进，每次改进都使$$J(\theta)$$\)更小，直到最小化$$J(\theta)$$的$$\theta$$的值收敛。

  从初始$$\theta$$开始，不断更新，**此公式就是梯度下降核心算法，具体可见《统计学习》的推导**：$$\theta_j:=\theta_j-\alpha \frac{\delta}{\delta\theta_j}J(\theta)$$

  注意，更新是**同时**对所有$$j=0,…,n$$的$$\theta_j$$值进行，即对所有的参数。$$\alpha$$被称作学习率\(learning rate\)，也是梯度下降的长度，若$$\alpha$$取值较小，则收敛的时间较长；相反，若$$\alpha$$取值较大，则可能错过最优值。**常用的学习率数值：0.01,0.03,0.1,0.3,1,3,10.**
##code
Normalization在数据跨度不一的情况下对机器学习有很重要的作用。特别是各种数据属性还会互相影响的情况之下。**Scikit-learn中标准化的语句是preprocessing.scale()，其实也可以自己进行归一化**。scale以后，model就更能从标准化数据中学到东西。
    
    1. 引入相关包from sklearn import preprocessing

    ```python
    X_scaled=preprocessing.scale(X)#normalization step
    ```
    2. 处理后数据的均值和方差

    ```python
    X_scaled.mean(axis=0)
    array([ 0.,  0.,  0.])
    X_scaled.std(axis=0)
    array([ 1.,  1.,  1.])
    ```

6. **归一化Normalization和标准化Standardization**：**这两者都是缩放，但是目的不同而区别。**
    
    1. 归一化：**将数据映射到(0,1)范围之内（也可以放在其他范围内），目的是为了消除不同数据之间的量纲，方便数据比较和共同处理**，比如在梯度下降中，归一化可以加快收敛。
    2. 标准化：数据的标准化是将数据按比例缩放，**使之落入一个小的特定区间，目的是为了方便数据的下一步处理，而进行的数据缩放等变换，并不是为了方便与其他数据一同处理或比较**，比如数据经过零-均值标准化后，更利于使用标准正态分布的性质，进行处理。

7. 梯度下降与正规方程的比较。


| 梯度下降 | 正规方程 |
| --- | --- |
| 需要选择学习率α | 不需要 |
| 需要多次迭代 | 一次运算得出 |
| 当特征数量$$n$$大时也能较好适用 | 如果特征数量$$n$$较大则运算代价大，因为矩阵逆的计算时间复杂度为$$O(n^3)$$通常来说当$$n$$小于10000时还是可以接受的 |
| 适用于各种类型的模型 | 只适用于线性模型，不适合逻辑回归模型等其他模型 |

# Code

```python
#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import linear_model
#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays
x_train=input_variables_values_training_datasets
y_train=target_variables_values_training_datasets
x_test=input_variables_values_test_datasets
# Create linear regression object
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)
#Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
#Predict Output
predicted= linear.predict(x_test)
```

# Reference

- [知乎：最小二乘法和梯度下降法有哪些区别？](https://www.zhihu.com/question/20822481)
- [机器学习笔记1 有监督学习 线性回归 LMS算法 正规方程](http://nanshu.wang/post/2015-02-10)
- [归一化、标准化和正则化的关系](http://blog.csdn.net/zyf89531/article/details/45922151)

