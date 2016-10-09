# Representation

$$f(x_{i}) = w^{T}x_{i} + b_{i}$$

使得$$f(x_{i}) \simeq y_{i}$$

线性回归主要有两种类型：简单线性回归和多元线性回归。简单线性回归只有一个自变量。而多元线性回归有多个自变量。

# Evalution

回归学习最常用的损失函数（cost function）是平方损失函数，在此情况下，回归问题可以用著名的最小二乘法来解决。

注意：**广义的最小二乘准则，是一种对于偏差程度的评估准则，本质上是一种evaluation rule或者说objective funcion**，这里的「最小二乘法」应叫做「最小二乘法则」或者「最小二乘准则」，英文可呼为LSE\(**least square error**\)。可以理解为是通过平方损失函数建立模型优化目标函数的一种思路。

# Optimization

1. 通常我们所说的**狭义的最小二乘，指的是在线性回归下采用最小二乘准则（或者说叫做最小平方），进行线性拟合参数求解的、矩阵形式的公式方法**。所以，这里的「最小二乘法」应叫做「最小二乘算法」或者「最小二乘方法」，百度百科「最小二乘法」词条中对应的英文为**The least square method**。

    ** 所以狭义的最小二乘法是一种数学优化技术**。它通过最小化误差的平方和寻找数据的最佳函数匹配。

2. 由上述分析可知，广义的最小二乘法是通过平方损失函数建立模型优化目标函数的一种思路，此时求解最优模型过程便具体化为最优化目标函数的过程了。**在具体求解过程中最小二乘有两种情形：线性和非线性。线性最小二乘的解是closed-form即$$\theta=(X^TX)^{-1}X^T\overrightarrow y$$，即狭义的最小二乘法或正规方程。而非线性最小二乘没有closed-form，通常用迭代法求解。而梯度下降法便对应最优化目标函数的一种迭代优化算法\*\*，具体求解的是使得目标函数能达到最优或者近似最优的参数集。

3. 非线性最小二乘方法是以误差的平方和最小为准则来估计非线性静态模型参数的一种参数估计方法。设非线性系统的模型为$$y=f(x，θ)$$，式中y是系统的输出，x是输入，θ是参数（它们可以是向量）。这里的非线性是指对参数θ的非线性模型。

4. **正规方程**：_对于那些不可逆的矩阵（通常是因为特征之间不独立，如同时包含英尺为单位的尺寸和米
为单位的尺寸两个特征，也有可能是特征数量大于训练集的数量），正规方程方法是不能用的。_

    先简单将y表示为x的线性函数，则cost function为：
    
    $$J(\theta)=\sum_{i=1}^m(h_{\theta}(x^{(i)}-y^{(i)}))^2=\frac12(X\theta-\overrightarrow y)^T(X\theta-\overrightarrow y)$$

    令其对每个参数求导（J求偏导）并令导数为0，即：
    
    $$\nabla_{\theta}J(\theta)=0$$
    
    解之，$$\nabla_{\theta}J(\theta) = X^TX\theta-X^T\overrightarrow y=0$$

    得到正规方程，$$X^TX\theta=X^T\overrightarrow y$$

    求解参数，$$\theta=(X^TX)^{-1}X^T\overrightarrow y$$

5. **梯度下降**的解法：
    
6. 梯度下降与正规方程的比较。

| 梯度下降       | 正规方程           |
| 需要选择学习率α |不需要|
| 需要多次迭代      | 一次运算得出 |
| 当特征数量 n 大时也能较好适用      | 如果特征数量 n 较大则运算代价大，因为矩阵逆的计算时间复杂度为 O(n3)通常来说当 n 小于 10000 时还是可以接受的      |
| 适用于各种类型的模型 | 只适用于线性模型，不适合逻辑回归模型等其他模型      |

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

* [https:\/\/www.zhihu.com\/question\/20822481](https://www.zhihu.com/question/20822481) 

