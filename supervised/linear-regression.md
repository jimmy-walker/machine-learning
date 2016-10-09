# Representation

$$f(x_{i}) = w^{T}x_{i} + b_{i}$$，使得$$f(x_{i}) \simeq y_{i}$$

线性回归主要有两种类型：简单线性回归和多元线性回归。简单线性回归只有一个自变量。而多元线性回归有多个自变量。

# Evalution

回归学习最常用的损失函数（cost function）是平方损失函数，在此情况下，回归问题可以用著名的最小二乘法来解决。

注意：**广义的最小二乘准则，是一种对于偏差程度的评估准则，本质上是一种evaluation rule或者说objective funcion**，这里的「最小二乘法」应叫做「最小二乘法则」或者「最小二乘准则」，英文可呼为LSE(**least square error**)。可以理解为是通过平方损失函数建立模型优化目标函数的一种思路。

# Optimization

1. 通常我们所说的**狭义的最小二乘，指的是在线性回归下采用最小二乘准则（或者说叫做最小平方），进行线性拟合参数求解的、矩阵形式的公式方法**。所以，这里的「最小二乘法」应叫做「最小二乘算法」或者「最小二乘方法」，百度百科「最小二乘法」词条中对应的英文为**The least square method**。

  ** 所以狭义的最小二乘法是一种数学优化技术**。它通过最小化误差的平方和寻找数据的最佳函数匹配。

2. 最小二乘法的目标：求误差的最小平方和，对应有两种：线性和非线性。线性最小二乘的解是closed-form即$$x=(A^T A)^{-1}A^Tb$$，而非线性最小二乘没有closed-form，通常用迭代法求解。

3. 由上述分析可知，广义的最小二乘法是通过平方损失函数建立模型优化目标函数的一种思路，此时求解最优模型过程便具体化为最优化目标函数的过程了。**在具体求解过程中最小二乘有两种情形：线性和非线性。线性最小二乘的解是closed-form即$$x=(A^T A)^{-1}A^Tb$$，而非线性最小二乘没有closed-form，通常用迭代法求解。而梯度下降法便对应最优化目标函数的一种迭代优化算法**，具体求解的是使得目标函数能达到最优或者近似最优的参数集。

4. 非线性最小二乘方法是以误差的平方和最小为准则来估计非线性静态模型参数的一种参数估计方法。设非线性系统的模型为y=f\(x，θ\)，式中y是系统的输出，x是输入，θ是参数（它们可以是向量）。这里的非线性是指对参数θ的非线性模型。

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

#Reference
- https://www.zhihu.com/question/20822481 

