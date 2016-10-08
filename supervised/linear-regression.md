# Representation

$$f(x_{i}) = w^{T}x_{i} + b_{i}$$，使得$$f(x_{i}) \simeq y_{i}$$

线性回归主要有两种类型：简单线性回归和多元线性回归。简单线性回归只有一个自变量。而多元线性回归有多个自变量。

# Evalution

回归学习最常用的损失函数（cost function）是平方损失函数，在此情况下，回归问题可以用著名的最小二乘法来解决。

# Optimization

1. 最小二乘法（又称最小平方法）是一种数学**优化**技术。它通过最小化误差的平方和寻找数据的最佳函数匹配。
  J所以最小二乘法是一种优化方法。最小二乘法的目标：求误差的最小平方和，对应有两种：线性和非线性。线性最小二乘的解是closed-form即$$x=(A^T A)^{-1}A^Tb$$，而非线性最小二乘没有closed-form，通常用迭代法求解。

2. 把最小二乘看做是优化问题的话，那么梯度下降是求解方法的一种。最小二乘法是直接对$$\Delta$$求导找出全局最小，是非迭代法。而梯度下降法是一种迭代法，先给定一个$$\beta$$ ，然后向$$\Delta$$下降最快的方向调整$$\beta$$ ，在若干次迭代之后找到局部最小。梯度下降法的缺点是到最小点的时候收敛速度变慢，并且对初始点的选择极为敏感，其改进大多是在这两方面下功夫。

以误差的平方和最小为准则来估计非线性静态模型参数的一种参数估计方法。设非线性系统的模型为y=f(x，θ)，常用于传感器参数设定。式中y是系统的输出，x是输入，θ是参数（它们可以是向量）。这里的非线性是指对参数θ的非线性模型。

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

[http:\/\/www.cnblogs.com\/lvlvlvlvlv\/p\/5578805.html](http://www.cnblogs.com/lvlvlvlvlv/p/5578805.html)
[https:\/\/www.zhihu.com\/question\/20822481](https://www.zhihu.com/question/20822481)
http://blog.csdn.net/wsj998689aa/article/details/41558945
http://lanbing510.info/2016/03/28/Least-Squares-Parameter-Estimation.html
