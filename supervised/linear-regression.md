#Representation
$$f(x_{i}) = w^{T}x_{i} + b_{i}$$，使得$$f(x_{i}) \simeq y_{i}$$

线性回归主要有两种类型：简单线性回归和多元线性回归。简单线性回归只有一个自变量。而多元线性回归有多个自变量。

#Evalution

#Optimization
最小二乘法（又称最小平方法）是一种数学优化技术。它通过最小化误差的平方和寻找数据的最佳函数匹配。


python代码
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
http://www.cnblogs.com/lvlvlvlvlv/p/5578805.html
