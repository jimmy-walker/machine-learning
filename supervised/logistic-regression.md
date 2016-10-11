# Representation
logistic函数定义为：$$g(z)=\frac{1}{1+e^{-z}}$$

logistic函数的图形为：
![](/assets/logistic-function.png)

可以看到，当自变量取值0时，函数值为0.5，自变量趋向于正无穷和负无穷时函数值趋近于1和0。因此我们用下式表示对一个样本$$x$$的预测：

$$h_\theta(x)=g({\theta^T}{x})=\frac{1}{1+e^{-{\theta^T}{x}}}$$

不要被这个名字给混淆，**这是一种分类而不是回归算法**。简单来讲，它通过把给定的数据输入进一个评定模型方程来预测一个事件发生的可能性。因此它又被称为逻辑回归模型。 因为它是预测可能性的，它的输出值介于0和1之间。
# Evalution

# Optimization

# Code
```python
#Import Library
from sklearn.linear_model import LogisticRegression
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create logistic regression object
model = LogisticRegression()
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)
#Predict Output
predicted= model.predict(x_test)
```
# Reference
- [机器学习之逻辑回归](http://zhikaizhang.cn/2016/06/10/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B9%8B%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92/?utm_source=tuicool&utm_medium=referral)
