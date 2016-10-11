# Representation
logistic函数定义为：$$g(z)=\frac{1}{1+e^{-z}}$$

logistic函数的图形为：
![](/assets/logistic-function.png)

可以看到，当自变量取值0时，函数值为0.5，自变量趋向于正无穷和负无穷时函数值趋近于1和0。因此我们用下式表示对一个样本$$x$$的预测，这也就是逻辑回归的模型，**记住此模型公式**：

$$h_\theta(x)=g({\theta^T}{x})=\frac{1}{1+e^{-{\theta^T}{x}}}$$

不要被这个名字给混淆，**这是一种分类而不是回归算法**。简单来讲，它通过把给定的数据输入进一个评定模型方程来预测一个事件发生的可能性。因此它又被称为逻辑回归模型。 因为它是预测可能性的，它的输出值介于0和1之间。

# Evalution
对于线性回归模型，我们定义的代价函数是所有模型误差的平方和。理论上来说，我们也可以对逻辑回归模型沿用这个定义，但是问题在于，当我们将模型带入到这样定义了的代价函数中时，我们得到的代价函数将是一个非凸函数(non-convex function)。
![](/assets/non-convex function.PNG)
这意味着我们的代价函数有许多局部最小值，这将影响梯度下降算法寻找全局最小值。因此我们重新定义参数的似然性来作为目标函数，然后学习得到使得这个似然性不断增大最终收敛时的参数$$\theta$$。注意：在数理统计学中，似然函数是一种关于统计模型中的参数的函数，表示模型参数中的似然性。而似然性可以理解为概率的意思。

$$L(\theta)=p(Y\mid{X};\theta)=\prod_{i=1}^{m}p(y^{(i)}{\mid}x^{(i)};\theta)=\prod_{i=1}^{m}{(h_\theta(x^{(i)}))^{y(i)}}{(1-h_\theta(x^{(i)}))^{1-y^{(i)}}}$$

在上述式子的推到过程中，用到了以下已有结论：

逻辑回归做出了如下假设：

$$P(y=1{\mid}{x};{\theta})=h_\theta(x)$$

$$P(y=0{\mid}{x};{\theta})=1-h_\theta(x)$$
将logistic函数得到的函数值认为是类别为1的概率，1减去这个值就是类别为0的概率。这两个概率可以写成一个式子来表示：

$$P(y{\mid}{x};{\theta})={(h_\theta(x))}^{y}{(1-h_\theta(x))}^{1-y}$$

经过上述分析，通过对数进一步将代价函数化简为，**记住此代价函数**：
$$l(\theta)=log(L(\theta))=\sum_{i=1}^{m}y^{(i)}logh_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)}))$$

# Optimization
为了最大化这个对数似然函数，显然可以使用梯度下降法，只不过这一次是梯度上升。


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
