# Representation

logistic函数定义为：$$g(z)=\frac{1}{1+e^{-z}}$$

logistic函数的图形为：
![](/assets/logistic-function.png)

可以看到，当自变量取值0时，函数值为0.5，自变量趋向于正无穷和负无穷时函数值趋近于1和0。因此我们用下式表示对一个样本$$x$$的预测，这也就是逻辑回归的模型，**记住此模型公式**：

$$h_\theta(x)=g({\theta^T}{x})=\frac{1}{1+e^{-{\theta^T}{x}}}$$

不要被这个名字给混淆，**这是一种分类而不是回归算法**。简单来讲，它通过把给定的数据输入进一个评定模型方程来预测一个事件发生的可能性。因此它又被称为逻辑回归模型。 因为它是预测可能性的，它的输出值介于0和1之间。

# Evalution

1. 对于线性回归模型，我们定义的代价函数是所有模型误差的平方和。理论上来说，我们也可以
  对逻辑回归模型沿用这个定义，但是问题在于，当我们将模型带入到这样定义了的代价函数中时，我们得到的代价函数将是一个非凸函数\(non-convex function\)。

  ![](/assets/non-convex function.PNG)

  这意味着我们的代价函数有许多局部最小值，这将影响梯度下降算法寻找全局最小值。
   此外，一个理解角度是为什么Logistic Regression不使用最小二乘做cost function呢？**答案是各自的响应变量服从不同的概率分布。在Linear Regression中，前提假设是服从正态分布，而Logistic中的是服从二项分布的。**(为什么不服从正态？因为非0即1啊！)

  因此我们重新定义参数的似然性来作为目标函数，然后学习得到使得这个似然性不断增大最终收敛时的参数$$\theta$$。注意：统计学中常用的一种方法是最大似然估计（极大似然估计），即找到一组参数，使得在这组参数下，我们的数据的似然度（概率）越大。即似然性可以理解为概率的意思。

  $$L(\theta)=p(Y\mid{X};\theta)=\prod_{i=1}^{m}p(y^{(i)}{\mid}x^{(i)};\theta)=\prod_{i=1}^{m}{(h_\theta(x^{(i)}))^{y(i)}}{(1-h_\theta(x^{(i)}))^{1-y^{(i)}}}$$

  在上述式子的推到过程中，用到了以下已有结论：

  逻辑回归做出了如下假设：

  $$P(y=1{\mid}{x};{\theta})=h_\theta(x)$$

  $$P(y=0{\mid}{x};{\theta})=1-h_\theta(x)$$

  将logistic函数得到的函数值认为是类别为1的概率，1减去这个值就是类别为0的概率。这两个概率可以写成一个式子来表示：

  $$P(y{\mid}{x};{\theta})={(h_\theta(x))}^{y}{(1-h_\theta(x))}^{1-y}$$

  经过上述分析，通过对数进一步将代价函数化简为，**记住此代价函数**：

  $$l(\theta)=log(L(\theta))=\sum_{i=1}^{m}y^{(i)}logh_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)}))$$

2. **对偶问题**：相对于求解对数似然函数的最大值，我们当然可以将该目标转换为对偶问题，即求解代价函数$$J(\theta)=-log(\ell(\theta))$$的最小值。因此，我们定义logistic回归的代价函数为：

    $$cost=J(\theta)=-log(\ell(\theta))＝－\frac{1}{m} \sum^m_{i=1} \lgroup y^{(i)} log(g(x)) + (1-y^{(i)})log(1-g(x)) \rgroup$$

3. **极大似然估计（最大似然估计）**：最大似然估计就是估计未知参数的一种方法，最大似然估计（Maximum Likelihood Method）是建立在各样本间相互独立且样本满足随机抽样（可代表总体分布）下的估计方法，它的核心思想是如果现有样本可以代表总体，那么**最大似然估计就是找到一组参数使得出现现有样本的可能性最大**，即从**统计学角度需要使得所有观测样本的联合概率最大化，又因为样本间是相互独立的，所以所有观测样本的联合概率可以写成各样本出现概率的连乘积**，即：


# Optimization

**为了最大化这个对数似然函数，显然可以使用梯度下降法，只不过这一次是梯度上升。注意这是没有对偶前的优化方法。**对对数似然求参数导数，得到:

$$\frac{\partial}{\partial{\theta_j}}l(\theta)=\sum_{i=1}^{m}(y^{(i)}-h_\theta(x^{(i)}))x^{(i)}_j$$

从而可以得到批梯度下降的参数更新公式，**记住此优化方法**：

$$\theta_j=\theta_j+\alpha\sum_{i=1}^{m}(y^{(i)}-h_\theta(x^{(i)}))x^{(i)}_j=\theta_j+\alpha\sum_{i=1}^{m}(y^{(i)}-\frac{1}{1+e^{-{\theta^T}{x^{(i)}}}})x^{(i)}_j$$

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

* [机器学习之逻辑回归](http://zhikaizhang.cn/2016/06/10/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B9%8B%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92/?utm_source=tuicool&utm_medium=referral)

