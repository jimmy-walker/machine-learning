# Representation

logistic函数定义为：$$g(z)=\frac{1}{1+e^{-z}}$$

logistic函数的图形为：

![](/assets/logistic-function.png)

可以看到，当自变量取值0时，函数值为0.5，自变量趋向于正无穷和负无穷时函数值趋近于1和0。因此我们用下式表示对一个样本$$x$$的预测，这也就是逻辑回归的模型，**记住此模型公式**：

$$h_\theta(x)=g({\theta^T}{x})=\frac{1}{1+e^{-{\theta^T}{x}}}$$

$$P(y=1{\mid}{x};{\theta})=h_\theta(x)$$

$$P(y=0{\mid}{x};{\theta})=1-h_\theta(x)$$

将logistic函数得到的函数值认为是类别为1的概率，1减去这个值就是类别为0的概率。这两个概率可以写成一个式子来表示：

$$P(y{\mid}{x};{\theta})={(h_\theta(x))}^{y}{(1-h_\theta(x))}^{1-y}$$

不要被这个名字给混淆，**这是一种分类而不是回归算法**。简单来讲，它通过把给定的数据输入进一个评定模型方程来预测一个事件发生的可能性。因此它又被称为逻辑回归模型。 因为它是预测可能性的，它的输出值介于0和1之间。

# Evalution

1. 对于线性回归模型，我们定义的代价函数是所有模型误差的平方和。理论上来说，我们也可以
  对逻辑回归模型沿用这个定义，但是问题在于，当我们将模型带入到这样定义了的代价函数中时，我们得到的代价函数将是一个非凸函数\(non-convex function\)。

  ![](/assets/non-convex function.PNG)

  这意味着我们的代价函数有许多局部最小值，这将影响梯度下降算法寻找全局最小值。
   此外，一个理解角度是为什么Logistic Regression不使用最小二乘做cost function呢？**答案是各自的响应变量服从不同的概率分布。在Linear Regression中，前提假设是服从正态分布，而Logistic中的是服从二项分布的。**(为什么不服从正态？因为非0即1啊！)

  因此我们重新定义参数的似然性来作为目标函数，然后学习得到使得这个似然性不断增大最终收敛时的参数$$\theta$$。注意：统计学中常用的一种方法是最大似然估计（极大似然估计），即找到一组参数，使得在这组参数下，我们的数据的似然度（概率）越大。即似然性可以理解为概率的意思。

  $$L(\theta)=p(Y\mid{X};\theta)=\prod_{i=1}^{m}p(y^{(i)}{\mid}x^{(i)};\theta)=\prod_{i=1}^{m}{(h_\theta(x^{(i)}))^{y(i)}}{(1-h_\theta(x^{(i)}))^{1-y^{(i)}}}$$

  经过上述分析，通过对数进一步将代价函数化简为，**记住此代价函数**：

  $$l(\theta)=log(L(\theta))=\sum_{i=1}^{m}y^{(i)}logh_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)}))$$

2. **对偶问题**：相对于求解对数似然函数的最大值，我们当然可以将该目标转换为对偶问题，即求解代价函数$$J(\theta)=-log(\ell(\theta))$$的最小值。因此，我们定义logistic回归的代价函数为，注意此处的m分之1是用于抵消最大似然估计求极值时常会求导得到的值（但如果用梯度下降来优化则不用）：

    $$cost=J(\theta)=-log(\ell(\theta))＝－\frac{1}{m} \sum^m_{i=1} \lgroup y^{(i)} log(g(x)) + (1-y^{(i)})log(1-g(x)) \rgroup$$
    
    从代价函数的直观表达上来看，当$$y^{(i)}=1, g(x)=1$$时(预测类别和真实类别相同)，$$J(\theta｜x^{(i)})=0$$；当$$y^{(i)}=1, g(x) \rightarrow 0$$时(预测类别和真实类别相反)，$$J(\theta｜x^{(i)}) \rightarrow \infty$$（注意对数函数前有个负号）。这意味着，当预测结果和真实结果越接近时，预测产生的代价越小，当预测结果和真实结果完全相反时，预测会产生很大的惩罚。该理论同样适用于$$y^{(i)}=0$$的情况。

3. **极大似然估计（最大似然估计）**：最大似然估计就是估计未知参数的一种方法，最大似然估计（Maximum Likelihood Method）是建立在各样本间相互独立且样本满足随机抽样（可代表总体分布）下的估计方法，它的核心思想是如果现有样本可以代表总体，那么**最大似然估计就是找到一组参数使得出现现有样本的可能性最大**，即从**统计学角度需要使得所有观测样本的联合概率最大化，又因为样本间是相互独立的，所以所有观测样本的联合概率可以写成各样本出现概率的连乘积**。
    
    **所以我们使用极大似然估计的思想，得出代价函数**。
    
    此外，**最小二乘估计是最大似然估计的一种**，有心的人还记得上面提到过的线性回归必须满足的条件，即误差项均服从正态分布的假设，如果线性回归记为$$y=\theta x + \epsilon$$的话，对于误差函数$$\epsilon$$，其服从正态分布$$\epsilon \sim N(0, \sigma^2)$$，因此利用正态分布的性质，我们可以得到$$y-\theta x \sim N(0, \sigma^2) \Rightarrow y \sim N(\theta x, \sigma^2)$$。
    因此，根据极大似然估计的定义，我们要获得产生样本$$y$$可能性最大的一组参数$$\theta$$，因此，似然函数可写为：
    
    $$\ell(\theta)=\prod^m_{i=1} \frac{1}{\sqrt{2\pi}\sigma} exp(- \frac{(y^{(i)}-\theta x)^2}{2 \sigma})$$
    
    与logistic回归类似，我们仍然将似然函数变换为对数似然函数求解极值，此时，

    $$log(\ell(\theta))=mlog(\frac{1}{\sqrt{2\pi}}) + \sum^m_{i=1} -\frac{(y^{(i)}-\theta x)^2}{2 \sigma}$$

    综上所述，要让$$log(\ell(\theta))$$最大，我们需要让$$\sum^m_{i=1}(y^{(i)}-\theta x)^2$$最小(因为另外一项是固定值)，**该式即为我们经常提及的线性回归的代价函数**，所以，线性回归的求解过程也利用最大似然估计的思想。

4. **正则化(Regularization)**：**正则化不是只有逻辑回归存在，它是一个通用的算法和思想，所以会产生过拟合现象的算法都可以使用正则化来避免过拟合**。
    
    过拟合现象是指对训练数据预测很好但是对未知数据预测不行的现象，通常都是因为模型过于复杂，或者训练数据太少。模型复杂体现在两个方面，一是参数过多，二是参数值过大。参数值过大会导致导数非常大，那么拟合的函数波动就会非常大，即下图所示，从左到右分别是欠拟合、拟合和过拟合。
    
    ![](/assets/outfit.jpg)
    
    从而，**解决过拟合可以从两个方面入手，一是减少模型复杂度，一是增加训练集个数**。而正则化就是减少模型复杂度的一个方法。
    
    由于模型的参数个数一般是由人为指定和调节的，所以正则化常常是用来限制模型参数值不要过大，也被称为惩罚项。一般是在目标函数(经验风险)中加上一个正则化项。
    

# Optimization

**为了最大化这个对数似然函数，显然可以使用梯度下降法，只不过这一次是梯度上升。注意这是没有对偶前的优化方法，对偶的话可以直接梯度下降。**对对数似然求参数导数，得到:

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

在sklearn中，函数原型如下，其中penalty表示惩罚项选用何种范数，C表示Inverse of regularization strength即正则化参数的导数，tol表示Tolerance for stopping criteria即算法停止的条件。


```python
LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
```

# Reference

- [机器学习之逻辑回归](http://zhikaizhang.cn/2016/06/10/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B9%8B%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92/?utm_source=tuicool&utm_medium=referral)
- [机器学习－逻辑回归与最大似然估计](http://www.hanlongfei.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/2015/08/05/mle)
- [正则化、归一化含义解析](http://sobuhu.com/ml/2012/12/29/normalization-regularization.html)
- [【机器学习算法系列之二】浅析Logistic Regression](http://chenrudan.github.io/blog/2016/01/09/logisticregression.html#4)