# Representation

1. GBDT这个算法有很多名字，但都是同一个算法：

  * GBRT \(Gradient BoostRegression Tree\) 渐进梯度回归树
  * GBDT \(Gradient BoostDecision Tree\) 渐进梯度决策树
  * MART \(MultipleAdditive Regression Tree\) 多决策回归树
  * TN \(Tree Net\) 决策树网络

2. Gradient Boosting是一种Boosting的方法，它主要的思想是，**每一次建立模型是在之前建立模型损失函数的梯度下降方向（负梯度方向）**。它的每一次计算都是为了减少上一次的residual，而为了减少这些residual，它会在residual减少的gradient方向上建立一个新的model。**新model建立的目的是为了使先前模型得残差往梯度方向减少，与传统的boosting算法对正错样本赋予不同加权的做法有着极大的区别**。算法描述如下，**记住算法思想**：

    - Input:

        * 训练数据集：$$T=\{(x_1,y_1),(x_2,y_2)…(x_N,y_N)\}$$
        * 可导的损失函数：$$L(y,f(x))$$
        * 迭代的次数：$$M$$

    - Output:

        * 最终模型：$$f(x)$$

    - Procedure:

        1. 使用一个常量进行模型的初始化：$$f_0(x) = \underset{\gamma}{argmin} \sum_{i=1}^n L(y_i,\gamma)$$

        2. 循环：$$m \in \{1….M\}$$

            1. 计算残差：$$r_{im}=-\left[ \frac{\partial L(y_i,f(x_i))}{\partial f(x_i)}  \right]_{f(x_i)=f_{m-1}(x_i)} \quad i=1…n$$

            2. 使用训练集$$\{(x_i,r_{im})\}$$对弱分类器$$G_m(x)$$进行拟合

            3. 通过线性搜索进行乘子$$\gamma_m$$的计算：$$\gamma_m = \underset{\gamma}{argmin} \sum_{i=1}^n L \left(y_i,f_{m-1}(x_i)+\gamma_m G_m(x_i) \right)$$

            4. 进行模型的更新：$$f_m(x) = f_{m-1}(x)+\gamma_m G_m(x)$$

        3. 输出最终的模型：$$f_M(x)$$

3. Gradient Boosting算法解释

    - 第(1)步初始化，估计使损失函数极小化的常数值（是一个只有根结点的树），其中**argmin表示取是目标函数最小值的参数值**；

    - 第(2)(a)步计算损失函数的负梯度在当前模型的值，将它作为残差的估计。(对于平方损失函数，他就是残差；对于一般损失函数，它就是残差的近似值)，其中**残差近似等于当前模型中损失函数的负梯度值，由于我们在计算第k棵树，所以使用第k-1棵树的函数**；
    - 第(2)(b)步估计回归树的结点区域，以拟合残差的近似值；
    - 第(2)(c)步利用线性搜索估计叶结点区域的值，使损失函数极小化；
    - 第(2)(d)步更新回归树。


# Algorithm: the same as random forest.

1. 决策树分为回归树和分类树：
  回归树用于预测实数值，如明天温度、用户年龄
  ；分类树用于分类标签值，如晴天\/阴天\/雾\/雨、用户性别
  。注意前者结果加减是有意义的，如10岁+5岁-3岁=12岁，后者结果加减无意义，如男+女=到底是男还是女？GBDT的核心在于累加所有树的结果作为最终结果，而**分类树是没有办法累加的，所以GBDT中的树都是回归树而非分类树**。

2. 残差：第一棵树是正常的，之后所有的树的决策全是由残差（此次的值与上次的值之差）来作决策。


# Code

```python
#Import Library
from sklearn.ensemble import GradientBoostingClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Gradient Boosting Classifier object
model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)
```

# Reference

* [【机器学习】迭代决策树GBRT（渐进梯度回归树）](http://blog.csdn.net/dianacody/article/details/40688783)
* [Boosted Tree](http://www.52cs.org/?p=429)
* [第06章：深入浅出ML之Boosting家族](http://www.52caml.com/head_first_ml/ml-chapter6-boosting-family/)
* [从Gradient Boosting 到GBT](http://kubicode.me/2016/04/24/Machine%20Learning/From-Gradient-Boosting-to-GBT/)
