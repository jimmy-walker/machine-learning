# Representation

1. GBDT这个算法有很多名字，但都是同一个算法：

  * GBRT \(Gradient BoostRegression Tree\) 渐进梯度回归树
  * GBDT \(Gradient BoostDecision Tree\) 渐进梯度决策树
  * MART \(MultipleAdditive Regression Tree\) 多决策回归树
  * TN \(Tree Net\) 决策树网络

2. Gradient Boosting是一种Boosting的方法，它主要的思想是，**每一次建立模型是在之前建立模型损失函数的梯度下降方向（负梯度方向）**。它的每一次计算都是为了减少上一次的residual，而为了减少这些residual，它会在residual减少的gradient方向上建立一个新的model。**新model建立的目的是为了使先前模型得残差往梯度方向减少，与传统的boosting算法对正错样本赋予不同加权的做法有着极大的区别**。算法描述如下，**J具体只记住算法思想，下方公式等有用再记**：


$$\{ \\   \quad\, 输入：训练数据集D=\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \cdots, (x^{(M)}, y^{(M)})\}, x^{(i)} \in \mathcal{X} \subseteq R^n, y^{(i)} \in \mathcal{Y}; \\   \qquad\quad\; 损失函数L(y, f(x)); \\   \quad 输出：提升树\hat{f}(x). \\   \quad 过程: \\   \qquad (1). 初始化模型 \\   \qquad\qquad\qquad f_0(x) = \arg \min_c \sum_{i=1}^{M} L(y^{(i)}, c)； \\   \qquad\; (2). 循环训练K个模型 k=1,2,\cdots,K \\   \qquad\qquad (a). 计算残差：对于i=1,2,\cdots,M \\   \qquad\qquad\qquad\qquad r_{ki} = -\left[ \frac{\partial L(y^{(i)}, \; f(x^{(i)}))} {\partial f(x^{(i)})} \right]_{f(x) = f_{k-1}(x)} \\   \qquad\qquad (b). 拟合残差r_{ki}学习一个回归树，得到第k颗树的叶结点区域R_{kj}，\quad j=1,2,\cdots,J \\   \qquad\qquad (c). 对j=1,2,\cdots,J, 计算：\\   \qquad\qquad\qquad\qquad c_{kj} = \arg \min_c \sum_{x^{(i)} \in R_{kj}} L(y^{(i)}, \; f_{k-1}(x^{(i)}) + c)\\   \qquad\qquad (d). 更新模型：\\   \qquad\qquad\qquad\qquad    f_k(x) = f_{k-1}(x) + \sum_{j=1}^{J} c_{kj} I(x \in R_{kj}) \\   \qquad\; (3). 得到回归提升树 \\   \qquad\qquad\qquad \hat{f}(x) = f_K(x) = \sum_{k=1}^{K} \sum_{j=1}^{J} c_{kj} I(x \in R_{kj}) \\    \}$$

- 第(1)步初始化，估计使损失函数极小化的常数值（是一个只有根结点的树）；
- 第(2)(a)步计算损失函数的负梯度在当前模型的值，将它作为残差的估计（**因为我们将残差近似等于当前模型中损失函数的负梯度值，因为要求第k个，所以使用第k-1的模型来求**）。(对于平方损失函数，他就是残差；对于一般损失函数，它就是残差的近似值)
- 第(2)(b)步估计回归树的结点区域，以拟合残差的近似值；
- 第(2)(c)步利用线性搜索估计叶结点区域的值，使损失函数极小化；
- 第(2)(d)步更新回归树。

# Algorithm: the same as random forest.

1. 决策树分为回归树和分类树：回归树用于预测实数值，如明天温度、用户年龄；分类树用于分类标签值，如晴天/阴天/雾/雨、用户性别。注意前者结果加减是有意义的，如10岁+5岁-3岁=12岁，后者结果加减无意义，如男+女=到底是男还是女？GBDT的核心在于累加所有树的结果作为最终结果，而**分类树是没有办法累加的，所以GBDT中的树都是回归树而非分类树**。

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

