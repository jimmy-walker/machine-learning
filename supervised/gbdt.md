# Representation

1. GBDT这个算法有很多名字，但都是同一个算法：

    - GBRT (Gradient BoostRegression Tree) 渐进梯度回归树
    - GBDT (Gradient BoostDecision Tree) 渐进梯度决策树
    - MART (MultipleAdditive Regression Tree) 多决策回归树
    - TN (Tree Net) 决策树网络

2. Gradient Boosting是一种Boosting的方法，它主要的思想是，**每一次建立模型是在之前建立模型损失函数的梯度下降方向（负梯度方向）**。它的每一次计算都是为了减少上一次的residual，而为了减少这些residual，它会在residual减少的gradient方向上建立一个新的model。**新model建立的目的是为了使先前模型得残差往梯度方向减少，与传统的boosting算法对正错样本赋予不同加权的做法有着极大的区别**。算法描述如下：

$$\{ \\   \quad\, 输入：训练数据集D=\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \cdots, (x^{(M)}, y^{(M)})\}, x^{(i)} \in \mathcal{X} \subseteq R^n, y^{(i)} \in \mathcal{Y}; \\   \qquad\quad\; 损失函数L(y, f(x)); \\   \quad 输出：提升树\hat{f}(x). \\   \quad 过程: \\   \qquad (1). 初始化模型 \\   \qquad\qquad\qquad f_0(x) = \arg \min_c \sum_{i=1}^{M} L(y^{(i)}, c)； \\   \qquad\; (2). 循环训练K个模型 k=1,2,\cdots,K \\   \qquad\qquad (a). 计算残差：对于i=1,2,\cdots,M \\   \qquad\qquad\qquad\qquad r_{ki} = -\left[ \frac{\partial L(y^{(i)}, \; f(x^{(i)}))} {\partial f(x^{(i)})} \right]_{f(x) = f_{k-1}(x)} \\   \qquad\qquad (b). 拟合残差r_{ki}学习一个回归树，得到第k颗树的叶结点区域R_{kj}，\quad j=1,2,\cdots,J \\   \qquad\qquad (c). 对j=1,2,\cdots,J, 计算：\\   \qquad\qquad\qquad\qquad c_{kj} = \arg \min_c \sum_{x^{(i)} \in R_{kj}} L(y^{(i)}, \; f_{k-1}(x^{(i)}) + c)\\   \qquad\qquad (d). 更新模型：\\   \qquad\qquad\qquad\qquad    f_k(x) = f_{k-1}(x) + \sum_{j=1}^{J} c_{kj} I(x \in R_{kj}) \\   \qquad\; (3). 得到回归提升树 \\   \qquad\qquad\qquad \hat{f}(x) = f_K(x) = \sum_{k=1}^{K} \sum_{j=1}^{J} c_{kj} I(x \in R_{kj}) \\    \}$$

# Algorithm: the same as random forest.
1. 


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
