# Representation

1. GBDT这个算法有很多名字，但都是同一个算法：

    - GBRT (Gradient BoostRegression Tree) 渐进梯度回归树
    - GBDT (Gradient BoostDecision Tree) 渐进梯度决策树
    - MART (MultipleAdditive Regression Tree) 多决策回归树
    - TN (Tree Net) 决策树网络

2. Gradient Boosting是一种Boosting的方法，它主要的思想是，**每一次建立模型是在之前建立模型损失函数的梯度下降方向（负梯度方向）**。它的每一次计算都是为了减少上一次的residual，而为了减少这些residual，它会在residual减少的gradient方向上建立一个新的model。**新model建立的目的是为了使先前模型得残差往梯度方向减少，与传统的boosting算法对正错样本赋予不同加权的做法有着极大的区别**。

$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]{F(x)=F{m-1}(x)} \quad \mbox{for } i=1,\ldots,n$$

# Algorithm: the same as random forest.

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
