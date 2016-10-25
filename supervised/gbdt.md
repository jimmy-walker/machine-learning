# Representation

1. GBDT这个算法有很多名字，但都是同一个算法：

    - GBRT (Gradient BoostRegression Tree) 渐进梯度回归树
    - GBDT (Gradient BoostDecision Tree) 渐进梯度决策树
    - MART (MultipleAdditive Regression Tree) 多决策回归树
    - TN (Tree Net) 决策树网络

2. Gradient Boosting是一种Boosting的方法，它主要的思想是，**每一次建立模型是在之前建立模型损失函数的梯度下降方向（负梯度方向）**。


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

