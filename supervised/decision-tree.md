# Representation
**决策树模型呈树形结构**。决策树可以认为是if-then规则集合，也可以认为是定义在特征空间与类空间上的条件概率分布。

决策树有内部节点和叶子节点组成，内部节点表示特征，叶子节点表示一个类，决策时，从根节点开始对实例的测试，根据测试结果把实例分配到子节点继续测试，直到叶子节点即可确定其类型。

![](/assets/decision tree.png)

# Evalution
决策树定义损失函数来评价模型好坏，那么学习的策略就是损失函数的最小化，现实中通常采用启发式方法，求解这一最优化值。

对于决策树的概率模型，可以由极大似然估计来估计模型参数，正则化极大似然函数作为决策树损失函数，**决策树的策略就变成损失函数的最小化问题**。

# Optimization

# Code
```python
#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import tree
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create tree object 
model = tree.DecisionTreeClassifier(criterion='gini') 
# for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
# model = tree.DecisionTreeRegressor() for regression
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(x_test)
```
# Reference
* [决策树](https://github.com/darlinglele/classification/wiki/%E5%86%B3%E7%AD%96%E6%A0%91)