# Representation

**决策树模型呈树形结构**。决策树可以认为是if-then规则集合，也可以认为是定义在特征空间与类空间上的条件概率分布。

决策树有内部节点和叶子节点组成，内部节点表示特征，叶子节点表示一个类，决策时，从根节点开始对实例的测试，根据测试结果把实例分配到子节点继续测试，直到叶子节点即可确定其类型。

![](/assets/decision tree.png)

# Evalution

决策树定义损失函数来评价模型好坏，那么学习的策略就是损失函数的最小化，现实中通常采用启发式方法，求解这一最优化值，比如ID3等。

对于决策树的概率模型，可以由极大似然估计来估计模型参数，正则化极大似然函数作为决策树损失函数，**决策树的策略就变成损失函数的最小化问题**。

# Optimization

**决策树的学习通常包含3个部分：特征选择、决策树生成和决策树的修剪。**

决策树学习**常用的算法有ID3、C4.5与CART**。

1. **决策树构建的一般过程**。

  1. 开始构建根结点，将**所有的训练数据都放入根结点**；

  2. **选择一个最优特征，按照这一特征将训练数据分割成子集**，使得各个子集有一个在当前条件下最好的分类；

  3. 如果这些子集已经基本被正确分类，那么就把这些子集分到所对应的叶节点中去；

  4. 如果还有子集未能基本正确分类，那么就对这些子集选择新的最优特征，继续对其进行分割

  5. 如此递归下去，直到全部基本正确分类，最后每一个子集都被分配到叶节点上，即都有了明确的分类，这就生成了一棵决策树。

  6. 以上生成的决策树对训练数据有很好的分类能力，但**可能发生过拟合的情况。我们需要对生成的决策树进行自下而上的剪枝**（去掉过于细分的叶结点，使其回退到父节点或者更高的结点，使树变得更简单），使其具有更好的泛化能力。

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

* [决策树](http://www.wengweitao.com/jue-ce-shu.html)
* [随机森林与决策树](https://clyyuanzi.gitbooks.io/julymlnotes/content/rf.html)

