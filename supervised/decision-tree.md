# Representation

**决策树模型呈树形结构，记住此模型**。决策树可以认为是if-then规则集合，也可以认为是定义在特征空间与类空间上的条件概率分布。

决策树有内部节点和叶子节点组成，内部节点表示特征，叶子节点表示一个类，决策时，从根节点开始对实例的测试，根据测试结果把实例分配到子节点继续测试，直到叶子节点即可确定其类型。

![](/assets/decision tree.png)

# Evalution

决策树定义损失函数来评价模型好坏，那么学习的策略就是损失函数的最小化，现实中通常采用启发式方法，求解这一最优化值，比如ID3等。

对于决策树的概率模型，可以**由极大似然估计来估计模型参数，正则化极大似然函数作为决策树损失函数，记住此策略**，**决策树的策略就变成损失函数的最小化问题**。

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

2. 特征选择：特征的选择在于选取对训练数据具有分类能力的特征，这样可以提高决策树学习的效率。**通常特征选择的准则是信息增益或信息增益比**。

    1. 熵：**熵**（entropy）**表示随机变量不确定性的度量**。
    
    假设X是一个取有限个值的离散变量，X的熵定义如下，式中的对数以2为底或者以e为底，因为X的熵只依赖于X的分布，而与X的取值无关，改成关于p的函数：

    $$H(X)=-\sum_{i=1}^{n}p_ilogp_i$$

    $$H(p)=-\sum_{i=1}^{n}p_ilogp_i$$

    2. 条件熵：表示在已知随机变量X的条件下随机变量Y的不确定性。

    定义为X给定条件下Y的条件概率分布的熵对X的数学期望：

    $$H(Y|X)=\sum_{i=1}^{n}p_iH(Y|X=x_i)\\
    p_i=P(X=x_i), i=1,2,...,n
    $$
    
    3. 当熵和条件熵中的概率由数据统计得到时，所对应的熵与条件熵分别称为经验熵（empirical entropy）和经验条件熵（empirical conditional entropy）。

    4. **信息增益**（information gain）**表示得知特征X的信息而使得类Y的信息不确定性的减少的程度**。
    
    特征A对训练数据集D的信息增益g(D,A)，定义为集合D的经验熵H(D)与特征A给定条件下D的经验条件熵H(D|A)之差：

    $$g(D|A)=H(D)-H(D|A)$$

    5. 信息增益的算法：对训练数据集D，计算其每个特征的信息增益，并比较它们的大小，选择信息增益最大的特征。
    
        1. 计算数据集D的经验熵：

            $$H(D)=-\sum_{k=1}^{K}\frac{C_k}{|D|}log_2\frac{C_k}{|D|}$$
        
        2. 计算特征A对数据集D的经验条件熵

            $$H(D|A)=\sum_{i=1}^{n}\frac{|D_i|}{|D|}H(D_i)=-\sum_{i=1}^{n}\frac{|D_i|}{|D|}\sum_{k=1}^{K}\frac{|D_{ik}|}{|D_i|}log_2\frac{|D_{ik}|}{|D_i|}$$

        3. 计算信息增益

            $$g(D|A)=H(D)-H(D|A)$$

3. 决策树生成

4. 决策树的修剪

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

