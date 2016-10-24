# Representation

**随机森林是决策树集合的术语**。

随机森林是有很多随机得决策树构成，**它们之间没有关联**。得到随机森林后，在预
测时分别对每一个决策树进行判断，最后使用Bagging的思想进行结果的输出
（也就是投票的思想）。

# Evalution

# Optimization

# Code

```python
#Import Library
from sklearn.ensemble import RandomForestClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Random Forest object
model= RandomForestClassifier()
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)
```

```python
sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', 
max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, 
n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
```

参数调参中主要需要调整3个参数：

1. max\_features
    随机森林允许单个决策树使用特征的最大数量，python中常见的选项有:

    * Auto\/None:简单地选取所有特征，每棵树都没有限制

    * sqrt：每棵子树可以利用总特征数的平方根个，同log2

    * 0.2\(0.X\): 允许每个随机森林的子树可以利用特征数目20%

    增加max\_features一般能提高模型的性能，因为在每个节点上，我们有更多的选择可以考虑。然而，这未必完全是对的，因为它降低了单个树的多样性，而这正是随机森林独特的优点。但是，可以肯定，增加max\_features会降低算法的速度。因此，需要适当的平衡和选择最佳max\_features。

2. n\_estimators 
    子树数量。在允许的范围内应选择尽可能高的值。

3. min\_sample\_leaf 
    较小的叶子使模型更容易捕捉训练数据中的噪声。推荐50，实际中应测试多种叶子大小。


# Reference

