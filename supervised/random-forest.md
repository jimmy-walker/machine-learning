# Representation

1. 在机器学习算法中，有一类算法比较特别，叫组合算法\(Ensemble\)，即将多个基算法\(Base\)组合起来使用。每个基算法单独预测，最后的结论由全部基算法进行投票（用于分类问题）或者求平均（包括加权平均，用于回归问题）。例如，如果你训练了3个树，其中有2个树的结果是A，1个数的结果是B，那么最终结果会是A。

2. **随机森林在以以决策树为基学习器构建Bagging集成的基础上，进一步在决策树的训练过程中引入随机属性选择，记住此模型**。

3. 集成学习分为两类：

  1. 个体学习器间存在强大依赖关系、必须串行生成的序列化方法，代表算法：Boosting；

  2. 个体学习器间不存在依赖关系、可同时生成的并行化方法，代表算法Bagging和“随机森林”RF。

4. **提升(Boosting)方法是串行式集成学习代表方法**。在分类问题中，它通过改变训练样本的权重，学习多个分类器，并将这些分类器进行线性组合,提高分类性能。代表算法为AdaBoost。下图中红色和蓝色的点就是数据，点越大表示权重越高。**记住boosting思想。**

    1. 在算法初始化阶段，为每一个样本赋予一个相等的weight，换言之，每个样本在开始都是一样重要的。
    
    2. 接下来，每一次训练后得到的模型，对数据点的估计会有所差异，所以在每一步结束后，我们需要对weight进行处理，而处理的方式就是通过增加错分类的样本点的weight，同时减少分类正确的样本点的weight。这样能够确保，如果某些点经常被分错，那么就会被“严重关注”，也就会被赋予一个很高的weight。

    3. 然后等进行了N次迭代（迭代次数由用户指定），将会得到N个简单的base learner，最后将它们组合起来，可以**对它们进行加权（错误率越大的base learner 其权重值越小，错误率越小的基分类器权重值越大）、或者让它们进行投票等得到一个最终的模型**。

    ![](/assets/boosting.png)
    ![](/assets/boosting formula.png)

5. AdaBoost VS GBDT：**AdaBoost采用原始的boosting算法，而GBDT采用改进的Gradient boosting**。

6. **Bagging是并行式集成学习代表方法**。**基于“自助采样法”（bootstrap sampling）**。

  自助采样法机制：给定包含m个样本的数据集，我们先随机取出一个样本放入采样集中，再把该样本放回初始数据集，使得下一次采样时该样本还会被采到。这样，经过m次样本采集，我们得到包含m个样本的采样集。采样集中，有的样本出现过很多次，有的没有出现过。Bagging机制：我们采样出T个含m个样本的采样集。然后基于每个采样集训练出一个学习器，再将学习器进行结合。对分类任务使用投票法，对回归任务采用平均值法。**J一句话就是m个样本，有放回地取m次。记住此bagging思想，加上取t次，就明白所谓的集成了。**

  若m个样本，有放回地取m次，那么大约$$\frac{1}{3}$$没被选择(如果m很大时的时候），可以作为验证集，证明如下：
  
  $$(1-\frac{1}{N})^N = \frac{1}{(\frac{N}{N-1})^N} = \frac{1}{(1+\frac{1}{N-1})^N} \approx \frac{1}{e} = 0.368\cdots $$

7. Bagging VS 随机森林：Bagging中基学习器的“多样性”通过样本扰动（对初始训练集采样）而获得；**随机森林中基学习器的“多样性”不仅来自于样本扰动，而且还可以通过随机属性扰动来获得。**这使得随机森立的泛化性能可通过个体学习器之间差异程度的增加而进一步提升。


# Algorithm: this in not algorithm like others, so I combine the evaluation and optimisation into one algorithm.

在随机森林中，每个决策树的生成过程如下所示：（建立第i棵树，原始训练集为S）

1. 用N来表示训练单个决策树所用样本的数目，M表示原始特征维度（输入矩阵N\*M）。

2. 输入特征维度m，用于决策树中一个节点的决策结果，其中m应远小于M， 如sqrt\(M\), 1\/2sqrt\(M\), 2sqrt\(M\)等。

3. **从原始训练集S中，以有放回抽样的方式，取样N次，形成一个训练集S\(i\)（J这就是bagging）**，作为根节点的样本，从根节点开始训练。

4. **对于每一个节点，随机选择m个特征，运用这m个变量来决定最佳的分裂点。（J这就是随机）**在决策树的生成过中，m的值是保持不变的。在**这m个特征中根据信息增益来找到该节点的最佳维度的特征k及其阈值th**，当k&lt;th时划到左节点，其余划到右节点。（列采样）

5. 用未抽到样本做预测，评估其误差。（行采样）

6. **每棵树都会完整生长而不会剪枝**，因为是随机采样，所以不会出现过拟合（overfitting）。

7. 每次根据从M维特征中随机抽取的m个特征生成一个决策树，共生成t个不同的决策树，这样**决策树通过投票对测试数据分类（回归时采用平均）**。


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

参数调参中主要需要调整3个参数，**记住此调参方法**：

1. max\_features

  随机森林允许单个决策树使用特征的最大数量，python中常见的选项有:

  1. Auto\/None:简单地选取所有特征，每棵树都没有限制

  2. sqrt：每棵子树可以利用总特征数的平方根个，同log2

  3. 0.2\(0.X\): 允许每个随机森林的子树可以利用特征数目20%


  增加max\_features一般能提高模型的性能，因为在每个节点上，我们有更多的选择可以考虑。然而，这未必完全是对的，因为它降低了单个树的多样性，而这正是随机森林独特的优点。但是，可以肯定，增加max\_features会降低算法的速度。因此，需要适当的平衡和选择最佳max\_features。

2. n\_estimators

  子树数量。在允许的范围内应选择尽可能高的值。

3. min\_sample\_leaf

  较小的叶子使模型更容易捕捉训练数据中的噪声。**推荐50**，实际中应测试多种叶子大小。


# Reference

* [Random Forest入门](https://www.zybuluo.com/hshustc/note/179319)
* [集成学习（Boosting,Bagging和随机森林）](http://blog.csdn.net/u014665416/article/details/51557318)
* [boosting方法(Adaboost,GBDT)](http://www.duzelong.com/wordpress/201509/archives1614/)
* [Gradient Tree Boosting Algorithm](http://www.csuldw.com/2015/08/19/2015-08-19%20GBDT/)
* [机器学习中的数学(3)-模型组合(Model Combining)之Boosting与Gradient Boosting](http://www.cnblogs.com/LeftNotEasy/archive/2011/01/02/machine-learning-boosting-and-gradient-boosting.html)