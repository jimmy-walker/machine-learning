#模型选择

模型选择核心思想就是从某个模型类中选择最佳模型。当我们使用一种新的学习模型或者算法时，那么可是使用交叉验证来对模型进行评价。

下面伪代码表示了模型选择的**一般流程**。在这个算法中，最重要的就是第三个步骤中的误差评价。**记住此流程，具体可见交叉验证中的详细阐述**。

1. 准备候选的$$\ell$$个模型：$$M_{1},\cdot\cdot\cdot,M_{\ell}$$。

2. 对每个模型$$M_{1},\cdot\cdot\cdot, M_{\ell}$$求解它的学习结果。

3. 对每个学习结果的误差$$e_{1},\cdot\cdot\cdot,e_{\ell}$$进行计算。这里可以使用交叉验证方法。

4. 选择误差$$e_{1},\cdot\cdot\cdot,e_{\ell}$$最小的模型作为最终的模型。

此外，Sklearn提供了一张非常有用的流程图,供我们选择合适的学习方法。
![](/assets/model selection.png)

#交叉验证

1. **Holdout验证或称为简单交叉验证**

  **方法为将原始数据随机分为两组,一组做为训练集,一组做为验证集**,利用训练集训练分类器,然后**利用验证集验证模型,记录最后的分类准确率**为此HoldOutMethod下分类器的性能指标。

  优点是简单，只需随机把原始数据分为两组即可；

  缺点为一般来说，Holdout验证并非一种交叉验证，因为数据并没有交叉使用。 随机从最初的样本中选出部分，形成交叉验证数据，而剩余的就当做训练数据。 一般来说，少于原本样本三分之一的数据被选做验证数据。

2. **K-fold cross-validation**

  **方法为初始采样分割成K个子样本，一个单独的子样本被保留作为验证模型的数据，其他K-1个样本用来训练**。交叉验证重复K次，每个子样本验证一次，**平均K次的结果**或者使用其它结合方式，最终得到一个单一估测。

  优点是可以有效的避免过学习以及欠学习状态的发生。**10折交叉验证是最常用的**。

  缺点是训练和验证次数过多。

3. **留一验证**

  方法是只使用原本样本中的一项来当做验证资料，而剩余的则留下来当做训练资料。 这个步骤一直持续到每个样本都被当做一次验证资料。事实上，这等同于 K-fold 交叉验证是一样的，只是这里的K被设置为原本样本个数。

4. **交叉验证应用于模型选择，各个数据集的作用**

    1. 用测试集training set对多个模型(比如直线、二次曲线、三次曲线)进行训练；

    2.** 用交叉验证集cross validation set验证step1得到的多个假设函数，选择交叉验证集误差最小的模型**；

    3. 用测试集test set对step2选择的最优模型进行预测；

5. sklearn中的cross validation交叉验证对于我们选择正确的model和model的参数是非常有帮助的，我们能直观的看出不同model或者参数对结构准确度的影响。
##code
    1. 不需要使用train_test_split分割，而直接使用cross_val_score，传入模型和组数自动帮你得出各个组的得分。
    ```python
	from sklearn.cross_validation import cross_val_score 
	knn = KNeighborsClassifier(n_neighbors=5) 
	scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
	print(scores)

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4) 
	knn = KNeighborsClassifier(n_neighbors=5) 
	knn.fit(X_train, y_train) 
	y_pred = knn.predict(X_test) 
	print(knn.score(X_test, y_test))
    ```
    2. 还可以用来选参数和模型，用for循环传递进去，注意regression时的评分需要加负号。参见：[Model evaluation: quantifying the quality of predictions](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
        ```python
	# this is how to use cross_val_score to choose model and configs #
	from sklearn.cross_validation import cross_val_score
	import matplotlib.pyplot as plt
	k_range = range(1, 31)
	k_scores = []
	for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            ##loss = -cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error') # for regression
            scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy') # for classification
            k_scores.append(scores.mean())
        ```

#训练集train set、 验证集validation set、测试集test set

在监督机器学习中，数据集常被分成2~3个部分。
- 训练集(train set)：用来估计模型；

- 验证集(validation set)：确定网络结构或者控制模型复杂程度的参数；（J比如判断过拟合等）

- 测试集(test set)：检验最终选择最优的模型的性能如何。

一个典型的划分是训练集占总样本的50％，而其它各占25％，三部分都是从样本中随机抽取。样本少的时候，上面的划分就不合适了。**常用的是留少部分做测试集。然后对其余N个样本采用K折交叉验证法。**就是将样本打乱，然后均匀分成K份，轮流选择其中K－1份训练，剩余的一份做验证，计算预测误差平方和，最后把K次的预测误差平方和再做平均作为选择最优模型结构的依据。

**但实际应用中，一般只将数据集分成两类，即training set 和test set，大多数文章并不涉及validation set。所以我常常没做K折交叉验证，直接训练完，就测试了。**

#交叉验证的一个目的是为了验证模型是否过拟合

1. 过拟合产生的原因：**没有任何的learning algorithm可以彻底避免overfitting**

  1. 因为参数太多，会导致我们的模型复杂度上升，容易过拟合。

  2. 权值学习迭代次数足够多\(Overtraining\),拟合了训练数据中的噪声和训练样例中没有代表性的特征。

2. 过拟合的解决方法，**记住这些方法**。

  1. **减少特征**

  2. **正则化**

  3. **增加训练实例**

  4. **交叉验证法**

#学习曲线

学习曲线是将训练集误差和交叉验证集误差作为训练集实例数量的函数绘制的图表，分为高偏差和高方差两种情况\(欠拟合和过拟合\)。

1. 高偏差（欠拟合）

2. 

#误差分析
1. 

# Reference
- [ 训练集\(train set\) 验证集\(validation set\) 测试集\(test set\)](http://www.cnblogs.com/xfzhang/archive/2013/05/24/3096412.html)

- 

