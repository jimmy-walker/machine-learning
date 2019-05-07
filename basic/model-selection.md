#模型选择

模型选择核心思想就是从某个模型类中选择最佳模型。当我们使用一种新的学习模型或者算法时，那么可是使用交叉验证来对模型进行评价。

下面伪代码表示了模型选择的**一般流程**。在这个算法中，最重要的就是第三个步骤中的误差评价。**记住remember此流程，具体可见交叉验证中的详细阐述**。

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

    2. ** 用交叉验证集cross validation set验证step1得到的多个假设函数，选择交叉验证集误差最小的模型**；

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
    
    3. 此外还有一种选参数的方式：GridSearchCV(estimator, param_grid, ...)第一个参数是估计器，第二个参数是包含参数的字典。**可以利用grid.best_estimator_得到最佳的模型及参数。**
        ```python
        from sklearn.grid_search import GridSearchCV
    
        parameter_space = {
                           "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                           }
        clf = DecisionTreeClassifier(random_state=14)
        grid = GridSearchCV(clf, parameter_space)
        grid.fit(X_homehigher, y_true)
        print("Accuracy: {0:.1f}%".format(grid.best_score_ * 100))
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

2. 过拟合的解决方法，**记住remember这些方法**。

  1. **减少特征**

  2. **正则化**

  3. **增加训练实例**

  4. **交叉验证法**

#学习曲线

**学习曲线是一种诊断方法**，将训练集误差和交叉验证集误差作为**训练集实例数量**的函数绘制的图表，分为高偏差和高方差两种情况\(欠拟合和过拟合\)。
![](/assets/learning curve.png)

1. 高偏差（欠拟合）：通过增加样本量两者误差都很大，即训练集实例数量的增加对于算法的改进无益。
![](/assets/high bias.jpg)
2. 高方差(过拟合)：通过增加样本量训练集样本拟合程度很好(过拟合)，训练集误差很小，即训练集实例数量的增加对于算法的改进有一些帮助。
![](/assets/high variance.jpg)
3. 借助学习曲线进行决策
    - **训练集误差大、交叉验证集误差也大：欠拟合、高偏差、多项式次数d太小、$$\lambda$$太大**；
    - **训练集误差小、交叉验证集误差却很大：过拟合、高方差、多项式次数d太大、$$\lambda$$太小、样本量太少**。

4. sklearn.learning_curve中的learning curve可以很直观的看出我们的model学习的进度,对比发现有没有overfitting的问题。

    ```python
    learning_curve(estimator, X, y, train_sizes=array([ 0.1  ,  0.325,  0.55 ,  0.775,  1.   ]), cv=None, scoring=None, exploit_incremental_learning=False, n_jobs=1, pre_dispatch='all', verbose=0)
    ```
    **通过画出不同训练集大小对应的训练集和验证集准确率，我们能够很轻松滴检测模型是否方差偏高或偏差过高，以及增大训练集是否有用**。

    对于每个train_size，结合公式中的解释A cross-validation generator splits the whole dataset k times in training and test data. Subsets of the training set with varying sizes will be used to train the estimator and a score for each training subset size and the test set will be computed. Afterwards, the scores will be averaged over all k runs for each training subset size.**我的理解是先用cv将数据分成k份，之前在K折交叉验证中轮流选择其中K－1份训练，剩余的一份做验证，计算预测误差平方和，最后把K次的预测误差平方和再做平均作为选择最优模型结构的依据；而现在这里的training是训练集，test其实是验证集，那么就是说对K-1比例的训练集取不同规模进行训练模型，此时可以得到训练集误差，得到一份验证集误差，然后K次循环取平均值，接着再提高规模从而得到新值**。

    对于不同大小的训练集，确定交叉验证训练和测试的分数。一个交叉验证发生器将整个数据集分割k次，分割成训练集和测试集。不同大小的训练集的子集将会被用来训练评估器并且对于每一个大小的训练子集都会产生一个分数，然后测试集的分数也会计算。然后，对于每一个训练子集，运行k次之后的所有这些分数将会被平均。
##code
    ```python
        train_sizes, train_loss, test_loss= learning_curve(SVC(gamma=0.01), X, y, cv=10, scoring='mean_squared_error',train_sizes=[0.1, 0.25, 0.5, 0.75, 1])
        train_loss_mean = -np.mean(train_loss, axis=1)
        test_loss_mean = -np.mean(test_loss, axis=1)
        plt.plot(train_sizes, train_loss_mean, 'o-', color="r", label="Training")
        plt.plot(train_sizes, test_loss_mean, 'o-', color="g", label="Cross-validation")
    ```

#误差分析
误差分析可以帮助我们系统化地选择该做什么。**J这里将本页的内容串联起来了**

1. **从一个简单的能快速实现的算法开始，实现该算法并用交叉验证集数据测试这个算法**
2. **绘制学习曲线，决定是增加更多数据，或者添加更多特征，还是其他选择，J比如换算法**
3. **进行误差分析：**人工检查交叉验证集中我们算法中产生预测误差的实例，看看这些实
例是否有某种系统化的趋势

# 精确率、召回率、F1 值、ROC、AUC
假定有一个二分类问题，比如判定商品是否是假货。给系统一个样本，系统将会判断该样本为“真”（Predicted positive），或“假”（Predicted Negative）。但是当然，系统的判断与真实判断（actual positive/negative）是有误差的，将原本是真的判为真，就是TP（True Positive），原本真的判为假，就是FN（False Negative），原本假的判为真，就是FP（False Negative），原本假的判为假，就是TN（True Negative）。 

**注意背诵方式PPAP**
![](/assets/confusion.png)

**精确率(Precision）**是指在所有系统判定的“真”的样本中，确实是真的的占比，就是TP/(TP+FP)。

**召回率（Recall）**是指在所有确实为真的样本中，被判为的“真”的占比，就是TP/(TP+FN)。

**TPR（True Positive Rate）**的定义，跟Recall一样。

**FPR（False Positive Rate）**，又被称为“Probability of False Alarm”，就是所有确实为“假”的样本中，被误判真的样本，或者FP/(FP+TN)

**F1值**是为了综合考量精确率和召回率而设计的一个指标，一般公式为取P和R的harmonic mean:2*Precision*Recall/(Precision+Recall)。

**ROC**=Receiver Operating Characteristic，是TPR vs FPR的曲线；与之对应的是Precision-Recall Curve，展示的是Precision vs Recall的曲线。


# Reference
- [ 训练集\(train set\) 验证集\(validation set\) 测试集\(test set\)](http://www.cnblogs.com/xfzhang/archive/2013/05/24/3096412.html)

- [Stanford机器学习-第六周.学习曲线、机器学习系统的设计](http://www.myexception.cn/other/2060415.html)

- [精确率、召回率、F1 值、ROC、AUC 各自的优缺点是什么](https://www.zhihu.com/question/30643044/answer/161955532)
