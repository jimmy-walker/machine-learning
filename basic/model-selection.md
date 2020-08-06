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

**FPR（False Positive Rate）**，又被称为“Probability of False Alarm”，就是所有确实为假的样本中，被误判为真的样本，或者FP/(FP+TN)

**F1值**是为了综合考量精确率和召回率而设计的一个指标，一般公式为取P和R的harmonic mean:2*Precision*Recall/(Precision+Recall)。

**ROC**=Receiver Operating Characteristic，是TPR vs FPR的曲线；与之对应的是Precision-Recall Curve，展示的是Precision vs Recall的曲线。

具体画ROC图时这么画：

将置信度（认同为1的概率）从大到小排，从而画出ROC曲线。

```python
tpr.append(np.sum((y_sort[:i] == 1)) / pos)
fpr.append(np.sum((y_sort[:i] == 0)) / neg)
```

置信度很大时，大部分都被判定为0，因此纵坐标的TPR为0，横坐标的FPR为0。

置信度很小时，大部分都被判定为1，因此纵坐标的TPR为1，横坐标的FPR为1。

**可以看出两者是同步增长的！**

```python
import numpy as np
import matplotlib.pyplot as plt
data_len = 50
label = np.random.randint(0, 2, size=data_len)
score = np.random.choice(np.arange(0.1, 1, 0.01), data_len)
#随机设置标签label和分类器判定为1的置信度score
def ROC_curve(y,pred):
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    pred_sort = np.sort(pred)[::-1]  #从大到小排序
    index = np.argsort(pred)[::-1]#从大到小排序
    y_sort = y[index]
    print(y_sort)
    tpr = []
    fpr = []
    thr = []
    for i,item in enumerate(pred_sort):
        tpr.append(np.sum((y_sort[:i] == 1)) / pos)
        fpr.append(np.sum((y_sort[:i] == 0)) / neg)
        thr.append(item)
    print(fpr)
    print(tpr)
    print(thr)
    return fpr, tpr, thr

fpr, tpr, thr = ROC_curve(label, score)
plt.plot(fpr, tpr, 'k')
plt.title('Receiver Operating Characteristic')
plt.plot([(0,0),(1,1)],'r--')
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,01.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


```

这里给出一个简单的例子：

**画线的时候就是链接点，如果是硬分类器，直接连，如果是概率分类器，则自己定阈值。auc好处是更好的面对样本的平衡。**

首先对于硬分类器（例如SVM，NB），预测类别为离散标签，对于8个样本的预测情况如下：

![](/assets/auc.png)

得到混淆矩阵如下：

![](/assets/auc2.png)

进而算得TPRate=3/4，FPRate=2/4，得到ROC曲线：

![](/assets/auc3.png)

最终得到AUC为0.625。

对于LR等预测类别为概率的分类器，依然用上述例子，假设预测结果如下：

![](/assets/auc4.png)

这时，需要设置阈值来得到混淆矩阵，不同的阈值会影响得到的TPRate，FPRate，如果阈值取0.5，小于0.5的为0，否则为1，那么我们就得到了与之前一样的混淆矩阵。其他的阈值就不再啰嗦了。依次使用所有预测值作为阈值，得到一系列TPRate，FPRate，描点，求面积，即可得到AUC。

------

最后说说AUC的优势，AUC的计算方法同时考虑了分类器对于正例和负例的分类能力，在样本不平衡的情况下，依然能够对分类器作出合理的评价。

例如在反欺诈场景，设欺诈类样本为正例，正例占比很少（假设0.1%），如果使用准确率评估，把所有的样本预测为负例，便可以获得**99.9%的准确率**。

但是如果使用AUC，把所有样本预测为负例，TPRate和FPRate同时为0（没有Positive），与(0,0) (1,1)连接，得出**AUC仅为0.5**，成功规避了样本不均匀带来的问题。

# AUC排序

AUC这个指标有两种解释方法，一种是传统的“曲线下面积”解释，另一种是关于排序能力的解释。例如0.7的AUC，**其含义可以大概理解为：给定一个正样本和一个负样本，在70%的情况下，模型对正样本的打分高于对负样本的打分。可以看出在这个解释下，我们关心的只有正负样本之间的分数高低，而具体的分值则无关紧要。** 

这个解释是对应于AUC即ROC曲线下的面积，而ROC曲线的横轴是FPRate，纵轴是TPRate，当二者相等时，即y=x。

![](/assets/auc5.png)

表示的意义是：对于不论真实类别是1还是0的样本，分类器预测为1的概率是相等的。

换句话说，分类器对于正例和负例毫无区分能力，和**抛硬币**没什么区别，一个抛硬币的分类器是我们能想象的最差的情况，因此一般来说我们认为AUC的最小值为0.5（当然也存在预测相反这种极端的情况，AUC小于0.5，这种情况相当于分类器**总是**把对的说成错的，错的认为是对的，那么只要把预测类别取反，便得到了一个AUC大于0.5的分类器）。

## AUC在排序中的目的

**AUC就是用来比较有点击记录的得分与无点击记录的得分是否满足上述关系的指标。**

**我们需要对比有点击行为的记录与无点击行为的记录，要求有点击行为记录的预测分数高于无点击行为记录的预测分数，AUC就是衡量满足要求的对比数占总对比数的指标。** 

## **与ndcg区别**

**AUC更关注模型在给定阈值下区分正负样本的能力，而不关注正样本相互之间（或者负样本相互之间）的比较。**

计算广告场景下（尤其是电商广告）用户的行为往往是有偏的，即便同样是正样本，用户偏好程度往往不同，甚至差异明显。正样本排序结果的好坏往往也对性能产生较大影响。

可以尝试下其他更关注样本相对样本排序好坏的指标，例如搜索重排序中很常用的NDCG.



通常在训练模型时，我们需要从日志里获得某一天或者某几天的数据作为训练集，选取另外某一天或者某几天的作为验证集（通常训练集时间先于验证集时间）。日志里是一条一条的记录(通常包括用户标识、item标识、操作时间、操作类型等等)，对于某一条记录，有用户点击的我们标记为1，代表用户对当前的item感兴趣，点击观看了（当然不考虑手抽点到的）；对于展示但是用户未点击的我们标记为0，代表用户不感兴趣。在LTR中，根据训练集对模型进行训练，训练好的模型会为验证集的每一条记录打分，那么对于用户行为是点击的记录，我们希望得分高一些；而对于用户行为是不点击的记录，我们希望得分低一些，这样预测分数就可以代表用户对当前item的兴趣大小。**AUC就是用来比较有点击记录的得分与无点击记录的得分是否满足上述关系的指标**。**这里需要强调的是AUC仅仅用来比较有点击与无点击的记录，不用来比较有点击之间或者无点击之间的关系（比如A记录中用户点击并且对该次展示极感兴趣**，而B记录中用户对该次展示没有兴趣只是不小心点到，所以A记录得分理应高于B，但这不属于AUC的考虑范围）。 

举个例子如下:

下表中有如下6条记录：

![](/assets/auc6.png)

这里我们无法预知同为用户点击过的A和D两条记录到底谁的得分更应该高一些，也无法预知其余四条未点击的记录谁的得分应该更低一些。但是根据AUC的概念，A和D的得分应该高于其余四条记录中的任意一条。下面开始计算AUC的流程:

我们需要将记录A、D分别与另外四条记录比较，一共有8组对比。这里计算AUC的分母就是8；那么共有多少组对比是满足要求的呢?记录A比另外四组(B、C、E、F)得分都高，记录D只比另外二组(E、F)得分都高，所以八组对比中满足条件的只有6组，那么分子就是6。所以我们计算得到的AUC就是6/8 = 0.75。简单吧？好像确实不是很难耶！

下面对AUC的计算做一个简单总结：通过模型对验证集中的每条记录做一个预测。这些记录中有点击行为的记录数为M，未点击的记录数为N，则用有M*N组对比。对于有点击的M条记录分别记为p1、p2、……pM，对于其中任意一条记录Pi，其预测分数高于未点击记录的个数为记做$\alpha_i$，则满足条件的对比组为的$\alpha_1, \alpha_2, ... \alpha_m,$累积和，除以M*N就是本次记录组中AUC的结果。我们需要对比有点击行为的记录与无点击行为的记录，要求有点击行为记录的预测分数高于无点击行为记录的预测分数，AUC就是衡量满足要求的对比数占总对比数的指标。

**J虽然计算方式不同于以往的auc计算**，但是其实是一样的，具体可见参考文章(LTR那点事—AUC及其与线上点击率的关联详解)。

## AUC与线上点击率:AUC毕竟是线下离线评估指标，与线上真实业务指标有差别 

**LTR就是对于某一个特定用户，根据模型对召回的不同item的预测分数进行降序排列，如果AUC足够高，则根据预测分数进行排序的结果与用户真实兴趣排序的结果相符**。

这也是为什么我们要提高AUC的原因。将用户感兴趣的item优先展示，可以提高用户的点击数同时降低入屏但是不点击的数目，从而提高点击率，产生较好的用户体验。 

AUC毕竟是线下离线评估指标，与线上真实业务指标有差别。差别越小则AUC的参考性越高。比如上文提到的点击率模型和购买转化率模型，虽然购买转化率模型的AUC会高于点击率模型，但往往都是点击率模型更容易做，线上效果更好。

购买决策比点击决策过程长、成本重，且用户购买决策受很多场外因素影响，比如预算不够、在别的平台找到更便宜的了、知乎上看了评测觉得不好等等原因，这部分信息无法收集到，导致最终样本包含的信息缺少较大，模型的离线AUC与线上业务指标差异变大。

总结起来，样本数据包含的信息越接近线上，则离线指标与线上指标gap越小。而决策链路越长，信息丢失就越多，则更难做到线下线上一致。

# Reference

- [ 训练集\(train set\) 验证集\(validation set\) 测试集\(test set\)](http://www.cnblogs.com/xfzhang/archive/2013/05/24/3096412.html)
- [Stanford机器学习-第六周.学习曲线、机器学习系统的设计](http://www.myexception.cn/other/2060415.html)
- [精确率、召回率、F1 值、ROC、AUC 各自的优缺点是什么](https://www.zhihu.com/question/30643044/answer/161955532)
- [LTR那点事—AUC及其与线上点击率的关联详解](https://www.jiqizhixin.com/articles/2019-10-14-4 )
- [乱弹机器学习评估指标AUC](https://zhuanlan.zhihu.com/p/52930683 )
