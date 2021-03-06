# Representation

**决策树模型呈树形结构，记住remember此模型**。决策树可以认为是if-then规则集合，也可以认为是定义在特征空间与类空间上的条件概率分布。

决策树有内部节点和叶子节点组成，内部节点表示特征，叶子节点表示一个类，决策时，从根节点开始对实例的测试，根据测试结果把实例分配到子节点继续测试，直到叶子节点即可确定其类型。

![](/assets/decision tree.png)
![](/assets/decision tree.gif)

# Evalution

决策树定义损失函数来评价模型好坏，那么学习的策略就是损失函数的最小化，现实中通常采用启发式方法，求解这一最优化值，比如ID3等。

对于决策树的概率模型，可以**由极大似然估计来估计模型参数，正则化极大似然函数作为决策树损失函数，记住remember此策略**，**决策树的策略就变成损失函数的最小化问题**。

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
    p_i=P(X=x_i), i=1,2,...,n$$

  3. 当熵和条件熵中的概率由数据统计得到时，所对应的熵与条件熵分别称为经验熵（empirical entropy）和经验条件熵（empirical conditional entropy）。

  4. **信息增益**（information gain）**表示得知特征X的信息而使得类Y的信息不确定性的减少的程度**。

    特征A对训练数据集D的信息增益g\(D,A\)，定义为集合D的经验熵H\(D\)与特征A给定条件下D的经验条件熵H\(D\|A\)之差：

    $$g(D|A)=H(D)-H(D|A)$$

  5. 信息增益的算法：**对训练数据集D，计算其每个特征的信息增益，并比较它们的大小，选择信息增益最大的特征**。$$|D|$$表示样本容量，设有K个类$$C_k，|C_k|$$属于类$$C_k$$的样本数量。特征A将D划分为n个子集$$D_1,D_2,...,D_n，|D_i|$$为$$D_i$$的样本数。子集$$D_i$$中属于类$$C_k$$的样本的集合为$$D_{ik}$$。

    1. 计算数据集D的经验熵：
      $$H(D)=-\sum_{k=1}^{K}\frac{C_k}{|D|}log_2\frac{C_k}{|D|}$$
    2. 计算特征A对数据集D的经验条件熵
      $$H(D|A)=\sum_{i=1}^{n}\frac{|D_i|}{|D|}H(D_i)=-\sum_{i=1}^{n}\frac{|D_i|}{|D|}\sum_{k=1}^{K}\frac{|D_{ik}|}{|D_i|}log_2\frac{|D_{ik}|}{|D_i|}$$

    3. 计算信息增益
      $$g(D|A)=H(D)-H(D|A)$$

  6. 信息增益比：以信息增益作为划分训练数据集的特征，存在偏向于选择取值较多的特征的问题。使用信息增益比（information gain ratio）可以对这一问题进行校正。这是特征选择的另一准则。特征A对训练数据集D的信息增益比$$g_R(D,A)$$定义为其信息增益$$g(D,A)$$与训练数据集D关于特征A的值的熵$$H_A(D)$$之比，其中$$H_A(D)=-\sum_{i=1}^{n}\frac{|D_{i}|}{|D|}log_2\frac{|D_{i}|}{|D|}$$，n为特征A的取值个数，$$D_i$$表示特征A将D分成的子集：

    $$g_R(D,A)=\frac{g(D,A)}{H_A(D)}$$

3. 决策树生成

  1. ID3算法：**ID3算法的核心是在决策树各个结点上应用信息增益准则选择特征，递归地构建决策树**。具体方法如下：
     1. 从根结点开始，对结点计算所有可能的特征的信息增益，选择信息增益最大的特征作为结点的特征，由该特征的不同取值建立子节点。

    1. 对子节点递归的调用以上方法，构建决策树。

    2. 直到所有特征的信息增益均很小或没有特征选择为止。

  2. C4.5算法：**C4.5算法与ID3算法类似，在生成的过程中，用信息增益比来选择特征**。

4. 决策树的修剪：**决策树生成算法产生的决策树，会出现过拟合的现象**。因为在学习的过程中过多地考虑如何提高对训练数据的正确分类，从而构建出过于复杂的决策树。**解决这个问题的办法就是考虑决策树的复杂度，对已生成的决策树进行简化**。对已生成的决策树进行简化的过程称为剪枝（pruning）。

  1. 决策树的剪枝往往通过极小化决策树整体的损失函数或代价函数来实现。决策树的损失函数可以定义为：

    $$C_\alpha(T)=\sum_{t=1}^{|T|}N_tH_t(T)+\alpha|T|$$

    其中，树T的叶节点数为\|T\|，叶节点t有$$|N_t|$$个样本点，其中属于k类的数目为$$|N_t|$$个，$$|N_{tk}|$$为叶节点t的经验熵：$$H_t(T)=-\sum_{k=1}^{K}\frac{|N_{tk}|}{|N_t|}log_2\frac{|N_{tk}|}{|N_t|}$$

    决策树的损失函数可以表示为$$C_\alpha(T)=C(T)+\alpha|T|$$，**记住remember该公式，虽然在第三环节但是应该在第二环节的策略的，记住remember最小化此损失函数，等价于正则化的极大似然估计，J背后的数学原理不去细推**。这时C\(T\)表示模型对训练数据的预测误差，即模型与训练数据的拟合程度，**\|T\|表示模型的复杂度，J即叶结点个数，不要搞错了！**参数α控制二者直接的影响。较大的$$\alpha$$促使选择较简答的模型，较小反之。

    **J因为叶结点就是最底层的结点，那么熵越大表示不确定度即误差越大，所以以此作为损失函数，计算每个叶结点的结果之和**。

    剪枝，就是当$$\alpha$$确定时，选择损失函数最小的模型，即损失函数最小的子树。利用损失函数最小原则进行剪枝就是用正则化的极大似然估计进行模型选择。

  2. 树的剪枝算法。输入：生成算法产生的整棵树T，参数$$\alpha$$；
    输出：修剪后的子树$$T_\alpha$$。**J记住remember这个算法，因为就是损失函数的优化。**

    1. 计算每个结点的经验熵

    2. 递归地从叶节点向上回缩。设叶节点回到到其父节点之前与之后的整体树分别为$$T_B$$和$$T_A$$，如果其对应的损失函数有：$$C_\alpha(T_A) \leq C_\alpha(T_B)$$，则进行剪枝，即将父节点变为新的叶结点。

    3. 返回第二步直至不能继续，得到损失函数最小的子树。

5. CART算法，**与前两个算法不同（CART是二叉树），这是重要区别**，CART单独拿出来讲。分类与回归树（classification and regression tree, CART）是应用广泛的决策树学习方法。既可以用于分类也可以用于回归。

  1. CART决策树的生成：决策树的生成就是递归地构建二叉决策树的过程，**对回归树用平方误差最小化准则，对分类树用基尼指数（Gini index）最小化准则**，进行特征选择，生成二叉树。

    1. 回归树的生成

      遍历所有输入变量，找到最优的切分变量j和切分点（可以用平方误差来表示回归树对于训练数据的预测误差，使预测误差最小），构成一个对**（j,s）**（第j个变量$$x^{(j)}$$和**它取的值**）。依次将输入空间划分为两个区域，接着对每个区域重复上述划分过程，直到满足停止条件为止。这样就生成一棵回归树。这样的回归树通常称为最小二乘回归树（least squares regression tree）。**注意：回归树输出叶子节点中各个样本值的平均值！**

      ![](/assets/cart regression tree.PNG)

    2. 分类树的生成

      1. 分类树用基尼指数选择最优特征，同时决定该特征的最优二值切分点。**基尼指数Gini\(D\)表示集合D的不确定性，基尼指数Gini\(D,A\)表示经A=a分割后集合D的不确定性。基尼指数越大，样本集合的不确定性也就越大（与熵类似）**。

        在分类问题中，假设有K个类，样本点属于第k类的概率为$$p_k$$，则概率分布的基尼指数定义为$$Gini(p)=\sum_{k=1}^{K}p_k(1-p_k)=1-\sum_{k=1}^{K}p_k^2$$

        对于给定的样本集合D，其基尼指数为$$Gini(p)=1-\sum_{k=1}^{K}(\frac{|C_k|}{|D|})^2$$

        假设样本集合D根据特征A是否取某一可能值a被分割成两个部分$$D_1$$和$$D_2$$，则在特征A的条件下，集合D的基尼系数定义为$$Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2)$$

      2. CART分类树生成算法

        1. 设结点的训练数据集为D，计算现有特征对该数据集的基尼指数，对每一个特征A，对其可能的每一个取值a，根据样本点对A=a的测试为“是”或“否”将D分割成$$D_1$$和$$D_2$$两部分，计算A=a时的基尼指数。

        2. 在所有的特征A以及它们所有可能的切分点a中，选择基尼指数最小的特征及其对应的切分点作为最优特征与最优切分点。依最优特征与最优切分点，从现结点生成两个子结点，将训练数据集依特征分配到两个子结点中去。

  2. CART决策树的剪枝,分以下两步走：

    1. 剪枝，形成一个子树序列
        从生成算法产生的$$T_0$$**决策树底端开始不断剪枝**，到$$T_0$$根结点，形成一个子树序列$${T_0,T_1,...,T_n}$$

        对$$T_0$$中每一内部结点t，计算$$g(t) = \frac{C(t)-C(T_t)}{|T_t|-1}$$，**它表示剪枝后整体损失函数减少的程度**。在$$T_0$$中剪去g(t)最小的$$T_t$$，将得到子树作为$$T_1$$，同时将最小的g(t)设为$$\alpha_1$$。$$T_1$$为区间$$[\alpha_1,\alpha_2)$$的最优子树。**记住rememberCART这个优化过程，就是每次最小的g(t)的结点减去得到新的树。**

        其中的公式由来如下：

        在剪枝的过程中，计算子树的损失函数：$$C_\alpha(T)=C(T)+\alpha|T|$$

        其中，T为任意子树，C(T)为对训练数据的预测误差（如基尼指数），|T|为子树的叶结点个数（这个子树代表序列中的子树）。

        对$$T_0$$的任意内部结点t:

        以t为单结点树的损失函数是（叶结点个数为0）：$$C_\alpha(t)=C(t)+\alpha$$

        以t为根结点的子树$$T_t$$的损失函数是：$$C_\alpha(T_t)=C(T_t)+\alpha|T_t|$$

        只要$$\alpha = \frac{C(t)-C(T_t)}{|T_t|-1}$$，$$T_t与t$$有相同的损失函数值，而t的结点少，因此t比$$T_t$$更可取，对$$T_t$$进行剪枝。

    2. 在剪枝得到的子树序列$$T_0,T_1,...,T_n$$中通过交叉验证得到最优子树$$T_n$$

        测试子树序列$$T_0,T_1,...,T_n$$中各棵子树的平方误差或基尼指数。平方误差或基尼指数最小的决策树被认为是最优的决策树。在子树序列中每棵子树$$T_0,T_1,...,T_n$$都对应于一个参数$$\alpha_0,\alpha_1,...,\alpha_n$$。所以当最优子树$$T_k$$确定时，对应的$$\alpha_k$$也确定了，即得到最优决策树$$T_\alpha$$。

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

sklearn中的决策树算法就是CART的加强版。scikit-learn uses an optimised version of the CART algorithm.

# Reference

* [决策树](http://www.wengweitao.com/jue-ce-shu.html)

