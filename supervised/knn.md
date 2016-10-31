# Representation

1. 给定一个训练数据集，对新的输入实例，在训练数据集中找出与这个新实例最近的$$k$$个训练实例，然后统计最近的$$k$$个训练实例中所属类别计数最多的那个类。

    根据给定的距离度量（如欧式距离），在训练集$$T$$中找出与$$x$$距离最近的$$k$$个点，并把涵盖这些点的领域记为$$N_k(x)$$，根据决策规则（如多数表决）得到类别$$y$$。**记住该模型公式**。

    $$y = \arg\max_{c_j}\sum_{x_i \in N_k(x)} I(y_i=c_j), \ i = 1, 2, \dots, N; \ j = 1, 2, \dots, K$$

    其中训练集$$T=\{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\}$$；实例$$y_i$$的类别为$$\{ c_1,c_2,\dots,c_K\}$$；待分类样本$$x$$；设定好的最近邻个数$$k$$。

2. 一般k会取一个较小的值，然后用过交叉验证来确定。这里所谓的**交叉验证就是将样本划分一部分出来为预测样本，比如95%训练，5%预测，然后k分别取1，2，3，4，5之类的，进行预测，计算最后的分类误差，选择误差最小的k**。

3. 距离的度量（常见的距离度量有**欧式距离，马氏距离**等）。

4. k近邻法没有显示的学习过程。

# Evalution
k近邻法中的分类决策规则往往是**多数表决，等价于经验风险最小化**。**记住此策略**。

# Optimization
KD树（K-dimensional tree)即**K维树**：考虑这样的问题， 给定一个数据集$$D$$和某个点$$x$$，找出与$$x$$距离最近的$$k$$个点。这是一类很常见的问题，最简单的方法是暴力搜索，直接线性搜索所有数据，直到找到这$$k$$个点为止。对少量数据而言，这种方法还不错，但对大量、高纬度的数据集来说，这种方法的效率就差了。我们知道，排序好的数据查找起来效率很高，所以，可以**利用某种规则把已有的数据“排列”起来，再按某种特定的搜索算法（通过KD树的搜索找到与搜索目标最近的k个点，这样KNN的搜索就可以被限制在空间的局部区域上了，可以大大增加效率。），以达到高效搜索的目的，记住kd树思想**。

数据集$$T=(x_1, x_2, \dots, x_n)$$，其中$$x_i=(a_1, a_2, \dots, a_k)$$。

1. 构造KD树

    （**记住三点即可：下面的维度选择公式、划分是对当前区域内的实例点、实例保存在该节点**）

    1. 开始构造根节点，根节点对应包含数据集的k维空间超矩形区域

    选择$$a_1$$作为坐标轴，以数据集中所有实例的$$a_1$$坐标的中位数作为切分点，将根节点对应的超矩形区域切分为两个子区域。切分由通过切分点并与坐标轴$$a_1$$垂直的超平面实现；**由根节点生成深度为1的左右子节点**，左子节点对应坐标$$a_1$$小于切分点的子区域，右子节点对应坐标$$a_1$$大于切分点的子区域。**将落在切分超平面上的实例点保存在根节点**。

    2. 重复：对于深度为$$j$$的节点，为了生成$$j+1$$节点，选择$$a_l$$作为切分的坐标轴，$$l= j(mod\ k) +1$$，**以该节点区域中所有实例**的$$a_l$$坐标的中位数为切分点，将该节点对应的超巨型区域分为两个子区域，切分有通过切分点并与坐标轴$$a_l$$垂直的超平面实现。**将落在切分超平面上的实例点保存在该节点**。

    3. 直到**两个子区域没有实例存在**时停止。

    例子：有6个二维数据点：{(2,3),(5,4),(9,6),(4,7),(8,1),(7,2)}。

    ![](/assets/knn.jpg)

2. 搜索KD树（**J知道是先找到包含目标点的叶节点，然后逐步回撤回去，找更新点；找到就往上回到更上一级父节点**）

    1. 首先从根节点开始递归往下找到包含目标点的叶子节点。

    2. 将这个叶子节点认为是当前的“近似最近点”。

    3. 递归向上回退，如果以目标点圆心，以“近似最近点”为半径的球与根节点的另一半子区域边界相交，那么在相交的区域内寻找与目标点更近的实例点，**如果存在这样的点，将此点作为新的”近似最近点“，算法找到更上一级的父节点**。

    4. 重复3的步骤，直到另一子区域与球体不相交或者不存在比当前最近点更近的点。

    5. 最后更新的”近似最近点“是与目标点真正的最近点。

    ![](/assets/knn2.jpg)

# Code
```python
#Import Library
from sklearn.neighbors import KNeighborsClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create KNeighbors classifier object model
KNeighborsClassifier(n_neighbors=6)
# default value for n_neighbors is 5
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)
```

# Reference
- [机器学习之KNN（K近邻）](http://blog.csdn.net/zhang20072844/article/details/51704544)