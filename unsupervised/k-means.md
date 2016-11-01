# Representation

聚类是一种无监督的学习算法，它将相似的数据归纳到同一簇中。K-均值是因为它可以按照k个不同的簇来分类，并且**不同的簇中心采用簇中所含的均值计算而成**。

算法如下，**记住remember该模型**：

![](/assets/K-Means.jpg)

1. **随机**在图中取K（这里K=2）个质心$$\mu_k$$。

2. 然后对图中的所有点求到这K个质心的距离$$d_{ik} = ||x^{(i)}-\mu_k||^2$$，假如点Pi离质心点Si最近，那么Pi属于Si点群。（上图中，我们可以看到A,B属于上面的质心，C,D,E属于下面中部的质心）

3. 接下来，**计算每个聚类中所有点的坐标平均值，并将这个平均值作为新的聚类中心（质心）**。（见图上的第三步）

4. 然后重复第2）和第3）步，直到，**质心没有大范围移动**（我们可以看到图中的第四步上面的质心聚合了A,B,C，下面的质心聚合了D，E）。

# Evalution

K-means算法希望最小化平方误差（SSE），即最小化目标函数: $$SSE = \sum^{K}_{k=1}\sum_{x \in \mu_k} ||x-\mu_k||^2$$，但是最小化这个公式并不容易，因此K均值采用贪心策略，通过迭代优化来求近似解，这也就是为什么K-means有时候求得的划分是次优解。**记住remember该策略**。

# Optimization

1. 初始质心的随机初始化：多次运行，每次随机的选择一组初始质心，然后选择具有最小误差的簇集。

2. K-Means主要有两个最重大的缺陷——都和初始值有关：

    1. K 是事先给定的，这个 K 值的选定是非常难以估计的。很多时候，事先并不知道给定的数据集应该分成多少个类别才最合适。（ ISODATA 算法通过类的自动合并和分裂，得到较为合理的类型数目 K）

    2. K-Means算法需要用初始随机种子点来搞，这个随机种子点太重要，不同的随机种子点会有得到完全不同的结果。（K-Means++算法可以用来解决这个问题，其可以有效地选择初始点）

# Code
```python
#Import Library
from sklearn.cluster import KMeans
#Assumed you have, X (attributes) for training data set and x_test(attributes) of test_dataset
# Create object model
k_means = KMeans(n_clusters=3, random_state=0)
# Train the model using the training sets and check score
model.fit(X)
#Predict Output
predicted= model.predict(x_test)
```
```python
sklearn.cluster.KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
```
n_clusters : int, optional, default: 8 表示簇类个数

init : {‘k-means++’, ‘random’ or an ndarray}
Method for initialization, defaults to ‘k-means++’: J具体就不管k-means++的原理了

n_init : int, default: 10 重复次数。为了弥补初始质心的影响，算法默认会初始10组质心，实现算法，然后返回最好的结果，所谓的随机初始化。J就是说初始不同的质心，返回最好的一个

# Reference
- [ K-means(K均值）](http://blog.csdn.net/u012328159/article/details/51377896)
- [K-Means 算法](http://coolshell.cn/articles/7779.html)
