# Representation

PCA（Principal component analysis），主成分分析。PCA的思想是将n维特征映射到k维上（k&lt;n），这k维是全新的正交特征。这k维特征称为主元，是重新构造出来的k维特征，而不是简单地从n维特征中去除其余n-k维特征。

PCA是一种统计技术，**经常应用于人面部识别和图像压缩以及信号去噪等领域**，是在高维数据中提取模式的一种常用技术。

例子：由图，我们用三个维度X,Y,Z去描述这个三维空间中的点。然后，当我们仔细观察这些点后，发现这些点几乎都在如图的蓝色平面上，只有很少的点在蓝色平面外。对于蓝色平面，我们建立正交的坐标向量X,Y，同时对于蓝色平面外的点，我们增加坐标向量Z去描述。在这个问题中，\[X,Y\]可以说是问题的真正输入模式，是主成分，即用减少后的2维平面同样可以很好地描述原数据集。

![](/assets/PCA.jpg)

1. 协方差矩阵

    协方差\(Covariance\)用于衡量两个变量的总体误差。设两个随机变量$$X$$和$$Y$$的期望值分别为$$\mathrm{E}(X)$$和$$\mathrm{E}(Y)$$，则其协方差定义为：$$\operatorname{cov}(X, Y) = \mathrm{E}((X - \mathrm{E}(X)) (Y - \mathrm{E}(Y)))$$

  ```
  X 1.1 1.9 3
  Y 5.0 10.4 14.6
  E(X) = (1.1+1.9+3)/3=2
  E(Y) = (5.0+10.4+14.6)/3=10
  E(XY)=(1.1*5.0+1.9*10.4+3*14.6)/3=23.02
  Cov(X,Y)=E(XY)-E(X)E(Y)=23.02-2*10=3.02
  ```

    方差是协方差的一种特殊情况，即当两个变量是相同时：$$\operatorname{var}(X) = \operatorname{cov}(X, X) = \mathrm{E}((X - \mathrm{E}(X))^2)$$

    协方差从随机标量推广到随机向量，则得到协方差矩阵\(Covariance Matrix\)定义：$$\operatorname{cov}(\mathrm{X}) = \mathrm{E}((\mathrm{X}-\mathrm{E}(\mathrm{X}))(\mathrm{X}-\mathrm{E}(\mathrm{X}))^\mathsf{T})$$

  其中，$$X$$是一个随机向量。显然一个随机向量的协方差矩阵是一个方阵。`协方差矩阵是一个矩阵，其每个元素是向量元素之间的协方差`。计算协方差矩阵时候，计算的是不同维度之间的协方差，比如$$X$$中的$$x_i$$与$$x_j$$。

2. 线性变换、特征值和特征向量

  线性变换\(线性映射\)是在作用于两个向量空间之间的函数，它保持向量加法和标量乘法的运算。实际上线性变换表现出来的就是一个矩阵。

  特征值和特征向量是一体的概念：对于一个给定的线性变换，它的特征向量$$\xi$$经过这个线性变换之后，得到的新向量仍然与原来的$$\xi$$保持在同一條直線上，但其长度也许會改变。`一个特征向量的长度在该线性变换下缩放的比例称为其特征值（本征值）`。数学描述如下：$$\mathbf{A} \xi = \lambda \xi$$

  在线性变换$$\mathbf{A}$$的作用下，向量$$\xi$$仅仅在尺度上变为原来的$$\lambda$$倍。称$$\xi$$是线性变换$$\mathbf{A}$$的一个特征向量，$$\lambda$$ 是对应的特征值。

  求解线性变换$$\mathbf{A}$$的特征向量和特征值：$$\begin{align} & \mathbf{A} \mathrm{x} = \lambda \mathrm{x} \\ \Rightarrow & \mathbf{IA} \mathrm{x} =\mathbf{I} \cdot \lambda \mathrm{x} \\ \Rightarrow & (\mathbf{A} - \lambda \mathbf{I}) \mathrm{x} = 0 \end{align}$$

  根据线性方程组理论，如果上式有非零解，则矩阵$$(\mathbf{A} - \lambda \mathbf{I})$$的行列式为0：$$|\mathbf{A} - \lambda \mathbf{I}| = 0$$

  该方程组称作矩阵的特征多项式，解该方程组可以求得所有的特征值$$\lambda$$。矩阵 $$\mathbf{A}$$的非零特征值最大数目是该矩阵的秩 $$\operatorname{rank}(\mathbf{A})$$。对于每个特征值$$\lambda_i$$都有如下特征方程\(Characteristic equation\)成立：$$(\mathbf{A} - \lambda_i \mathbf{I}) \mathrm{x} = 0$$进一步可以解得相应的特征向量$$X$$。

  顾名思义，特征值和特征向量表达了一个线性变换的特征。**在物理意义上，一个高维空间的线性变换可以想象是在对一个向量在各个方向上进行了不同程度的变换**，而特征向量之间是线性无关的，它们对应了最主要的变换方向，同时特征值表达了相应的变换程度。

3. 特征值分解

  矩阵对角化定理\(Matrix diagonalization theorem\)：对于$$N \times N$$方阵 $$\mathbf{A}$$ ，如果它有$$N$$个线性无关的特征向量，那么存在一个特征分解：$$\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{-1}$$

  其中，$$\mathbf{Q}$$是$$N \times N$$的方阵，且其第$$i$$列为 $$\mathbf{A}$$的特征向量$$\mathrm{q}_i$$ 。$$\mathbf{\Lambda}$$是对角矩阵，其对角线上的元素为对应的特征值，即$$\mathbf{\Lambda}_{ii}=\mathbf{\lambda}_i$$。

  对称对角化定理\(Symmetric diagonalization theorem\)：更进一步，如果方阵$$\mathbf{A}$$是对称方阵，可得$$\mathbf{Q}$$的每一列都是$$\mathbf{A}$$的互相正交且归一化（单位长度）的特征向量，即$$\mathbf{Q}^{-1} = \mathbf{Q}^\mathsf{T}$$ 。

4. 主成分分析

  1. 数据规范化：将数据规范为均值为0，方差为1的数据。

  2. 计算协方差矩阵：$$\Sigma {\rm{ = }}{1 \over {\rm{m}}}{X^T}X$$

  3. 计算协方差矩阵的特征向量和特征值

  4. 组成特征向量矩阵：将特征值按照从大到小的顺序排序，选择其中最大的k个，然后将其对应的k个特征向量分别作为列向量组成特征向量矩阵。

  5. 将样本点投影（即相乘）到选取的特征向量上。假设样例数为m，特征数为n，减去均值后的样本矩阵为DataAdjust\(m\*n\)，协方差矩阵是n\*n，选取的k个特征向量组成的矩阵为EigenVectors\(n\*k\)。那么投影后的数据`FinalData(m*k) = DataAdjust(m*n)×EigenVectors(n*k)`，这样就将原始样例的n维特征变成了k维，这k维就是原始特征在k维上的投影。（**J记住该公式，协方差矩阵的特征向量就是k维理想特征，而最大特征值得特征向量即是数据的主要成分或叫主成分**）



# Code

```python
#Import Library
from sklearn import decomposition
#Assumed you have training and test data set as train and test
# Create PCA obeject pca= decomposition.PCA(n_components=k)
#default value of k =min(n_sample, n_features)
# For Factor analysis
#fa= decomposition.FactorAnalysis()
# Reduced the dimension of training dataset using PCA
train_reduced = pca.fit_transform(train)
#Reduced the dimension of test dataset
test_reduced = pca.transform(test)
```

```python
sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False)
```

n\_components : int, float, None or string：**若赋值为int，则表示PCA算法中所要保留的主成分个数n，也即保留下来的特征个数k**

copy : bool \(default True\)：表示是否在运行算法时，将原始训练数据复制一份。

whiten : bool, optional \(default False\)：**白化**，使得每个特征具有相同的方差，因为许多点具有强相关性，降低输入冗余性，**特别用于图像中**。

fit\_transform\(X\)：用X来训练PCA模型，同时返回降维后的数据。

使用实例：

```
>>> from sklearn.decomposition import PCA
>>> pca=PCA(n_components=1)
>>> newData=pca.fit_transform(data)
>>> newData
array([[-2.12015916],
 [-2.22617682],
 [-2.09185561],
 [-0.70594692],
 [-0.64227841],
 [-0.79795758],
 [ 0.70826533],
 [ 0.76485312],
 [ 0.70139695],
 [ 2.12247757],
 [ 2.17900746],
 [ 2.10837406]])
>>> data
array([[ 1. , 1. ],
 [ 0.9 , 0.95],
 [ 1.01, 1.03],
 [ 2. , 2. ],
 [ 2.03, 2.06],
 [ 1.98, 1.89],
 [ 3. , 3. ],
 [ 3.03, 3.05],
 [ 2.89, 3.1 ],
 [ 4. , 4. ],
 [ 4.06, 4.02],
 [ 3.97, 4.01]])
```

# Reference

* [ K-means\(K均值）](http://blog.csdn.net/u012328159/article/details/51377896)
* [K-Means 算法](http://coolshell.cn/articles/7779.html)
* [从协方差到PCA算法步骤详解](http://blog.csdn.net/fngy123/article/details/45153163)
* [scikit-learn中PCA的使用方法](http://blog.csdn.net/u012162613/article/details/42192293)

