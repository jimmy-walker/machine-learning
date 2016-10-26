# Representation

1. 贝叶斯定理：**已知某条件概率，如何得到两个事件交换后的概率，也就是在已知P(B|A)的情况下如何求得P(A|B)**。

  因为事件A和B同时发生的概率为（在A发生的情况下发生B）或者（在B发生的情况下发生A）：$$P(A \cap B) = P(A)*P(B|A) = P(B)*P(A|B)$$

    那么可以得到：$$P(A|B)=\frac{P(B|A)*P(A)}{P(B)}$$

    由此得到启示，可以利用贝叶斯定理进行分类，**对于给出的待分类项，求解在此项出现的条件下各个目标类别出现的概率，哪个最大，就认为此待分类项属于哪个类别**。

2. 朴素贝叶斯法是基于贝叶斯定理与特征条件独立假设的分类方法。对于给定的训练数据集，首先基于特征条件独立假设学习输入\/输出的联合概率分布；然后基于此模型，对于给定的输入x，利用贝叶斯定理求出后验概率最大的输出y。

    输入: 线性可分训练集$$T={(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)}$$, 其中$$x_i\in{\scr {X}}=R^n, y_i\in{\scr {Y}}={c_1,\cdots,c_K}$$，$$X,Y$$分别是定义在$${\scr{X,Y}}$$上的随机向量和随机变量。假设$$x^{(j)}$$可取值有$$S_j$$个。

    **注意：随机变量的意义在于产生一系列样本，因为不能说样本属于某集合。**$$x_i=(x_i^{(1)},x_i^{(2)},\cdots,x_i^{(n)})^T \\x—_i^{(j)}=(a_{j1},a_{j2},\cdots,a_{jS_j})$$

    先验概率分布：$$P(Y=c_k), k = 1,2,...,K$$

    条件概率分布：$$P(X=x \mid Y=c_k)=P(X^{(1)}=x^{(1)},...,X^{(n)}=x^{(n)}|Y=c_k)$$

    后验概率分布：$$P(Y=c_k\mid X=x)=\frac{P(X=X\mid Y=c_k)P(Y=c_k)}{P(X)}\\qquad\qquad\qquad\qquad =\frac{P(X=x\mid Y=c_k)P(Y=c_k)}{\sum_kP(X=x\mid Y=c_k)P(Y=c_k)}\\qquad\qquad\qquad\stackrel{条件独立性}=\frac{P(Y=c_k)\prod_j P(X^{(j)}=x^{(j)}\mid Y=c_k)}{\sum_kP(Y=c_k)\prod_jP(X^{(j)}=x^{(j)}\mid Y=c_k)}$$

    

# Evalution

# Optimization
1. 判别模型与生成模型：**建模及优化的对象不一样，一个是条件概率，一个是联合概率。**

    1. 判别模型：
    
    简单的说就是分类的最终结果是以某个函数或者是假设函数的取值范围来表示它属于那一类的，例如H(x)>0就是第一类，H(x)<0。**判别模型主要对p(y|x)建模（优化条件概率分布），通过x来预测y**。在建模的过程中不需要关注联合概率分布。只关心如何优化p(y|x)使得数据可分。通常，判别模型在分类任务中的表现要好于生成模型。但判别模型建模过程中通常为有监督的，而且难以被扩展成无监督的。比如决策树、逻辑回归、SVM等。

    2. 生成模型：

    该模型对观察序列的**联合概率分布p(x,y)建模（优化训练数据的联合分布概率）**，在获取联合概率分布之后，可以通过贝叶斯公式得到条件概率分布。生成式模型所带的信息要比判别模型更丰富。除此之外，生成模型较为容易的实现增量学习。比如朴素贝叶斯等。

2. 

# Code

```python
#Import Library
from sklearn.naive_bayes import GaussianNB
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object model = GaussianNB() 
# there is other distribution for multinomial classes like Bernoulli Naive Bayes, Refer link
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)
```

# Reference

* [朴素贝叶斯法](http://www.wengweitao.com/po-su-bei-xie-si-fa.html)

