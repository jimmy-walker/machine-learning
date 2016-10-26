# Representation

1. 贝叶斯定理：**已知某条件概率，如何得到两个事件交换后的概率，也就是在已知P(B|A)的情况下如何求得P(A|B)**。

  因为事件A和B同时发生的概率为（在A发生的情况下发生B）或者（在B发生的情况下发生A）：$$P(A \cap B) = P(A)*P(B|A) = P(B)*P(A|B)$$

    那么可以得到：$$P(A|B)=\frac{P(B|A)*P(A)}{P(B)}$$

    可以利用贝叶斯定理进行分类，**对于给出的待分类项，求解在此项出现的条件下各个目标类别出现的概率，哪个最大，就认为此待分类项属于哪个类别**。
2. 

朴素贝叶斯法是基于贝叶斯定理与特征条件独立假设的分类方法。对于给定的训练数据集，首先基于特征条件独立假设学习输入\/输出的联合概率分布；然后基于此模型，对于给定的输入x，利用贝叶斯定理求出后验概率最大的输出y。

# Evalution

# Optimization
1. 判别模型与生成模型

    1. 判别模型：
    
    简单的说就是分类的最终结果是以某个函数或者是假设函数的取值范围来表示它属于那一类的，例如H(x)>0就是第一类，H(x)<0。该模型主要对p(y|x)建模，通过x来预测y。在建模的过程中不需要关注联合概率分布。只关心如何优化p(y|x)使得数据可分。通常，判别式模型在分类任务中的表现要好于生成式模型。但判别模型建模过程中通常为有监督的，而且难以被扩展成无监督的。

    2. 生成模型

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

