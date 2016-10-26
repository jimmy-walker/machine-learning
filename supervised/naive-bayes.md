# Representation
1. 贝叶斯定理：**已知某条件概率，如何得到两个事件交换后的概率，也就是在已知$$P(A|B)$$的情况下如何求得$$P(B|A)$$**。
    $$P(A \cap B) = P(A)*P(B|A) = P(B)*P(A|B)$$
    
2. 

朴素贝叶斯法是基于贝叶斯定理与特征条件独立假设的分类方法。对于给定的训练数据集，首先基于特征条件独立假设学习输入/输出的联合概率分布；然后基于此模型，对于给定的输入x，利用贝叶斯定理求出后验概率最大的输出y。

# Evalution

# Optimization

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