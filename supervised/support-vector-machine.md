# Representation
支持向量机，因其英文名为support vector machine，故一般简称SVM，通俗来讲，它是一种二类分类模型，**其基本模型定义为特征空间上的间隔最大的线性分类器，其学习策略便是间隔最大化，最终可转化为一个凸二次规划问题的求解**。

SVM的超平面
$$w^Tx+b=0$$

SVM的基本想法就是求解能正确划分训练样本并且其几何间隔最大化的超平面。

# Evalution

# Optimization

# Code
```python
#Import Library
from sklearn import svm
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.svc() # there is various option associated with it, this is simple for classification. You can refer the link(http://scikit-learn.org/stable/modules/svm.html), for more detail.
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(x_test)
```
在sklearn中，svm.svc()不需要设置参数，直接使用即可。

# Reference
- [支持向量机通俗导论（理解SVM的三层境界）](http://blog.csdn.net/v_july_v/article/details/7624837)
- [机器学习常见算法个人总结（面试用）](http://kubicode.me/2015/08/16/Machine%20Learning/Algorithm-Summary-for-Interview/)