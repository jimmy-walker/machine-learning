# Representation
1. 支持向量机，因其英文名为support vector machine，故一般简称SVM，通俗来讲，它是一种二类分类模型，**其基本模型定义为特征空间上的间隔最大的线性分类器，其学习策略（即评价）便是间隔最大化，最终可转化为一个凸二次规划问题的求解（即优化）**。

2. **记住此模型**：SVM的基本想法就是求解能正确划分训练样本并且其几何间隔最大化的超平面。
   
    SVM的超平面：$$w^Tx+b=0$$

    分类决策函数是：$$f(x)=sign(wx+b)$$

3. 其与logistic regression的区别在于，logistic regression需要学习到$$\theta$$，使得正例的特征远大于0，负例的特征远小于0，**强调在全部训练实例上达到这个目标**，而**SVM更关心靠近中间分割线上的点，不要求在所有点上达到最优**。在形式上，SVM使用$$b$$代替$$\theta_0$$，由于$$x_0=1$$，所以得到$$\theta^Tx=w^Tx+b$$。

4. 支持向量机一共分为三种情况：
    - **线性可分支持向量机**：针对训练数据线性可分
        
        硬间隔最大化 (hard margin maximization)
    - **线性支持向量机**：针对训练数据近似线性可分
        
        软间隔最大化 (soft margin maximization)
    - **非线性支持向量机**：针对训练数据线性不可分
        
        核函数 (kernel function)

# Evalution
1.函数间隔代表我们认为特征是正例还是反例的确信度

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
- [支持向量机SVM](https://clyyuanzi.gitbooks.io/julymlnotes/content/svm.html)
- [C-SVM模型](https://json0071.gitbooks.io/svm/content/c-svm.html)
- 