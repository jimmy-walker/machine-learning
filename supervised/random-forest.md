# Representation
**随机森林是决策树集合的术语**。

随机森林是有很多随机得决策树构成，**它们之间没有关联**。得到随机森林后，在预测时分别对每一个决策树进行判断，最后使用Bagging的思想进行结果的输出（也就是投票的思想）。

# Evalution

# Optimization

# Code
```python
#Import Library
from sklearn.ensemble import RandomForestClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Random Forest object
model= RandomForestClassifier()
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)
```
# Reference