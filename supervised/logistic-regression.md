# Representation
不要被这个名字给混淆，**这是一种分类而不是回归算法**。简单来讲，它通过把给定的数据输入进一个评定模型方程来预测一个事件发生的可能性。因此它又被称为逻辑回归模型。 因为它是预测可能性的，它的输出值介于0和1之间。
# Evalution

# Optimization

# Code
```python
#Import Library
from sklearn.linear_model import LogisticRegression
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create logistic regression object
model = LogisticRegression()
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)
#Predict Output
predicted= model.predict(x_test)
```
# Reference
