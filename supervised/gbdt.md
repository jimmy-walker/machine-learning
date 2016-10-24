# Representation
GBDT这个算法有很多名字，但都是同一个算法：
GBRT (Gradient BoostRegression Tree) 渐进梯度回归树
GBDT (Gradient BoostDecision Tree) 渐进梯度决策树
MART (MultipleAdditive Regression Tree) 多决策回归树
Tree Net决策树网络

# Evalution
# Optimization

# Code
```python
#Import Library
from sklearn.ensemble import GradientBoostingClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Gradient Boosting Classifier object
model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)
```
# Reference