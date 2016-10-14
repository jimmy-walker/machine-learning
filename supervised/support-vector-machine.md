# Representation

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