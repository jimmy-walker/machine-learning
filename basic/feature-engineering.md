#特征预处理
1. 非数值类型转成数值类型。使用sklearn中的LabelEncoder，Encode labels with value between 0 and n_classes-1. 

    **注意先fit训练（输入所有字符串），然后再传入要转换的数据结构进行transform，得到最终结果。**

    ```python
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"]) 
    array([2, 2, 1]...)
    ```

2. 数值类型转成二进制数字，消除潜在的邻近性。使用sklearn中的OneHotEncoder， Encode categorical integer features using a one-hot aka one-of-K scheme.

    ```python
    onehot = OneHotEncoder()
    X_teams = onehot.fit_transform(X_teams).todense()
    ```

