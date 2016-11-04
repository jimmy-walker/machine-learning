#特征预处理
1. 字符串类型转成整型。使用sklearn中的LabelEncoder，Encode labels with value between 0 and n_classes-1. **注意先fit训练（输入所有字符串），然后再传入要转换的数据结构进行transform，得到最终结果。**

    ```python
>>> le = preprocessing.LabelEncoder()
>>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()
>>> list(le.classes_)
['amsterdam', 'paris', 'tokyo']
>>> le.transform(["tokyo", "tokyo", "paris"]) 
array([2, 2, 1]...)
    ```