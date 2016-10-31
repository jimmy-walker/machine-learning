# Representation
如果概率模型的变量都是观测变量，那么给定数据之后就可以直接使用极大似然法或者贝叶斯估计模型参数。

但是当模型含有隐含变量的时候就不能简单的用这些方法来估计，EM就是一种含有隐含变量的概率模型参数的极大似然估计法。

EM算法是一种迭代算法，用于含有隐变量(hidden variable)的概率模型参数的极大似然估计，或极大后验概率估计。EM算法的每次迭代由两步组成：E步，求期望(expectation)；M步，求极大( maximization )，所以这一算法称为期望极大算法(expectation maximization algorithm)，简称EM算法。

# Reference
- [EM算法学习笔记](http://blog.csdn.net/mytestmy/article/details/38778147)
- [从最大似然到EM算法浅解](http://blog.csdn.net/zouxy09/article/details/8537620)
- [统计学习方法 李航---第9章 EM算法及其推广](http://blog.csdn.net/demon7639/article/details/51011424)
- [EM算法简介](https://ask.julyedu.com/article/73)
