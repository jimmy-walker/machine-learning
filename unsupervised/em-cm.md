## 问题阐述

用户点击会容易受到位置偏向的影响，排序在前的结果更容易获得用户的点击。这就是position bias effect（位置偏见效应）。

为了剔除这种效应，我们的思路就是对点击偏向作一些惩罚，比如排在前列的结果被用户跳过了，将会比后面被跳过的结果降权更多。

## 解决方案

###方案介绍

对于点击行为使用点击模型（user browsing model）预估其实际attractiveness分值。

### 原理推导

假设有三个事件：$$C_u = 1 \Leftrightarrow  E_u = 1 \  and \ A_u = 1 $$

那么对于点击的概率就可以进行分解：
$$
\begin{align}
P(C_u = 1) &= P(E_u = 1)\cdot P(A_u = 1)\\
P(A_u = 1) &= \alpha_{uq}\\
P(E_u = 1) &= \gamma_{\gamma_u}
\end{align}
$$
目的就是使用$$P(A_u = 1)$$进行排序。

我们使用user browsing model点击模型。
$$
\begin{align}
P(C_u = 1) &= P(E_r = 1 ; \ \pmb{C}_{<r})\cdot P(A_u = 1)\\
P(A_u = 1) &= \alpha_{uq}\\
P(E_r = 1;\ \pmb{C}_{<r}) &= P(E_r = 1 ;\ C_{\gamma'}=1,C_{\gamma'+1}=0,...,C_{\gamma-1}=0) = \gamma_{\gamma\gamma'}\\
\\
\gamma' &= max\{k \in \{0,...,\gamma-1\}:c_k=1\}, \ c_0 = 1
\end{align}
$$

利用EM算法对attractiveness进行推导，得到：

$$
\begin{align}
\alpha^{(t+1)}_{uq} &= \frac{1}{|S_{uq}|} \sum_{s \in S_{uq}} P(A_u = 1 |\ \pmb{C})\\
\\
S_{uq} &= \{s_q:u\in s_q\}
\end{align}
$$

其中$$P(A_u = 1|\ \pmb{C})$$推导如下：

$$
\begin{align}
P(A_u = 1|\ \pmb{C})&= P(A_u = 1 |\ C_u)\\
&= \mathcal{L}(C_u = 1)P(A_u = 1|\ C_u = 1) +  \mathcal{L}(C_u =0)P(A_u = 1|\ C_u = 0)\\
&=c_u + (1-c_u)\frac{P(C_u=0|\ A_u = 1)\cdot P(A_u = 1)}{P(C_u =0)}\\
&=c_u + (1-c_u)\frac{(1-\gamma_{\gamma\gamma'}) \alpha_{uq}}{1-\gamma_{\gamma\gamma'} \alpha_{uq}}\\
\\
 \mathcal{L}(expr)&=
  \begin{cases}
1 & \text{if expr is true,}\\
0 & \text{otherwise.}
  \end{cases}
\end{align}
$$

因此最终结果为：
$$
\begin{align}
&\alpha^{(t+1)}_{uq} = \frac{1}{|S_{uq}|} \sum_{s \in S_{uq}} \biggl(c_u^{(s)} + (1-c_u^{(s)})\frac{(1-\gamma^{(t)}_{\gamma\gamma'}) \alpha^{(t)}_{uq}}{1-\gamma^{(t)}_{\gamma\gamma'} \alpha^{(t)}_{uq}}\biggr)\\

\end{align}
$$
同理使用EM算法对examination进行推导，得到：
$$
\begin{align}
\gamma^{(t+1)}_{\gamma\gamma'} &= \frac{1}{|S_{\gamma\gamma'}|} \sum_{s \in S_{\gamma\gamma'}} P(E_r = 1 |\ \pmb{C})\\
\\
S_{\gamma\gamma'} &= \{s:c_{\gamma'}=1,c_{\gamma'+1}=0,...,c_{\gamma-1}=0\}\\
\\
P(E_r = 1|\ \pmb{C})&= P(E_r = 1 |\ C_u)\\
&= \mathcal{L}(C_u = 1)P(E_r = 1|\ C_u = 1) + \mathcal{L}(C_u =0)P(E_r = 1|\ C_u = 0)\\
&=c_u + (1-c_u)\frac{P(C_u=0|\ E_r = 1)\cdot P(E_r = 1)}{P(C_u =0)}\\
&=c_u + (1-c_u)\frac{(1-\alpha_{uq}) \gamma_{\gamma\gamma'}}{1- \alpha_{uq} \gamma_{\gamma\gamma'}}\\

\end{align}
$$

最终结果为
$$
\begin{align}
&\gamma^{(t+1)}_{\gamma\gamma'} = \frac{1}{|S_{\gamma\gamma'}|} \sum_{s \in S_{\gamma\gamma'}} \biggl(c_u^{(s)} + (1-c_u^{(s)})\frac{(1-\alpha^{(t)}_{uq}) \gamma^{(t)}_{\gamma\gamma'}}{1- \alpha^{(t)}_{uq} \gamma^{(t)}_{\gamma\gamma'}}\biggr)\\

\end{align}
$$

### EM算法

样本 $$x=(x^{(1)},x^{(2)},...x^{(m)})$$ ，对应的隐变量为 $$z=(z^{(1)},z^{(2)},...z^{(m)})$$，模型参数为 $$θ$$ , 则似然函数为 $$P(x^{(i)},z^{(i)};\theta)$$ 。

目的是找到合适的 $$\theta$$ 让对数似然函数极大。

####E步

$$
\begin{align} 
\sum\limits_{i=1}^m logP(x^{(i)};\theta) &= \sum\limits_{i=1}^m log\sum\limits_{z^{(i)}}P(x^{(i)}， z^{(i)};\theta)\\
& = \sum\limits_{i=1}^m log\sum\limits_{z^{(i)}}Q_i(z^{(i)})\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})}  \\ 
& \geq \sum\limits_{i=1}^m \sum\limits_{z^{(i)}}Q_i(z^{(i)})log\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})}  
\end{align}
$$

其中$$Q_i(z^{(i)})$$是引入的一个关于随机变量$$z^{(i)}$$的概率函数，满足$$\sum\limits_{z^{(i)}}Q_i(z^{(i)}) = 1$$

此处采用了Jensen不等式进行推导：

如果函数 $$f$$ 是凸函数， $$x$$ 是随机变量，假设有 0.5 的概率是 a，有 0.5 的概率是 b，那么：
$$E[f(x)] \ge f(E(x)) \\$$ 

特别地，如果函数 $$f$$ 是严格凸函数，当且仅当： $$p(x = E(x)) = 1$$ (即随机变量是常量) 时等号成立。

![](/assets/Jensen.jpg)

注：若函数 $$f$$ 是凹函数，Jensen不等式符号相反。

此处对数函数是凹函数：

$$log(E(y)) \ge E(log(y)) \\$$ 
其中：

$$f:\ log$$

$$E(y) = \sum\limits_i\lambda_iy_i, \lambda_i \geq 0, \sum\limits_i\lambda_i =1 $$

$$y_i = \frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})}$$

$$\lambda_i = Q_i(z^{(i)})$$

那么：$$E(log\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})}) = \sum\limits_{z^{(i)}}Q_i(z^{(i)}) log\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})} \\$$

此处设置了下界，又由于$$\sum\limits_{z^{(i)}}Q_i(z^{(i)}) = 1$$，因此这个下界可看成是$$log\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})}$$的期望。就是所谓的**Expectation**。为了取得下界，需要先固定住$$\theta$$，**J如果不固定住此$$\theta$$，那么就无法得到固定值$$Q_i(z)$$** 。然后寻找合适的 $$Q_i(z)$$ 来使得等号相等。

由 Jensen 不等式可知，等式成立的条件是随机变量是常数，则有：$$\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})} =c \\$$ 
其中 c 为常数，对于任意 $$i$$，得到：
$${P(x^{(i)}， z^{(i)};\theta)} =c{Q_i(z^{(i)})} \\$$ 
方程两边同时累加和：
$$\sum\limits_{z} {P(x^{(i)}， z^{(i)};\theta)} = c\sum\limits_{z} {Q_i(z^{(i)})} \\$$ 
因此：
$$\sum\limits_{z} {P(x^{(i)}， z^{(i)};\theta)} = c \\$$

$$Q_i(z^{(i)}) = \frac{P(x^{(i)}， z^{(i)};\theta)}{c} = \frac{P(x^{(i)}， z^{(i)};\theta)}{\sum\limits_{z}P(x^{(i)}， z^{(i)};\theta)} = \frac{P(x^{(i)}， z^{(i)};\theta)}{P(x^{(i)};\theta)} = P( z^{(i)}|x^{(i)};\theta) \\$$

 $$Q(z)$$是已知样本和模型参数下的隐变量分布。

#### M步

由上所述，需要极大化下式：$$arg \max \limits_{\theta} \sum\limits_{i=1}^m \sum\limits_{z^{(i)}}Q_i(z^{(i)})log\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})} \\$$ 

这就是所谓的**Maximization** 。

固定 $$Q_i(z^{(i)})$$ 后，调整 $$\theta$$，去极大化$$logL(\theta)$$的下界。

去掉上式中常数的部分 $$Q_i(z^{(i)})$$ ，则需要极大化的对数似然下界为：$$arg \max \limits_{\theta} \sum\limits_{i=1}^m \sum\limits_{z^{(i)}}Q_i(z^{(i)})log{P(x^{(i)}， z^{(i)};\theta)} \\$$

#### EM在点击模型中的应用

$$
\begin{align}
&\sum\limits_{i=1}^m \sum\limits_{z^{(i)}}Q_i(z^{(i)})log{P(x^{(i)}， z^{(i)};\theta)}\\ 
&= \sum\limits_{i=1}^m E_{z^{(i)}|x^{(i)};\theta}[log{P(x^{(i)}， z^{(i)};\theta)}]\\
&\Rightarrow  \sum\limits_{s \in S} E_{\textbf{X}|\textbf{C}^{(s)};\Psi}[log{P(\textbf{C}^{(s)},\textbf{X};\Psi)}]\\
&= \sum\limits_{s \in S} E_{\textbf{X}|\textbf{C}^{(s)};\Psi}[log{\prod_{c_{i} \in s}P(\textbf{X}_{c_i}^{(s)};\Psi)}]\\
&= \sum\limits_{s \in S} E_{\textbf{X}|\textbf{C}^{(s)};\Psi}[\sum_{c_{i} \in s} log{P(\textbf{X}_{c_i}^{(s)};\Psi)}]\\
&= \sum\limits_{s \in S} E_{\textbf{X}|\textbf{C}^{(s)};\Psi}[\sum_{c_{i} \in s} log{P(X_{c_i}^{(s)}, \mathcal{P}(X_{c_i}^{(s)})=\pmb{p})}]\\
&= \sum\limits_{s \in S} E_{\textbf{X}|\textbf{C}^{(s)};\Psi}[\sum_{c_{i} \in s} log({P(X_{c_i}^{(s)}| \mathcal{P}(X_{c_i}^{(s)})=\pmb{p})\cdot P(\mathcal{P}(X_{c_i}^{(s)})=\pmb{p})}]\\
&= \sum\limits_{s \in S} E_{\textbf{X}|\textbf{C}^{(s)};\Psi}[\sum_{c_{i} \in s}\biggl(  \mathcal{L}(X_{c_i}^{(s)}=1, \mathcal{P}(X_{c_i}^{(s)})=\pmb{p}) log(\theta_c) + \mathcal{L}(X_{c_i}^{(s)}=0, \mathcal{P}(X_{c_i}^{(s)})=\pmb{p}) log(1-\theta_c) \biggl) + \mathcal{Z} ]\\
\end{align}
$$

将上式记录为$$Q(\theta_c)$$。其中$$P(X|\mathcal{P}{X}=\pmb{p}) \backsim Bernoulli(\theta) $$，因此对于点击行为c对应的参数为$$\theta_c$$，$$\mathcal{Z}$$表示无关项。

对于每一个点击行为其相应的参数，都进行导数求导，令其为0。$$\frac{\partial{Q(\theta_c)}}{\partial{\theta_c}}=0$$

得到结果为：
$$
\begin{align}
\theta_{c}^{(t+1)} &= \frac{ \sum_{s\in S} \sum_{c_i\in s} P(X_{c_i}^{(s)}=1, \mathcal{P}(X_{c_i}^{(s)})=\pmb{p}) | \pmb{C}^{(s)};\Psi) }{\sum_{s\in S} \sum_{c_i\in s} P(\mathcal{P}(X_{c_i}^{(s)})=\pmb{p}) | \pmb{C}^{(s)};\Psi)}\\

\alpha^{(t+1)}_{uq} &= \frac{ \sum_{s\in S_{uq}}  P(A_u = 1, \mathcal{P}(A_{u})=\pmb{p} | \pmb{C}) }{ \sum_{s\in S_{uq}}  P(\mathcal{P}(A_{u})=\pmb{p} | \pmb{C}) }\\
&=\frac{ \sum_{s\in S_{uq}}  P(A_u = 1 |\ \pmb{C}) }{ \sum_{s\in S_{uq}}  1 }\\

\gamma^{(t+1)}_{\gamma\gamma'} &= \frac{ \sum_{s\in  S_{\gamma\gamma'}}  P(E_r = 1, \mathcal{P}(E_r)=\pmb{p} | \pmb{C}) }{ \sum_{s\in  S_{\gamma\gamma'}}  P(\mathcal{P}(E_r)=\pmb{p} | \pmb{C}) }\\
&=\frac{ \sum_{s\in  S_{\gamma\gamma'}} P(E_r = 1 |\ \pmb{C}) }{ \sum_{s\in  S_{\gamma\gamma'}}  1 }\\

\gamma^{(t+1)}_{\gamma\gamma'} &= \frac{1}{|S_{\gamma\gamma'}|} \sum_{s \in S_{\gamma\gamma'}} P(E_r = 1 |\ \pmb{C})\\

\end{align}
$$
对于实际场景下，每一个对话s中只有一个关于文档或位置的点击，因此省略$$c_i \in s$$的计算。额外参数$$\psi$$可以去除。

对于$$A_u$$来说并没有父节点的约束，因此直接去除$$\mathcal{P}(A_u)$$的影响。

对于$$E_r$$来说其父节点的约束为$$\mathcal{P}(E_r) = {C_{\gamma'},...,C_{\gamma-1}}$$，对应的值为$$\pmb{p}=[1,0,...,0]$$，对于$$S_{\gamma\gamma'}$$会话而言，因此$$ \mathcal{P}(E_r)=\pmb{p}$$ ，因此$$ P(\mathcal{P}(E_r)=\pmb{p} | \pmb{C}) = 1$$，而对于其他会话而言，因此$$ \mathcal{P}(E_r)\ \neq\pmb{p}$$ ，因此$$ P(\mathcal{P}(E_r)=\pmb{p} | \pmb{C}) = 0$$。

此外
$$
\begin{align}
&P(E_r = 1, \mathcal{P}(E_r)=\pmb{p} | \pmb{C})\\
=&P(E_r = 1|\mathcal{P}(E_r)=\pmb{p} , \pmb{C}) \cdot P(\mathcal{P}(E_r)=\pmb{p} | \pmb{C}) \\
=&P(E_r = 1 |\ \pmb{C})
\end{align}
$$

###贝叶斯平均
对于小曝光的数据进行贝叶斯平均
$$
\begin{align}
&\frac{ numerator }{ denominator }\\
\Rightarrow & \frac{ numerator + bayes\_sum * bayes\_value }{ denominator + bayes\_sum}\\
\end{align}
$$

## Reference

- [人人都懂EM算法](https://zhuanlan.zhihu.com/p/36331115 )