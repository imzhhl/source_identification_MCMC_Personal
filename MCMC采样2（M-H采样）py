'''
Created on 2018年5月16日
p:输入的概率分布，离散情况采用元素为概率值的数组表示
N:认为迭代N次马尔可夫链收敛
Nlmax:马尔可夫链收敛后又取的服从p分布的样本数
isMH:是否采用MH算法，默认为True

1）输入我们任意选定的马尔科夫链状态转移矩阵Q，平稳分布π(x)，设定状态转移次数阈值n1，需要的样本个数n2
2）从任意简单概率分布采样得到初始状态值x0
3）for t=0 to n1+n2−1:
　a) 从条件概率分布Q(x|xt)中采样得到样本x∗
　b) 从均匀分布采样u∼uniform[0,1]
　c) 如果u<α(xt,x∗)=π(x∗)Q(x∗,xt), 则接受转移xt→x∗，即xt+1=x∗
　d) 否则不接受转移，即xt+1=xt
样本集(xn1,xn1+1,...,xn1+n2−1)即为我们需要的平稳分布对应的样本集。
'''


import matplotlib.pyplot as plt
import numpy as np
from array import array


Pi = np.array([0.5, 0.2, 0.3]) # 目标的概率分布
#马尔可夫链在平衡时有：π(i)P(i, j)=π(j)
#状态转移矩阵，但是不满足在平衡状态时和 Pi相符
#我们的目标是按照某种条件改造Q ，使其在平衡状态时和Pi相符
#改造方法就是，构造矩阵 P，使得有细致平衡条件：π(i)P(i, j)=π(j)P(j, i)
#对于任意的转移矩阵Q有：π(i)Q(i, j)α(i,j)=π(j)Q(j, i)α(j,i),且 P(i,j)=Q(i,j)α(i,j)                      
#α(i, j)取值有:            α(i, j) = π(j)Q(j, i)
#α(j, i)取值有:            α(j, i) = π(i)Q(i, j)
#任意转移矩阵如下:
Q = np.array([[0.9, 0.075, 0.025],
              [0.15, 0.8, 0.05],
              [0.25, 0.25, 0.5]])

N=10000
Nlmax=10000  
isMH=False 

# X0 = np.random.randint(len(Pi))# 第一步：从均匀分布（随便什么分布都可以）采样得到初始状态值x0
T = N+Nlmax-1
result = [0 for i in range(T)]
t = 0
while t < T-1:
    t = t + 1
    # 从条件概率分布Q(x|xt)中采样得到样本x∗
    # 该步骤是模拟采样，根据多项分布，模拟走到了下一个状态
    # (也可以将该步转换成一个按多项分布比例的均匀分布来采样)  
    x_cur = np.argmax(np.random.multinomial(1,Q[result[t-1]]))  # 第二步：取下一个状态，采样候选样本, x_cur = x*
    if isMH:
        '''
            细致平稳条件公式：πi*Pij = πj*Pji, ∀i,j
        '''
        a = (Pi[x_cur] * Q[x_cur][result[t-1]]) /(Pi[result[t-1]] * Q[result[t-1]][x_cur])  # 第三步：计算接受率
        acc = min(a ,1)
    else: #mcmc
        acc = Pi[x_cur] * Q[x_cur][result[t-1]]
    u = np.random.uniform(0 ,1)  # 第四步：生成阈值
    if u< acc:  # 第五步：是否接受样本
        result[t]=x_cur
    else:
        result[t]= result[t-1]
        
a = result
L = array("d")
l1 = array("d") #?
l2 = array("d") #?
for e in a:
    L.append(e)
for e in range(len(Pi)):
    l1.append(L.count(e))
for e in l1:
    l2.append(e / sum(l1))
l11 = ['state%d' %x for x in range(len(Pi))]
plt.pie(l1, labels=l11, labeldistance=0.3, autopct='%1.2f%%')
plt.title("markov:" +str(Pi))
plt.show()
