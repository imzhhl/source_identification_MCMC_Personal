"""
    马尔可夫链采样
    前提是得预先知道状态转移矩阵
    author: ZHHL
    
"""

import numpy as np
import matplotlib.pyplot as plt

transfer_matrix = np.array([[0.6,0.2,0.2],[0.3,0.4,0.3],[0,0.3,0.7]])

start_matrix = np.array([0.5,0.3,0.2])

value1 = []
value2 = []
value3 = []
value4 = []

#转移次数
n1=50
#采样样本个数
n2=200

elements=[1,2,3]

for i in range(n1+n2+1):
    #转移矩阵计算
    start_matrix = np.dot(start_matrix,transfer_matrix)
    value1.append(start_matrix[0])
    value2.append(start_matrix[1])
    value3.append(start_matrix[2])
    
    probabilities = start_matrix.tolist()   
    #采样
    sampling_value = np.random.choice(elements, 1, p=probabilities)
    value4.append(sampling_value.tolist())

value4 = sum(value4,[])
value4[-n2:]
x = np.arange(n1+n2+1)

fig,ax = plt.subplots()
fig,bx = plt.subplots()

fig.ax = ax.hist(x=value4,
                  bins=3,
                  color = 'steelblue',
                  edgecolor = 'black'       
                  )

fig.bx=bx.plot(x,value1,label='cheerful')
fig.bx=bx.plot(x,value2,label='so-so')
fig.bx=bx.plot(x,value3,label='sad')
fig.bx=bx.legend()
plt.show()
