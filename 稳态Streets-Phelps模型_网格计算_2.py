# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 22:31:33 2022

@author: zhhl_
"""
# 与上一版相比，使用了更多循环数组，使用了似然函数的封装，并且是为了测试fluent的数据做的预计算

from scipy.integrate import solve_bvp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
from mpl_toolkits.mplot3d import Axes3D

D = np.linspace(1.5,2.5,101)
k = np.linspace(0.01,0.02,101)
D, k = np.meshgrid(D, k)

def function(D = 2, k = 0.015):
    u = 1
    c_0 = 1
    L = 15
    loc_value = []
        
    def func(x,c):
        # 计算 dc0/dx, dc1/dx 的值
        dc0 = c[1]  # 计算 dh0/dx
        dc1 = (k * c[0] + u * c[1]) / D # 计算 dc1/dx
        return np.vstack((dc0, dc1))
       
    def bc(ca,cb):
        fa = c_0    # 边界条件：c(x=0)  = 1
        fb = 0.0    # 边界条件：c(x=15) = 0
        return np.array([ca[0]-fa,cb[0]-fb])
    
    xa, xb = 0, L  # 边界点 (xa=0, xb=15)
    x = np.linspace(xa, xb, 151)  # 设置网格 x 的序列
    c = np.zeros((2, x.size))      # 设置函数 c 的初值
    
    res = solve_bvp(func, bc, x, c)   # 求解 BVP
    # scipy.integrate.solve_bvp(fun, bc, x, y,..)
    #   fun(x, y, ..), 导数函数 f(y,x)，y在 x 处的导数。
    #   bc(ya, yb, ..), 边界条件，y 在两点边界的函数。
    #   x: shape (m)，初始网格的序列，起止于两点边界值 xa，xb。
    #   y: shape (n,m)，网格节点处函数值的初值，第 i 列对应于 x[i]。
    
    xSol = np.linspace(xa, xb, 151)  # 输出的网格节点
    cSol = res.sol(xSol)[0]  # 网格节点处的 h 值

    for i,j in zip([0, 1, 2, 3], [3, 6, 9, 12]):
        loc_value.append(np.array(list(zip(xSol,cSol)))[j*10,1])
        
    return(loc_value)

    # plt.plot(xSol, cSol, label='c(x)')
    # plt.xlabel("location")
    # plt.ylabel("concentration")
    # plt.title("稳态Streets-Phelps模型，河流中污染物衰减规律")
    # plt.show()
    
# 用于计算似然函数
def likelihood_func(sigma, c_predict, c_monitor):
    temp = 0
    for i in range(len(c_predict)):   
        temp = temp + (-(c_predict[i] - c_monitor[i])**2/(2*sigma**2))
    return (np.exp(temp))  
  
c_ture = [0.96, 0.91, 0.84, 0.67]
c_predict = np.zeros([4,101,101])


for i in range(0, 101, 1):
    for j in range(0, 101, 1):
        value = function(D = D[i,j], k = k[i,j])        
        for z in range(0, 4, 1):
            c_predict[z,i,j] = value[z]        

sigma = likelihood_func(0.01, c_predict, c_ture)

fig = plt.figure()
ax = Axes3D(fig)

surf = ax.plot_surface(D, k, sigma, rstride=1, cstride=1, edgecolor='black', cmap='rainbow')

font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}

ax.set_xlabel( 'D',font )
ax.set_ylabel( 'k',font )
ax.set_zlabel( 'Post-PDF',font )

# 设置标题
plt.title("A figure of 3D")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

temp_value_x = np.amax(sigma, axis=1)   #行最大值
temp_value_y = np.amax(sigma, axis=0)   #列最大值
temp_index_x = np.argmax(sigma, axis=1) #行最大值索引
temp_index_y = np.argmax(sigma, axis=0) #列最大值索引

temp_x = np.array(list(zip(temp_index_x,temp_value_x)))
temp_y = np.array(list(zip(temp_index_y,temp_value_y)))

D_find = temp_x[(np.argmax(temp_x, axis=0)[1]),0]/100 * 0.01 + 0.01
k_find = temp_y[(np.argmax(temp_y, axis=0)[1]),0]/100 * 1.0  + 1.5
print(D_find)
print(k_find)
