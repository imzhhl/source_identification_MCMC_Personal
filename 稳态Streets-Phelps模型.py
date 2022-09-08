# -*- coding: utf-8 -*-
# Solving ordinary differential equations (boundary value problem) with scipy.
# 参考：https://zhuanlan.zhihu.com/p/392234053
# 模型来自朱嵩博士论文《基于贝叶斯推理的环境水力反问题研究》. P79

from scipy.integrate import solve_bvp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

u = 1
D = 2
k = 0.015
c_0 = 1
l = 15

# 3. 求解微分方程边值问题，污染物衰减
# 导数函数，计算 c=[c0,c1] 点的导数 dc/dx
def func(x,c):
    # 计算 dc0/dx, dc1/dx 的值
    dc0 = c[1]  # 计算 dh0/dx
    dc1 = (k * c[0] + u * c[1]) / D # 计算 dc1/dx
    return np.vstack((dc0, dc1))

# 计算 边界条件
def bc(ca,cb):
    fa = c_0    # 边界条件：c(x=0)  = 1
    fb = 0.0    # 边界条件：c(x=15) = 0
    return np.array([ca[0]-fa,cb[0]-fb])

xa, xb = 0, 15  # 边界点 (xa=0, xb=15)
x = np.linspace(xa, xb, 1501)  # 设置网格 x 的序列
c = np.zeros((2, x.size))      # 设置函数 c 的初值

res = solve_bvp(func, bc, x, c)   # 求解 BVP
# scipy.integrate.solve_bvp(fun, bc, x, y,..)
#   fun(x, y, ..), 导数函数 f(y,x)，y在 x 处的导数。
#   bc(ya, yb, ..), 边界条件，y 在两点边界的函数。
#   x: shape (m)，初始网格的序列，起止于两点边界值 xa，xb。
#   y: shape (n,m)，网格节点处函数值的初值，第 i 列对应于 x[i]。

xSol = np.linspace(xa, xb, 1501)  # 输出的网格节点
cSol = res.sol(xSol)[0]  # 网格节点处的 h 值
plt.plot(xSol, cSol, label='c(x)')
plt.xlabel("location")
plt.ylabel("concentration")
plt.title("稳态Streets-Phelps模型，河流中污染物衰减规律")
plt.show()


