# -*- coding: utf-8 -*-
"""
author： Hongliang Zhang - WHU
date：   2022-09-26
log: 1. 2022-09-26 伴随方法进行采用，成功
     2. 2022-09-26 封装下函数，注释一下
     3. 写清晰的代码而不是简洁的代码
     ______  _    _   _    _   _      
    |___  / | |  | | | |  | | | |     
       / /  | |__| | | |__| | | |     
      / /   |  __  | |  __  | | |     
     / /__  | |  | | | |  | | | |____ 
    /_____| |_|  |_| |_|  |_| |______|

"""
#%% 导入库
import ansys.fluent.core as pyfluent
import numpy as np
import random
import time
import re
import os
from decimal import Decimal
from tqdm import tqdm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.stats import norm
import sys
from scipy.integrate import solve_bvp
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import csv
from scipy.interpolate import griddata 
import scipy.stats as stats 
import pymc as pm

matplotlib.rc("font", family='Microsoft YaHei')

#%% 一些需要的函数定义

# 将正态分布作为建议分布, 注意设置截断位置参数
def normal_proposal_distribution(t, trunc_X = [0, 1000], trunc_Y = [0, 1000], sigma_X = 0, sigma_Y = 0, X_history = [], Y_history = []):
    # 设置截断
    x_lower = trunc_X[0]; x_upper = trunc_X[1] 
    # 截断正态分布采样
    X_new = stats.truncnorm.rvs((x_lower - X_history[t - 1]) / sigma_X, (x_upper - X_history[t - 1]) / sigma_X, loc=X_history[t - 1], scale=sigma_X) #建议分布
    # 设置截断
    y_lower = trunc_Y[0]; y_upper = trunc_Y[1]
    # 截断正态分布采样
    Y_new = stats.truncnorm.rvs((y_lower - Y_history[t - 1]) / sigma_Y, (y_upper - Y_history[t - 1]) / sigma_Y, loc=Y_history[t - 1], scale=sigma_Y) #建议分布
 
    # 设置采样值保留的小数位数 
    X_new = Decimal(X_new).quantize(Decimal("0.0"))
    Y_new = Decimal(Y_new).quantize(Decimal("0.0"))
    
    # 返回值为新的污染物坐标 
    return float(X_new), float(Y_new)
      
# 将均匀分布作为建议分布, 注意设置截断位置参数
def uniform_proposal_distribution(t, trunc_X = [0, 1000], trunc_Y = [0, 1000], delta_X = 0, delta_Y = 0, X_history =[], Y_history = []):
    #建议分布为均匀分布采样
    X_new = random.uniform(X_history[t - 1] - delta_X, X_history[t - 1] + delta_X)   
    Y_new = random.uniform(Y_history[t - 1] - delta_Y, Y_history[t - 1] + delta_Y)

    # X坐标超边界罚回
    while X_new < trunc_X[0] or Y_new > trunc_X[1]:
        X_new = random.uniform(X_history[t - 1] - delta_X, X_history[t - 1] + delta_X)
    # Y坐标超边界罚回
    while Y_new < trunc_Y[0] or Y_new > trunc_Y[1]:
        Y_new = random.uniform(Y_history[t - 1] - delta_Y, Y_history[t - 1] + delta_Y)
        
    # 设置采样值保留的小数位数       
    X_new = Decimal(X_new).quantize(Decimal("0.0"))
    Y_new = Decimal(Y_new).quantize(Decimal("0.0"))
    
    # 返回值为新的污染物坐标    
    return float(X_new), float(Y_new)

# 用于计算似然函数, c_predict和c_true为存储预测点值和检测点值的数组
def likelihood_func(sigma, c_predict = [], c_true  = []):
    temp = 0
    for i in range(len(c_predict)):   
        temp = temp + (-(c_predict[i] - c_true[i])**2/(2*sigma**2))
    # 返回似然函数(无系数部分)
    return np.exp(temp)

# 用于计算伴随浓度场在预测点(X_current, Y_current)处的浓度值
def adjoint_results(X_current, Y_current):
    # 用于存储新预测位置下,各监测点处的值(对偶)
    data_predict_new = []
    for i in range(1, 10):
        data_predict_new.append(uds_adjoint[i][int(10*Y_current)-1][int(10*X_current)-1])
        
    # 返回存储各监测点浓度值的列表
    return data_predict_new

# 用于提取uds_0在监测点处的浓度值(真实值)
def direct_results(points_list):  
    # 保存监测点处的真实浓度值
    point_true_value = []
    for i in range(len(points_list)):
        point_true_value.append(uds_adjoint[0][10*points_list[i][1]-1][10*points_list[i][0]-1])
    # 浓度列表返回
    return point_true_value

# 从fluent中导出的数据导入到numpy数组中
def file_to_array():

    # 保存x坐标, 和y坐标的列表
    x=[]; y=[]

    # 创建10个二维空数组
    uds = [[] for i in range(10)]
    
    # 创建一维列表，用于存储二维np数组，进而转化为三维np数组
    uds_t = []

    # 读取csv文件，当然也可以读取类似txt之类的文件
    with open(r'F:/ZHHL/TE_Doctor/CASES/case220915/adjoint_method/fluent-adjoint-0924','r')  as  csvfile:   
        #指定分隔符为","，因为我们刚才导出时就是逗号
        plots=csv.reader(csvfile,delimiter=',')
        #循环读取文件各列
        for row in plots:
            #为了跳过文件前面的非数据行  
            if plots.line_num == 1:
                continue
            # 读取x和y坐标
            x.append(float(row[1]))
            y.append(float(row[2]))
            # 读取uds_0-uds_9的场值到二维数组uds中
            for i in range(10):
                uds[i].append(float(row[i+3]))
                
    # 将一维列表(元素为一维数组)转化为二维数组             
    uds = np.array(uds)
    
    # 如果数值过小，则视为0，防止出现非数的情况
    for i in range(10):
        for j in range(len(y)):
            if uds[i][j] < 1e-10:
                uds[i][j] =0.

    # 确定meshgrid的坐标范围，并离散坐标
    xi=np.linspace(min(x),max(x), 10000)
    yi=np.linspace(min(y),max(y), 5000)
     
    # grid_x,grid_y坐标必须维数一致，且为二维
    grid_x, grid_y = np.meshgrid(xi, yi)
     
    # 对x，y位置的浓度值进行插值，插值的方法有'cubic', 'linear', 'nearest'
    # 注意传入的坐标参数需要以元组的形式成对传入
    # 当然matplotlib也自带griddata插值函数，该函数每个坐标是一个参数
    # 但matplotlib自带的griddata插值函数只能使用默认的linear插值

    # griddata插值，并存入uds_t列表中，列表中的元素为二维数组
    for i in range(10):
        uds_t.append(griddata((x,y), uds[i], (grid_x,grid_y), method='linear', fill_value = np.nan, rescale = False))

    # 将uds_t转换为三维np数组
    uds_t = np.array(uds_t)
    
    # 保存到文件中，下次使用是直接读取文件，避免重复上述过程(太慢！)
    np.save('uds_adjoint.npy', uds_t)

#%% 生成监测点数据

# 设定真实污染源的位置 
X_true = 100.0
Y_true = 250.0

# 设定监测点的坐标
points_list = [[200,400], [500,400], [800,400],
               [200,250], [500,250], [800,250],
               [200,100], [500,100], [800,100]]

# 从文件中导入uds_adjoint数组，并反转y轴
uds_adjoint = []; uds_adjoint = np.load('uds_adjoint.npy', uds_adjoint); uds_adjoint = np.flip(uds_adjoint, 1)

# 真实污染源时，在监测点处污染物的浓度(uds_0在各监测点处的浓度值)
point_true_value = direct_results(points_list)

# 打印真实污染源时监测点处污染物的浓度
for i in range(len(point_true_value)):
    print(f"point_{i+1}_true_value = {point_true_value[i]}") 

# %%关键参数设定及变量声明   

T = 90000       # MCMC迭代总步数
sigma = 0.3     #似然函数的标准差
is_normal_proposal = True  #选择建议分布为正态分布还是均匀分布(True or False)

t = 0           # 循环迭代索引
count = 0       # 用于计算接受率标识
flag = True     # 用于记录是否接受了新值

X_init = 10.    #用于设定X迭代初始值
Y_init = 10.    #用于设定Y迭代初始值

X_history = [0 for i in range(T)]    # X迭代历史记录
Y_history = [0 for i in range(T)]    # Y迭代历史记录

################################ 迭代开始 ################################
for t in tqdm(range(T-1)): 
    t = t + 1
    
    # 计算新的建议值(j)
    if is_normal_proposal:
        # 建议分布采用正态分布
        sigma_X = 50       # 设置标准差
        sigma_Y = 25        # 设置标准差
        trunc_X = [0, 1000] # 设置截断范围
        trunc_Y = [0, 500]  # 设置截断范围
        X_new = normal_proposal_distribution(t, trunc_X, trunc_Y, sigma_X, sigma_Y, X_history, Y_history)[0]
        Y_new = normal_proposal_distribution(t, trunc_X, trunc_Y, sigma_X, sigma_Y, X_history, Y_history)[1]

    else:
        # 建议分布采用均匀分布    
        delta_X = 100       # 设置标准差
        delta_Y = 50        # 设置标准差
        trunc_X = [0, 1000] # 设置截断范围
        trunc_Y = [0, 500]  # 设置截断范围
        X_new = uniform_proposal_distribution(t, trunc_X, trunc_Y, delta_X, delta_Y, X_history, Y_history)[0]
        Y_new = uniform_proposal_distribution(t, trunc_X, trunc_Y, delta_X, delta_Y, X_history, Y_history)[1]
    
    # 初始值计算
    if t == 1:
        data_predict_now = adjoint_results(X_init, Y_init)
    
    # 计算污染物坐标为(X_new, Y_new)时,保存各监测点值的列表
    data_predict_new = adjoint_results(X_new, Y_new)
    
    # 计算似然函数
    likelihood_j = likelihood_func(sigma, data_predict_new, point_true_value)
    likelihood_i = likelihood_func(sigma, data_predict_now, point_true_value)
    alpha = min(1, likelihood_j / likelihood_i)
    
    # 产生0~1之前的随机数
    u = random.uniform(0, 1)
    
    if u < alpha: # 接收该点
        X_history[t]  = X_new
        Y_history[t]  = Y_new
        flag  = True
        count = count + 1
    else:         #拒绝该点
        X_history[t]  = X_history[t - 1]
        Y_history[t]  = Y_history[t - 1]
        flag  = False

    # 接收后将新数据存入旧数据    
    if flag == True:
        data_predict_now = data_predict_new
################################ 迭代结束 ################################   

print(f"最终接受率 = {round((count/T)*100, 4)}%")

#  绘图后处理

f, ax = plt.subplots(2, 2, figsize = (16, 8))

#直方图统计D
plt.subplot(2,2,1)
sns.distplot(X_history[-20000:], hist=True, bins=50, kde=True, color='red')
# plt.hist(X[-20000:], bins=100 , density=1, facecolor='red')
plt.title('X-plot')
plt.xlabel('X(m)')
plt.ylabel('KDE')
plt.axvline(X_true, color='k')

# 直方图统计k
plt.subplot(2,2,2)
# plt.hist(Y[-20000:], bins=100, density=1, facecolor='green')
sns.distplot(Y_history[-20000:], hist=True, bins=50, kde=True, color='green')
plt.title('Y-plot')
plt.xlabel('Y(m)')
plt.ylabel('KDE')
plt.axvline(Y_true, color='k')

#采样值变化D
plt.subplot(2,2,3)
plt.plot(list(range(1, 20001)), X_history[-20000:])
plt.title('X-plot')
plt.xlabel('iteration')
plt.ylabel('X(m)')
plt.axhline(X_true, color='k')

#采样值变化k
plt.subplot(2,2,4)
plt.plot(list(range(1, 20001)), Y_history[-20000:])
plt.title('Y-plot')
plt.xlabel('iteration')
plt.ylabel('Y(m)')
plt.axhline(Y_true, color='k')

plt.tight_layout()
plt.show()

# %% pymc进行高斯推断

# 显示核密度估计
sns.kdeplot(X_history[-20000:])
plt.xlabel('$x$', fontsize=16)
plt.show()

sns.kdeplot(Y_history[-20000:])
plt.xlabel('$x$', fontsize=16)
plt.show()

# 利用pymc进行x坐标推断
with pm.Model() as model_g_X:
    mu = pm.Uniform('mu', lower=0, upper=1000)
    sigma = pm.HalfNormal('sigma', sigma=50)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=X_history[-20000:])
    trace_g_X = pm.sample(1100)
    
az.plot_trace(trace_g_X, combined=True);
a = az.summary(trace_g_X, round_to=2)

# 利用pymc进行y坐标推断
with pm.Model() as model_g_Y:
    mu = pm.Uniform('mu', lower=0, upper=500)
    sigma = pm.HalfNormal('sigma', sigma=50)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=Y_history[-20000:])
    trace_g_Y = pm.sample(1100)
    
az.plot_trace(trace_g_Y, combined=True);
b = az.summary(trace_g_Y, round_to=2)

# 绘图
x = np.linspace(0, 1000, 1000)
y = stats.norm(a.loc["mu","mean"], a.loc["sigma","mean"]).pdf(x)
plt.axvline(X_true, color='b')
plt.plot(x, y)

x = np.linspace(0, 1000, 1000)
y = stats.norm(b.loc["mu","mean"], b.loc["sigma","mean"]).pdf(x)
plt.axvline(Y_true, color='b')
plt.plot(x, y)

