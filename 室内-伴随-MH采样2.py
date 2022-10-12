# -*- coding: utf-8 -*-
"""
author： Hongliang Zhang - WHU
date：   2022-09-26
log: 1. 2022-10-05 伴随方法用于室内污染物寻源
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
from matplotlib.pyplot import MultipleLocator

matplotlib.rc("font", family='Microsoft YaHei')

#%% 一些需要的函数定义

# 将正态分布作为建议分布, 注意设置截断位置参数
def normal_proposal_distribution(t, trunc_X = [0, 1], trunc_Y = [0, 1], trunc_S = [0, 1], sigma_X = 0., sigma_Y = 0., sigma_S = 0., X_history = [], Y_history = [], S_history = []):
    # 设置截断
    x_lower = trunc_X[0]; x_upper = trunc_X[1] 
    # 截断正态分布采样
    X_new = stats.truncnorm.rvs((x_lower - X_history[t - 1]) / sigma_X, (x_upper - X_history[t - 1]) / sigma_X, loc=X_history[t - 1], scale=sigma_X) #建议分布
    # 设置截断
    y_lower = trunc_Y[0]; y_upper = trunc_Y[1]
    # 截断正态分布采样
    Y_new = stats.truncnorm.rvs((y_lower - Y_history[t - 1]) / sigma_Y, (y_upper - Y_history[t - 1]) / sigma_Y, loc=Y_history[t - 1], scale=sigma_Y) #建议分布
    # 设置截断 
    s_lower = trunc_S[0]; s_upper = trunc_S[1] 
    # 截断正态分布采样
    S_new = stats.truncnorm.rvs((s_lower - S_history[t - 1]) / sigma_S, (s_upper - S_history[t - 1]) / sigma_S, loc=S_history[t - 1], scale=sigma_S) #建议分布
       
    
    # 设置采样值保留的小数位数 
    X_new = Decimal(X_new).quantize(Decimal("0.000"))
    Y_new = Decimal(Y_new).quantize(Decimal("0.000"))
    S_new = Decimal(S_new).quantize(Decimal("0.000"))
    
    # 返回值为新的污染物坐标 
    return float(X_new), float(Y_new), float(S_new)
      

# 用于计算似然函数, c_predict和c_true为存储预测点值和检测点值的数组
# 用了log，将相乘变成相加
def likelihood_func_log(sigma, c_predict, c_true):
    # c_predict = mu
    # sigma = sigma
    # c_true = the observation (new or current)
    return np.sum(-np.log(sigma * np.sqrt(2* np.pi) )-((c_true-c_predict)**2) / (2*sigma**2))

# 用于计算伴随浓度场在预测点(X_current, Y_current)处的浓度值
def adjoint_results(X_current, Y_current, S_current, points_list):
    # 用于存储新预测位置下,各监测点处的值(对偶)
    data_predict_new = []
    
    for i in range(1, len(points_list)+1):
        data_predict_new.append(S_current*uds_adjoint[i][int(1000*Y_current)-1][int(1000*X_current)-1])
    
    # 返回存储各监测点浓度值的列表
    return data_predict_new

# 用于提取uds_0在监测点处的浓度值(真实值)
def direct_results(points_list):  
    # 保存监测点处的真实浓度值
    point_true_value = []
    for i in range(len(points_list)):
        point_true_value.append(uds_adjoint[0][int(1000*points_list[i][1])][int(1000*points_list[i][0])])
    # 浓度列表返回
    return point_true_value

# 从fluent中导出的数据导入到numpy数组中
def file_to_array(points_list):

    # 保存x坐标, 和y坐标的列表
    x=[]; y=[]

    # 创建4个二维空数组
    uds = [[] for i in range(len(points_list)+1)]
    
    # 创建一维列表，用于存储二维np数组，进而转化为三维np数组
    uds_t = []

    # 读取csv文件，当然也可以读取类似txt之类的文件
    with open(r'F:/ZHHL/TE_Doctor/CASES/case220927/fluent16-uniformmesh-0928','r')  as  csvfile:   
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
            # 读取uds_0-uds_3的场值到二维数组uds中
            for i in range(len(points_list)+1):
                uds[i].append(float(row[i+3]))
                
    # 将一维列表(元素为一维数组)转化为二维数组             
    uds = np.array(uds)
    
    # 如果数值过小，则视为0，防止出现非数的情况
    for i in range(len(points_list)+1):
        for j in range(len(y)):
            if uds[i][j] < 1e-10:
                uds[i][j] =0.0

    # 确定meshgrid的坐标范围，并离散坐标
    xi=np.linspace(min(x),max(x), 9000)
    yi=np.linspace(min(y),max(y), 3000)
     
    # grid_x,grid_y坐标必须维数一致，且为二维
    grid_x, grid_y = np.meshgrid(xi, yi)
    
    # 对x，y位置的浓度值进行插值，插值的方法有'cubic', 'linear', 'nearest'
    # 注意传入的坐标参数需要以元组的形式成对传入
    # 当然matplotlib也自带griddata插值函数，该函数每个坐标是一个参数
    # 但matplotlib自带的griddata插值函数只能使用默认的linear插值

    # griddata插值，并存入uds_t列表中，列表中的元素为二维数组
    for i in range(len(points_list)+1):
        uds_t.append(griddata((x,y), uds[i], (grid_x,grid_y), method='linear', fill_value = np.nan, rescale = False))

    # 将uds_t转换为三维np数组
    uds_t = np.array(uds_t)
    
    # 保存到文件中，下次使用是直接读取文件，避免重复上述过程(太慢！)
    np.save('uds_adjoint.npy', uds_t)

#%% 生成监测点数据

# 设定真实污染源的位置 
X_true = 2.3
Y_true = 2.6
S_true = 10

# 设定监测点的坐标
# points_list = [[5.8, 2.8],
#                [2.3, 0.2],
#                [8.8, 0.24]]

points_list = [[5.8, 2.8],
               [2.3, 0.2],
               [8.8, 0.24],
               [3.2, 2.8],
               [0.2, 1.5]]
# file_to_array(points_list)

# 从文件中导入uds_adjoint数组，并反转y轴
uds_adjoint = []; uds_adjoint = np.load('uds_adjoint_1.npy', uds_adjoint); #uds_adjoint = np.flip(uds_adjoint, 1)

# 交互0和1轴的数据
# uds_adjoint[[0,1],:,:] = uds_adjoint[[1,0],:,:]

# 真实污染源时，在监测点处污染物的浓度(uds_0在各监测点处的浓度值)
point_true_value = np.array(direct_results(points_list))*S_true

# 打印真实污染源时监测点处污染物的浓度
for i in range(len(point_true_value)):
    print(f"point_{i+1}_true_value = {point_true_value[i]}") 

# %%关键参数设定及变量声明   

T = 50000       # MCMC迭代总步数
sigma = 2     # 似然函数的标准差

t = 0           # 循环迭代索引
count = 0       # 用于计算接受率标识
flag = True     # 用于记录是否接受了新值

X_init = 4.5    # 用于设定X迭代初始值
Y_init = 1.5    # 用于设定Y迭代初始值
S_init = 10.    # 用于设定Y迭代初始值

X_history = [0 for i in range(T)]    # X迭代历史记录
Y_history = [0 for i in range(T)]    # Y迭代历史记录
S_history = [0 for i in range(T)]    # S迭代历史记录

trunc_X = [0, 9]  # 设置建议分布截断范围
trunc_Y = [0, 3]  # 设置建议分布截断范围
trunc_S = [0, 20]  # 设置建议分布截断范围

sigma_X = 2     # 设置建议分布标准差
sigma_Y = 2     # 设置建议分布标准差
sigma_S = 1     # 设置建议分布标准差

accepted = []     # 记录接受的采样
rejected = []     # 记录拒绝的采样

################################ 迭代开始 ################################
for t in tqdm(range(T-1)): 
    t = t + 1
    
    # 计算新的建议值(j)
    # 建议分布采用正态分布
    X_new, Y_new, S_new = normal_proposal_distribution(t, trunc_X, trunc_Y, trunc_S, sigma_X, sigma_Y, sigma_S, X_history, Y_history, S_history)
 
    # 初始值计算
    if t == 1:
        data_predict_now = adjoint_results(X_init, Y_init, S_init, points_list)
    
    # 计算污染物坐标为(X_new, Y_new)，强度为S_new时,保存各监测点值的列表
    data_predict_new = adjoint_results(X_new, Y_new, S_new, points_list)
    
    # 计算似然函数
    likelihood_i = likelihood_func_log(sigma, data_predict_now, point_true_value)
    likelihood_j = likelihood_func_log(sigma, data_predict_new, point_true_value)
    alpha = min(1, np.exp(likelihood_j - likelihood_i))
    
    # 产生0~1之前的随机数
    u = random.uniform(0, 1)
    
    if u < alpha: # 接收该点
        X_history[t]  = X_new
        Y_history[t]  = Y_new
        S_history[t]  = S_new
        flag  = True
        count = count + 1
        accepted.append([X_new, Y_new, S_new])
        
    else:         #拒绝该点
        X_history[t]  = X_history[t - 1]
        Y_history[t]  = Y_history[t - 1]
        S_history[t]  = S_history[t - 1]        
        flag  = False
        rejected.append([X_new, Y_new, S_new])

    # 接收后将新数据存入旧数据    
    if flag == True:
        data_predict_now = data_predict_new
################################ 迭代结束 ################################   

accepted = np.array(accepted); rejected = np.array(rejected)
print(f"最终接受率 = {round((count/T)*100, 4)}%")

# %% 统一绘图

f, ax = plt.subplots(2, 3, figsize = (16, 8))

#直方图统计x
plt.subplot(2,3,1)
sns.distplot(X_history[-20000:], hist=True, bins=50, kde=True, color='red')
# plt.hist(X[-20000:], bins=100 , density=1, facecolor='red')
plt.title('X-plot')
plt.xlabel('X(m)')
plt.ylabel('KDE')
plt.axvline(X_true, color='k')

# 直方图统计y
plt.subplot(2,3,2)
# plt.hist(Y[-20000:], bins=100, density=1, facecolor='green')
sns.distplot(Y_history[-20000:], hist=True, bins=50, kde=True, color='green')
plt.title('Y-plot')
plt.xlabel('Y(m)')
plt.ylabel('KDE')
plt.axvline(Y_true, color='k')

# 直方图统计s
plt.subplot(2,3,3)
# plt.hist(Y[-20000:], bins=100, density=1, facecolor='green')
sns.distplot(S_history[-20000:], hist=True, bins=50, kde=True, color='blue')
plt.title('S-plot')
plt.xlabel('S(m)')
plt.ylabel('KDE')
plt.axvline(S_true, color='k')

#采样值变化x
plt.subplot(2,3,4)
plt.plot(list(range(1, 20001)), X_history[-20000:])
plt.title('X-plot')
plt.xlabel('iteration')
plt.ylabel('X(m)')
plt.axhline(X_true, color='k')

#采样值变化y
plt.subplot(2,3,5)
plt.plot(list(range(1, 20001)), Y_history[-20000:])
plt.title('Y-plot')
plt.xlabel('iteration')
plt.ylabel('Y(m)')
plt.axhline(Y_true, color='k')

#采样值变化s
plt.subplot(2,3,6)
plt.plot(list(range(1, 20001)), S_history[-20000:])
plt.title('s-plot')
plt.xlabel('iteration')
plt.ylabel('S(m)')
plt.axhline(S_true, color='k')

plt.tight_layout()
plt.show()
# %% X采用分布绘图

f, ax = plt.subplots(1, 1, figsize = (10, 4))
# 设置字体
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}

#直方图统计X
plt.subplot(1,1,1)
sns.distplot(X_history[-20000:], hist=True, bins=60, kde=True, fit=None, norm_hist=False,color="b",hist_kws={'alpha':0.6,'color':'gray'})
# plt.hist(X_history[-20000:], bins=100)
# sns.kdeplot(X_history[-20000:],shade=True,color="g")
# plt.hist(X[-20000:], bins=100 , density=1, facecolor='red')
plt.xlabel('$X$(m)',font,labelpad=(10))
plt.ylabel('$P$',font,labelpad=(10))
plt.tick_params(labelsize=18,which='major',width=2,colors='k')
plt.tick_params(labelsize=18,which='minor',width=2,colors='k')

x_major_locator=MultipleLocator(1)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.1)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为1的倍数

ax=plt.gca();#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1);####设置上部坐标轴的粗细
ax.xaxis.set_major_formatter('{x:.01f}')
ax.yaxis.set_major_formatter('{x:.01f}')

plt.xlim(-1,9)
plt.ylim(0,0.25)

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.axvline(X_true, color='red',linestyle='--',label='$X$ location')
plt.grid(b=1, which='both',color='k',linestyle='--', linewidth=0.5,alpha=0.6)
plt.legend([r'$KDE$', r'$x$ location'], loc=1,frameon=True, framealpha=1, edgecolor = 'k', prop = {"family": "Times New Roman", "size":15})
plt.tight_layout()
plt.savefig('1.svg', figsize=(10, 5))

# %% Y采样分布绘图

f, ax = plt.subplots(1, 1, figsize = (10, 4))
# 设置字体
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}

#直方图统计Y
plt.subplot(1,1,1)
sns.distplot(Y_history[-20000:], hist=True, bins=60, kde=True, fit=None, norm_hist=False,color="b",hist_kws={'alpha':0.6,'color':'gray'})
# plt.hist(X_history[-20000:], bins=100)
# sns.kdeplot(X_history[-20000:],shade=True,color="g")
# plt.hist(X[-20000:], bins=100 , density=1, facecolor='red')
plt.xlabel('$Y$(m)',font,labelpad=(10))
plt.ylabel('$P$',font,labelpad=(10))
plt.tick_params(labelsize=18,which='major',width=2,colors='k')
plt.tick_params(labelsize=18,which='minor',width=2,colors='k')

x_major_locator=MultipleLocator(0.5)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.2)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为1的倍数

ax=plt.gca();#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1);####设置上部坐标轴的粗细
ax.xaxis.set_major_formatter('{x:.01f}')
ax.yaxis.set_major_formatter('{x:.01f}')

plt.xlim(0,3)
plt.ylim(0,0.6)

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.axvline(Y_true, color='red',linestyle='--',label='$Y$ location')
plt.grid(b=1, which='both',color='k',linestyle='--', linewidth=0.5,alpha=0.6)
plt.legend([r'$KDE$', r'$y$ location'], loc=2,frameon=True, framealpha=1, edgecolor = 'k', prop = {"family": "Times New Roman", "size":15})
plt.tight_layout()
plt.savefig('1.svg', figsize=(10, 5))
# %% S采样分布绘图

f, ax = plt.subplots(1, 1, figsize = (10, 4))
# 设置字体
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}

#直方图统计S
plt.subplot(1,1,1)
sns.distplot(S_history[-20000:], hist=True, bins=60, kde=True, fit=None, norm_hist=False,color="b",hist_kws={'alpha':0.6,'color':'gray'})
# plt.hist(X_history[-20000:], bins=100)
# sns.kdeplot(X_history[-20000:],shade=True,color="g")
# plt.hist(X[-20000:], bins=100 , density=1, facecolor='red')
plt.xlabel('$S$ ($\mathregular{unit/m^{3}·kg}$)',font,labelpad=(10))
plt.ylabel('$P$',font,labelpad=(10))
plt.tick_params(labelsize=18,which='major',width=2,colors='k')
plt.tick_params(labelsize=18,which='minor',width=2,colors='k')

x_major_locator=MultipleLocator(2)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.05)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为1的倍数

ax=plt.gca();#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1);####设置上部坐标轴的粗细
ax.xaxis.set_major_formatter('{x:.01f}')
ax.yaxis.set_major_formatter('{x:.02f}')

plt.xlim(0,20)
plt.ylim(0,0.25)

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.axvline(S_true, color='red',linestyle='--')
plt.grid(b=1, which='both',color='k',linestyle='--', linewidth=0.5,alpha=0.6)
plt.legend([r'$KDE$', r'$s$ strength'], loc=1, frameon=True, framealpha=1, edgecolor = 'k', prop = {"family": "Times New Roman", "size":15})
plt.tight_layout()
plt.savefig('1.svg', figsize=(10, 5))

# %% 显示采样历史

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(2,1,1)
to_show=-int(0.5*accepted.shape[0])
ax.plot( rejected[to_show:, 0], 'rx', label='Rejected',alpha=0.5)
ax.plot( accepted[to_show:, 0], 'b.', label='Accepted',alpha=0.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("$x$")
ax.set_title("MCMC sampling for $x$ with Metropolis-Hastings. Half samples are shown.")
ax.grid()
ax.legend()

ax2 = fig.add_subplot(2,1,2)
to_show=-int(0.5*accepted.shape[0])
ax2.plot( rejected[to_show:, 1], 'rx', label='Rejected',alpha=0.5)
ax2.plot( accepted[to_show:, 1], 'b.', label='Accepted',alpha=0.5)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("$y$")
ax2.set_title("MCMC sampling for $y$ with Metropolis-Hastings. Half samples are shown.")
ax2.grid()
ax2.legend()

fig.tight_layout()
accepted.shape

# %% x和y的采样结果同时显示

fig = plt.figure(figsize=(10,4))

font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}

ax = fig.add_subplot(1,1,1)
# ax.plot(accepted[:,0], accepted[:,1], label="Path")
ax.plot(accepted[-5000:,0], accepted[-5000:,1], 'b.', label='Accepted',alpha=0.3)
ax.plot(rejected[-5000:,0], rejected[-5000:,1], 'rx', label='Rejected',alpha=0.3)
plt.xlabel("$X$(m)",font,labelpad=(10))
plt.ylabel("$Y$(m)",font,labelpad=(10))
x_major_locator=MultipleLocator(1)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(1)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为1的倍数

ax=plt.gca();#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1);####设置上部坐标轴的粗细
ax.xaxis.set_major_formatter('{x:.01f}')
ax.yaxis.set_major_formatter('{x:.01f}')

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

plt.xlim(0,9)
plt.ylim(0,3)

plt.annotate('$S$', xy=(2.2,2.5), textcoords='offset points', weight='heavy', bbox=dict(boxstyle='circle,pad=0.5', fc='yellow', ec='k', lw=1, alpha=1))

plt.tick_params(labelsize=18,which='major',width=2,colors='k')
plt.tick_params(labelsize=18,which='minor',width=2,colors='k')
plt.legend(loc=4,frameon=True, framealpha=1, edgecolor = 'k', prop = {"family": "Times New Roman", "size":15})
plt.tight_layout()
# ax.set_title("MCMC sampling for $x$ and $y$ with Metropolis-Hastings. All samples are shown.") 

# %% 显示自相关性

show=-int(0.5*accepted.shape[0])
mean_acc_0=accepted[show:,0].mean()
mean_acc_1=accepted[show:,1].mean()
mean_acc_2=accepted[show:,2].mean()
lag=np.arange(1,100)

def autocorr(accepted,lag):
    num_0=0
    denom_0=0
    num_1=0
    denom_1=0
    num_2=0
    denom_2=0
    for i in range(accepted.shape[0]-lag):
        num_0+=(accepted[i,0]-mean_acc_0)*(accepted[i+lag,0]-mean_acc_0)
        num_1+=(accepted[i,1]-mean_acc_1)*(accepted[i+lag,1]-mean_acc_1)
        num_2+=(accepted[i,2]-mean_acc_2)*(accepted[i+lag,2]-mean_acc_2)
        denom_0+=(mean_acc_0-accepted[i,0])**2
        denom_1+=(mean_acc_1-accepted[i,1])**2
        denom_2+=(mean_acc_2-accepted[i,2])**2
    rk_0=num_0/denom_0
    rk_1=num_1/denom_1
    rk_2=num_2/denom_2  
    return rk_0, rk_1, rk_2


accepted_reversed=accepted[show:,:]
result=np.zeros((3,lag.shape[0]))
#print(lag)
for l in lag:
    result[:,l-1]=autocorr(accepted_reversed,l)
    
    
###Instead of writing an autocorrelation function, one could simply use thee autocorr function provided in pymc3    
#from pymc3.stats import autocorr
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1,1,1)
#ax.plot(lag, [autocorr(accepted[show:,1], l) for l in lags], label='auto b')
#ax.plot(lag, [autocorr(accepted[show:,0], l) for l in lags], label='auto a')
ax.plot(lag, result[1,:], linestyle='--',color='r', linewidth=2, label='Auto correlation for $X$')
ax.plot(lag, result[0,:], linestyle='-', color='b',linewidth=2, label='Auto correlation for $Y$')
ax.plot(lag, result[2,:], linestyle='-.',color='k', linewidth=2, label='Auto correlation for $S$')
ax.legend(loc=0)

plt.xlabel("iteration",font,labelpad=(10))
plt.ylabel("autocorrelation",font,labelpad=(10))
x_major_locator=MultipleLocator(10)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.2)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为1的倍数

ax=plt.gca();#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1);####设置上部坐标轴的粗细
ax.xaxis.set_major_formatter('{x:.01f}')
ax.yaxis.set_major_formatter('{x:.01f}')

plt.xlim(0,100)
plt.ylim(-0.1,1)

plt.grid(b=1, which='both',color='k',linestyle='--', linewidth=0.5,alpha=0.6)

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

plt.tick_params(labelsize=18,which='major',width=2,colors='k')
plt.tick_params(labelsize=18,which='minor',width=2,colors='k')
plt.legend(loc=1,frameon=True, framealpha=1, edgecolor = 'k', prop = {"family": "Times New Roman", "size":15})
plt.tight_layout()

# %% 二维联合分布图

show=int(-0.5*accepted.shape[0])
hist_show=int(-0.50*accepted.shape[0])
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1,1,1)
xbins, ybins = np.linspace(0,9,300), np.linspace(0,3,300)
counts, xedges, yedges, im = ax.hist2d(accepted[hist_show:,0], accepted[hist_show:,1], density=True, bins=[xbins, ybins])
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(im, ax=ax)
ax.set_title("2D histogram showing the joint distribution of $x$ and $y$")

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
    mu = pm.Uniform('mu', lower=0, upper=9)
    sigma = pm.HalfNormal('sigma', sigma=1)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=X_history[-20000:])
    trace_g_X = pm.sample(1100)
    
az.plot_trace(trace_g_X, combined=True);
a = az.summary(trace_g_X, round_to=2)

# 利用pymc进行y坐标推断
with pm.Model() as model_g_Y:
    mu = pm.Uniform('mu', lower=0, upper=3)
    sigma = pm.HalfNormal('sigma', sigma=1)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=Y_history[-20000:])
    trace_g_Y = pm.sample(1100)
    
az.plot_trace(trace_g_Y, combined=True);
b = az.summary(trace_g_Y, round_to=2)

# 绘图
x = np.linspace(0, 9, 1000)
y = stats.norm(a.loc["mu","mean"], a.loc["sigma","mean"]).pdf(x)
plt.axvline(X_true, color='b')
plt.plot(x, y)

x = np.linspace(0, 3, 1000)
y = stats.norm(b.loc["mu","mean"], b.loc["sigma","mean"]).pdf(x)
plt.axvline(Y_true, color='b')
plt.plot(x, y)

