# -*- coding: utf-8 -*-
"""
author： Hongliang Zhang - WHU
date：   2022-10-05
log: 1. 2022-10-05 室内二维网格计算

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
matplotlib.rc("font", family='Microsoft YaHei')
from mpl_toolkits.mplot3d import Axes3D
import csv
from scipy.interpolate import griddata 

#%% 一些需要的函数定义

# 用于计算似然函数
def likelihood_func(sigma, c_predict, c_true):
    temp = 0
    for i in range(len(c_predict)):   
        temp = temp + (-(c_predict[i] - c_true[i])**2/(2*sigma**2))
    return (np.exp(temp))  

# 从fluent中导出的数据导入到numpy数组中
def file_to_array(points_list):

    # 保存x坐标, 和y坐标的列表
    x=[]; y=[]

    # 创建10个二维空数组
    uds = [[] for i in range(10)]
    
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
            # 读取uds_0-uds_9的场值到二维数组uds中
            for i in range(len(points_list)+1):
                uds[i].append(float(row[i+3]))
                
    # 将一维列表(元素为一维数组)转化为二维数组             
    uds = np.array(uds)
    
    # 如果数值过小，则视为0，防止出现非数的情况
    for i in range(len(points_list)+1):
        for j in range(len(y)):
            if uds[i][j] < 1e-10:
                uds[i][j] =0.

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

# 用于计算伴随浓度场在预测点(X_current, Y_current)处的浓度值
def adjoint_results():
    # 截取掉uds_0
    c_predict = uds_adjoint[1:len(points_list)+1]
    # 返回三维数组
    return c_predict

# 用于提取uds_0在监测点处的浓度值(真实值)
def direct_results(points_list):  
    # 保存监测点处的真实浓度值
    point_true_value = []
    for i in range(len(points_list)):
        point_true_value.append(uds_adjoint[0][int(1000*points_list[i][1])][int(1000*points_list[i][0])])
    # 浓度列表返回
    return point_true_value

#%% 生成监测点数据

# 设定真实污染源的位置 
X_true = 2.3
Y_true = 2.6

# 设定监测点的坐标
points_list = [[5.8, 2.8],
               [2.3, 0.2],
               [8.8, 0.24]]

# file_to_array(points_list)

# 从文件中导入uds_adjoint数组，并反转y轴
uds_adjoint = []; uds_adjoint = np.load('uds_adjoint.npy', uds_adjoint); #uds_adjoint = np.flip(uds_adjoint, 1)

# 真实污染源时，在监测点处污染物的浓度(uds_0在各监测点处的浓度值)
point_true_value = direct_results(points_list)

# 打印真实污染源时监测点处污染物的浓度
for i in range(len(point_true_value)):
    print(f"point_{i+1}_true_value = {point_true_value[i]}") 
# %%网格计算方法

# 生成网格坐标
xi = np.linspace(0, 9, 9000)
yi = np.linspace(0, 3, 3000)
X, Y = np.meshgrid(xi, yi)

# 定义预测值
c_predict = np.zeros([3,3000,9000])

c_predict = adjoint_results()

# %%计算似然函数
sigma = likelihood_func(0.001, c_predict, point_true_value)
sigma = np.flip(sigma, 0)

#%% 显示图

# 污染物扩散二维等值线图
# matplotlib.pyplot.contour(X, Y, c_predict[4,:,:], colors=list(["purple","blue","cyan", "green","yellow","orange","red"]), levels = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4], linestyles=['-'])

# 二维后验密度图
matplotlib.pyplot.contour(X, Y, sigma, colors=list(["purple","blue","cyan", "green","yellow","orange","red"]), levels = 7, linestyles=['-'])

# %%三维后验密度图
fig = plt.figure(figsize=(15,12))
# 转换为三维
ax = Axes3D(fig)

# set the limits
ax.set_xlim([0,9])
ax.set_ylim([0,3])
ax.set_box_aspect((3, 1, 0.5))
surf = ax.plot_surface(X, Y, sigma, rstride=50, cstride=50, edgecolor='black', cmap='rainbow')

# 设置字体
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}

# 设置label
ax.set_xlabel( 'X',font,labelpad=(30) )
ax.set_ylabel( 'Y',font,labelpad=(10) )
ax.set_zlabel( 'Post-PDF',fontdict = font, labelpad=(10) )
plt.tick_params(labelsize=18)

config = {
    "font.family":'family', # sans-serif/serif/cursive/fantasy/monospace
    "font.size": 30, # medium/large/small
    'font.style':'normal', # normal/italic/oblique
    'font.weight':'normal', # bold
    "mathtext.fontset":'cm',# 'cm' (Computer Modern)
    "font.family": ['Times New Roman'], # 'Simsun'宋体
    "axes.unicode_minus": False,# 用来正常显示负号
}

ax.xaxis.set_major_formatter('{x:.01f}')
ax.yaxis.set_major_formatter('{x:.01f}')
ax.zaxis.set_major_formatter('{x:.01f}')

plt.rcParams.update(config)
# 设置标题
plt.title("A figure of 3D")
# fig.colorbar(surf, shrink=0.2, aspect=15, pad=0.05, ticks=[0,0.2,0.4,0.6,0.8,1.0])
plt.tight_layout()
plt.show()

# #%% 寻找后验概率密度最大点

# temp_value_x = np.amax(sigma, axis=1)   #行最大值
# temp_value_y = np.amax(sigma, axis=0)   #列最大值
# temp_index_x = np.argmax(sigma, axis=1) #行最大值索引
# temp_index_y = np.argmax(sigma, axis=0) #列最大值索引

# temp_x = np.array(list(zip(temp_index_x,temp_value_x)))
# temp_y = np.array(list(zip(temp_index_y,temp_value_y)))

# # X_find = temp_x[(np.argmax(temp_x, axis=0)[1]),0]/100 * 0.01 + 0.01
# # Y_find = temp_y[(np.argmax(temp_y, axis=0)[1]),0]/100 * 1.0  + 1.5
# # print(X_find)
# # print(Y_find)
