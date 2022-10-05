# -*- coding: utf-8 -*-
"""
author： Hongliang Zhang - WHU
date：   2022-09-04
log: 1. 2022-09-04 attempt to couple with pyfluent
     2. 2022-09-09 fix code adapted to MH method
     3. 2022-09-12 function encapsulation 
     4. 2022-09-13 fix code adapted to mesh method   
     5. 2022-09-14 fineshed and successful
     6. 2022-09-27 函数封装，并简化代码
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

# 用于计算伴随浓度场在预测点(X_current, Y_current)处的浓度值
def adjoint_results():
    # 截取掉uds_0
    c_predict = uds_adjoint[1:10]
    # 返回三维数组
    return c_predict

# 用于提取uds_0在监测点处的浓度值(真实值)
def direct_results(points_list):  
    # 保存监测点处的真实浓度值
    point_true_value = []
    for i in range(len(points_list)):
        point_true_value.append(uds_adjoint[0][10*points_list[i][1]-1][10*points_list[i][0]-1])
    # 浓度列表返回
    return point_true_value

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
# %%网格计算方法

# 生成网格坐标
xi = np.linspace(0, 1000, 10000)
yi = np.linspace(-250, 250, 5000)
X, Y = np.meshgrid(xi, yi)

# 定义预测值
c_predict = np.zeros([9,5000,10000])

c_predict = adjoint_results()

# %%计算似然函数
sigma = likelihood_func(0.1, c_predict, point_true_value)

#%% 显示图

# 污染物扩散二维等值线图
# matplotlib.pyplot.contour(X, Y, c_predict[4,:,:], colors=list(["purple","blue","cyan", "green","yellow","orange","red"]), levels = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4], linestyles=['-'])

# 二维后验密度图
matplotlib.pyplot.contour(X, Y, sigma, colors=list(["purple","blue","cyan", "green","yellow","orange","red"]), levels = 7, linestyles=['-'])

# %%三维后验密度图
fig = plt.figure(figsize=(12,6))
# 转换为三维
ax = Axes3D(fig)

# set the limits
ax.set_xlim([0,1000])
ax.set_ylim([-250,250])
ax.set_box_aspect((2, 1, 0.2))
surf = ax.plot_surface(X, Y, sigma, rstride=50, cstride=100, edgecolor='black', cmap='rainbow')
# 设置字体
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}

# 设置label
ax.set_xlabel( 'X',font )
ax.set_ylabel( 'Y',font )
ax.set_zlabel( 'Post-PDF',font )

# 设置标题
plt.title("A figure of 3D")
# fig.colorbar(surf, shrink=0.5, aspect=5)
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
