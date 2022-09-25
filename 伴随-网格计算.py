# -*- coding: utf-8 -*-
"""
author： Hongliang Zhang - WHU
date：   2022-09-04
log: 1. 2022-09-04 attempt to couple with pyfluent
     2. 2022-09-09 fix code adapted to MH method
     3. 2022-09-12 function encapsulation 
     4. 2022-09-13 fix code adapted to mesh method   
     5. 2022-09-14 fineshed and successful
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

# 用于计算正态建议分布值
def normal_proposal_distribution(t, sigma_X = 0, sigma_Y = 0):
    X_new = norm.rvs(loc=X[t - 1], scale=sigma_X, size=1, random_state=None)[0] #建议分布
    Y_new = norm.rvs(loc=Y[t - 1], scale=sigma_Y, size=1, random_state=None)[0] #建议分布
    # X坐标超标罚回
    while X_new < 0 or X_new > 1000:
        X_new = norm.rvs(loc=X[t - 1], scale=sigma_X,   size=1, random_state=None)[0]
    # Y坐标超标罚回
    while Y_new < -250 or Y_new > 250:
        Y_new = norm.rvs(loc=Y[t - 1], scale=sigma_Y, size=1, random_state=None)[0]
    # 返回值为新的污染物坐标    
    return (X_new, Y_new)
      
# 用于计算均匀建议分布值  
def uniform_proposal_distribution(t, delta_X = 0, delta_Y = 0):
    X_new = random.uniform(X[t - 1] - delta_X, X[t - 1] + delta_X)   #建议分布
    Y_new = random.uniform(Y[t - 1] - delta_Y, Y[t - 1] + delta_Y)     #建议分布
    # X坐标超标罚回
    while X_new < 0 or Y_new > 1000:
        X_new = random.uniform(X[t - 1] - delta_X, X[t - 1] + delta_X)
    # Y坐标超标罚回
    while Y_new < -250 or Y_new > 250:
        Y_new = random.uniform(Y[t - 1] - delta_Y, Y[t - 1] + delta_Y)
    # 返回值为新的污染物坐标   
    return (X_new, Y_new)

# 用于计算似然函数
def likelihood_func(sigma, c_predict, c_true):
    temp = 0
    for i in range(len(c_predict)):   
        temp = temp + (-(c_predict[i] - c_true[i])**2/(2*sigma**2))
    return (np.exp(temp))  

def adjoint_results():
    #保存x坐标
    x=[]

    #保存y坐标
    y=[]

    #保存导出的uds_0-uds_9
    uds_0=[]
    uds_1=[]
    uds_2=[]
    uds_3=[]
    uds_4=[]
    uds_5=[]
    uds_6=[]
    uds_7=[]
    uds_8=[]
    uds_9=[]

    #读取csv文件，当然也可以读取类似txt之类的文件
    with open(r'F:/ZHHL/TE_Doctor/CASES/case220915/adjoint_method/fluent-adjoint-0924','r')  as  csvfile:
    #指定分隔符为","，因为我们刚才导出时就是逗号
        plots=csv.reader(csvfile,delimiter=',')

        #循环读取到的文件
        for row in plots:
        #为了跳过文件前面的非数据行  
            if plots.line_num == 1:
                continue
            x.append(float(row[1]))
            y.append(float(row[2]))
            
            uds_0.append(float(row[3]))
            uds_1.append(float(row[4]))
            uds_2.append(float(row[5]))
            uds_3.append(float(row[6]))
            uds_4.append(float(row[7]))
            uds_5.append(float(row[8]))
            uds_6.append(float(row[9]))
            uds_7.append(float(row[10]))
            uds_8.append(float(row[11]))
            uds_9.append(float(row[12]))

    #如果数值过小，则视为0，防止出现非数的情况
    for i in range(len(y)):
        if uds_0[i] < 1e-10:
            uds_0[i] = 0.
            
        if uds_1[i] < 1e-10:
            uds_1[i] = 0.
            
        if uds_2[i] < 1e-10:
            uds_2[i] = 0.
            
        if uds_3[i] < 1e-10:
            uds_3[i] = 0.
            
        if uds_4[i] < 1e-10:
            uds_4[i] = 0.
            
        if uds_5[i] < 1e-10:
            uds_5[i] = 0.
            
        if uds_6[i] < 1e-10:
            uds_6[i] = 0.
            
        if uds_7[i] < 1e-10:
            uds_7[i] = 0.
            
        if uds_8[i] < 1e-10:
            uds_8[i] = 0.
            
        if uds_9[i] < 1e-10:
            uds_9[i] = 0.   
            
            
    xi=np.linspace(min(x),max(x), 10000)
    yi=np.linspace(min(y),max(y), 5000)
     
    #X,Y坐标必须维数一致，且为二维
    grid_x,grid_y = np.meshgrid(xi,yi)
     
    #对x，y的速度进行插值，插值的方法有'cubic(1-d)','cubic(2-d)','linear','nearest'
    #注意传入的坐标参数需要以元组的形式成对传入
    #当然matplotlib也自带griddata插值函数，该函数每个坐标是一个参数
    #但matplotlib自带的griddata插值函数只能使用默认的linear插值
    # uds_1 = griddata((x,y),uds_1,(grid_x,grid_y),method='cubic')
    # uds_1 = griddata((x,y),uds_1,(grid_x,grid_y),method='nearest')
    uds_0 = griddata((x,y),uds_0,(grid_x,grid_y), method='linear', fill_value = np.nan, rescale = False)
    uds_1 = griddata((x,y),uds_1,(grid_x,grid_y), method='linear', fill_value = np.nan, rescale = False)
    uds_2 = griddata((x,y),uds_2,(grid_x,grid_y), method='linear', fill_value = np.nan, rescale = False)
    uds_3 = griddata((x,y),uds_3,(grid_x,grid_y), method='linear', fill_value = np.nan, rescale = False)
    uds_4 = griddata((x,y),uds_4,(grid_x,grid_y), method='linear', fill_value = np.nan, rescale = False)
    uds_5 = griddata((x,y),uds_5,(grid_x,grid_y), method='linear', fill_value = np.nan, rescale = False)
    uds_6 = griddata((x,y),uds_6,(grid_x,grid_y), method='linear', fill_value = np.nan, rescale = False)
    uds_7 = griddata((x,y),uds_7,(grid_x,grid_y), method='linear', fill_value = np.nan, rescale = False)
    uds_8 = griddata((x,y),uds_8,(grid_x,grid_y), method='linear', fill_value = np.nan, rescale = False)
    uds_9 = griddata((x,y),uds_9,(grid_x,grid_y), method='linear', fill_value = np.nan, rescale = False)
    
    c_predict = np.array([uds_1, uds_2, uds_3, uds_4, uds_5, uds_6, uds_7, uds_8, uds_9])
    
    return c_predict

def direct_results(points_list):
    # 保存真实值
    point_true_value=[]
    
    #保存x坐标
    x=[]

    #保存y坐标
    y=[]

    #保存导出的uds_0-uds_9
    uds_0=[]


    #读取csv文件，当然也可以读取类似txt之类的文件
    with open(r'F:/ZHHL/TE_Doctor/CASES/case220915/adjoint_method/fluent-adjoint-0924','r')  as  csvfile:
    #指定分隔符为","，因为我们刚才导出时就是逗号
        plots=csv.reader(csvfile,delimiter=',')

        #循环读取到的文件
        for row in plots:
        #为了跳过文件前面的非数据行  
            if plots.line_num == 1:
                continue
            x.append(float(row[1]))
            y.append(float(row[2]))
            
            uds_0.append(float(row[3]))



    #如果数值过小，则视为0，防止出现非数的情况
    for i in range(len(y)):
        if uds_0[i] < 1e-10:
            uds_0[i] = 0.
            
    xi=np.linspace(min(x),max(x), 10000)
    yi=np.linspace(min(y),max(y), 5000)
     
    #X,Y坐标必须维数一致，且为二维
    grid_x,grid_y = np.meshgrid(xi,yi)
     
    #对x，y的速度进行插值，插值的方法有'cubic(1-d)','cubic(2-d)','linear','nearest'
    #注意传入的坐标参数需要以元组的形式成对传入
    #当然matplotlib也自带griddata插值函数，该函数每个坐标是一个参数
    #但matplotlib自带的griddata插值函数只能使用默认的linear插值
    # uds_1 = griddata((x,y),uds_1,(grid_x,grid_y),method='cubic')
    # uds_1 = griddata((x,y),uds_1,(grid_x,grid_y),method='nearest')
    uds_0 = griddata((x,y),uds_0,(grid_x,grid_y), method='linear', fill_value = np.nan, rescale = False)
    for i in range(len(points_list)):
        point_true_value.append(uds_0[10*points_list[i][1], 10*points_list[i][0]])

    return point_true_value

#%% 生成监测点数据

# 设定真实污染源的位置 
X_true = 300.0
Y_true = 0.0

# 设定监测点的坐标
points_list = [[200,100], [500,100], [800,100],
               [200,250], [500,250], [800,250],
               [200,400], [500,400], [800,400]]

# fluent计算，返回值为真实污染源时——监测点处污染物的浓度
point_true_value = direct_results(points_list)

# 打印真实污染源时监测点处污染物的浓度
for i in range(len(point_true_value)):
    print(f"point_{i+1}_true_value = {point_true_value[i]}") 

# %%网格计算方法

# 生成网格坐标
X = np.linspace(0, 1000, 10000)
Y = np.linspace(-250, 250, 5000)
X, Y = np.meshgrid(X, Y)

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
