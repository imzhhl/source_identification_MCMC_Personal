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

#%% 利用pyFluent包连接fluent,该包的使用仅限于Fluent 2022R2以后版本

# 定义工作目录
base_path =  r'F:\ZHHL\TE_Doctor\CASES\case220904\python'
import_filename = base_path + r'\fluent'
XY_file = base_path + r'\interactive_files\X_Y_position.txt'
# UDF_Path = r'F:\ZHHL\TE_Doctor\CASES\case220626\python_fluent\python\udf_source.c'

session = pyfluent.launch_fluent(version = "2d", precision='double',processor_count = 2, show_gui=True)

# 用工作站远程计算，连接现有的窗口
# session = pyfluent.launch_fluent(ip='192.168.31.230', port=63993, start_instance=False)

# 从session中提取tui和root命令
tui = session.solver.tui
root = session.solver.root

#%% 一些需要的函数定义

# consol输出重定向类定义
class RedirectStdout:
    def __init__(self):
        self.default_stdout = sys.stdout
        sys.stdout = self
        self.buffer =[]         
    def flush(self):
        pass   
    def write(self,*args):
        if args != ('\n',):
            self.buffer.append(args[0])                   
    def restore(self):
        sys.stdout = self.default_stdout

# 通过重定向方法提取监测点变量的值
# 污染物浓度值放大了1倍，保留小数后4位
def get_points_value_redirect(points_list):
    point_result = []
    point_true_value = []
    i = 0
    for i in range(len(points_list)):
        index = f'(point-{i+1})'
        tmp_output = RedirectStdout()
        tui.report.surface_integrals.sum(f"{index}" , 'uds-0-scalar')
        point_result = tmp_output.buffer
        point_true_value.append(round((1.0 * float(re.findall(r"\d+\.?\d*e?-?\d*", point_result[4])[1])), 4) ) 
        tmp_output.restore()
    return (point_true_value)

# 通过I/O接口获取监测点变量的值
# 污染物浓度值放大了10倍，保留小数后4位
def get_points_value_IO(points_list):
    point_result = []
    point_true_value = []
    i = 0
    j = 0
    s =''
    for i in range(len(points_list)):
        s = s + f'point-{i+1}'
        if i < len(points_list)-1:
            s = s + ' '
    index_1 = '(' + s + ')'
    points_file = base_path + r'\interactive_files\points_value'
    if os.path.exists(f"{points_file}"):
        os.remove(f"{points_file}")
    tui.report.surface_integrals.sum(f"{index_1}" , 'uds-0-scalar', 'yes', f'{points_file}')    
    with open(points_file, 'r') as file_read:
        point_result = file_read.readlines()
    for j in range(len(points_list)):
        temp = round(1.0 * float(re.findall(r"-?\d+\.?\d*e?-?\d*", point_result[5+j])[1]), 5)
        if temp < 0:
            temp = 0
        point_true_value.append(temp)
    return (point_true_value)
        
# 污染物云图显示
def show_contour():
    # root.results.graphics.contour['contour-1'].get_active_child_names()
    # root.results.graphics.contour['contour-1']()
    # root.results.graphics.contour()   
    root.results.graphics.contour['contour-1'] = {'name': 'contour-1'}
    root.results.graphics.contour['contour-1'] = {'field': 'uds-0-scalar'}
    root.results.graphics.contour['contour-1'] = {'filled' : False}
    root.results.graphics.contour['contour-1'] = {'range_option': {'option': 'auto-range-on', 'auto_range_on': {'global_range': True}}}
    root.results.graphics.contour['contour-1'] = {'surfaces_list': ['air', 'bottom', 'left', 'right', 'top']}
    root.results.graphics.contour['contour-1'].color_map = {'format': '%0.3f'}
    root.results.graphics.contour['contour-1'].display()

# 根据坐标创建监测点
def create_points(points_list):   
    for i in range(len(points_list)):
        index = f'point-{i+1}'
        X_position = f'{points_list[i][0]}'
        Y_position = f'{points_list[i][1]}'
        tui.surface.point_surface(f"{index}", f'{X_position}', f'{Y_position}')

# 用于返回监测点处的浓度值
def direct_fluent(X_current = 0.0, Y_current = 0.0): 
    #------------------------------------------------------------------------------
    #下面进行UDF的数据操作...执行此语句前一定先运行socket函数
    with open(XY_file, 'w') as file_write:
        space = ' '
        file_write.write(str(X_current) + space + str(Y_current))
    # 执行Define_on_demand宏
    tui.define.user_defined.execute_on_demand ('"python_udf_socket::libudf_socket"')  
    #UDF数据操作结束...
    # -----------------------------------------------------------------------------
    # 初始化patch
    tui.solve.patch('air', [], 'uds-0', '0')
    # fluent计算
    tui.solve.set.equations('flow', 'no')
    tui.solve.set.equations('ke', 'no')
    tui.solve.set.equations('uds-0', 'yes')
    tui.solve.iterate(10)   
    # 统计真实污染源时监测点处污染物的浓度
    point_true_value = get_points_value_IO(points_list)
    # 返回值为每个测点处污染物的浓度
    return point_true_value

# 用于计算正态建议分布值
def normal_proposal_distribution(t, sigma_X = 0, sigma_Y = 0):
    X_new = norm.rvs(loc=X[t - 1], scale=sigma_X, size=1, random_state=None)[0] #建议分布
    Y_new = norm.rvs(loc=Y[t - 1], scale=sigma_Y, size=1, random_state=None)[0] #建议分布
    # X坐标超标罚回
    while X_new < 10 or X_new > 990:
        X_new = norm.rvs(loc=X[t - 1], scale=sigma_X,   size=1, random_state=None)[0]
    # Y坐标超标罚回
    while Y_new < -240 or Y_new > 240:
        Y_new = norm.rvs(loc=Y[t - 1], scale=sigma_Y, size=1, random_state=None)[0]
    # 返回值为新的污染物坐标    
    return (X_new, Y_new)
      
# 用于计算均匀建议分布值  
def uniform_proposal_distribution(t, delta_X = 0, delta_Y = 0):
    X_new = random.uniform(X[t - 1] - delta_X, X[t - 1] + delta_X)   #建议分布
    Y_new = random.uniform(Y[t - 1] - delta_Y, Y[t - 1] + delta_Y)     #建议分布
    # X坐标超标罚回
    while X_new < 10 or Y_new > 990:
        X_new = random.uniform(X[t - 1] - delta_X, X[t - 1] + delta_X)
    # Y坐标超标罚回
    while Y_new < -240 or Y_new > 240:
        Y_new = random.uniform(Y[t - 1] - delta_Y, Y[t - 1] + delta_Y)
    # 返回值为新的污染物坐标   
    return (X_new, Y_new)

# 用于计算似然函数
def likelihood_func(sigma, c_predict, c_true):
    temp = 0
    for i in range(len(c_predict)):   
        temp = temp + (-(c_predict[i] - c_true[i])**2/(2*sigma**2))
    return (np.exp(temp))  

#%% 读入case和data文件, 创建监测点
root.file.read(file_type="case", file_name=import_filename)
root.file.read(file_type="data", file_name=import_filename)

#%% 生成监测点数据

# 首先加载污染源UDF
# 不能在此进行编译，因UDF采用C++，所以需要进行外部边界，在此只进行加载
# tui.define.user_defined.compiled_functions('compile', 'libudf', 'yes', 'udf_source.cpp')
tui.define.use_defined.compiled_functions('load' , 'libudf_socket')

# 设定真实污染源的位置 
X_true = 500.0
Y_true = 120.0

# 设定监测点的坐标
points_list = [[200,150],
               [500,150],
               [800,150],
               [200,0],
               [500,0],
               [800,0],
               [200,-150],
               [500,-150],
               [800,-150],]

# fluent中创建监测点
create_points(points_list)

# fluent计算，返回值为真实污染源时——监测点处污染物的浓度
point_true_value = direct_fluent(X_true, Y_true)

# 打印真实污染源时监测点处污染物的浓度
for i in range(len(point_true_value)):
    print(f"point_{i+1}_true_value = {point_true_value[i]}") 

# contour显示
show_contour()

# %%网格计算方法

# 生成网格坐标
X = np.linspace(50, 950, 91)
Y = np.linspace(-200, 200, 41)
X, Y = np.meshgrid(X, Y)

# 定义预测值
c_predict = np.zeros([9,41,91])

################################### 迭代开始 ###################################
start = time.perf_counter() 
for i in tqdm(range(41)):
    for j in range(91):      
        value = direct_fluent(X_current = X[i, j], Y_current = Y[i, j])
        for z in range(9):
            c_predict[z,i,j] = value[z]
            
end = time.perf_counter()
print("运行时间为", round(((end-start)/60), 2), 'mins')           
################################### 迭代结束 ###################################   

# %%计算似然函数
sigma = likelihood_func(0.01, c_predict, point_true_value)

#%% 显示图

# 污染物扩散二维等值线图
matplotlib.pyplot.contour(X, Y, c_predict[4,:,:], colors=list(["purple","blue","cyan", "green","yellow","orange","red"]), levels = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4], linestyles=['-'])

# 二维后验密度图
matplotlib.pyplot.contour(X, Y, sigma, colors=list(["purple","blue","cyan", "green","yellow","orange","red"]), levels = 7, linestyles=['-'])

# 三维后验密度图
fig = plt.figure(figsize=(12,6))
# 转换为三维
ax = Axes3D(fig)

# set the limits
ax.set_xlim([0,1000])
ax.set_ylim([-250,250])
ax.set_box_aspect((2, 1, 0.2))
surf = ax.plot_surface(X, Y, sigma, rstride=1, cstride=1, edgecolor='black', cmap='rainbow')
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

#%% 寻找后验概率密度最大点

temp_value_x = np.amax(sigma, axis=1)   #行最大值
temp_value_y = np.amax(sigma, axis=0)   #列最大值
temp_index_x = np.argmax(sigma, axis=1) #行最大值索引
temp_index_y = np.argmax(sigma, axis=0) #列最大值索引

temp_x = np.array(list(zip(temp_index_x,temp_value_x)))
temp_y = np.array(list(zip(temp_index_y,temp_value_y)))

# X_find = temp_x[(np.argmax(temp_x, axis=0)[1]),0]/100 * 0.01 + 0.01
# Y_find = temp_y[(np.argmax(temp_y, axis=0)[1]),0]/100 * 1.0  + 1.5
# print(X_find)
# print(Y_find)
