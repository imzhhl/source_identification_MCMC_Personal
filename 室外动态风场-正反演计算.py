# %% 导入库

import ansys.fluent.core as pyfluent
import numpy as np
import random
import time
import re
import os
import pathlib
from tqdm import tqdm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.stats import norm
import sys
import csv
from scipy.integrate import solve_bvp
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
from mpl_toolkits.mplot3d import Axes3D

#%% 利用pyFluent包连接fluent,该包的使用仅限于Fluent 2022R2以后版本

# 定义工作目录
import_filename = pathlib.Path(r"D:\ZHHL_date\case22-06-23\case_cal_again\fluent")
UDP_Path = pathlib.Path(r'D:\ZHHL_date\case22-06-23\case_cal_again\udf_boundary.c')

session = pyfluent.launch_fluent(version = "2d", precision='double',processor_count = 10, show_gui=True)

# 用工作站远程计算，连接现有的窗口
# session = pyfluent.launch_fluent(ip='192.168.31.230', port=63993, start_instance=False)

# 从session中提取tui和root命令
tui = session.solver.tui
root = session.solver.root

# 读入case和data文件
root.file.read(file_type="case", file_name=import_filename)

# %% 一些需要的函数定义

#更改UDF文件函数
def change_UDF_file_x_velocity(x):
    with open(UDP_Path, 'r') as old_file:
        with open(UDP_Path, 'r+') as new_file:
            current_line = 0
            # 定位到需要删除的行
            while current_line < (17 - 1):
                old_file.readline()
                current_line += 1
         
            seek_point = old_file.tell() # 当前光标在被删除行的行首，记录该位置       
            new_file.seek(seek_point, 0) # 设置光标位置            
            new_file.write(f'		F_PROFILE(f,t,i) = {x};\n')  # 在该行设置新内容      
            old_file.readline() # 读需要删除的行，光标移到下一行行首      
            next_line = old_file.readline() # 被删除行的下一行读给 next_line
            # 连续覆盖剩余行，后面所有行上移一行
            while next_line:
                new_file.write(next_line)
                next_line = old_file.readline()
            new_file.truncate() # 写完最后一行后截断文件，因为删除操作，文件整体少了一行，原文件最后一行需要去掉 
            
#更改UDF文件函数
def change_UDF_file_y_velocity(y):
    with open(UDP_Path, 'r') as old_file:
        with open(UDP_Path, 'r+') as new_file:
            current_line = 0
            # 定位到需要删除的行
            while current_line < (27 - 1):
                old_file.readline()
                current_line += 1
         
            seek_point = old_file.tell() # 当前光标在被删除行的行首，记录该位置       
            new_file.seek(seek_point, 0) # 设置光标位置            
            new_file.write(f'		F_PROFILE(f,t,i) = {y};\n')  # 在该行设置新内容      
            old_file.readline() # 读需要删除的行，光标移到下一行行首      
            next_line = old_file.readline() # 被删除行的下一行读给 next_line
            # 连续覆盖剩余行，后面所有行上移一行
            while next_line:
                new_file.write(next_line)
                next_line = old_file.readline()
            new_file.truncate() # 写完最后一行后截断文件，因为删除操作，文件整体少了一行，原文件最后一行需要去掉   
            
# 通过I/O接口获取监测点变量的值
def get_points_value_IO(points_list):
    point_result = []
    point_forward_value = []
    point_inverse_value = []
    i = 0
    j = 0
    k = 0
    s =''
    for i in range(len(points_list)):
        s = s + f'point-{i+1}'
        if i < len(points_list)-1:
            s = s + ' '
    index_1 = '(' + s + ')'
    points_file = r'D:\ZHHL_date\case22-06-23\case_cal_again\points_value'
    if os.path.exists(f"{points_file}"):
        os.remove(f"{points_file}")
    tui.report.surface_integrals.sum(f"{index_1}" , 'uds-0-scalar', 'yes', f'{points_file}')
    
    tui.report.surface_integrals.sum('(point-0)' , 'uds-1-scalar', 'yes', f'{points_file}')
    tui.report.surface_integrals.sum('(point-0)' , 'uds-2-scalar', 'yes', f'{points_file}')
    tui.report.surface_integrals.sum('(point-0)' , 'uds-3-scalar', 'yes', f'{points_file}')
    
    with open(points_file, 'r') as file_read:
        point_result = file_read.readlines()
    for j in range(len(points_list)):
        temp = round(1.0 * float(re.findall(r"-?\d+\.?\d*e?-?\d*", point_result[5+j])[1]), 5)
        if temp < 0:
            temp = 0
        point_forward_value.append(temp)
    
    for k in range(len(points_list)):
        temp = round(1.0 * float(re.findall(r"-?\d+\.?\d*e?-?\d*", point_result[14+5*k])[1]), 5)
        if temp < 0:
            temp = 0
        point_inverse_value.append(temp)
        
    return point_forward_value, point_inverse_value

# %% 主过程

#从csv文件中提取风向和风速信息
filename = 'D:\ZHHL_date\case22-06-23\case_cal_again\风向风速信息.csv'
with open(filename) as f:
    reader = csv.reader(f)
    head_row = next(reader)
    
    x_velocitys = []
    y_velocitys = []
    for row in reader:
        x_velocity = float(row[4])
        y_velocity = float(row[5])
        x_velocitys.append(x_velocity)  
        y_velocitys.append(y_velocity)

points_list = [[600, 150],
               [850, 80],
               [300, -140]]
        
#scheme.execSchemeToString(r'(read-case "D:\ZHHL_date\case22-06-23\fluent.cas")')

# 编译污染源UDF并加载
# tui.define.user_defined.compiled_functions('compile', 'libudf-s', 'yes', 'udf_source.c', '')
tui.define.user_defined.compiled_functions('load' , 'libudf-s')

# 存储逐时污染物浓度的数组
points_forward_values_history = []
points_inverse_values_history = []

# 初始化
# tui.solve.patch('(air air:005 air:015 air:019 air:023)', 'uds-0', '0')
# tui.solve.patch('(air air:005 air:015 air:019 air:023)', 'uds-1', '0')
# tui.solve.patch('(air air:005 air:015 air:019 air:023)', 'uds-2', '0')
# tui.solve.patch('(air air:005 air:015 air:019 air:023)', 'uds-3', '0')

for i in range(300):
#    if i==1:
#        scheme.doMenuCommandToString("/define/user-defined/compiled-functions unload libudf-s")
       
    # 正演过程更改UDF文件
    change_UDF_file_x_velocity(x_velocitys[i])
    change_UDF_file_y_velocity(y_velocitys[i])
    
    # 编译修改后的风场UDF并加载
    tui.define.user_defined.compiled_functions('compile', 'libudf-b', 'yes', 'udf_boundary.c', '')
    tui.define.user_defined.compiled_functions('load' , 'libudf-b')
    
    # 改为稳态计算风场
    tui.solve.set.equations('flow', 'yes')
    tui.solve.set.equations('ke', 'yes')
    tui.solve.set.equations('uds-0', 'no')
    tui.solve.set.equations('uds-1', 'no')
    tui.solve.set.equations('uds-2', 'no')
    tui.solve.set.equations('uds-3', 'no')
    tui.define.models.steady('yes')   
    tui.solve.iterate(200) 
    
    # 改为瞬态计算污染物
    tui.solve.set.equations('flow', 'no')  
    tui.solve.set.equations('ke', 'no')
    tui.solve.set.equations('uds-0', 'yes')
    tui.solve.set.equations('uds-1', 'yes')
    tui.solve.set.equations('uds-2', 'yes')
    tui.solve.set.equations('uds-3', 'yes')    
    
    tui.define.models.unsteady_1st_order('yes')
    tui.solve.set.transient_controls.number_of_time_steps(10)
    tui.solve.set.transient_controls.time_step_size(0.1)
    tui.solve.set.transient_controls.max_iterations_per_time_step(20)
    tui.solve.dual_time_iterate(10, 20)
    
    # 统计污染物的浓度
    points_forward_values_history.append(get_points_value_IO(points_list))
    points_inverse_values_history.append(get_points_value_IO(points_list))
    
    # 存储每个时刻计算的结果
    tui.file.write_case(f"cuttent_time={i}")
    tui.file.write_data(f"cuttent_time={i}")
    print(f"current i = {i}")
    
