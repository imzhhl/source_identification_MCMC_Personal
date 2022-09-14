# %%
# 升级版，速度提升8倍（主要是function函数的调用次数减少很多）
# python库导入
import time
import random
import seaborn as sns
from tqdm import tqdm
from scipy.integrate import solve_bvp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
matplotlib.rc("font", family='Microsoft YaHei')

# %%
# 函数定义

# 用于计算稳态Streets-Phelps模型
def function(D_current = 2, k_current = 0.015, loc_1 = 0, loc_2 = 0, loc_3 = 0, loc_4 = 0):
    u = 1
    c_0 = 1
    L = 15
        
    def func(x,c):
        # 计算 dc0/dx, dc1/dx 的值
        dc0 = c[1]  # 计算 dh0/dx
        dc1 = (k_current * c[0] + u * c[1]) / D_current # 计算 dc1/dx
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
    loc_1_value = np.array(list(zip(xSol,cSol)))[loc_1*10, 1]
    loc_2_value = np.array(list(zip(xSol,cSol)))[loc_2*10, 1]
    loc_3_value = np.array(list(zip(xSol,cSol)))[loc_3*10, 1]
    loc_4_value = np.array(list(zip(xSol,cSol)))[loc_4*10, 1]
    
    return (loc_1_value, loc_2_value, loc_3_value, loc_4_value)

# 用于计算正态建议分布值
def normal_proposal_distribution(t, sigma_D = 0.1, sigma_k = 0.01, D = [], k = []):
    D_new = norm.rvs(loc=D[t - 1], scale=sigma_D, size=1, random_state=None)[0] #建议分布
    k_new = norm.rvs(loc=k[t - 1], scale=sigma_k, size=1, random_state=None)[0] #建议分布
    
    while D_new < 0 or D_new > 10:
        D_new = norm.rvs(loc=D[t - 1], scale=sigma_D,   size=1, random_state=None)[0]

    while k_new < 0 or k_new > 0.1:
        k_new = norm.rvs(loc=k[t - 1], scale=sigma_k, size=1, random_state=None)[0]
        
    return (D_new, k_new)
      
# 用于计算均匀建议分布值  
def uniform_proposal_distribution(t, delta_D = 1, delta_k = 0.01,  D = [], k = []):
    D_new = random.uniform(D[t - 1] - delta_D, D[t - 1] + delta_D)   #建议分布
    k_new = random.uniform(k[t - 1] - delta_k, k[t - 1] + delta_k)     #建议分布
    
    while D_new < 0 or D_new > 10:
        D_new = random.uniform(D[t - 1] - delta_D, D[t - 1] + delta_D)
    
    while k_new < 0 or k_new > 0.1:
        k_new = random.uniform(k[t - 1] - delta_k, k[t - 1] + delta_k)
        
    return (D_new, k_new)

# 用于计算似然函数
def likelihood_func(sigma, c, c_monitor):
    temp = 0
    for i in range(len(c)):   
        temp = temp + (-(c[i] - c_monitor[i])**2/(2*sigma**2))
    return (np.exp(temp))

# %%关键参数设定及变量声明   

T = 50000    # MCMC迭代总步数
sigma = 0.05 #似然函数的标准差
is_normal_proposal = True  #选择建议分布为正态分布还是均匀分布(True or False)
 
# 监测点值
c_1_monitor = 0.96
c_2_monitor = 0.91
c_3_monitor = 0.84
c_4_monitor = 0.67
c_monitor   = [c_1_monitor, c_2_monitor, c_3_monitor, c_4_monitor]

D_real_value = 2.0   # D-真值
k_real_value = 0.015 # k-真值

t = 0       # 循环迭代索引
count = 0   # 用于计算接受率标识
flag = True # 用于记录是否接受了新值

D_init = 5    # D初始点(需在先验范围内)
k_init = 0.01 # k初始点(需在先验范围内)

D = [0 for i in range(T)] # 迭代历史记录
k = [0 for i in range(T)] # 迭代历史记录

#%%

######## 迭代开始 ##########
start = time.perf_counter() 

# while t < T-1: 
for t in tqdm(range(T-1)): 
    t = t + 1 
    if is_normal_proposal:
        # 建议分布采用正态分布
        sigma_D = 1    # 设置标准差
        sigma_k = 0.01 # 设置标准差
        D_new = normal_proposal_distribution(t, sigma_D, sigma_k, D, k)[0]
        k_new = normal_proposal_distribution(t, sigma_D, sigma_k, D, k)[1]
    else:
        # 建议分布采用均匀分布    
        delta_D = 1
        delta_k = 0.01
        D_new = uniform_proposal_distribution(t, delta_D, delta_k, D, k)[0]
        k_new = uniform_proposal_distribution(t, delta_D, delta_k, D, k)[1]
    
    c_new = function(D_current = D_new,  k_current = k_new, loc_1 = 3, loc_2 = 6, loc_3 = 9, loc_4 = 12 )
    
    if t == 1:
        c_now = function(D_current = D_init,  k_current = k_init, loc_1 = 3, loc_2 = 6, loc_3 = 9, loc_4 = 12 )

    likelihood_j = likelihood_func(sigma, c_new, c_monitor)
    likelihood_i = likelihood_func(sigma, c_now, c_monitor)
    alpha = min(1, likelihood_j / likelihood_i)
    
    u = random.uniform(0, 1)
    if u < alpha:    # accept
        D[t]  = D_new
        k[t]  = k_new
        flag  = True
        count = count + 1
    else:           #reject
        D[t]  = D[t - 1]
        k[t]  = k[t - 1]
        flag  = False
        
    # 接受后将新数据存入旧数据
    if flag == True:
        c_now = c_new
    
    # print(f"current i = {t}/{T}")
    # print(f"当前接受率 = {round((count/t), 2)*100}%")

######## 迭代结束 ##########    
end = time.perf_counter()
print(f"最终接受率 = {round((count/T)*100, 2)}%")
print("运行时间为", round(((end-start)/60), 2), 'mins')
# %%
# 绘图后处理
f, ax = plt.subplots(2, 2, figsize = (16, 8))

#直方图统计D
plt.subplot(2,2,1)
sns.distplot(D[-20000:], hist=True, bins=50, kde=True, color='red')
# plt.hist(D[-20000:], bins=100 , density=1, facecolor='red')
plt.title('D-plot')
plt.xlabel('D(km$^2$/h)')
plt.ylabel('KDE')
plt.axvline(D_real_value, color='k')

# 直方图统计k
plt.subplot(2,2,2)
# plt.hist(k[-20000:], bins=100, density=1, facecolor='green')
sns.distplot(k[-20000:], hist=True, bins=50, kde=True, color='green')
plt.title('k-plot')
plt.xlabel('k(h$^{-1}$)')
plt.ylabel('KDE')
plt.axvline(k_real_value, color='k')

#采样值变化D
plt.subplot(2,2,3)
plt.plot(list(range(1, 20001)), D[-20000:])
plt.title('D-plot')
plt.xlabel('iteration')
plt.ylabel('D(km$^2$/h)')
plt.axhline(D_real_value, color='k')

#采样值变化k
plt.subplot(2,2,4)
plt.plot(list(range(1, 20001)), k[-20000:])
plt.title('k-plot')
plt.xlabel('iteration')
plt.ylabel('k(h$^{-1}$)')
plt.axhline(k_real_value, color='k')

plt.tight_layout()
plt.show()

# plt.ylim([0,0.1])
# plt.xlim([0,10])
# plt.plot(D[-5000:],  k[-5000:])
