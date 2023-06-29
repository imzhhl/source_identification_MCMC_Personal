import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# 目标分布
mean = [0, 0]
cov = [[1, 0], [0, 1]]  # 协方差矩阵
target = stats.multivariate_normal(mean, cov).pdf

# 初始化参数
n_samples = 10000  # 采样次数
n_chains = 5  # Markov 链的数量
n_dims = 2  # 参数的维度
z_res = np.random.rand(n_chains, n_samples)
samples = np.random.rand(n_chains, n_samples, n_dims)  # 每条链的初始点

gamma = 2.38 / np.sqrt(2 * n_dims)  # 更新规则中的因子
epsilon = 1e-6  # 保证生成的提议点与当前点不同的小常数

# DREAM
for i in range(1, n_samples):
    for j in range(n_chains):
        # 从剩下的链中选择两条
        r1, r2 = np.random.choice([k for k in range(n_chains) if k != j], size=2, replace=False)
        
        # 计算差分向量
        diff = samples[r1, i - 1, :] - samples[r2, i - 1, :]
        
        # 生成新的提议点
        z = samples[j, i - 1, :] + gamma * diff + epsilon * np.random.randn(n_dims)
        
        # 计算接受概率
        alpha = min(1, target(z) / target(samples[j, i - 1, :]))
        
        # 以 alpha 的概率接受提议点
        if np.random.rand() < alpha:
            samples[j, i, :] = z
        else:
            samples[j, i, :] = samples[j, i - 1, :]
            
        z_res[j,i] = stats.multivariate_normal(mean, cov).pdf([samples[j, i, 0], samples[j, i, 1]])

# 烧入期
burn_in = int(0.1 * n_samples)  # 烧入期，例如前10%的样本
samples = samples[:, burn_in:, :]
z_res = z_res[:, burn_in:]

#%% 绘制图形
plt.figure(figsize=(10, 10))
for i in range(n_chains):
    plt.plot(samples[i, :, 0], samples[i, :, 1], 'b.')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('DREAM sampling from a 2D Gaussian distribution')
plt.show()

#%% 绘制图形
num_bins = 50
for i in range(n_chains):
    plt.hist(samples[i, :, 0], num_bins, density=1, facecolor='green', alpha=0.5)
    plt.hist(samples[i, :, 1], num_bins, density=1, facecolor='red', alpha=0.5)
plt.title('Histogram')
plt.show()

#%% 绘制图形,3维
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
for i in range(1):
    ax.scatter(samples[i, :, 0], samples[i, :, 1], z_res[i,:], marker='o')
plt.show()
