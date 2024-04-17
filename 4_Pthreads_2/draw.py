# 画一个第一象限的半径为1的四分之一圆与边长为1的正方形
# 用于展示蒙特卡洛方法的计算过程
import matplotlib.pyplot as plt
import numpy as np

# 画圆
theta = np.linspace(0, np.pi/2, 100)
x = np.cos(theta)
y = np.sin(theta)
plt.plot(x, y, 'r')

# 画正方形
plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'black')

# # 画随机点
# n = 1000
# x = np.random.rand(n)
# y = np.random.rand(n)

# # 落在圆内的点标红
# r = np.sqrt(x**2 + y**2)
# plt.scatter(x[r <= 1], y[r <= 1], color='red')

# # 落在圆外的点标蓝
# plt.scatter(x[r > 1], y[r > 1], color='blue')

# 设置坐标轴范围
plt.xlim(0, 1)
plt.ylim(0, 1)

# 设置图像为正方形
plt.gca().set_aspect('equal', adjustable='box')

# 显示图像
plt.show()