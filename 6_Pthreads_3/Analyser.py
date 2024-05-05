import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 使用pd读取数据
pthreads_data = pd.read_csv('output/heated_plate_pthreads.txt',sep=" ",names=["iteration","time"])
openmp_data = pd.read_csv('output/heated_plate_openmp.txt',sep=" ",names=["iteration","time"])

# 将迭代次数转换为字符串，方便绘图
pthreads_data["iteration"] = pthreads_data["iteration"].apply(lambda x: str(x))
openmp_data["iteration"] = openmp_data["iteration"].apply(lambda x: str(x))

# 绘制图像
sns.set(style="whitegrid")
plt.figure()
sns.lineplot(data=pthreads_data,x="iteration",y="time",marker="o", label="pthreads")
sns.lineplot(data=openmp_data,x="iteration",y="time",marker="o", label="openmp")
plt.title("openmp vs pthreads")
plt.xlabel("iteration")
plt.ylabel("time")

# 添加图例
plt.legend(title="Method",loc="upper left")

plt.show()