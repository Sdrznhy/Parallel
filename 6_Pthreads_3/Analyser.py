import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 使用pd读取数据
pthreads_data = pd.read_csv('output/heated_plate_pthreads.txt',sep=" ",names=["thread_num","time"])
openmp_data = pd.read_csv('output/heated_plate_openmp.txt',sep=" ",names=["thread_num","time"])

# 将线程数转换为字符串
pthreads_data["thread_num"] = pthreads_data["thread_num"].astype(str)
openmp_data["thread_num"] = openmp_data["thread_num"].astype(str)

# 使用seaborn绘制图表
sns.set(style="whitegrid")
plt.figure(figsize=(10,6))
sns.set_context("paper", font_scale=1.5)
sns.lineplot(x="thread_num",y="time",data=pthreads_data,marker="o",label="Pthreads")
sns.lineplot(x="thread_num",y="time",data=openmp_data,marker="o",label="OpenMP")

# 设置标题
plt.title("Heated Plate Execution Time")
plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (s)")
# 添加图例
plt.legend(title="Method",loc="upper left")

plt.show()