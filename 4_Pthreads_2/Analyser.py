import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file = "output/MonteCarlo.txt"

# 读取数据
data = pd.read_csv(file, sep=" ", names=["Point_num", "Number_of_threads", "Time"])

# print(data)

# 按矩阵规模和进程数分组，计算平均时间
grouped = data.groupby(["Point_num", "Number_of_threads"]).mean().reset_index()
# print(grouped)

# 为每一行数据计算加速比，以每种规模的单线程时间为基准
for size in grouped["Point_num"].unique():
    base_time = grouped[
        (grouped["Point_num"] == size) & (grouped["Number_of_threads"] == 1)
    ]["Time"].values[0]
    grouped.loc[grouped["Point_num"] == size, "Speedup"] = base_time / grouped["Time"]

# 将向量规模用2的幂次方表示，并转换为字符串，方便绘图
grouped["Point_num"] = grouped["Point_num"].apply(lambda x: str(int(np.log2(x))))

# 按照规模绘制加速比曲线
sns.set(style="whitegrid")
plt.figure()
sns.lineplot(
    data=grouped, x="Point_num", y="Speedup", hue="Number_of_threads", marker="o"
)
plt.title("Speedup powered by pthreads")
plt.xlabel("Point_num(log2)")
plt.ylabel("Speedup")
plt.legend(title="Number of Threads", loc="upper left")
plt.show()



