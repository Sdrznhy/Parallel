import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file = 'output/Array.txt'

# 读取数据
data = pd.read_csv(file, sep=' ', names=['Array_size', 'Number_of_threads', 'Time'])

# 按矩阵规模和进程数分组，计算平均时间
grouped = data.groupby(['Array_size', 'Number_of_threads']).mean().reset_index()

# 为每一行数据计算加速比，以每种规模的单线程时间为基准
for size in grouped['Array_size'].unique():
    base_time = grouped[(grouped['Array_size'] == size) & (grouped['Number_of_threads'] == 1)]['Time'].values[0]
    grouped.loc[grouped['Array_size'] == size, 'Speedup'] = base_time / grouped['Time']

# 将向量规模转换为字符串，方便绘图
grouped['Array_size'] = grouped['Array_size'].astype(str)

# 按照规模绘制加速比曲线
sns.set(style='whitegrid')
plt.figure()
sns.lineplot(data=grouped, x='Array_size', y='Speedup', hue='Number_of_threads', marker='o')
plt.title('Array Sum Speedup powered by pthreads')
plt.xlabel('Array Size(M)')
plt.ylabel('Speedup')
plt.legend(title='Number of Threads', loc='upper left')
plt.show()

# print(grouped)