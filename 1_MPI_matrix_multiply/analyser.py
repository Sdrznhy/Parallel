import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file = '1_MPI_matrix_multiply/output/P2P.txt'
# file = '1_MPI_matrix_multiply/output/GrpMsg.txt'

# 读取数据
data = pd.read_csv(file, sep=' ', names=['Matrix_size', 'Number_of_Processes', 'Time'])

# 按矩阵规模分组
grouped = data.groupby('Matrix_size')

# 对每个组计算相对加速比，并重置索引
relative_speedup = grouped.apply(lambda g: g[g['Number_of_Processes'] == 1]['Time'].values[0] / g['Time']).reset_index(drop=True)

# 将相对加速比插入到 data DataFrame 中
data['Relative_Speedup'] = relative_speedup

# 计算线程增加到平均加速比
average_speedup = data.groupby('Number_of_Processes')['Relative_Speedup'].mean()

# 创建一个箱线图，显示每个矩阵规模下的加速比的分布
plt.figure(figsize=(10, 6))
sns.boxplot(x='Matrix_size', y='Relative_Speedup', data=data)
plt.title('Relative Speedup by Matrix Size')
plt.show()

print(data)
print(average_speedup)