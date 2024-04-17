import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 载入数据
file = 'output/loss.txt'
data = pd.read_csv(file, sep=' ', header=None, names=['size', 'loss'])

# 按照size分组，计算loss的均值
data = data.groupby('size').mean().reset_index()
# print(data)

# 将向量规模用2的幂次方表示，并转换为字符串，方便绘图
data['size'] = data['size'].apply(lambda x: str(int(np.log2(x))))

# 绘制图像
sns.set(style='whitegrid')
plt.figure()
sns.lineplot(data=data, x='size', y='loss', marker='o')

plt.xlabel('Point_num(log2)')
plt.ylabel('Loss')
plt.title('Loss of different point_num')
plt.show()
