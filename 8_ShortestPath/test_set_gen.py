# 向test_data.txt中写入测试数据
# 生成测试数据的格式为：每行两个整数，表示两个节点的编号

import random

file = open("test_data.txt", "w")
dataset = set()
for i in range(100):
    a = random.randint(1, 930)
    b = random.randint(1, 930)
    if a != b:
        dataset.add((a, b))

for data in dataset:
    file.write(str(data[0]) + " " + str(data[1]) + "\n")
file.close()
