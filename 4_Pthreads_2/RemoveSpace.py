with open("output/MonteCarlo.txt", "r") as f:
    lines = f.readlines()

# 去除每一行末尾的空格
lines = [line.rstrip() for line in lines]

# 将处理后的行写回文件
with open("output/MonteCarlo.txt", "w") as f:
    f.write("\n".join(lines))