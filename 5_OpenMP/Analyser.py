import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# file = "output/MM_static.txt"
# file = "output/MM_dynamic_256.txt"
# file = "output/MatrixMultiply.txt"
# file = "output/MM_PthreadFor.txt"


def analyse(file):
    # 读取数据
    data = pd.read_csv(
        file, sep=" ", names=["Matrix_size", "Number_of_threads", "Time"]
    )

    # 按照矩阵规模和进程数分组，计算平均时间
    grouped = data.groupby(["Matrix_size", "Number_of_threads"]).mean().reset_index()

    # 为每一行数据计算加速比，以每种规模的单线程时间为基准
    for size in grouped["Matrix_size"].unique():
        base_time = grouped[
            (grouped["Matrix_size"] == size) & (grouped["Number_of_threads"] == 1)
        ]["Time"].values[0]
        grouped.loc[grouped["Matrix_size"] == size, "Speedup"] = (
            base_time / grouped["Time"]
        )

    # 将矩阵规模用2的幂次方表示，并转换为字符串，方便绘图
    # grouped["Matrix_size"] = grouped["Matrix_size"].apply(lambda x: str(int(np.log2(x))))

    # 将矩阵规模转换为字符串，方便绘图
    grouped["Matrix_size"] = grouped["Matrix_size"].apply(lambda x: str(x))

    # 按照规模绘制加速比曲线
    sns.set(style="whitegrid")
    plt.figure()
    sns.lineplot(
        data=grouped, x="Matrix_size", y="Speedup", hue="Number_of_threads", marker="o"
    )
    # plt.title("OpenMP, schedule=dynamic, chunk=256")
    # 去掉MM_前缀作为标题
    title = file[7:-4]
    title = title[3:]
    plt.title(title)
    # plt.title("Pthread For")
    plt.xlabel("Matrix_size")
    plt.ylabel("Speedup")
    plt.legend(title="Number of Threads", loc="upper left")

    # plt.show()
    # 保存图片
    plt.savefig("assets/{}.png".format(title))


if __name__ == "__main__":
    for file in [
        "output/MM_static.txt",
        "output/MM_dynamic_1.txt",
        "output/MM_dynamic_4.txt",
        "output/MM_dynamic_16.txt",
        "output/MM_dynamic_64.txt",
        "output/MM_dynamic_256.txt",
        "output/MM_PthreadFor.txt",
    ]:
        analyse(file)

    # analyse("output/MM_static.txt")
# 使用pd输出markdown表格，只输出1024规模的数据
# print(grouped[grouped["Matrix_size"] == "1024"].to_markdown(index=False))
