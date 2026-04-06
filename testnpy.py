import numpy as np

# 在这里修改为你要测试的npy文件路径
npy_path = "/media/wsl/SANDISK ELE/dataset/tartanair/carwelding/Hard/P001/depth_left/000057_left_depth.npy"

# 加载npy文件
data = np.load(npy_path)

# 打印基本信息
print("=== 基本信息 ===")
print(f"数据类型: {data.dtype}")
print(f"数据形状: {data.shape}")
print(f"数据维度: {data.ndim}")
print(f"数据大小: {data.size} 个元素")

# 打印统计信息
print("\n=== 统计信息 ===")
print(f"最小值: {data.min()}")
print(f"最大值: {data.max()}")
print(f"均值: {data.mean()}")
print(f"中位数: {np.median(data)}")
print(f"标准差: {data.std()}")

# 打印数据分布
print("\n=== 数据分布 ===")
print(f"零值数量: {np.sum(data == 0)} ({np.sum(data == 0) / data.size * 100:.2f}%)")
print(f"非零值数量: {np.sum(data != 0)} ({np.sum(data != 0) / data.size * 100:.2f}%)")

# 打印一些样本值
print("\n=== 样本值 ===")
if data.ndim == 1:
    print(f"前5个值: {data[:5]}")
    print(f"后5个值: {data[-5:]}")
elif data.ndim == 2:
    print(f"第一行前5个值: {data[0, :5]}")
    print(f"第一列前5个值: {data[:5, 0]}")
elif data.ndim == 3:
    print(f"第一个通道前5x5值:\n{data[0, :5, :5]}")
