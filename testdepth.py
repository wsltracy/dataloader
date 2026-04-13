import numpy as np
import os
from PIL import Image
def read_pfm(filename):
    """读取 PFM 格式的深度图文件"""
    with open(filename, 'rb') as f:
        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file')
        
        while True:
            dims_line = f.readline().rstrip()
            if dims_line and dims_line[0:1] != b'#':
                break
        
        dims = dims_line.split()
        width = int(dims[0])
        height = int(dims[1])
        scale = float(f.readline().rstrip())
        
        if scale < 0:
            endian = '<'
            scale = -scale
        else:
            endian = '>'
        
        data = np.fromfile(f, dtype=endian + 'f')
        data = data.reshape((height, width))
        
        if color:
            data = data.reshape((height, width, 3))
        # 垂直翻转深度图，因为PFM格式是bottom-up存储的
        data = np.flipud(data)
        return data, scale
def read_matterport_depth(filename):
    """读取 Matterport 格式的深度图文件（PNG格式，单位mm，需要转换为m）"""
    # 使用PIL读取PNG图像
    depth_img = Image.open(filename)
    
    # 转换为numpy数组
    depth = np.array(depth_img, dtype=np.float32)
    # Matterport深度图是16位PNG，需要除以4000转换为米
    depth = depth / 4000.0
    return depth
# 在这里修改为你要测试的npy文件路径
npy_path = "/media/wsl/SANDISK ELE/dataset/tartanair/carwelding/Hard/P001/depth_right/000004_right_depth.npy"
# 在这里修改为你要测试的pfm文件路径
pfm_path = "/media/weishanling/SANDISK ELE/dataset/BlendedMVS/58f7f7299f5b5647873cb110/rendered_depth_maps/00000013.pfm"
matterport_path = "/media/wsl/SANDISK ELE/dataset/matterport/data/v1/scans/1LXtFkjw3qL/undistorted_depth_images/1LXtFkjw3qL/undistorted_depth_images/ddb93f6063d54365bc6e8c751fd2e698_d2_2.png"

# 加载pfm文件
# data,_ = read_pfm(pfm_path)
# 加载npy文件
# data = np.load(npy_path)
# 加载Matterport深度图
data = read_matterport_depth(matterport_path)


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
max_coords = np.where(data == data.max())
print(f"最大值坐标: {tuple(zip(*max_coords))}")  # 输出最大值的所有坐标位置
# 打印数据分布
print("\n=== 数据分布 ===")
print(f"零值数量: {np.sum(data == 0)} ({np.sum(data == 0) / data.size * 100:.2f}%)")
print(f"非零值数量: {np.sum(data != 0)} ({np.sum(data != 0) / data.size * 100:.2f}%)")

print(f"大于50的值数量: {np.sum(data > 50)} ({np.sum(data > 50) / data.size * 100:.2f}%)")

print(f"大于80的值数量: {np.sum(data > 80)} ({np.sum(data > 80) / data.size * 100:.2f}%)")


# 统计
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
