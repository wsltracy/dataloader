import open3d as o3d
import numpy as np

def visualize_npz_pointcloud(npz_path):
    """
    可视化.npz格式的点云文件
    
    Args:
        npz_path: .npz文件的路径
    """
    # 加载.npz文件
    data = np.load(npz_path)
    
    # 打印文件中的所有键名，以便调试
    print("Keys in npz file:", list(data.keys()))
    
    
    # 获取点云数据
    points = data['data']
    print(f"Point cloud shape: {points.shape}")
        # 检查点云数据的维度
    if points.shape[1] >= 3:
        # 只使用前三个坐标（x, y, z）
        xyz = points[:, :3]
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # 如果点云有颜色信息，也可以添加颜色
    # if 'colors' in data:
    #     colors = data['colors']
    #     pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 可视化点云
    o3d.visualization.draw_geometries([pcd])

# 使用示例
visualize_npz_pointcloud('/media/wsl/SANDISK ELE/dataset/DDAD/ddad_train_val/000012/point_cloud/LIDAR/15621795824731086.npz')
