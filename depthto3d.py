import numpy as np
import open3d as o3d
import cv2
import torch
from tartan import TartanAirV1Dataset
import matplotlib.pyplot as plt

def depth_to_pointcloud(rgb_image, depth_image, K, depth_scale=1.0):
    """
    将RGB-D图像转换为点云
    
    Args:
        rgb_image: RGB图像 (H, W, 3) numpy数组，值域0-255
        depth_image: 深度图 (H, W) numpy数组，单位米
        K: 相机内参矩阵 (3, 3)
    
        depth_scale: 深度缩放因子（TartanAir已经是米，所以=1.0）
    """
    h, w = depth_image.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # 生成像素坐标网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # 获取深度值
    depth = depth_image.astype(np.float32)
    
    # 计算3D坐标
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 过滤无效点（深度为0、超出范围或无效）
    valid_mask = (depth > 0) & np.isfinite(depth)
    
    # 提取有效点
    points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=-1)
    colors = rgb_image[valid_mask].astype(np.float32) / 255.0
    
    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd, valid_mask

def visualize_stereo_pointclouds(left_rgb, left_depth, right_rgb, right_depth, 
                                 K, baseline=0.25):
    """
    可视化双目点云（左目坐标系为参考）
    
    Args:
        baseline: 基线距离（TartanAir V1 的基线距离）
    """
    # 生成左右目点云
    left_pcd, _ = depth_to_pointcloud(left_rgb, left_depth, K)
    right_pcd, _ = depth_to_pointcloud(right_rgb, right_depth, K)
    
    # 将右目点云变换到左目坐标系
    # TartanAir V1 是标准的水平双目，右目在左目右侧
    transform = np.eye(4)
    transform[0, 3] = -baseline  # X轴负方向平移
    right_pcd.transform(transform)
    
    # 可选：给左右点云不同颜色以便区分
    # left_pcd.paint_uniform_color([1, 0, 0])   # 红色
    # right_pcd.paint_uniform_color([0, 1, 0])  # 绿色
    
    # 合并点云
    merged_pcd = left_pcd + right_pcd
    
    return left_pcd, right_pcd, merged_pcd

def load_specific_frame(dataset, traj_name, frame_idx):
    """
    从dataset中加载特定帧
    """
    # 在all_frames中查找特定轨迹和帧索引
    for i, frame in enumerate(dataset.all_frames):
        if frame['traj'] == traj_name and frame['frame_idx'] == frame_idx:
            # 手动加载该帧
            left_img = dataset._load_image(frame['left_image_path'])
            left_depth = dataset._load_depth(frame['left_depth_path'])
            right_img = dataset._load_image(frame['right_image_path'])
            right_depth = dataset._load_depth(frame['right_depth_path'])
            K = dataset.K
            
            return {
                'left_image': left_img,
                'left_depth': left_depth,
                'right_image': right_img,
                'right_depth': right_depth,
                'K': K,
                'traj': traj_name,
                'frame_idx': frame_idx,
                'left_path': frame['left_image_path'],
                'right_path': frame['right_image_path']
            }
    
    raise ValueError(f"未找到帧: traj={traj_name}, frame_idx={frame_idx}")

def visualize_frame_2d(frame_data):
    """
    可视化2D图像和深度图用于检查
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 左目RGB
    axes[0, 0].imshow(frame_data['left_image'])
    axes[0, 0].set_title(f"Left RGB - {frame_data['traj']} frame {frame_data['frame_idx']}")
    axes[0, 0].axis('off')
    
    # 右目RGB
    axes[0, 1].imshow(frame_data['right_image'])
    axes[0, 1].set_title("Right RGB")
    axes[0, 1].axis('off')
    
    # 左目深度 - 使用图像自身的最小最大值进行归一化
    left_depth = frame_data['left_depth']
    left_min, left_max = left_depth[left_depth > 0].min(), left_depth[left_depth > 0].max()
    left_depth_disp = (left_depth - left_min) / (left_max - left_min)
    left_depth_disp[left_depth <= 0] = 0  # 保持无效深度为0
    im1 = axes[1, 0].imshow(left_depth_disp, cmap='jet')
    axes[1, 0].set_title(f"Left Depth ({left_min:.2f}m - {left_max:.2f}m)")
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # 右目深度 - 使用图像自身的最小最大值进行归一化
    right_depth = frame_data['right_depth']
    right_min, right_max = right_depth[right_depth > 0].min(), right_depth[right_depth > 0].max()
    right_depth_disp = (right_depth - right_min) / (right_max - right_min)
    right_depth_disp[right_depth <= 0] = 0  # 保持无效深度为0
    im2 = axes[1, 1].imshow(right_depth_disp, cmap='jet')
    axes[1, 1].set_title(f"Right Depth ({right_min:.2f}m - {right_max:.2f}m)")
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()


def main():
    # 初始化数据集（设置为test模式以便固定索引）
    dataset = TartanAirV1Dataset(
        split="test",  # 使用test模式，按顺序读取
        root_dir="/media/wsl/SANDISK ELE/dataset/tartanair",
        env="carwelding",  # 根据你的数据修改
        difficulty="Hard",  # 根据你的数据修改
        traj_id=None,  # 可以指定特定轨迹，如 "P001"
        len_test=1000,
    )
    
    # 方式1：通过索引读取（test模式下按顺序）
    # frame_idx_in_dataset = 0  # 第一个样本
    # sample = dataset[frame_idx_in_dataset]
    
    # 方式2：通过轨迹名和帧号读取
    frame_data = load_specific_frame(dataset, traj_name="P001", frame_idx=200)
    

    
    # 先查看2D图像和深度图
    visualize_frame_2d(frame_data)
    
    # 生成3D点云
    print("正在生成点云...")
    baseline = 0.25  # TartanAir V1 的基线距离（米）
    
    left_pcd, right_pcd, merged_pcd = visualize_stereo_pointclouds(
        frame_data['left_image'],
        frame_data['left_depth'],
        frame_data['right_image'],
        frame_data['right_depth'],
        frame_data['K'],
        baseline=baseline,
        
    )
    
    print(f"左目点数: {len(left_pcd.points)}")
    print(f"右目点数: {len(right_pcd.points)}")
    print(f"合并点数: {len(merged_pcd.points)}")
    
    # 统计深度值分布
    valid_left = frame_data['left_depth'][frame_data['left_depth'] > 0]
    if len(valid_left) > 0:
        print(f"左目深度范围: {valid_left.min():.2f} - {valid_left.max():.2f} 米")
        print(f"左目深度平均值: {valid_left.mean():.2f} 米")
        print(f"左目深度中值: {np.median(valid_left):.2f} 米")

    valid_right = frame_data['right_depth'][frame_data['right_depth'] > 0]
    if len(valid_right) > 0:
        print(f"右目深度范围: {valid_right.min():.2f} - {valid_right.max():.2f} 米")
        print(f"右目深度平均值: {valid_right.mean():.2f} 米")
        print(f"右目深度中值: {np.median(valid_right):.2f} 米")
    
    # 可视化点云
    # 选项1：分别显示左右目点云（便于观察对齐）
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    
    print("\n正在显示左右目点云（红色=左目，绿色=右目）...")
    # 给点云着色以便区分
    left_pcd.paint_uniform_color([1, 0, 0])   # 红色
    right_pcd.paint_uniform_color([0, 1, 0])  # 绿色
    
    o3d.visualization.draw_geometries(
        [left_pcd, right_pcd, coord_frame],
        window_name=f"Stereo Point Clouds - {frame_data['traj']} frame {frame_data['frame_idx']}",
        width=1280, height=720,
        point_show_normal=False
    )
    

    
 



if __name__ == "__main__":
    main()