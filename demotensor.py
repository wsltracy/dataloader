# demo_tensorboard.py

import sys
sys.path.append('/home/wsl/dataloader')

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from blendedmvs import BlendedMVSDataset  # 请根据实际情况导入你的数据集
from tartan import TartanAirV1Dataset
from matterport import Matterport3DDataset
from dgp.datasets import SynchronizedSceneDataset

random.seed(42)
np.random.seed(42)

BATCH_SIZE = 1
NUM_BATCHES_TO_LOG = 3
LOG_DIR = "runs/mat"

# DDAD
# # 相机配置
# CAMERA_IDS = ['01', '05', '06', '07', '08', '09']
# dataset = SynchronizedSceneDataset(
#     '/media/wsl/SANDISK ELE/dataset/DDAD/ddad_train_val/ddad_2.json',
#     datum_names=('lidar','CAMERA_01', 'CAMERA_05','CAMERA_06','CAMERA_07','CAMERA_08','CAMERA_09'),
#     generate_depth_from_datum='lidar',
#     split='train'
# )
# k_recorded = False


# dataset = BlendedMVSDataset(
#     root_dir="/media/wsl/SANDISK ELE/dataset/BlendedMVS",
#     split="train",
#     depth_max=200.0,
#     len_train=10000,
# )
# dataset = TartanAirV1Dataset(
#     split='train',
#     root_dir='/media/wsl/SANDISK ELE/dataset/tartanair',
#     env='carwelding',
#     difficulty='Hard',
#     traj_id=None,
# )
dataset = Matterport3DDataset(
    root_dir="/media/wsl/SANDISK ELE/dataset/matterport/data/v1/scans",
    split="train",
    depth_max=100.0,
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


writer = SummaryWriter(log_dir=LOG_DIR)

def pil_to_numpy(pil_image):
    """将 PIL Image 转换为 numpy array (H, W, C) 值域 [0,1]"""
    return np.array(pil_image).astype(np.float32) / 255.0
def tensor_to_numpy(tensor):
    """将 (C, H, W) tensor 转换为 (H, W, C) numpy，值域 [0,1]"""
    if tensor.ndim == 3 and tensor.shape[0] in (1,3):
        arr = tensor.permute(1, 2, 0).cpu().numpy()
    else:
        arr = tensor.cpu().numpy()
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr

def depth_to_colormap_numpy(depth_tensor, cmap='jet'):
    """深度图转伪彩色 numpy array (H, W, 3) 值域 [0,1]"""
    if depth_tensor.ndim == 3 and depth_tensor.shape[0] == 1:
        depth_np = depth_tensor.squeeze(0).cpu().numpy()
    else:
        depth_np = depth_tensor.cpu().numpy()
    vmin, vmax = depth_np.min(), depth_np.max()
    if vmax - vmin < 1e-6:
        norm = np.zeros_like(depth_np)
    else:
        norm = (depth_np - vmin) / (vmax - vmin)
    colormap = cm.get_cmap(cmap)
    colored = colormap(norm)[:, :, :3]   # (H, W, 3)
    return colored
def pil_to_colormap(pil_image, cmap='jet'):
    """将PIL Image格式的深度图转换为伪彩色 numpy array (H, W, 3) 值域 [0,1]
    
    Args:
        pil_image: PIL.Image格式的深度图
        cmap: 使用的颜色映射，默认为'jet'
    
    Returns:
        colored: 伪彩色图像，numpy数组，形状为(H, W, 3)，值域[0,1]
    """
    # 将PIL Image转换为numpy数组
    depth_np = np.array(pil_image, dtype=np.float32)
    
    # 创建一个掩码，标识深度值为0的像素
    zero_mask = (depth_np == 0)
    
    # 根据当前深度图的最大最小值进行归一化
    vmin, vmax = depth_np.min(), depth_np.max()
    if vmax - vmin > 1e-6:
        norm = (depth_np - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(depth_np)
    
    # 应用颜色映射
    colormap = cm.get_cmap(cmap)
    colored = colormap(norm)[:, :, :3]   # (H, W, 3)
    
    # 将深度值为0的像素设置为白色
    colored[zero_mask] = [1.0, 1.0, 1.0]  # 白色
    
    return colored
print(f"Dataset size: {len(dataset)}")
print(f"Logging to: {LOG_DIR}")
print(f"Logging first {NUM_BATCHES_TO_LOG} batches...")

# 单目
# for batch_idx, batch in enumerate(dataloader):
#     if batch_idx >= NUM_BATCHES_TO_LOG:
#         break

#     images = batch['image']   # (B, C, H, W)
#     depths = batch['depth']   # (B, 1, H, W)
#     Ks     = batch['K']       # (B, 3, 3)
#     seq_names = batch.get('seq_name', [f"sample_{i}" for i in range(BATCH_SIZE)])

#     B = images.shape[0]
#     for i in range(B):
#         # ---- 图像 I：带标题的 figure ----
#         img_arr = tensor_to_numpy(images[i])
#         fig, ax = plt.subplots(figsize=(6, 4))
#         ax.imshow(img_arr)
#         ax.set_title(seq_names[i], fontsize=10)
#         ax.axis('off')
#         writer.add_figure(f'I/sample_{i}', fig, global_step=batch_idx)
#         plt.close(fig)   # 释放内存

#         # ---- 深度 D：伪彩色，也可以加标题（可选，这里不加） ----
#         depth_color = depth_to_colormap_numpy(depths[i])
#         fig_d, ax_d = plt.subplots(figsize=(6, 4))
#         ax_d.imshow(depth_color)
#         ax_d.set_title(f"Depth of {seq_names[i]}", fontsize=10)
#         ax_d.axis('off')
#         writer.add_figure(f'D/sample_{i}', fig_d, global_step=batch_idx)
#         plt.close(fig_d)

#         # ---- 内参 K：文本记录 ----
#         K = Ks[i].cpu().numpy()
#         k_text = f"fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}"
#         print(f"Batch {batch_idx}, Sample {i} - K: {k_text}")
#         writer.add_text(f'K/sample_{i}', k_text, global_step=batch_idx)

#     print(f"Logged batch {batch_idx+1}/{NUM_BATCHES_TO_LOG}")

# 双目
# for batch_idx, batch in enumerate(dataloader):
#     if batch_idx >= NUM_BATCHES_TO_LOG:
#         break

#     images_left = batch['image_1']   # (B, C, H, W)
#     images_right = batch['image_2']  # (B, C, H, W)
#     depths_left = batch['depth_1']   # (B, 1, H, W)
#     depths_right = batch['depth_2']  # (B, 1, H, W)
#     Ks_left = batch['K_1']           # (B, 3, 3)
#     Ks_right = batch['K_2']          # (B, 3, 3)
#     seq_names = batch.get('seq_name', [f"sample_{i}" for i in range(BATCH_SIZE)])

#     B = images_left.shape[0]
#     for i in range(B):
#         # ---- 左右目图像并排显示 ----
#         img_left_arr = tensor_to_numpy(images_left[i])
#         img_right_arr = tensor_to_numpy(images_right[i])
        
#         # 创建并排图
#         fig_stereo, axes = plt.subplots(1, 2, figsize=(12, 5))
        
#         axes[0].imshow(img_left_arr)
#         axes[0].set_title(f"Left: {seq_names[i]}", fontsize=10)
#         axes[0].axis('off')
        
#         axes[1].imshow(img_right_arr)
#         axes[1].set_title(f"Right: {seq_names[i]}", fontsize=10)
#         axes[1].axis('off')
        
#         fig_stereo.suptitle(f"Stereo Images - Batch {batch_idx}, Sample {i}")
#         writer.add_figure(f'Stereo_Images/sample_{i}', fig_stereo, global_step=batch_idx)
#         plt.close(fig_stereo)
        
#         # ---- 左右目深度并排显示 ----
#         depth_left_color = depth_to_colormap_numpy(depths_left[i])
#         depth_right_color = depth_to_colormap_numpy(depths_right[i])
        
#         fig_depth, axes_d = plt.subplots(1, 2, figsize=(12, 5))
        
#         axes_d[0].imshow(depth_left_color)
#         axes_d[0].set_title(f"Left Depth: {seq_names[i]}", fontsize=10)
#         axes_d[0].axis('off')
        
#         axes_d[1].imshow(depth_right_color)
#         axes_d[1].set_title(f"Right Depth: {seq_names[i]}", fontsize=10)
#         axes_d[1].axis('off')
        
#         fig_depth.suptitle(f"Stereo Depths - Batch {batch_idx}, Sample {i}")
#         writer.add_figure(f'Stereo_Depths/sample_{i}', fig_depth, global_step=batch_idx)
#         plt.close(fig_depth)
        
#         # ---- 内参信息 ----
#         K_left = Ks_left[i].cpu().numpy()
#         K_right = Ks_right[i].cpu().numpy()
#         k_text = f"Left: fx={K_left[0,0]:.2f}, fy={K_left[1,1]:.2f}, cx={K_left[0,2]:.2f}, cy={K_left[1,2]:.2f}\n"
#         k_text += f"Right: fx={K_right[0,0]:.2f}, fy={K_right[1,1]:.2f}, cx={K_right[0,2]:.2f}, cy={K_right[1,2]:.2f}"
#         writer.add_text(f'K_stereo/sample_{i}', k_text, global_step=batch_idx)

#     print(f"Logged batch {batch_idx+1}/{NUM_BATCHES_TO_LOG}")

# 三目
for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= NUM_BATCHES_TO_LOG:
        break

    # 提取三目数据
    images_1 = batch['image_1']   # (B, C, H, W) 相机1
    images_2 = batch['image_2']   # (B, C, H, W) 相机2
    images_3 = batch['image_3']   # (B, C, H, W) 相机3
    
    depths_1 = batch['depth_1']   # (B, 1, H, W)
    depths_2 = batch['depth_2']   # (B, 1, H, W)
    depths_3 = batch['depth_3']   # (B, 1, H, W)
    
    Ks_1 = batch['K_1']           # (B, 3, 3)
    Ks_2 = batch['K_2']           # (B, 3, 3)
    Ks_3 = batch['K_3']           # (B, 3, 3)
    
    seq_names = batch.get('seq_name', [f"sample_{i}" for i in range(BATCH_SIZE)])

    B = images_1.shape[0]
    # ========== 分开显示每个相机 ==========
    for i in range(B):
        # ---- 相机1 (View0) ----
        img_1_arr = tensor_to_numpy(images_1[i])
        fig_cam1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.imshow(img_1_arr)
        ax1.set_title(f"Camera1 (View0): {seq_names[i]}", fontsize=10)
        ax1.axis('off')
        writer.add_figure(f'Camera1_Images/sample_{i}', fig_cam1, global_step=batch_idx)
        plt.close(fig_cam1)
        
        # 相机1深度
        depth_1_color = depth_to_colormap_numpy(depths_1[i])
        fig_d1, ax_d1 = plt.subplots(figsize=(6, 4))
        ax_d1.imshow(depth_1_color)
        ax_d1.set_title(f"Camera1 Depth: {seq_names[i]}", fontsize=10)
        ax_d1.axis('off')
        writer.add_figure(f'Camera1_Depths/sample_{i}', fig_d1, global_step=batch_idx)
        plt.close(fig_d1)
        
        # ---- 相机2 (View1) ----
        img_2_arr = tensor_to_numpy(images_2[i])
        fig_cam2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.imshow(img_2_arr)
        ax2.set_title(f"Camera2 (View1): {seq_names[i]}", fontsize=10)
        ax2.axis('off')
        writer.add_figure(f'Camera2_Images/sample_{i}', fig_cam2, global_step=batch_idx)
        plt.close(fig_cam2)
        
        # 相机2深度
        depth_2_color = depth_to_colormap_numpy(depths_2[i])
        fig_d2, ax_d2 = plt.subplots(figsize=(6, 4))
        ax_d2.imshow(depth_2_color)
        ax_d2.set_title(f"Camera2 Depth: {seq_names[i]}", fontsize=10)
        ax_d2.axis('off')
        writer.add_figure(f'Camera2_Depths/sample_{i}', fig_d2, global_step=batch_idx)
        plt.close(fig_d2)
        
        # ---- 相机3 (View2) ----
        img_3_arr = tensor_to_numpy(images_3[i])
        fig_cam3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.imshow(img_3_arr)
        ax3.set_title(f"Camera3 (View2): {seq_names[i]}", fontsize=10)
        ax3.axis('off')
        writer.add_figure(f'Camera3_Images/sample_{i}', fig_cam3, global_step=batch_idx)
        plt.close(fig_cam3)
        
        # 相机3深度
        depth_3_color = depth_to_colormap_numpy(depths_3[i])
        fig_d3, ax_d3 = plt.subplots(figsize=(6, 4))
        ax_d3.imshow(depth_3_color)
        ax_d3.set_title(f"Camera3 Depth: {seq_names[i]}", fontsize=10)
        ax_d3.axis('off')
        writer.add_figure(f'Camera3_Depths/sample_{i}', fig_d3, global_step=batch_idx)
        plt.close(fig_d3)
        
        # 内参信息（每个相机单独）
        K_1 = Ks_1[i].cpu().numpy()
        K_2 = Ks_2[i].cpu().numpy()
        K_3 = Ks_3[i].cpu().numpy()
        
        writer.add_text(f'Camera1_K/sample_{i}', 
                        f"fx={K_1[0,0]:.2f}, fy={K_1[1,1]:.2f}, cx={K_1[0,2]:.2f}, cy={K_1[1,2]:.2f}", 
                        global_step=batch_idx)
        writer.add_text(f'Camera2_K/sample_{i}', 
                        f"fx={K_2[0,0]:.2f}, fy={K_2[1,1]:.2f}, cx={K_2[0,2]:.2f}, cy={K_2[1,2]:.2f}", 
                        global_step=batch_idx)
        writer.add_text(f'Camera3_K/sample_{i}', 
                        f"fx={K_3[0,0]:.2f}, fy={K_3[1,1]:.2f}, cx={K_3[0,2]:.2f}, cy={K_3[1,2]:.2f}", 
                        global_step=batch_idx)
    writer.close()
    print(f"Logged batch {batch_idx+1}/{NUM_BATCHES_TO_LOG}")

# # 6目
# for sample_idx, sample in enumerate(dataset):
#     if sample_idx >= NUM_BATCHES_TO_LOG:
#         break
    
#     # 提取相机数据
#     cameras = {
#         '01': sample[0][0],
#         '05': sample[0][1],
#         '06': sample[0][2],
#         '07': sample[0][3],
#         '08': sample[0][4],
#         '09': sample[0][5],
#     }
    
#     # 准备6目图像
#     images = []
#     depths = []
#     depth_ranges = {}  # 存储每个相机深度图的范围

    
#     for cam_id in CAMERA_IDS:
#         cam_data = cameras[cam_id]
#         # 图像
#         img_np = np.array(cam_data['rgb']).astype(np.float32) / 255.0
#         images.append(img_np)
#         # 深度图
#         depth = cam_data['depth']
#         # 计算深度图的范围
#         depth_np = np.array(depth, dtype=np.float32)
#         vmin, vmax = depth_np.min(), depth_np.max()
#         depth_ranges[cam_id] = (vmin, vmax)
#         # 转换为伪彩色
#         depth_colormap = pil_to_colormap(depth)
#         depths.append(depth_colormap)
    
#     # 6目图像并排显示
#     fig_img, axes = plt.subplots(2, 3, figsize=(18, 12))
#     for idx, (cam_id, img) in enumerate(zip(CAMERA_IDS, images)):
#         row, col = idx // 3, idx % 3
#         axes[row, col].imshow(img)
#         axes[row, col].set_title(f"Camera {cam_id}")
#         axes[row, col].axis('off')
#     writer.add_figure(f'Images/sample', fig_img, global_step=sample_idx)
#     plt.close(fig_img)
    
#     # 6目深度并排显示
#     fig_depth, axes_d = plt.subplots(2, 3, figsize=(18, 12))
#     for idx, (cam_id, dep) in enumerate(zip(CAMERA_IDS, depths)):
#         row, col = idx // 3, idx % 3
#         axes_d[row, col].imshow(dep)
#         # 在标题中添加深度范围
#         vmin, vmax = depth_ranges[cam_id]
#         axes_d[row, col].set_title(f"Camera {cam_id} Depth [{vmin:.2f}m - {vmax:.2f}m]")
#         axes_d[row, col].axis('off')
#     writer.add_figure(f'Depths/sample', fig_depth, global_step=sample_idx)
#     plt.close(fig_depth)
    
#     # 只在第一次记录K
#     if not k_recorded:
#         for cam_id in CAMERA_IDS:
#             K = cameras[cam_id]['intrinsics']
#             k_text = f"Camera {cam_id}: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}"
#             writer.add_text(f'K/Camera_{cam_id}', k_text, global_step=sample_idx)
#         k_recorded = True
    
#     print(f"Logged sample {sample_idx+1}")

writer.close()
print(f"Done! Run: tensorboard --logdir={LOG_DIR} to visualize.")