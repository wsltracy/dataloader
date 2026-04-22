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

# 导入简化版数据集
from blendedmvs import BlendedMVSDataset
from tartan import TartanAirV1Dataset
from matterport import Matterport3DDataset
from dgp.datasets import SynchronizedSceneDataset

random.seed(42)
np.random.seed(42)

BATCH_SIZE = 1
NUM_BATCHES_TO_LOG = 3
LOG_DIR = "runs/mat2"

# ==================== 选择数据集 ====================

# # 选项1: DDAD (六目，字典格式)
# CAMERA_IDS = ['01', '05', '06', '07', '08', '09']
# dataset = SynchronizedSceneDataset(
#     '/media/wsl/SANDISK ELE/dataset/DDAD/ddad_train_val/ddad_2.json',
#     datum_names=('lidar','CAMERA_01', 'CAMERA_05','CAMERA_06','CAMERA_07','CAMERA_08','CAMERA_09'),
#     generate_depth_from_datum='lidar',
#     split='train'
# )

# # 选项2: BlendedMVS (单目，元组格式)
# dataset = BlendedMVSDataset(
#     root_dir="/media/wsl/SANDISK ELE/dataset/BlendedMVS",
#     split="train",
#     depth_max=200.0,
#     len_train=10000,
# )

# 选项3: TartanAir (双目，元组格式)
# dataset = TartanAirV1Dataset(
#     split='train',
#     root_dir='/media/wsl/SANDISK ELE/dataset/tartanair',
#     env='carwelding',
#     difficulty='Hard',
#     traj_id=None,
    
# )

# 选项4: Matterport3D (三目，元组格式)
dataset = Matterport3DDataset(
    root_dir="/media/wsl/SANDISK ELE/dataset/matterport/data/v1/scans",
    split="train",
    depth_max=100.0,

)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
writer = SummaryWriter(log_dir=LOG_DIR)

# ==================== 辅助函数 ====================

def tensor_to_numpy(tensor):
    """将 (C, H, W) tensor 转换为 (H, W, C) numpy，值域 [0,1]"""
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
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
    colored = colormap(norm)[:, :, :3]
    
    # 将无效深度（0）设为白色
    colored[depth_np == 0] = [1.0, 1.0, 1.0]
    return colored

def pil_to_colormap(pil_image, cmap='jet'):
    """将PIL Image格式的深度图转换为伪彩色"""
    depth_np = np.array(pil_image, dtype=np.float32)
    zero_mask = (depth_np == 0)
    vmin, vmax = depth_np.min(), depth_np.max()
    if vmax - vmin > 1e-6:
        norm = (depth_np - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(depth_np)
    colormap = cm.get_cmap(cmap)
    colored = colormap(norm)[:, :, :3]
    colored[zero_mask] = [1.0, 1.0, 1.0]
    return colored

def add_image_depth_lidar_to_tensorboard(writer, tag, image, lidar, depth, step,sample_idx):
    """添加图像、LiDAR、深度图到 TensorBoard"""
    
    # # 1. RGB 图像
    img_arr = tensor_to_numpy(image)
    # fig_img, ax_img = plt.subplots(figsize=(6, 4))
    # ax_img.imshow(img_arr)
    # ax_img.set_title(f"RGB Image", fontsize=10)
    # ax_img.axis('off')
    # writer.add_figure(f'{tag}/RGB/sample_{sample_idx}', fig_img, global_step=step)
    # plt.close(fig_img)
    
    # # 2. LiDAR 稀疏深度图
    lidar_color = depth_to_colormap_numpy(lidar)
    # fig_lidar, ax_lidar = plt.subplots(figsize=(6, 4))
    # ax_lidar.imshow(lidar_color)
    # ax_lidar.set_title(f"LiDAR Sparse Depth", fontsize=10)
    # ax_lidar.axis('off')
    # writer.add_figure(f'{tag}/LiDAR/sample_{sample_idx}', fig_lidar, global_step=step)
    # plt.close(fig_lidar)
    
    # # 3. 密集深度图 (GT)
    depth_color = depth_to_colormap_numpy(depth)
    # fig_depth, ax_depth = plt.subplots(figsize=(6, 4))
    # ax_depth.imshow(depth_color)
    # ax_depth.set_title(f"Dense Depth (GT)", fontsize=10)
    # ax_depth.axis('off')
    # writer.add_figure(f'{tag}/Depth/sample_{sample_idx}', fig_depth, global_step=step)
    # plt.close(fig_depth)
    
    # 4. 并排对比 (RGB + LiDAR + Depth)
    fig_compare, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(img_arr)
    axes[0].set_title("RGB", fontsize=10)
    axes[0].axis('off')
    axes[1].imshow(lidar_color)
    axes[1].set_title("LiDAR", fontsize=10)
    axes[1].axis('off')
    axes[2].imshow(depth_color)
    axes[2].set_title("Dense Depth", fontsize=10)
    axes[2].axis('off')
    fig_compare.suptitle(f"Comparison - {tag}")
    writer.add_figure(f'Comparison/sample_{sample_idx}', fig_compare, global_step=step)
    plt.close(fig_compare)

def get_k_text(K):
    """从内参矩阵提取文本"""
    return f"fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}"

# ==================== 主循环 ====================
print(f"Dataset size: {len(dataset)}")
print(f"Logging to: {LOG_DIR}")
print(f"Logging first {NUM_BATCHES_TO_LOG} batches...")

# 判断数据集类型
is_ddad = isinstance(dataset, SynchronizedSceneDataset)

if is_ddad:
    # ==================== DDAD (六目，字典格式) ====================
    print("Processing DDAD dataset (6 cameras)")
    CAMERA_IDS = ['01', '05', '06', '07', '08', '09']
    # k_recorded = False
    
    # 为每个相机创建图像列表
    rgb_images = {cam_id: [] for cam_id in CAMERA_IDS}
    lidar_images = {cam_id: [] for cam_id in CAMERA_IDS}
    depth_images = {cam_id: [] for cam_id in CAMERA_IDS}
    
    for batch_idx in range(NUM_BATCHES_TO_LOG):
        print(f"Processing batch {batch_idx + 1}/{NUM_BATCHES_TO_LOG}")
        sample = dataset[batch_idx]
        cameras = {
            '01': sample[0][0],
            '05': sample[0][1],
            '06': sample[0][2],
            '07': sample[0][3],
            '08': sample[0][4],
            '09': sample[0][5],
        }
        
        for cam_id in CAMERA_IDS:
            cam_data = cameras[cam_id]
            
            # 图像
            img_np = np.array(cam_data['rgb']).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            
            # LiDAR (深度图)
            depth_pil = cam_data['depth']
            depth_np = np.array(depth_pil, dtype=np.float32)
            depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
            
            # DDAD 没有单独的密集深度图，用 LiDAR 代替
            lidar_tensor = depth_tensor
            
            # # 收集图像
            # rgb_images[cam_id].append(img_tensor)
            # lidar_images[cam_id].append(lidar_tensor)
            # depth_images[cam_id].append(depth_tensor)

            # 创建三图并列显示
            # tag: f'Camera_{cam_id}'，这样每个相机有独立的标签
            # step: batch_idx 用于滑块切换批次
            # sample_idx: batch_idx 作为样本索引
            add_image_depth_lidar_to_tensorboard(
                writer=writer,
                tag=f'Camera_{cam_id}',
                image=img_tensor,
                lidar=lidar_tensor,
                depth=depth_tensor,
                step=batch_idx,      # 使用 batch_idx 实现滑块切换
                sample_idx=0
            )
            # 记录内参（只在第一个batch记录）
            if batch_idx == 0:
                K = cam_data['intrinsics']
                writer.add_text(f'Camera_{cam_id}/Intrinsics', 
                              f"fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}", 
                              global_step=0)
    
    # # 将收集的图像堆叠为批次
    # for cam_id in CAMERA_IDS:
    #     # 堆叠图像为 (N, C, H, W) 格式
    #     rgb_batch = torch.stack(rgb_images[cam_id])
    #     lidar_batch = torch.stack(lidar_images[cam_id])
    #     depth_batch = torch.stack(depth_images[cam_id])
        
    #     # 添加到 TensorBoard，使用滑块切换
    #     writer.add_images(f'Camera_{cam_id}/RGB', rgb_batch, global_step=0, dataformats='NCHW')
    #     writer.add_images(f'Camera_{cam_id}/LiDAR', lidar_batch, global_step=0, dataformats='NCHW')
    #     writer.add_images(f'Camera_{cam_id}/Depth', depth_batch, global_step=0, dataformats='NCHW')
    
    # # 记录内参（只记录一次）
    # if not k_recorded:
    #     for cam_id in CAMERA_IDS:
    #         K = cameras[cam_id]['intrinsics']
    #         writer.add_text(f'Camera_{cam_id}/K', get_k_text(K), global_step=0)
    #     k_recorded = True
    
    print(f"Logged {NUM_BATCHES_TO_LOG} DDAD samples with slider")

else:
    # ==================== 元组格式处理 ====================
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= NUM_BATCHES_TO_LOG:
            break
        
        # 根据元组长度判断数据集类型
        num_elements = len(batch)
        
        if num_elements == 5:
            # ==================== 单目 (BlendedMVS) ====================
            # 输出格式: (image, lidar, depth, K, seq_name)
            print(f"Processing Monocular data (BlendedMVS) - Batch {batch_idx}")
            
            images = batch[0]      # (B, C, H, W)
            lidars = batch[1]      # (B, 1, H, W)
            depths = batch[2]      # (B, 1, H, W)
            Ks = batch[3]          # (B, 3, 3)
            seq_names = batch[4]   # tuple of strings
 
            
            B = images.shape[0]
            for i in range(B):
                seq_name = seq_names[i] if isinstance(seq_names, (list, tuple)) else seq_names
                
                add_image_depth_lidar_to_tensorboard(
                    writer, f'Monocular/{seq_name}',
                    images[i], lidars[i], depths[i],
                    batch_idx, i
                )
                
                # 记录内参
                K = Ks[i].cpu().numpy()
                writer.add_text(f'Monocular/{seq_name}/K', get_k_text(K), global_step=batch_idx)
            
            print(f"  Logged {B} samples")
        
        elif num_elements == 9:
            # ==================== 双目 (TartanAir) ====================
            # 输出格式: (image_1, lidar_1, depth_1, K_1, image_2, lidar_2, depth_2, K_2, seq_name)
            print(f"Processing Stereo data (TartanAir) - Batch {batch_idx}")
            
            images_left = batch[0]   # (B, C, H, W)
            lidars_left = batch[1]   # (B, 1, H, W)
            depths_left = batch[2]   # (B, 1, H, W)
            Ks_left = batch[3]       # (B, 3, 3)
            images_right = batch[4]  # (B, C, H, W)
            lidars_right = batch[5]  # (B, 1, H, W)
            depths_right = batch[6]  # (B, 1, H, W)
            Ks_right = batch[7]      # (B, 3, 3)
            seq_names = batch[8]     # tuple of strings
            
            B = images_left.shape[0]
            for i in range(B):
                seq_name = seq_names[i] if isinstance(seq_names, (list, tuple)) else seq_names
                
                # 左目
                add_image_depth_lidar_to_tensorboard(
                    writer, f'Stereo/{seq_name}/Left',
                    images_left[i], lidars_left[i], depths_left[i],
                    batch_idx, i
                )
                
                # 右目
                add_image_depth_lidar_to_tensorboard(
                    writer, f'Stereo/{seq_name}/Right',
                    images_right[i], lidars_right[i], depths_right[i],
                    batch_idx, i
                )
                
                # 记录内参
                K_left = Ks_left[i].cpu().numpy()
                K_right = Ks_right[i].cpu().numpy()
                writer.add_text(f'Stereo/{seq_name}/Left_K', get_k_text(K_left), global_step=batch_idx)
                writer.add_text(f'Stereo/{seq_name}/Right_K', get_k_text(K_right), global_step=batch_idx)
            
            print(f"  Logged {B} stereo pairs")
        
        elif num_elements == 13:
            # ==================== 三目 (Matterport3D) ====================
            # 输出格式: (image_1, lidar_1, depth_1, K_1, image_2, lidar_2, depth_2, K_2,
            #           image_3, lidar_3, depth_3, K_3, seq_name)
            print(f"Processing Triclops data (Matterport3D) - Batch {batch_idx}")
            
            images_1 = batch[0]   # (B, C, H, W)
            lidars_1 = batch[1]   # (B, 1, H, W)
            depths_1 = batch[2]   # (B, 1, H, W)
            Ks_1 = batch[3]       # (B, 3, 3)
            images_2 = batch[4]   # (B, C, H, W)
            lidars_2 = batch[5]   # (B, 1, H, W)
            depths_2 = batch[6]   # (B, 1, H, W)
            Ks_2 = batch[7]       # (B, 3, 3)
            images_3 = batch[8]   # (B, C, H, W)
            lidars_3 = batch[9]   # (B, 1, H, W)
            depths_3 = batch[10]  # (B, 1, H, W)
            Ks_3 = batch[11]      # (B, 3, 3)
            seq_names = batch[12] # tuple of strings

            
            B = images_1.shape[0]
            for i in range(B):
                seq_name = seq_names[i] if isinstance(seq_names, (list, tuple)) else seq_names
                
                # 相机1
                add_image_depth_lidar_to_tensorboard(
                    writer, f'Triclops/{seq_name}/Camera1',
                    images_1[i], lidars_1[i], depths_1[i],
                    batch_idx, i
                )
                
                # 相机2
                add_image_depth_lidar_to_tensorboard(
                    writer, f'Triclops/{seq_name}/Camera2',
                    images_2[i], lidars_2[i], depths_2[i],
                    batch_idx, i
                )
                
                # 相机3
                add_image_depth_lidar_to_tensorboard(
                    writer, f'Triclops/{seq_name}/Camera3',
                    images_3[i], lidars_3[i], depths_3[i],
                    batch_idx, i
                )
                
                # 记录内参
                K_1 = Ks_1[i].cpu().numpy()
                K_2 = Ks_2[i].cpu().numpy()
                K_3 = Ks_3[i].cpu().numpy()
                
                writer.add_text(f'Triclops/{seq_name}/Camera1_K', get_k_text(K_1), global_step=batch_idx)
                writer.add_text(f'Triclops/{seq_name}/Camera2_K', get_k_text(K_2), global_step=batch_idx)
                writer.add_text(f'Triclops/{seq_name}/Camera3_K', get_k_text(K_3), global_step=batch_idx)
            
            print(f"  Logged {B} triclops samples")
        
        else:
            print(f"Unknown output format with {num_elements} elements")
        
        print(f"Completed batch {batch_idx+1}/{NUM_BATCHES_TO_LOG}")

writer.close()
print(f"\nDone! Run: tensorboard --logdir={LOG_DIR} to visualize.")