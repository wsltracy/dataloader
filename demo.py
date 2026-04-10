# demo.py

import sys
sys.path.append('/home/wsl/dataloader')

import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
import numpy as np 
import random
from tartan import TartanAirV1Dataset

from blendedmvs import BlendedMVSDataset
from matterport import Matterport3DDataset
from ddadm import DDADDataset

# 设置所有随机种子
random.seed(42)          # Python random 模块
np.random.seed(42)       # NumPy 随机数生成器


stereo=False
# 创建数据集
# dataset = BlendedMVSDataset(
#     root_dir="/media/weishanling/SANDISK ELE/dataset/BlendedMVS",
#     split="train",
#     depth_max=200.0,
#     len_train=10000,
# )

# dataset = Matterport3DDataset(
#     root_dir="/media/wsl/SANDISK ELE/dataset/matterport/data/v1/scans",
#     split="train",
#     depth_max=20.0,
# )
# dataset = DDADDataset(
#     root_dir="/media/weishanling/SANDISK ELE/dataset/DDAD/ddad_train_val",
#     ddad_json_path="/media/weishanling/SANDISK ELE/dataset/DDAD/ddad.json",
#     split="0",
#     cam_names=['CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09'],
#     generate_depth=True,
#     depth_max=250.0,
#     len_train=100000,
# )

# 创建数据集（单目模式）
dataset = TartanAirV1Dataset(
    split='train',
    root_dir='/media/weishanling/SANDISK ELE/dataset/tartanair',
    env='carwelding',
    difficulty='Hard',
    traj_id=None,
)

print(f"Dataset size: {len(dataset)}")

# 测试加载
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

for batch_idx, batch in enumerate(dataloader):
    # 打印 batch 中所有的键名
    # print("Batch keys:", batch.keys())

    # print("Image shape:", batch['image'].shape)
    # print("Depth shape:", batch['depth'].shape)
    
    # 检查是否为双目模式
    is_stereo = stereo  # 双目模式下图像通道数为6 (3+3)
    
    if is_stereo:
        # 获取第一个样本进行可视化（双目模式）

        K = batch['K'][0].numpy()  # 取第一个样本的相机内参矩阵
        
        # 分离左目和右目
        img_left = batch['image'][0].numpy()  # (3,H,W)->(H,W,3)
        img_right = batch['right_image'][0].numpy()  # (3,H,W)->(H,W,3)
        depth_left = batch['depth'][0].numpy()  # (H,W)
        depth_right = batch['right_depth'][0].numpy()  # (H,W)
        print("frame:", batch['seq_name'][0],batch['frame_idx'][0].numpy())
        print("K:", K)
        
        # 归一化图像到[0,1]范围
        img_left = (img_left - img_left.min()) / (img_left.max() - img_left.min())
        img_right = (img_right - img_right.min()) / (img_right.max() - img_right.min())
        
        # 归一化深度图用于可视化
        depth_left_vis = (depth_left - depth_left.min()) / (depth_left.max() - depth_left.min())
        depth_right_vis = (depth_right - depth_right.min()) / (depth_right.max() - depth_right.min())
        print("Depth Left - min:", depth_left.min(), "max:", depth_left.max())
        print("Depth Right - min:", depth_right.min(), "max:", depth_right.max())
        # 计算深度差值（左目 - 右目）
        depth_diff = depth_left - depth_right

        
        # 创建3x2的可视化布局
        fig, axes = plt.subplots(2, 2, figsize=(12, 15))
        
        # 显示左目RGB图像
        axes[0, 0].imshow(img_left)
        axes[0, 0].set_title('Left RGB Image')
        axes[0, 0].axis('off')
        
        # 显示右目RGB图像
        axes[0, 1].imshow(img_right)
        axes[0, 1].set_title('Right RGB Image')
        axes[0, 1].axis('off')
        
        # 显示左目深度图
        im_left = axes[1, 0].imshow(depth_left_vis, cmap='jet')
        axes[1, 0].set_title('Left Depth Map')
        axes[1, 0].axis('off')
        plt.colorbar(im_left, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 显示右目深度图
        im_right = axes[1, 1].imshow(depth_right_vis, cmap='jet')
        axes[1, 1].set_title('Right Depth Map')
        axes[1, 1].axis('off')
        plt.colorbar(im_right, ax=axes[1, 1], fraction=0.046, pad=0.04)
        

        
        plt.tight_layout()
        plt.savefig(f'visualization_stereo_batch_{batch_idx}.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        # 单目模式的原有可视化代码
        # 获取第一个样本进行可视化
        # img = batch['image'][0].numpy()  # (C,H,W)->(H,W,C)格式
        # depth = batch['depth'][0].numpy()  # 取第一个样本的深度图
        # K=batch['K'][0].numpy()  # 取第一个样本的相机内参矩阵
        img = batch['image_1'][0].numpy()  # (C,H,W)->(H,W,C)格式
        # depth = batch['sparse_1'][0].numpy()  # 取第一个样本的深度图
        depth = batch['depth_1'][0].numpy()  # 取第一个样本的深度图
        K=batch['K_1'][0].numpy()  # 取第一个样本的相机内参矩阵
        print("frame:", batch['seq_name'][0],batch['frame_idx'][0])
        print("Depth - min:", depth.min(), "max:", depth.max())
        print("K:", K)
        
        # 归一化图像到[0,1]范围
        img = (img - img.min()) / (img.max() - img.min())
        
        # 归一化深度图用于可视化
        depth_vis = (depth - depth.min()) / (depth.max() - depth.min())
        
        # 创建可视化，添加sharex和sharey参数确保子图共享坐标轴
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        
        # 显示RGB图像
        axes[0].imshow(img)
        axes[0].set_title('RGB Image')
        axes[0].axis('off')
        
        # 显示深度图
        im = axes[1].imshow(depth_vis, cmap='jet')
        axes[1].set_title('Depth Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 调整布局，确保子图间距合适
        plt.tight_layout(pad=2.0)
        plt.savefig(f'visualization_mono_batch_{batch_idx}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # 只可视化第一个batch
    if batch_idx == 0:
        break