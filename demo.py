# demo.py

import sys
sys.path.append('/home/wsl/dataloader')

import logging
logging.basicConfig(level=logging.INFO)

from base_dataset import BaseDataset
from tartan import TartanAirV1Dataset
import matplotlib.pyplot as plt
import numpy as np 
# 配置对象（模拟 common_conf）
class Config:
    def __init__(self):
        self.img_size = 518
        self.patch_size = 14
        self.training = True
        self.rescale = True
        self.rescale_aug = True
        self.landscape_check = True
        self.augs = type('obj', (object,), {'scales': [0.8, 1.2]})()

common_conf = Config()

# 创建数据集（单目模式）
dataset = TartanAirV1Dataset(
    common_conf=common_conf,
    split='test',
    root_dir='/media/wsl/SANDISK ELE/dataset/tartanair',
    env='carwelding',
    difficulty='Hard',
    traj_id=['P000','P001'],
    stereo=True,  # 单目，只输出左目
    depth_max=80.0,
)

print(f"Dataset size: {len(dataset)}")

# 测试加载
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

for batch_idx, batch in enumerate(dataloader):
    print("Image shape:", batch['image'].shape)
    print("Depth shape:", batch['depth'].shape)
    
    # 检查是否为双目模式
    is_stereo = dataset.stereo  # 双目模式下图像通道数为6 (3+3)
    
    if is_stereo:
        # 获取第一个样本进行可视化（双目模式）

        K = batch['K'][0].numpy()  # 取第一个样本的相机内参矩阵
        
        # 分离左目和右目
        img_left = batch['image'][0].numpy()  # (3,H,W)->(H,W,3)
        img_right = batch['right_image'][0].numpy()  # (3,H,W)->(H,W,3)
        depth_left = batch['depth'][0].numpy()  # (H,W)
        depth_right = batch['right_depth'][0].numpy()  # (H,W)
        print("frame:", batch['seq_name'][0],batch['frame_idx'][0].numpy())
        print("origin Image shape:", batch['original_size'][0].numpy())
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
        img = batch['image'][0].numpy()  # (C,H,W)->(H,W,C)格式
        depth = batch['depth'][0].numpy()  # 取第一个样本的深度图
        K=batch['K'][0].numpy()  # 取第一个样本的相机内参矩阵
        print("origin Image shape:", batch['original_size'][0].numpy())
        print("K:", K)
        
        # 归一化图像到[0,1]范围
        img = (img - img.min()) / (img.max() - img.min())
        
        # 归一化深度图用于可视化
        depth_vis = (depth - depth.min()) / (depth.max() - depth.min())
        
        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 显示RGB图像
        axes[0].imshow(img)
        axes[0].set_title('RGB Image')
        axes[0].axis('off')
        
        # 显示深度图
        im = axes[1].imshow(depth_vis, cmap='jet')
        axes[1].set_title('Depth Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f'visualization_mono_batch_{batch_idx}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # 只可视化第一个batch
    if batch_idx == 0:
        break