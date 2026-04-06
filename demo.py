# demo.py

import sys
sys.path.append('/home/wsl/dataloader')

import logging
logging.basicConfig(level=logging.INFO)

from base_dataset import BaseDataset
from tartan import TartanAirV1Dataset
import matplotlib.pyplot as plt

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
    split='train',
    root_dir='/media/wsl/SANDISK ELE/dataset/tartanair',
    env='carwelding',
    difficulty='Hard',
    traj_id=['P000','P001'],
    stereo=False,  # 单目，只输出左目
    depth_max=80.0,
)

print(f"Dataset size: {len(dataset)}")

# 测试加载
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

for batch_idx, batch in enumerate(dataloader):
    print("Image shape:", batch['image'].shape)
    print("Depth shape:", batch['depth'].shape)
    print("Intrinsics shape:", batch['intrinsics'].shape)
    
    
    # 获取第一个样本进行可视化
    img = batch['image'][0].numpy()  # 已经是(H,W,C)格式
    depth = batch['depth'][0].numpy()  # 取第一个样本的深度图
    K=batch[K][0].numpy()  # 取第一个样本的相机内参矩阵
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
    plt.savefig(f'visualization_batch_{batch_idx}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 只可视化第一个batch
    if batch_idx == 0:
        break
