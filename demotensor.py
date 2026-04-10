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

random.seed(42)
np.random.seed(42)

BATCH_SIZE = 4
NUM_BATCHES_TO_LOG = 10
LOG_DIR = "runs/demo_tensorboard"

dataset = BlendedMVSDataset(
    root_dir="/media/wsl/SANDISK ELE/dataset/BlendedMVS",
    split="train",
    depth_max=200.0,
    len_train=10000,
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
writer = SummaryWriter(log_dir=LOG_DIR)

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

print(f"Dataset size: {len(dataset)}")
print(f"Logging to: {LOG_DIR}")
print(f"Logging first {NUM_BATCHES_TO_LOG} batches...")

for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= NUM_BATCHES_TO_LOG:
        break

    images = batch['image']   # (B, C, H, W)
    depths = batch['depth']   # (B, 1, H, W)
    Ks     = batch['K']       # (B, 3, 3)
    seq_names = batch.get('seq_name', [f"sample_{i}" for i in range(BATCH_SIZE)])

    B = images.shape[0]
    for i in range(B):
        # ---- 图像 I：带标题的 figure ----
        img_arr = tensor_to_numpy(images[i])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(img_arr)
        ax.set_title(seq_names[i], fontsize=10)
        ax.axis('off')
        writer.add_figure(f'I/sample_{i}', fig, global_step=batch_idx)
        plt.close(fig)   # 释放内存

        # ---- 深度 D：伪彩色，也可以加标题（可选，这里不加） ----
        depth_color = depth_to_colormap_numpy(depths[i])
        fig_d, ax_d = plt.subplots(figsize=(6, 4))
        ax_d.imshow(depth_color)
        ax_d.set_title(f"Depth of {seq_names[i]}", fontsize=10)
        ax_d.axis('off')
        writer.add_figure(f'D/sample_{i}', fig_d, global_step=batch_idx)
        plt.close(fig_d)

        # ---- 内参 K：文本记录 ----
        K = Ks[i].cpu().numpy()
        k_text = f"fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}"
        writer.add_text(f'K/sample_{i}', k_text, global_step=batch_idx)

    print(f"Logged batch {batch_idx+1}/{NUM_BATCHES_TO_LOG}")

writer.close()
print("Done! Run: tensorboard --logdir=runs/demo_tensorboard")