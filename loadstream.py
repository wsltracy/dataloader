from torch.utils.data import DataLoader
from streaming import StreamingDataset,Stream
from torchvision import transforms
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.cm as cm
import cv2


# 自定义 collate 函数
def collate_fn(batch):
    images = torch.stack([transforms.ToTensor()(item['image']) for item in batch])
    # 深度图处理：从 PIL Image 转为 Tensor
    depths = torch.stack([torch.from_numpy(np.array(item['depth'])).float().unsqueeze(0) for item in batch])
    # 处理完深度图数值范围：uint16
    depths = depths / 256.0
    return images, depths


def depth_to_colormap(depth_tensor):
    """将深度图转为彩色图 (JET colormap)"""
    # 归一化到 [0, 1]
    depth_np = depth_tensor.squeeze().numpy()
    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)

    # 应用 colormap
    colormap = cm.jet(depth_norm)  # (H, W, 4) RGBA
    colormap_rgb = colormap[:, :, :3]  # 去掉 alpha 通道

    # 转为 tensor (C, H, W)
    result = torch.from_numpy(colormap_rgb).permute(2, 0, 1).float()
    return result


# def ColorizeNew(depth, min_distance=None, max_distance=None, radius=None, norm_type='LogNorm', cmap=cm.jet, offset=1.):
#     # depth: numpy array (H, W) 或 (H, W, 1)
#     if len(depth.shape) == 3:
#         depth = depth.squeeze()
#
#     # 归一化处理
#     depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) + offset
#     Norm = getattr(cm.colors, norm_type)
#     norm = Norm(vmin=depth.min(), vmax=depth.max(), clip=True)
#     m = cm.ScalarMappable(norm=norm, cmap=cmap)
#
#     if radius is None:
#         depth_color = (255 * m.to_rgba(depth)[:, :, 0:3]).astype(np.uint8)
#     else:
#         pos = np.argwhere(depth > 1e-8)
#         print(pos.shape[0])
#         depth_color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
#         for i in range(pos.shape[0]):
#             color = tuple([int(255 * value) for value in m.to_rgba(depth[pos[i, 0], pos[i, 1]])[0:3]])
#             cv2.circle(depth_color, (pos[i, 1], pos[i, 0]), radius, (color[0], color[1], color[2]), -1)
#     return depth_color  # (H, W, 3) uint8
#
#
# def depth_to_tensorboard_colormap(depth_tensor, radius=None, norm_type='LogNorm'):
#     """
#     将深度图转为 TensorBoard 可用的彩色图
#     depth_tensor: torch Tensor (1, H, W) 或 (H, W)
#     """
#     # 转为 numpy (H, W)
#     if depth_tensor.dim() == 3:
#         depth_np = depth_tensor.squeeze().cpu().numpy()
#     else:
#         depth_np = depth_tensor.cpu().numpy()
#
#     # 使用 ColorizeNew 生成彩色图
#     depth_colored = ColorizeNew(depth_np, radius=radius, norm_type=norm_type)
#
#     # 转为 tensor (C, H, W) 并归一化到 [0, 1]
#     result = torch.from_numpy(depth_colored).permute(2, 0, 1).float() / 255.0
#     return result

data_dir_blended = './datas/mds_datasets/BlendedMVS'
data_dir_mat ='./datas/mds_datasets/Matterport'
data_dir_tar='./datas/mds_datasets/Tartanair'
data_dir_ddad='./datas/mds_datasets/DDAD_train'
# 定义多个数据流，每个都用 repeat=1（表示使用全部数据一次）
streams = [
    # Stream(
    #     local=data_dir_blended,
    #     repeat=1,        # 轮次：用全部数据
    # ),
    # Stream(
    #     local=data_dir_mat,
    #     repeat=1,
    # ),
    # Stream(
    #     local=data_dir_tar,
    #     repeat=1,
    # ),
    Stream(
        local=data_dir_ddad,
        repeat=1,
    ),
]
# 创建数据集
dataset = StreamingDataset(
    streams=streams,
    shuffle=False,
    batch_size=32,
    batching_method='per_stream',  #每个batch单数据源,'stratified'（分层采样）,'random'（默认）
)

sample=dataset[1]
print(f"数据集总样本数: {len(dataset)}")
print(f"样本字段: {sample.keys()}")
print(f"图片类型: {type(sample['image'])}")
print(f"图片尺寸: {sample['image'].size}")
print(f"深度图尺寸: {sample['depth'].size}")
print(f"深度图类型: {sample['depth'].mode}")  # 应该是 'I;16' (uint16)

# # 创建 DataLoader
# dataloader = DataLoader(
#     dataset,
#     batch_size=32,
#     num_workers=2,
#     collate_fn=collate_fn,
#     # shuffle=False  # StreamingDataset 已经设置了 shuffle，这里必须为 False
# )
#
# # 测试读取
# for images, depths in dataloader:
#     print(f"Images shape: {images.shape}")
#     print(f"Depths shape: {depths.shape}")
#     print(f"Images value range: [{images.min():.3f}, {images.max():.3f}]")
#     print(f"Depths value range: [{depths.min():.3f}, {depths.max():.3f}]")
#     break
#
# print("\n✅ 读取成功！")

# 创建 TensorBoard writer
dir='runs/stream_demo2'
writer = SummaryWriter(dir)

# 获取第一个样本
image = transforms.ToTensor()(sample['image'])  # (C, H, W)
depth = torch.from_numpy(np.array(sample['depth'])).float().unsqueeze(0) / 256.0  # (1, H, W)
print(f"Depths value range: [{depth.min():.3f}, {depth.max():.3f}]")
print(f"\n图像 shape: {image.shape}")
print(f"深度图 shape: {depth.shape}")
print(f"深度图范围: [{depth.min():.3f}, {depth.max():.3f}]")

# ========== 方法1：添加到 TensorBoard ==========
writer.add_image('RGB_Image', image, 0)
# writer.add_image('Depth_Map', depth, 0)

# 也可以把深度图伪彩色显示（更直观）
depth_colored = depth_to_colormap(depth, radius=3, norm_type='LogNorm')
writer.add_image('Depth_Colored', depth_colored, 0)

print("\n✅ 已添加到 TensorBoard！")
print("运行以下命令查看：")
print(f"tensorboard --logdir={dir} --port=6006")