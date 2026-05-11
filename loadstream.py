from torch.utils.data import DataLoader
from streaming import StreamingDataset,Stream
from torchvision import transforms
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.cm as cm
import cv2
import torchvision.transforms.functional as TF
import random
random.seed(42)
from PIL import Image



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

#定义 transform
class DepthDataTransform:
    """
    同步对图像、深度图、K矩阵进行数据增强
    """

    def __init__(self,
                 image_size=None,  # 输出尺寸 (H, W)，None 表示不resize
                 normalize=False,  # 是否对RGB进行ImageNet归一化
                 augment=False,  # 是否启用数据增强
                 generate_sparse=False, #是否自己随机采样生成稀疏深度图
                 flip_prob=0.5,  # 随机翻转概率
                 scale_range=(1.0, 1.5),  # 随机缩放范围
                 num_sample=500 #
                 ):
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        self.generate_sparse = generate_sparse
        self.num_sample = num_sample
        # ImageNet 归一化参数
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]

    def __call__(self, image, depth, K):
        """
        Args:
            image: PIL Image (RGB)
            depth: PIL Image (I;16) 深度图
            K: numpy array or list (3x3) 相机内参
        Returns:
            image: Tensor (3, H, W)
            depth: Tensor (1, H, W)
            sparse_depth: Tensor (1, H, W)
            K: Tensor (3, 3)
        """
        # 保存原始尺寸（用于 K 的调整）
        orig_w, orig_h = image.size

        # 转换 K 为 numpy 数组方便操作
        if not isinstance(K, np.ndarray):
            K = np.array(K, dtype=np.float32)
        else:
            K = K.copy().astype(np.float32)

        # ========== 数据增强 ==========
        if self.augment:
            # 1. 随机水平翻转
            if random.random() < self.flip_prob:
                image = TF.hflip(image)
                depth = TF.hflip(depth)
                # 翻转后调整 cx: cx = width - 1 - cx
                K[0, 2] = orig_w - 1 - K[0, 2]

            # 2. 随机缩放（参考KITTI，缩放范围 1.0~1.5）
            scale = random.uniform(*self.scale_range)
            # print(scale)
            if scale != 1.0:
                new_size = (int(orig_h * scale), int(orig_w * scale))
                image = TF.resize(image, new_size, interpolation=Image.BILINEAR)
                depth = TF.resize(depth, new_size, interpolation=Image.BILINEAR)
                # 缩放后调整 K: fx, fy, cx, cy 都乘以 scale
                K[0, 0] *= scale  # fx
                K[1, 1] *= scale  # fy
                K[0, 2] *= scale  # cx
                K[1, 2] *= scale  # cy

        # ========== Resize 到目标尺寸 ==========
        if self.image_size is not None:
            out_h, out_w = self.image_size
            width, height = image.size
            pad_w = max(out_w - width, 0)
            pad_h = max(out_h - height, 0)
            if height < out_h or width < out_w:
                # Pad right and bottom
                pad_h = max(out_h - height, 0)
                pad_w = max(out_w - width, 0)
                image = TF.pad(image, (0, 0, pad_w, pad_h), fill=0)
                depth = np.pad(depth, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            elif height > out_h or width > out_w:
                # 选项A：中心裁剪（推荐，保持原始分辨率）
                top = (height - out_h) // 2
                left = (width - out_w) // 2
                image = TF.crop(image, top, left, out_h, out_w)
                depth = TF.crop(depth, top, left, out_h, out_w)                # ✅ 裁剪需要调整 K：平移 cx, cy
                K[0, 2] = K[0, 2] - left
                K[1, 2] = K[1, 2] - top

        # ========== 转换为 Tensor ==========
        # 图像: (H,W,C) uint8 -> (C,H,W) float [0,1]
        image = TF.to_tensor(image)

        # 图像归一化（可选）
        if self.normalize:
            image = TF.normalize(image, mean=self.rgb_mean, std=self.rgb_std)

        # 深度图: uint16 -> (1,H,W) float /256
        depth = torch.from_numpy(np.array(depth)).float().unsqueeze(0) / 256.0

        # sparse_depth:
        if self.generate_sparse:
            # 这里可以对深度图进行稀疏采样，比如随机采样 N 个点
            # print(self.num_sample)
            sparse_depth = self.get_sparse_depth(depth,self.num_sample)
        else:

            sparse_depth = depth
        # K 转换为 tensor
        K_tensor = torch.from_numpy(K).float()

        return image, sparse_depth, depth, K_tensor

    # Example utility: generate a sparse depth map with a given number of random samples
    @staticmethod
    def get_sparse_depth(depth_map, num_samples=500):
        """
        Randomly sample a given number of valid depth points from a dense depth map.
        Args:
            depth_map (ndarray): 2D array of depth (zeros indicate invalid).
            num_samples (int): number of points to sample.
        Returns:
            sparse_map (ndarray): same shape, with values at sampled points and 0 elsewhere.
        """
        sparse = np.zeros_like(depth_map)
        valid = np.array(np.where(depth_map > 0)).T
        if len(valid) == 0:
            return sparse
        # Limit samples to available points
        num = min(num_samples, len(valid))
        idxs = np.random.choice(len(valid), num, replace=False)
        sel = valid[idxs]
        sparse[sel[:,0], sel[:,1]] = depth_map[sel[:,0], sel[:,1]]
        return sparse




class MultiDataset(StreamingDataset):
    def __init__(self, streams,shuffle, batch_size,batching_method, transform):
        super().__init__(streams=streams,shuffle=shuffle, batch_size=batch_size,batching_method=batching_method)
        self.transform = transform

    def __getitem__(self, idx):
        obj = super().__getitem__(idx)
        image = obj['image']  # PIL Image
        depth = obj['depth']  # PIL Image (I;16)
        K=obj['K']
        #应用 transforms
        image, sparse_depth, depth, K = self.transform(image, depth, K)

        # sparse_depth = self.spare_transform(depth)

        # sparse_depth=depth
        return image, sparse_depth,depth,K


#使用
transform = DepthDataTransform(
    image_size=(1200,1900),        # 输出guding尺寸
    normalize=False,               # 是否 ImageNet 归一化（当前不需要）
    augment=True,                  # 是否启用数据增强
    generate_sparse=True,  # 是否自己随机采样生成稀疏深度图
    flip_prob=0.5,                 # 翻转概率
    scale_range=(0.8,1.2),       # 缩放范围（训练时用 1.0-1.5）
    num_sample=500
)

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
dataset = MultiDataset(
    streams=streams,
    shuffle=False,
    batch_size=32,
    batching_method='per_stream',  #每个batch单数据源,'stratified'（分层采样）,'random'（默认）
    transform=transform
)


dataloader = DataLoader(dataset=dataset,batch_size=32)

for batch_idx, batch in enumerate(dataloader):
    images, sparse_depths, depths, Ks = batch

    print("=" * 50)
    print(f"第一个 Batch 信息 (batch_size=32):")
    print("=" * 50)

    print(f"\n1. Images:")
    print(f"   - Shape: {images.shape}")  # [32, C, H, W]
    print(f"   - Range: [{images.min():.3f}, {images.max():.3f}]")

    print(f"\n2. Depths (完整深度图):")
    print(f"   - Shape: {depths.shape}")  # [32, 1, H, W]
    print(f"   - Range: [{depths.min():.3f}, {depths.max():.3f}]")

    print(f"\n3. Sparse Depths (稀疏深度图):")
    print(f"   - Shape: {sparse_depths.shape}")
    print(f"   - Range: [{sparse_depths.min():.3f}, {sparse_depths.max():.3f}]")

    print(f"\n4. K matrices:")
    print(f"   - Shape: {Ks.shape}")  # [32, 3, 3]
    print(f"   - First K matrix:\n{Ks[0]}")

    # 创建 TensorBoard writer
    dir='runs/stream_demo3'
    writer = SummaryWriter(dir)



    # ========== 方法1：添加到 TensorBoard ==========
    writer.add_image('RGB_Image', images[0], 0)
    # writer.add_image('Depth_Map', depth, 0)

    # 也可以把深度图伪彩色显示（更直观）
    depth_colored = depth_to_colormap(depths[0])
    writer.add_image('Depth_Colored', depth_colored, 0)
    # diff=
    sparse_colored = depth_to_colormap(sparse_depths[0])
    writer.add_image('sparse_colored', sparse_colored, 0)
    print("\n✅ 已添加到 TensorBoard！")
    print("运行以下命令查看：")
    print(f"tensorboard --logdir={dir} --port=6006")

    break  # 只打印第一个 batch
# sample=dataset[1]
# print(f"数据集总样本数: {len(dataset)}")
# print(f"样本字段: {sample.keys()}")
# print(f"图片类型: {type(sample['image'])}")
# print(f"图片尺寸: {sample['image'].size}")
# print(f"深度图尺寸: {sample['depth'].size}")
# print(f"深度图类型: {sample['depth'].mode}")  # 应该是 'I;16' (uint16)
#
#
#
# # 创建 TensorBoard writer
# dir='runs/stream_demo2'
# writer = SummaryWriter(dir)
#
# # 获取第一个样本
# image = transforms.ToTensor()(sample['image'])  # (C, H, W)
# depth = torch.from_numpy(np.array(sample['depth'])).float().unsqueeze(0) / 256.0  # (1, H, W)
# print(f"Depths value range: [{depth.min():.3f}, {depth.max():.3f}]")
# print(f"\n图像 shape: {image.shape}")
# print(f"深度图 shape: {depth.shape}")
# print(f"深度图范围: [{depth.min():.3f}, {depth.max():.3f}]")
#
# # ========== 方法1：添加到 TensorBoard ==========
# writer.add_image('RGB_Image', image, 0)
# # writer.add_image('Depth_Map', depth, 0)
#
# # 也可以把深度图伪彩色显示（更直观）
# depth_colored = depth_to_colormap(depth, radius=3, norm_type='LogNorm')
# writer.add_image('Depth_Colored', depth_colored, 0)
#
# print("\n✅ 已添加到 TensorBoard！")
# print("运行以下命令查看：")
# print(f"tensorboard --logdir={dir} --port=6006")