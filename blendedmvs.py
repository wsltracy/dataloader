# blendedmvs_simple.py

import os
import os.path as osp
import logging
import random
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def read_pfm(filename):
    """读取 PFM 格式的深度图文件"""
    with open(filename, 'rb') as f:
        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file')
        
        while True:
            dims_line = f.readline().rstrip()
            if dims_line and dims_line[0:1] != b'#':
                break
        
        dims = dims_line.split()
        width = int(dims[0])
        height = int(dims[1])
        scale = float(f.readline().rstrip())
        
        if scale < 0:
            endian = '<'
            scale = -scale
        else:
            endian = '>'
        
        data = np.fromfile(f, dtype=endian + 'f')
        data = data.reshape((height, width))
        
        if color:
            data = data.reshape((height, width, 3))
        data = np.flipud(data)
        return data, scale


def read_cam_file(filename):
    """读取 BlendedMVS 相机参数文件，返回内参矩阵 K (3x3)"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    lines = [line.strip() for line in lines if line.strip()]
    
    for i, line in enumerate(lines):
        if line.lower() == 'intrinsic':
            K = np.zeros((3, 3), dtype=np.float32)
            for j in range(3):
                values = list(map(float, lines[i + 1 + j].split()))
                if len(values) >= 3:
                    K[j] = values[:3]
            return K
    
    raise ValueError(f"无法在文件中找到 'intrinsic' 标记: {filename}")


class BlendedMVSDataset(Dataset):
    """
    BlendedMVS 数据集加载器（简化版）
    输出格式: (image, lidar, depth, K, seq_name, scene, frame_idx)
    """
    def __init__(
        self,
        root_dir: str = "/path/to/BlendedMVS",
        split: str = "train",
        depth_max: float = 200.0,
        len_train: int = 10000,
        len_test: int = 1000,

    ):
        super().__init__()
        
        self.root_dir = root_dir
        self.depth_max = depth_max
        self.training = (split == "train")


        
        # 直接扫描所有图像
        self.images = sorted([
            f for f in glob.glob(osp.join(root_dir, "*", "blended_images", "*.jpg"))
            if os.path.basename(f).split('.')[0].isdigit()  # 只保留纯数字文件名
        ])
        
        # 构建对应的深度图和相机参数路径
        self.depths = []
        self.cams = []
        self.scenes = []
        self.frame_idxs = []
        
        for img_path in self.images:
            # 解析路径
            parts = img_path.split(os.sep)
            scene = parts[-3]  # 场景名
            basename = osp.basename(img_path)
            frame_idx = basename.replace('.jpg', '').replace('.png', '')
            
            # 构建深度图路径
            depth_path = osp.join(root_dir, scene, "rendered_depth_maps", f"{frame_idx}.pfm")
            
            # 构建相机参数路径
            cam_path = osp.join(root_dir, scene, "cams", f"{frame_idx}_cam.txt")
            
            # 验证文件存在
            if osp.exists(depth_path) and osp.exists(cam_path):
                self.depths.append(depth_path)
                self.cams.append(cam_path)
                self.scenes.append(scene)
                self.frame_idxs.append(frame_idx)
        
        # # 过滤掉无效的帧
        assert len(self.images) == len(self.depths), "图像和深度图数量不匹配"
        
        # 设置数据集长度
        if split == "train":
            self.dataset_len = len_train
        else:
            self.dataset_len = len_test
        
        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: BlendedMVS {split} split")
        logging.info(f"{status}: Total frames: {len(self.images)}")
        logging.info(f"{status}: Dataset length: {len(self)}")
    
    def _load_image(self, path):
        """加载 RGB 图像"""
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"无法读取图像: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _load_depth(self, path):
        """加载深度图"""
        if path.endswith('.pfm'):
            depth, _ = read_pfm(path)
        elif path.endswith('.npy'):
            depth = np.load(path)
        else:
            raise ValueError(f"不支持的深度图格式: {path}")
        
        depth = depth.astype(np.float32)
        depth[depth > self.depth_max] = 0
        return depth
    
    def _load_intrinsics(self, path):
        """加载相机内参 K (3x3)"""
        K = read_cam_file(path)
        return K
    
    def get_biased_angles(self, v_bias_range=(-1, 1), h_bias_range=(-5, 5)):
        """生成带偏置的 LiDAR 扫描角度"""
        if self.training:
            v_bias = np.random.uniform(*v_bias_range)
            h_bias = np.random.uniform(*h_bias_range)
        else:
            v_bias = 0
            h_bias = 0
        
        upper_angles = np.linspace(1.0, -9, 32) + v_bias
        lower_angles = np.linspace(-9.17, -12.8, 32) + v_bias
        v_angles_deg = np.concatenate([upper_angles, lower_angles])
        v_angles = np.deg2rad(v_angles_deg)
        
        h_angles_deg = np.arange(-45, 45, 0.2) + h_bias
        h_angles = np.deg2rad(h_angles_deg)
        
        return v_angles, h_angles
    
    def sample_hdl64e_lidar_new(self, depth_map, intrinsics):
        """从深度图采样生成 LiDAR 数据"""
        depth_map = depth_map[0]
        H, W = depth_map.shape
        
        # LiDAR 到相机的转换矩阵
        R_v2c = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        T_v2c = np.array([[0], [0], [0]])
        
        # 生成扫描角度
        v_angles, h_angles = self.get_biased_angles()
        
        v_grid, h_grid = np.meshgrid(v_angles, h_angles)
        v_flat = v_grid.flatten()
        h_flat = h_grid.flatten()
        
        # LiDAR 坐标系下的单位射线
        x_l = np.cos(v_flat) * np.cos(h_flat)
        y_l = np.cos(v_flat) * np.sin(h_flat)
        z_l = np.sin(v_flat)
        
        points_lidar = np.vstack([x_l, y_l, z_l])
        
        # 转换到相机坐标系
        points_cam = R_v2c @ points_lidar + T_v2c
        
        z_c = points_cam[2, :]
        
        # 投影到图像平面
        u = (intrinsics[0, 0] * points_cam[0, :] / z_c) + intrinsics[0, 2]
        v = (intrinsics[1, 1] * points_cam[1, :] / z_c) + intrinsics[1, 2]
        
        u_int = np.round(u).astype(int)
        v_int = np.round(v).astype(int)
        
        # 筛选在图像范围内的点
        mask = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        u_final = u_int[mask]
        v_final = v_int[mask]
        
        # 采样深度值
        sampled_depth = np.zeros_like(depth_map)
        sampled_depth[v_final, u_final] = depth_map[v_final, u_final]
        
        return sampled_depth[None, ...]
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        """返回单目数据"""
        if self.training:
            idx = random.randint(0, len(self.images) - 1)
        else:
            idx = idx % len(self.images)
        
        # 加载数据
        image = self._load_image(self.images[idx])
        depth = self._load_depth(self.depths[idx])
        K = self._load_intrinsics(self.cams[idx])
        
        # 生成 LiDAR 数据
        
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)
        lidar = self.sample_hdl64e_lidar_new(depth_tensor.numpy(), K)
        lidar = torch.from_numpy(lidar).float()
        
        
        # 转换为 tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        depth = torch.from_numpy(depth).float().unsqueeze(0)
        K = torch.from_numpy(K).float()
        
        seq_name = f"{self.scenes[idx]}_{self.frame_idxs[idx]}"
        
        # 输出: (image, lidar, depth, K, seq_name, scene, frame_idx)
        return (image, lidar, depth, K, 
                seq_name)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建数据集实例（测试模式）
    print("=" * 60)
    print("创建 BlendedMVS 测试数据集...")
    print("=" * 60)
    
    dataset = BlendedMVSDataset(
        root_dir="/media/wsl/SANDISK ELE/dataset/BlendedMVS",
        split="train",  # 使用测试模式，不随机采样
        depth_max=200.0,
        len_train=10000,
        len_test=5,    # 只测试5个样本
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    print(f"实际可用帧数: {len(dataset.images)}")