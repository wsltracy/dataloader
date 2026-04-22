# tartan_simple.py

import os
import os.path as osp
import logging
import random
import glob
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class TartanAirV1Dataset(Dataset):
    """
    TartanAir V1 数据集加载器（简化版）
    输出格式: (image_1, lidar_1, depth_1, K_1, image_2, lidar_2, depth_2, K_2, seq_name, frame_idx)
    """
    def __init__(
        self,
        split: str = "train",
        root_dir: str = "/media/wsl/SANDISK ELE/dataset/tartanair",
        env: str = "carwelding",
        difficulty: str = "Easy",
        traj_id: str = None,
        len_train: int = 10000,
        len_test: int = 1000,

    ):
        super().__init__()
        
        self.root_dir = root_dir
        self.env = env
        self.difficulty = difficulty
        self.traj_id = traj_id
        self.training = (split == "train")

        
        # 固定内参（640x480）
        self.K = np.array([[320.0, 0.0, 320.0],
                          [0.0, 320.0, 240.0],
                          [0.0, 0.0, 1.0]], dtype=np.float32)
        
        # 直接扫描所有左目图像
        base_path = osp.join(root_dir, env, difficulty)
        
        # 确定轨迹
        if traj_id:
            traj_dirs = [str(traj_id)] if not isinstance(traj_id, list) else [str(t) for t in traj_id]
        else:
            traj_dirs = [d for d in os.listdir(base_path) if d.startswith('P')]
        
        # 构建文件列表
        self.left_images = []
        self.right_images = []
        self.left_depths = []
        self.right_depths = []
        self.trajs = []
        self.frame_idxs = []
        
        for traj in traj_dirs:
            traj_path = osp.join(base_path, traj)
            left_dir = osp.join(traj_path, "image_left")
            
            if not osp.exists(left_dir):
                continue
            
            # 获取左目图像
            left_imgs = sorted(glob.glob(osp.join(left_dir, "*.png")))
            
            for left_path in left_imgs:
                # 提取帧索引
                basename = osp.basename(left_path)
                match = re.search(r'(\d+)', basename)
                frame_idx = int(match.group(1)) if match else 0
                
                # 右目图像路径
                right_path = left_path.replace("image_left", "image_right")
                right_path = right_path.replace("_left.png", "_right.png")
                
                # 深度图路径
                left_depth_path = osp.join(traj_path, "depth_left", f"{frame_idx:06d}_left_depth.npy")
                if not osp.exists(left_depth_path):
                    left_depth_path = osp.join(traj_path, "depth_left", f"{frame_idx:06d}.npy")
                
                right_depth_path = osp.join(traj_path, "depth_right", f"{frame_idx:06d}_right_depth.npy")
                if not osp.exists(right_depth_path):
                    right_depth_path = osp.join(traj_path, "depth_right", f"{frame_idx:06d}.npy")
                
                # 验证文件存在
                if (osp.exists(right_path) and osp.exists(left_depth_path) and 
                    osp.exists(right_depth_path)):
                    self.left_images.append(left_path)
                    self.right_images.append(right_path)
                    self.left_depths.append(left_depth_path)
                    self.right_depths.append(right_depth_path)
                    self.trajs.append(traj)
                    self.frame_idxs.append(frame_idx)
        
        # 设置数据集长度
        if split == "train":
            self.dataset_len = len_train
        else:
            self.dataset_len = len_test
        
        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: TartanAir V1 {env}/{difficulty}")
        logging.info(f"{status}: Total frames: {len(self.left_images)}")
        logging.info(f"{status}: Dataset length: {len(self)}")
    
    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"无法读取图像: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _load_depth(self, path):
        depth = np.load(path).astype(np.float32)
        # 深度过滤
        valid_depths = depth[depth > 0]
        if valid_depths.size > 0:
            global_median = np.median(valid_depths)
            lower_bound = global_median / 12.0
            upper_bound = global_median * 12.0
            depth[(depth <= 0) | (depth < lower_bound) | (depth > upper_bound)] = 0
        return depth
    
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
        
        R_v2c = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        T_v2c = np.array([[0], [0], [0]])
        
        v_angles, h_angles = self.get_biased_angles()
        
        v_grid, h_grid = np.meshgrid(v_angles, h_angles)
        v_flat = v_grid.flatten()
        h_flat = h_grid.flatten()
        
        x_l = np.cos(v_flat) * np.cos(h_flat)
        y_l = np.cos(v_flat) * np.sin(h_flat)
        z_l = np.sin(v_flat)
        
        points_lidar = np.vstack([x_l, y_l, z_l])
        points_cam = R_v2c @ points_lidar + T_v2c
        
        z_c = points_cam[2, :]
        u = (intrinsics[0, 0] * points_cam[0, :] / z_c) + intrinsics[0, 2]
        v = (intrinsics[1, 1] * points_cam[1, :] / z_c) + intrinsics[1, 2]
        
        u_int = np.round(u).astype(int)
        v_int = np.round(v).astype(int)
        
        mask = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        u_final = u_int[mask]
        v_final = v_int[mask]
        
        sampled_depth = np.zeros_like(depth_map)
        sampled_depth[v_final, u_final] = depth_map[v_final, u_final]
        
        return sampled_depth[None, ...]
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        """返回双目数据"""
        if self.training:
            idx = random.randint(0, len(self.left_images) - 1)
        else:
            idx = idx % len(self.left_images)
        
        # 加载左目数据
        left_img = self._load_image(self.left_images[idx])
        left_depth = self._load_depth(self.left_depths[idx])
        
        # 加载右目数据
        right_img = self._load_image(self.right_images[idx])
        right_depth = self._load_depth(self.right_depths[idx])
        
        K = self.K.copy()
        
        # 转换为 tensor
        left_img_tensor = torch.from_numpy(left_img).float().permute(2, 0, 1) / 255.0
        right_img_tensor = torch.from_numpy(right_img).float().permute(2, 0, 1) / 255.0
        
        left_depth_tensor = torch.from_numpy(left_depth).float().unsqueeze(0)
        right_depth_tensor = torch.from_numpy(right_depth).float().unsqueeze(0)
        
        # 生成 LiDAR 数据

        left_lidar = self.sample_hdl64e_lidar_new(left_depth_tensor.numpy(), K)
        right_lidar = self.sample_hdl64e_lidar_new(right_depth_tensor.numpy(), K)
        left_lidar = torch.from_numpy(left_lidar).float()
        right_lidar = torch.from_numpy(right_lidar).float()
        
        
        K_tensor = torch.from_numpy(K).float()
        
        seq_name = f"{self.env}_{self.difficulty}_{self.trajs[idx]}"
        frame_idx = self.frame_idxs[idx]
        
        # 输出: (image_1, lidar_1, depth_1, K_1, image_2, lidar_2, depth_2, K_2, seq_name, frame_idx)
        return (left_img_tensor, left_lidar, left_depth_tensor, K_tensor,
                right_img_tensor, right_lidar, right_depth_tensor, K_tensor,
                seq_name)