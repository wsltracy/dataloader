# tartanair_v1.py

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

# 设置所有随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


class TartanAirV1Dataset(Dataset):
    """
    TartanAir V1 数据集加载器（固定双目输出）
    输出格式类似 DDAD：
        - image_1, depth_1, K_1: 左目（相机1）
        - image_2, depth_2, K_2: 右目（相机2）
    不进行图像缩放，保持原始尺寸
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
        
        # 构建所有帧索引
        self._build_all_frames()
        
        # 原始内参（640x480 分辨率下的默认值）
        self.K = self._get_intrinsics()
        
        # 设置数据集长度
        if split == "train":
            self.dataset_len = len_train
        elif split == "test":
            self.dataset_len = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: TartanAir V1 {env}/{difficulty}")
        if traj_id:
            logging.info(f"{status}: Trajectory: {traj_id}")
        else:
            logging.info(f"{status}: Trajectories: {list(set([f['traj'] for f in self.all_frames]))}")
        logging.info(f"{status}: Total frames: {len(self.all_frames)}")
        logging.info(f"{status}: Dataset length: {len(self)}")
    
    def _get_intrinsics(self):
        """返回 TartanAir V1 固定内参（640x480 分辨率）"""
        return np.array([[320.0, 0.0, 320.0],
                         [0.0, 320.0, 240.0],
                         [0.0, 0.0, 1.0]], dtype=np.float32)
    
    def _build_all_frames(self):
        """构建所有帧的索引列表（必须同时包含左右目数据）"""
        base_path = osp.join(self.root_dir, self.env, self.difficulty)
        
        if not osp.isdir(base_path):
            raise FileNotFoundError(f"路径不存在: {base_path}")
        
        # 确定要加载的轨迹
        if self.traj_id:
            if isinstance(self.traj_id, list):
                traj_dirs = [str(t) for t in self.traj_id]
            else:
                traj_dirs = [str(self.traj_id)]
        else:
            traj_dirs = [d for d in os.listdir(base_path) 
                        if d.startswith('P') and osp.isdir(osp.join(base_path, d))]
            traj_dirs = sorted(traj_dirs)
        
        if not traj_dirs:
            raise ValueError(f"未找到任何轨迹: {base_path}")
        
        logging.info(f"找到轨迹: {traj_dirs}")
        
        self.all_frames = []
        
        for traj in traj_dirs:
            traj_path = osp.join(base_path, traj)
            
            if not osp.isdir(traj_path):
                logging.warning(f"轨迹路径不存在: {traj_path}")
                continue
            
            # 左目图像路径
            left_img_dir = osp.join(traj_path, "image_left")
            if not osp.isdir(left_img_dir):
                logging.warning(f"跳过 {traj}: 没有 image_left 目录")
                continue
            
            # 右目图像路径
            right_img_dir = osp.join(traj_path, "image_right")
            if not osp.isdir(right_img_dir):
                logging.warning(f"跳过 {traj}: 没有 image_right 目录")
                continue
            
            # 获取左目图像列表
            left_img_paths = sorted(glob.glob(osp.join(left_img_dir, "*.png")))
            
            if not left_img_paths:
                logging.warning(f"跳过 {traj}: 没有找到左目图像")
                continue
            
            logging.info(f"轨迹 {traj}: 找到 {len(left_img_paths)} 张图像对")
            
            for left_path in left_img_paths:
                # 提取帧索引
                basename = osp.basename(left_path)
                match = re.search(r'(\d+)', basename)
                frame_idx = int(match.group(1)) if match else len(self.all_frames)
                
                # 构建右目图像路径
                right_path = left_path.replace("image_left", "image_right")
                right_path = right_path.replace("_left.png", "_right.png")
                if not osp.exists(right_path):
                    right_path = osp.join(right_img_dir, osp.basename(left_path))
                if not osp.exists(right_path):
                    continue
                
                # 左目深度路径
                left_depth_path = osp.join(traj_path, "depth_left", f"{frame_idx:06d}_left_depth.npy")
                if not osp.exists(left_depth_path):
                    left_depth_path = osp.join(traj_path, "depth_left", f"{frame_idx:06d}.npy")
                if not osp.exists(left_depth_path):
                    continue
                
                # 右目深度路径
                right_depth_path = osp.join(traj_path, "depth_right", f"{frame_idx:06d}_right_depth.npy")
                if not osp.exists(right_depth_path):
                    right_depth_path = osp.join(traj_path, "depth_right", f"{frame_idx:06d}.npy")
                if not osp.exists(right_depth_path):
                    continue
                
                self.all_frames.append({
                    'left_image_path': left_path,
                    'right_image_path': right_path,
                    'left_depth_path': left_depth_path,
                    'right_depth_path': right_depth_path,
                    'traj': traj,
                    'frame_idx': frame_idx
                })
        
        if not self.all_frames:
            raise ValueError(f"没有找到任何有效帧对: {base_path}")
        
        logging.info(f"总共找到 {len(self.all_frames)} 个有效双目帧对")
    
    def _load_image(self, path):
        """加载 RGB 图像"""
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"无法读取图像: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _load_depth(self, path):
        """加载深度图"""
        if not osp.isfile(path):
            raise FileNotFoundError(f"深度文件不存在: {path}")
        depth = np.load(path).astype(np.float32)
        # depth = np.clip(depth, 0, self.depth_max)
        # 中值滤波
        # 创建一个掩码，只提取大于 0 的有效深度值
        valid_depths = depth[depth > 0]
        sigma = 12.0
        # 检查是否有有效值，避免除以0或空数组错误
        if valid_depths.size > 0:
            # 只基于有效深度计算全局中值
            global_median = np.median(valid_depths)
            
            # 计算范围
            lower_bound = global_median / sigma
            upper_bound = global_median * sigma

            # 过滤：将小于下限、大于上限，或者原本就是0的值，全部设为0
            # 这里加上 (depth <= 0) 是为了确保原本的无效值保持为0
            depth[(depth <= 0) | (depth < lower_bound) | (depth > upper_bound)] = 0
        return depth
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        """返回双目数据，格式类似 DDAD"""
        if self.training:
            frame = random.choice(self.all_frames)
        else:
            frame = self.all_frames[idx % len(self.all_frames)]
        
        # 加载左目数据
        left_img = self._load_image(frame['left_image_path'])
        left_depth = self._load_depth(frame['left_depth_path'])
        
        # 加载右目数据
        right_img = self._load_image(frame['right_image_path'])
        right_depth = self._load_depth(frame['right_depth_path'])
        
        # 使用固定内参
        K = self.K.copy()
        
        # 转换为 torch tensor
        # 图像: (H, W, C) -> (C, H, W)，值域 [0, 1]
        left_img_tensor = torch.from_numpy(left_img).float().permute(2, 0, 1)  / 255.0
        right_img_tensor = torch.from_numpy(right_img).float().permute(2, 0, 1)/ 255.0
        
        # 深度: (H, W) -> (1, H, W)
        left_depth_tensor = torch.from_numpy(left_depth).float().unsqueeze(0)
        right_depth_tensor = torch.from_numpy(right_depth).float().unsqueeze(0)
        
        # 内参
        K_tensor = torch.from_numpy(K).float()
        
        # 构建返回结果（类似 DDAD 格式）
        batch = {
            'image_1': left_img_tensor,
            'depth_1': left_depth_tensor,
            'K_1': K_tensor,
            'image_2': right_img_tensor,
            'depth_2': right_depth_tensor,
            'K_2': K_tensor,
            'seq_name': f"{self.env}_{self.difficulty}_{frame['traj']}",
            'frame_idx': frame['frame_idx'],
        }
        
        return batch