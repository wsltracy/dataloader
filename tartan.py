# tartanair_v1.py

import os
import os.path as osp
import logging
import random
import glob
import json
import re

import cv2
import numpy as np
import torch

from base_dataset import BaseDataset
from dataset_util import *


class TartanAirV1Dataset(BaseDataset):
    """
    TartanAir V1 数据集加载器
    继承 BaseDataset，复用图像处理流程
    支持单目（默认左目）和双目输出
    自动根据图像缩放调整内参
    """
    def __init__(
        self,
        common_conf,           # 配置对象（包含 img_size, patch_size, aug_scale 等）
        split: str = "train",  # 'train' 或 'test'
        root_dir: str = "/media/wsl/SANDISK ELE/dataset/tartanair",
        env: str = "carwelding",
        difficulty: str = "Easy",
        traj_id: str = None,   # None 表示加载所有轨迹
        stereo: bool = False,   # 是否输出双目，默认单目（左目）
        len_train: int = 10000,
        len_test: int = 1000,
        depth_max: float = 80.0,
    ):
        # 调用父类初始化，继承配置
        super().__init__(common_conf=common_conf)
        
        self.root_dir = root_dir
        self.env = env
        self.difficulty = difficulty
        self.traj_id = traj_id
        self.stereo = stereo
        self.depth_max = depth_max
        self.training = (split == "train")
        
        # 构建所有帧索引（支持多轨迹）
        self._build_all_frames()
        
        # 原始内参（640x480 分辨率下的默认值）
        self.original_K = self._get_original_intrinsics()
        
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
    
    def _get_original_intrinsics(self):
        """返回 TartanAir V1 原始内参（640x480 分辨率）"""
        return np.array([[320.0, 0.0, 320.0],
                         [0.0, 320.0, 240.0],
                         [0.0, 0.0, 1.0]], dtype=np.float32)
    
    def _adjust_intrinsics_for_resize(self, original_size, target_size):
        """
        根据图像缩放调整内参矩阵
        
        Args:
            original_size: (H, W) 原始图像尺寸
            target_size: (H, W) 目标图像尺寸（经过 process_one_image 处理后）
        
        Returns:
            调整后的内参矩阵 (3x3)
        """
        orig_h, orig_w = original_size
        target_h, target_w = target_size
        
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        K = self.original_K.copy()
        K[0, 0] *= scale_x  # fx
        K[1, 1] *= scale_y  # fy
        K[0, 2] *= scale_x  # cx
        K[1, 2] *= scale_y  # cy
        
        return K
    
    def _build_all_frames(self):
        """构建所有帧的索引列表"""
        base_path = osp.join(self.root_dir, self.env, self.difficulty)
        
        if not osp.isdir(base_path):
            raise FileNotFoundError(f"路径不存在: {base_path}")
        
        # 确定要加载的轨迹
        if self.traj_id:
            # 确保 traj_id 是字符串列表
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
            
            # 确保 traj_path 存在
            if not osp.isdir(traj_path):
                logging.warning(f"轨迹路径不存在: {traj_path}")
                continue
            
            # 图像路径
            img_dir = osp.join(traj_path, "image_left")
            if not osp.isdir(img_dir):
                logging.warning(f"跳过 {traj}: 没有 image_left 目录")
                continue
            
            # 获取所有图像
            img_paths = sorted(glob.glob(osp.join(img_dir, "*.png")))
            
            if not img_paths:
                logging.warning(f"跳过 {traj}: 没有找到图像")
                continue
            
            logging.info(f"轨迹 {traj}: 找到 {len(img_paths)} 张图像")
            
            for img_path in img_paths:
                # 提取帧索引
                basename = osp.basename(img_path)
                match = re.search(r'(\d+)', basename)
                frame_idx = int(match.group(1)) if match else len(self.all_frames)
                
                # 构建深度图路径
                depth_path = osp.join(traj_path, "depth_left", f"{frame_idx:06d}_left_depth.npy")
                if not osp.exists(depth_path):
                    depth_path = osp.join(traj_path, "depth_left", f"{frame_idx:06d}.npy")
                if not osp.exists(depth_path):
                    continue  # 跳过没有深度图的帧
                
                frame_info = {
                    'image_path': img_path,
                    'depth_path': depth_path,
                    'traj': traj,
                    'frame_idx': frame_idx
                }
                
                # 如果是双目模式，同时检查右目数据
                if self.stereo:
                    right_path = img_path.replace("image_left", "image_right")
                    right_path = right_path.replace("_left.png", "_right.png")
                    right_depth_path = depth_path.replace("depth_left", "depth_right")
                    right_depth_path = right_depth_path.replace("_left_depth.npy", "_right_depth.npy")
                    right_depth_path = right_depth_path.replace("_left.npy", "_right.npy")
                    
                    if osp.exists(right_path) and osp.exists(right_depth_path):
                        frame_info['right_image_path'] = right_path
                        frame_info['right_depth_path'] = right_depth_path
                    else:
                        # 如果没有右目数据，跳过该帧
                        continue
                
                self.all_frames.append(frame_info)
        
        if not self.all_frames:
            raise ValueError(f"没有找到任何有效帧: {base_path}")
        
        logging.info(f"总共找到 {len(self.all_frames)} 个有效帧")
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
        depth = np.load(path)
        depth = np.clip(depth, 0, self.depth_max)
        return depth
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        """
        被 DataLoader 调用
        返回单帧数据
        """
        if self.training:
            frame = random.choice(self.all_frames)
        else:
            frame = self.all_frames[idx % len(self.all_frames)]
        
        # 计算目标尺寸（使用父类方法，基于 img_size 和 patch_size）
        target_image_shape = self.get_target_shape(aspect_ratio=1.0)
        
        # ===== 加载左目数据 =====
        left_img = self._load_image(frame['image_path'])
        depth_left = self._load_depth(frame['depth_path'])
        original_size = np.array(left_img.shape[:2])
        
        # 原始内参
        intri_left = self.original_K.copy()
        
        # 调用父类的 process_one_image 处理左目
        # 注意：需要修改 process_one_image 的返回值，只取需要的部分
        # (
        #     left_img_processed,
        #     depth_left_processed,
        #     intri_left_processed,
        # ) = self.process_one_image(
        #     left_img,
        #     depth_left,
        #     intri_opencv=intri_left,
        #     original_size=original_size,
        #     target_image_shape=target_image_shape
        # )
        
        # 构建返回结果
        batch = {
            'seq_name': f"{self.env}_{self.difficulty}_{frame['traj']}",
            'frame_idx': frame['frame_idx'],
            'image': left_img,
            'depth': depth_left,
            'K': intri_left,
            'original_size': original_size,
        }
        
        # ===== 如果是双目模式，加载右目数据 =====
        if self.stereo and 'right_image_path' in frame:
            right_img = self._load_image(frame['right_image_path'])
            depth_right = self._load_depth(frame['right_depth_path'])
            original_size_right = np.array(right_img.shape[:2])
            
            intri_right = self.original_K.copy()
            
            # (
            #     right_img_processed,
            #     depth_right_processed,
            #     intri_right_processed,
            # ) = self.process_one_image(
            #     right_img,
            #     depth_right,
            #     intri_opencv=intri_right,
            #     original_size=original_size_right,
            #     target_image_shape=target_image_shape
            # )
            
            batch['right_image'] = right_img
            batch['right_depth'] = depth_right
            batch['right_K'] = intri_right
        
        return batch
    
