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
    """
    def __init__(
        self,
        common_conf,           # 配置对象（包含 img_size, patch_size, aug_scale 等）
        split: str = "train",  # 'train' 或 'test'
        root_dir: str = "/mnt/tartanair_data",
        env: str = "carwelding",
        difficulty: str = "Easy",
        traj_id: str = "P001",
        stereo: bool = False,   # 是否输出双目，默认单目（左目）
        len_train: int = 100000,
        len_test: int = 10000,
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
        self.training = split == "train"
        # 构建轨迹路径
        self.traj_path = osp.join(root_dir, env, difficulty, traj_id)
        if not osp.isdir(self.traj_path):
            raise FileNotFoundError(f"轨迹路径不存在: {self.traj_path}")
        
        # 构建帧索引（带缓存）
        self._build_index()
        
        # 加载相机内参
        self.K = self._load_intrinsics()
        
        # 设置数据集长度（采样次数）
        if split == "train":
            self.dataset_len = len_train
        elif split == "test":
            self.dataset_len = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        status = "Training" if split == "train" else "Testing"
        logging.info(f"{status}: TartanAir V1 {env}/{difficulty}/{traj_id}")
        logging.info(f"{status}: Total frames: {self.num_samples}")
        logging.info(f"{status}: Dataset length: {len(self)}")
    
    def _build_index(self):
        """扫描文件系统，构建帧索引"""
        # 获取左目图像列表
        self.left_paths = sorted(glob.glob(osp.join(self.traj_path, "image_left", "*.png")))
        if len(self.left_paths) == 0:
            raise ValueError(f"未找到左目图像: {self.traj_path}/image_left/*.png")
        
        # 提取帧索引
        self.frame_indices = []
        for path in self.left_paths:
            basename = osp.basename(path)
            match = re.search(r'(\d+)', basename)
            if match:
                self.frame_indices.append(int(match.group(1)))
            else:
                self.frame_indices.append(len(self.frame_indices))
        
        self.num_samples = len(self.left_paths)
        
        # 如果是双目模式，构建右目路径
        if self.stereo:
            self.right_paths = []
            for left_path in self.left_paths:
                right_path = left_path.replace("image_left", "image_right")
                right_path = right_path.replace("_left.png", "_right.png")
                if not osp.exists(right_path):
                    right_path = osp.join(self.traj_path, "image_right", osp.basename(left_path))
                if not osp.exists(right_path):
                    raise FileNotFoundError(f"右目图像不存在: {right_path}")
                self.right_paths.append(right_path)
    
    def _load_intrinsics(self):
        """加载相机内参"""
        # 尝试从 sensors.json 读取
        sensor_file = osp.join(self.root_dir, self.env, "sensors.json")
        if osp.isfile(sensor_file):
            try:
                with open(sensor_file, 'r') as f:
                    sensors = json.load(f)
                # 查找左目相机
                cam_name = None
                for key in sensors.keys():
                    if 'left' in key.lower() or 'cam0' in key.lower():
                        cam_name = key
                        break
                if cam_name is None:
                    cam_name = list(sensors.keys())[0]
                cam_data = sensors[cam_name]
                fx = cam_data.get('focal_length_x', 320.0)
                fy = cam_data.get('focal_length_y', 320.0)
                cx = cam_data.get('principal_point_x', 320.0)
                cy = cam_data.get('principal_point_y', 240.0)
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            except Exception as e:
                logging.warning(f"读取 sensors.json 失败: {e}")
                K = self._default_intrinsics()
        else:
            K = self._default_intrinsics()
        
        return K
    
    def _default_intrinsics(self):
        """默认内参（640x480 分辨率）"""
        return np.array([[320.0, 0.0, 320.0],
                         [0.0, 320.0, 240.0],
                         [0.0, 0.0, 1.0]], dtype=np.float32)
    
    def _load_image(self, path):
        """加载 RGB 图像"""
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"无法读取图像: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _load_depth(self, idx, side='left'):
        """加载深度图"""
        if side == 'left':
            depth_dir = "depth_left"
            prefix = f"{idx:06d}_left_depth.npy"
        else:
            depth_dir = "depth_right"
            prefix = f"{idx:06d}_right_depth.npy"
        
        depth_path = osp.join(self.traj_path, depth_dir, prefix)
        if not osp.isfile(depth_path):
            # 尝试另一种命名
            depth_path = osp.join(self.traj_path, depth_dir, f"{idx:06d}.npy")
        if not osp.isfile(depth_path):
            raise FileNotFoundError(f"深度文件不存在: {depth_path}")
        
        depth = np.load(depth_path).astype(np.float32)
        depth = np.clip(depth, 0, self.depth_max)
        return depth
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        """
        被 DataLoader 调用
        返回 (seq_index, img_per_seq, aspect_ratio) 给 get_data
        """
        # 随机选择序列（这里固定一个轨迹，所以 seq_index 始终为 0）
        seq_index = 0  # 固定序列索引为0，表示只使用一个固定的轨迹
        img_per_seq = 1  # 每次取一帧，即每次只处理一张图像
        aspect_ratio = 1.0  # 设置图像的宽高比为1.0，表示正方形图像
        return self.get_data(
            seq_index=seq_index,  # 传入固定的序列索引
            img_per_seq=img_per_seq,  # 传入每次处理的图像数量
            aspect_ratio=aspect_ratio  # 传入图像的宽高比
        )
    
    def get_data(self, seq_index=None, img_per_seq=None, aspect_ratio=1.0):
        """
        核心方法：加载数据
        类似 VKittiDataset 的 get_data
        """
        # 随机选择一帧（训练时随机，测试时可顺序）
        if self.training:
            frame_pos = random.randint(0, self.num_samples - 1)
        else:
            # 测试模式：使用 seq_index 作为帧位置（简化处理）
            frame_pos = seq_index if seq_index is not None else 0
            frame_pos = frame_pos % self.num_samples
        
        frame_idx = self.frame_indices[frame_pos]
        
        # 计算目标尺寸
        target_shape = self.get_target_shape(aspect_ratio)
        
        # ===== 加载左目数据 =====
        left_img = self._load_image(self.left_paths[frame_pos])
        depth_left = self._load_depth(frame_idx, side='left')
        original_size = np.array(left_img.shape[:2])
        
        # 左目相机参数（内参 + 外参）
        intri_left = self.K.copy()
        # TartanAir V1 的位姿文件是 pose_left.txt，这里简化处理，使用单位外参
        # 如果需要真实外参，可以从 pose_left.txt 读取
        extri_left = np.eye(4)[:3]  # 单位外参，表示相机在世界坐标系原点
        
        # 调用父类的 process_one_image 处理左目
        (
            left_img_processed,
            depth_left_processed,
            extri_left_processed,
            intri_left_processed,
            world_points_left,
            cam_points_left,
            point_mask_left,
            _,
        ) = self.process_one_image(
            left_img,
            depth_left,
            extri_left,
            intri_left,
            original_size,
            target_shape,
            filepath=self.left_paths[frame_pos],
        )
        
        # 构建返回结果
        batch = {
            'seq_name': f"{self.env}_{self.difficulty}_{self.traj_id}",
            'frame_idx': frame_idx,
            'image': left_img_processed,
            'depth': depth_left_processed,
            'K': intri_left_processed,
            'extrinsics': extri_left_processed,
            # 'world_points': world_points_left,
            # 'cam_points': cam_points_left,
            # 'point_mask': point_mask_left,
            'original_size': original_size,
        }
        
        # ===== 如果是双目模式，加载右目数据 =====
        if self.stereo:
            right_img = self._load_image(self.right_paths[frame_pos])
            depth_right = self._load_depth(frame_idx, side='right')
            original_size_right = np.array(right_img.shape[:2])
            
            # 右目相机参数（假设与左目相同，实际可能有基线偏移）
            intri_right = self.K.copy()
            extri_right = np.eye(4)[:3]  # 简化，实际应包含左右目基线
            
            (
                right_img_processed,
                depth_right_processed,
                extri_right_processed,
                intri_right_processed,
                world_points_right,
                cam_points_right,
                point_mask_right,
                _,
            ) = self.process_one_image(
                right_img,
                depth_right,
                extri_right,
                intri_right,
                original_size_right,
                target_shape,
                filepath=self.right_paths[frame_pos],
            )
            
            batch['right_image'] = right_img_processed
            batch['right_depth'] = depth_right_processed
            batch['right_intrinsics'] = intri_right_processed
            batch['right_extrinsics'] = extri_right_processed
        
        return batch