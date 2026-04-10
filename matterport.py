# matterport3d_dataset.py

import os
import os.path as osp
import logging
import random
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import re
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def read_camera_intrinsics(filepath):
    """
    读取 Matterport3D 相机内参文件（单行格式）
    文件内容示例：
        1280 1024 1076.45 1077.19 631.116 513.798 ...
    解析：width height fx fy cx cy
    """
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        if not line:
            raise ValueError(f"空文件: {filepath}")
        parts = list(map(float, line.split()))
        if len(parts) < 6:
            raise ValueError(f"内参文件格式错误，至少需要6个数值: {filepath}")
        width = int(parts[0])
        height = int(parts[1])
        fx = parts[2]
        fy = parts[3]
        cx = parts[4]
        cy = parts[5]
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    return K, width, height


class Matterport3DDataset(Dataset):
    """
    Matterport3D 数据集加载器
    使用原始 RGB (matterport_color_images) 和深度 (matterport_depth_images)，
    相机内参从 matterport_camera_intrinsics 读取。
    深度单位：每个值 = 0.25 mm，转换为米需除以 4000。
    """
    def __init__(
        self,
        root_dir: str = "/path/to/data/v1/scans",
        split: str = "train",
        scene_ids_file: str = None,
        depth_max: float = 20.0,
        len_train: int = 100000,
        len_val: int = 10000,
        len_test: int = 10000,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.depth_max = depth_max
        self.training = (split == "train")

        # 获取场景ID列表
        if scene_ids_file and osp.exists(scene_ids_file):
            with open(scene_ids_file, 'r') as f:
                self.scene_ids = [line.strip() for line in f if line.strip()]
            logging.info(f"从列表文件加载 {len(self.scene_ids)} 个场景")
        else:
            self.scene_ids = [d for d in os.listdir(root_dir)
                              if osp.isdir(osp.join(root_dir, d))]
            logging.info(f"自动扫描找到 {len(self.scene_ids)} 个场景")

        if not self.scene_ids:
            raise FileNotFoundError(f"在 {root_dir} 中没有找到场景文件夹")

        # 构建帧索引
        self._build_all_frames()

        # 数据集采样长度
        if split == "train":
            self.dataset_len = len_train
        elif split == "val":
            self.dataset_len = len_val
        else:
            self.dataset_len = len_test

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Matterport3D {split} split")
        logging.info(f"{status}: Total frames: {len(self.all_frames)}")
        logging.info(f"{status}: Dataset length: {len(self)}")

    def _build_all_frames(self):
        self.all_frames = []
        for scene_id in self.scene_ids:
            scene_path = osp.join(self.root_dir, scene_id)
            img_dir = osp.join(scene_path, "undistorted_color_images")
            depth_dir = osp.join(scene_path, "undistorted_depth_images")
            cam_dir = osp.join(scene_path, "undistorted_camera_parameters")

            if not (osp.isdir(img_dir) and osp.isdir(depth_dir) and osp.isdir(cam_dir)):
                logging.warning(f"跳过 {scene_id}: 缺少必要目录")
                continue

            img_paths = sorted(glob.glob(osp.join(img_dir, "*.jpg")))
            if not img_paths:
                logging.warning(f"跳过 {scene_id}: 没有图像")
                continue

            # logging.info(f"场景 {scene_id}: 找到 {len(img_paths)} 张图像")

            for img_path in img_paths:
                basename = osp.basename(img_path)
                # 匹配模式: {uuid}_i{view}_{index}.jpg
                match = re.match(r'([a-f0-9]+)_i(\d+)_(\d+)\.jpg', basename)
                if not match:
                    logging.warning(f"跳过无法解析的文件名: {basename}")
                    continue
                uuid = match.group(1)
                view = match.group(2)
                idx = match.group(3)

                # 深度图路径: {uuid}_d{view}_{idx}.png
                depth_filename = f"{uuid}_d{view}_{idx}.png"
                depth_path = osp.join(depth_dir, depth_filename)
                if not osp.exists(depth_path):
                    continue

                # 相机内参路径: {uuid}_intrinsics_{view}.txt
                cam_filename = f"{uuid}_intrinsics_{view}.txt"
                cam_path = osp.join(cam_dir, cam_filename)
                if not osp.exists(cam_path):
                    continue

                # 验证内参可读
                try:
                    read_camera_intrinsics(cam_path)
                except Exception as e:
                    logging.warning(f"跳过 {uuid}_view{view}_idx{idx}: 相机参数无效 - {e}")
                    continue

                self.all_frames.append({
                    'scene_id': scene_id,
                    'frame_id':  f"{uuid}_{view}_{idx}",
                    'image_path': img_path,
                    'depth_path': depth_path,
                    'cam_path': cam_path,
                })

        if not self.all_frames:
            raise ValueError(f"没有找到任何有效帧: {self.root_dir}")
        logging.info(f"总共找到 {len(self.all_frames)} 个有效帧")

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"无法读取图像: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_depth(self, path):
        # 16-bit PNG，每个值 = 0.25 mm
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise IOError(f"无法读取深度图: {path}")
        depth = depth.astype(np.float32) / 4000.0   # 转换为米
        depth = np.clip(depth, 0, self.depth_max)
        return depth

    def _load_intrinsics(self, path):
        K, _, _ = read_camera_intrinsics(path)
        return K

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.training:
            frame = random.choice(self.all_frames)
        else:
            frame = self.all_frames[idx % len(self.all_frames)]

        image = self._load_image(frame['image_path'])
        depth = self._load_depth(frame['depth_path'])
        
        K = self._load_intrinsics(frame['cam_path'])

        # 转为 torch tensor (保持 HWC 和 HW)
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)
        
        K_tensor = torch.from_numpy(K).float()

        sparse= depth
        sparse_tensor= torch.from_numpy(sparse).float()
        return {
            'seq_name': f"{frame['scene_id']}_{frame['frame_id']}",
            'scene_id': frame['scene_id'],
            'frame_idx': frame['frame_id'],
            'image': image_tensor,
            'depth': depth_tensor,
            'K': K_tensor,
            'sparse': sparse_tensor
        }