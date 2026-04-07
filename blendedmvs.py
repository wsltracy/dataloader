# blendedmvs_dataset.py

import os
import os.path as osp
import logging
import random
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# 设置所有随机种子
random.seed(42)          # Python random 模块
np.random.seed(42)       # NumPy 随机数生成器
torch.manual_seed(42)    # PyTorch CPU 随机数生成器
torch.cuda.manual_seed_all(42)  # PyTorch GPU 随机数生成器（如果使用GPU）

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
        # 垂直翻转深度图，因为PFM格式是bottom-up存储的
        data = np.flipud(data)
        return data, scale


def read_cam_file(filename):
    """
    读取 BlendedMVS 相机参数文件
    返回内参矩阵 K (3x3)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 跳过空行
    lines = [line.strip() for line in lines if line.strip()]
    
    K = None
    
    i = 0
    while i < len(lines):
        if lines[i].lower() == 'intrinsic':
            # 内参矩阵是接下来的3行
            i += 1
            K = np.zeros((3, 3), dtype=np.float32)
            for j in range(3):
                if i + j < len(lines):
                    values = list(map(float, lines[i + j].split()))
                    if len(values) >= 3:
                        K[j] = values[:3]
            break
        i += 1
    
    if K is None:
        raise ValueError(f"无法在文件中找到 'intrinsic' 标记: {filename}")
    
    return K


class BlendedMVSDataset(Dataset):
    """
    BlendedMVS 数据集加载器
    输出: image, depth, K (相机内参)
    单目模式，不做图像缩放
    """
    def __init__(
        self,
        root_dir: str = "/path/to/BlendedMVS",
        split: str = "train",           # 'train', 'val', 'test'
        scene_list_file: str = None,    # 场景列表文件
        depth_max: float = 200.0,       # 深度最大值
        len_train: int = 10000,
        # len_val: int = 1000,
        len_test: int = 1000,
    ):
        super().__init__()
        
        self.root_dir = root_dir
        self.split = split
        self.depth_max = depth_max
        self.training = (split == "train")
        
        # 如果没有提供列表文件，自动扫描目录
        if scene_list_file is None or not osp.exists(scene_list_file):
            # 自动扫描 root_dir 下的所有场景目录
            self.scenes = []
            for item in os.listdir(root_dir):
                item_path = osp.join(root_dir, item)
                if osp.isdir(item_path) :
                    # 检查是否有必要的数据目录
                    img_dir = osp.join(item_path, "blended_images")
                    depth_dir = osp.join(item_path, "rendered_depth_maps")
                    cam_dir = osp.join(item_path, "cams")
                    if osp.isdir(img_dir) and osp.isdir(depth_dir) and osp.isdir(cam_dir):
                        self.scenes.append(item)
            
            if not self.scenes:
                raise FileNotFoundError(f"在 {root_dir} 中没有找到有效的 BlendedMVS 场景")
            
            logging.info(f"自动扫描找到 {len(self.scenes)} 个场景")
        else:
            with open(scene_list_file, 'r') as f:
                self.scenes = [line.strip() for line in f.readlines()]
            logging.info(f"从列表文件加载 {len(self.scenes)} 个场景")
        
        # 构建所有帧索引
        self._build_all_frames()
        
        # 设置数据集长度
        if split == "train":
            self.dataset_len = len_train
        else:
            self.dataset_len = len_test
        
        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: BlendedMVS {split} split")
        logging.info(f"{status}: Total frames: {len(self.all_frames)}")
        logging.info(f"{status}: Dataset length: {len(self)}")
    
    def _build_all_frames(self):
        """构建所有帧的索引列表"""
        self.all_frames = []
        
        for scene in self.scenes:
            scene_path = osp.join(self.root_dir, scene)
            
            # 检查必要目录
            img_dir = osp.join(scene_path, "blended_images")
            depth_dir = osp.join(scene_path, "rendered_depth_maps")
            cam_dir = osp.join(scene_path, "cams")
            
            if not osp.isdir(img_dir):
                logging.warning(f"跳过 {scene}: 没有 blended_images 目录")
                continue
            if not osp.isdir(depth_dir):
                logging.warning(f"跳过 {scene}: 没有 rendered_depth_maps 目录")
                continue
            if not osp.isdir(cam_dir):
                logging.warning(f"跳过 {scene}: 没有 cams 目录")
                continue
            
            # 获取所有图像
            img_paths = sorted(glob.glob(osp.join(img_dir, "*.jpg")))
            
            if not img_paths:
                logging.warning(f"跳过 {scene}: 没有找到图像")
                continue

            
            for img_path in img_paths:
                # 提取帧索引（从文件名中提取数字）
                basename = osp.basename(img_path)
                # 文件名格式: 00000000.jpg
                name_without_ext = basename.replace('.jpg', '').replace('.png', '')
                frame_idx = name_without_ext
                
                # 构建深度图路径
                depth_path = osp.join(depth_dir, f"{frame_idx}.pfm")
                if not osp.exists(depth_path):
                    # 尝试其他格式
                    depth_path = osp.join(depth_dir, f"{frame_idx}.npy")
                if not osp.exists(depth_path):
                    continue  # 跳过没有深度图的帧
                
                # 构建相机参数路径
                cam_path = osp.join(cam_dir, f"{frame_idx}_cam.txt")
                if not osp.exists(cam_path):
                    continue  # 跳过没有相机参数的帧
                            # 验证相机参数文件是否可读
                try:
                    test_K = read_cam_file(cam_path)
                    if test_K is None:
                        logging.warning(f"跳过 {frame_idx}: 相机参数文件无效: {cam_path}")
                        continue
                except Exception as e:
                    logging.warning(f"跳过 {frame_idx}: 无法读取相机参数文件: {cam_path}, 错误: {e}")
                    continue
                self.all_frames.append({
                    'scene': scene,
                    'frame_idx': frame_idx,
                    'image_path': img_path,
                    'depth_path': depth_path,
                    'cam_path': cam_path,
                })
        
        if not self.all_frames:
            raise ValueError(f"没有找到任何有效帧: {self.root_dir}")
        
        logging.info(f"总共找到 {len(self.all_frames)} 个有效帧")
    
    def _load_image(self, path):
        """加载 RGB 图像，返回 (H, W, 3) 格式，值域 [0, 255]"""
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
        depth = np.clip(depth, 0, self.depth_max)
        return depth
    
    def _load_intrinsics(self, path):
        """加载相机内参 K (3x3)"""
        K = read_cam_file(path)
        return K
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        """
        返回单帧数据
        输出: image, depth, K
        """
        if self.training:
            frame = random.choice(self.all_frames)
        else:
            frame = self.all_frames[idx % len(self.all_frames)]
        
        # 加载图像
        image = self._load_image(frame['image_path'])
        
        # 加载深度图
        depth = self._load_depth(frame['depth_path'])
        
        # 加载相机内参
        K = self._load_intrinsics(frame['cam_path'])
        
        # 转换为 torch tensor
        # 图像: (H, W, C) 
        image_tensor = torch.from_numpy(image).float()/ 255.0
        
        # 深度: (H, W) 
        depth_tensor = torch.from_numpy(depth).float()
        
        # 内参: (3, 3)
        K_tensor = torch.from_numpy(K).float()
        
        batch = {
            'seq_name': f"{frame['scene']}_{frame['frame_idx']}",
            'scene': frame['scene'],
            'frame_idx': frame['frame_idx'],
            'image': image_tensor,
            'depth': depth_tensor,
            'K': K_tensor,
        }
        
        return batch