# matterport_simple.py

import os
import os.path as osp
import logging
import random
import glob
import re
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from collections import defaultdict

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def parse_camera_intrinsics_from_conf(conf_path, view_id):
    """从.conf文件解析相机内参"""
    if not osp.exists(conf_path):
        return None
    
    with open(conf_path, 'r') as f:
        lines = f.readlines()
    
    intrinsics_list = []
    for line in lines:
        line = line.strip()
        if line.startswith('intrinsics_matrix'):
            parts = list(map(float, line.split()[1:]))
            if len(parts) == 9:
                K = np.array([[parts[0], parts[1], parts[2]],
                              [parts[3], parts[4], parts[5]],
                              [parts[6], parts[7], parts[8]]], dtype=np.float32)
                intrinsics_list.append(K)
                if len(intrinsics_list) >= 3:
                    break
    
    if view_id < len(intrinsics_list):
        return intrinsics_list[view_id]
    return None


class Matterport3DDataset(Dataset):
    """
    Matterport3D 三目数据集加载器（简化版）
    输出格式: (image_1, lidar_1, depth_1, K_1, image_2, lidar_2, depth_2, K_2, 
               image_3, lidar_3, depth_3, K_3, seq_name, scene_id, uuid, frame_idx)
    """
    def __init__(
        self,
        root_dir: str = "/media/wsl/SANDISK ELE/dataset/matterport/data/v1/scans",
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
        
        # 获取场景列表
        if scene_ids_file and osp.exists(scene_ids_file):
            with open(scene_ids_file, 'r') as f:
                self.scene_ids = [line.strip() for line in f if line.strip()]
        else:
            self.scene_ids = [d for d in os.listdir(root_dir) 
                              if osp.isdir(osp.join(root_dir, d))]
        
        # 构建三目组
        self.tri_frames = []
        
        for scene_id in self.scene_ids:
            scene_path = osp.join(root_dir, scene_id)
            
            # 查找目录结构
            img_base = osp.join(scene_path, "undistorted_color_images", scene_id, "undistorted_color_images")
            depth_base = osp.join(scene_path, "undistorted_depth_images", scene_id, "undistorted_depth_images")
            cam_base = osp.join(scene_path, "undistorted_camera_parameters", scene_id, "undistorted_camera_parameters")
            
            if not osp.isdir(img_base):
                img_base = osp.join(scene_path, "undistorted_color_images")
                depth_base = osp.join(scene_path, "undistorted_depth_images")
                cam_base = osp.join(scene_path, "undistorted_camera_parameters")
            
            if not osp.isdir(img_base):
                continue
            
            # 查找.conf文件
            conf_files = glob.glob(osp.join(cam_base, "*.conf"))
            conf_path = conf_files[0] if conf_files else None
            
            # 按 uuid 分组
            grouped = defaultdict(lambda: {'views': {}})
            
            for img_path in glob.glob(osp.join(img_base, "*.jpg")):
                basename = osp.basename(img_path)
                match = re.match(r'([a-f0-9]+)_i(\d+)_(\d+)\.jpg', basename)
                if match:
                    uuid, view, idx = match.groups()
                    view = int(view)
                    idx = int(idx)
                    key = f"{uuid}_{idx}"
                    
                    grouped[key]['uuid'] = uuid
                    grouped[key]['index'] = idx
                    grouped[key]['views'][view] = {'image_path': img_path, 'view': view}
            
            # 为每个完整的三目组创建样本
            for key, data in grouped.items():
                if len(data['views']) != 3:
                    continue
                
                # 验证深度图和内参是否存在
                valid = True
                for view in [0, 1, 2]:
                    depth_filename = f"{data['uuid']}_d{view}_{data['index']}.png"
                    depth_path = osp.join(depth_base, depth_filename)
                    
                    if not osp.exists(depth_path):
                        valid = False
                        break
                    
                    K = None
                    if conf_path:
                        K = parse_camera_intrinsics_from_conf(conf_path, view)
                    
                    if K is None:
                        valid = False
                        break
                    
                    data['views'][view]['depth_path'] = depth_path
                    data['views'][view]['K'] = K
                
                if valid:
                    self.tri_frames.append({
                        'scene_id': scene_id,
                        'uuid': data['uuid'],
                        'index': data['index'],
                        'views': data['views']
                    })
        
        # 设置数据集长度
        if split == "train":
            self.dataset_len = len_train
        elif split == "val":
            self.dataset_len = len_val
        else:
            self.dataset_len = len_test
        
        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Matterport3D Tri-View Dataset")
        logging.info(f"{status}: Total tri-frames: {len(self.tri_frames)}")
        logging.info(f"{status}: Dataset length: {len(self)}")
    
    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"无法读取图像: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _load_depth(self, path):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise IOError(f"无法读取深度图: {path}")
        depth = depth.astype(np.float32) / 4000.0  # 转换为米
        depth[depth > self.depth_max] = 0
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
        """返回三目数据"""
        if self.training:
            frame = random.choice(self.tri_frames)
        else:
            frame = self.tri_frames[idx % len(self.tri_frames)]
        
        images = {}
        depths = {}
        lidars = {}
        Ks = {}
        
        for view in [0, 1, 2]:
            view_data = frame['views'][view]
            
            image = self._load_image(view_data['image_path'])
            depth = self._load_depth(view_data['depth_path'])
            K = view_data['K']
            
            # 转换为 tensor
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)
            K_tensor = torch.from_numpy(K).float()
            
            # 生成 LiDAR 数据
            
            lidar = self.sample_hdl64e_lidar_new(depth_tensor.numpy(), K)
            lidar_tensor = torch.from_numpy(lidar).float()
            
            
            images[view] = image_tensor
            depths[view] = depth_tensor
            lidars[view] = lidar_tensor
            Ks[view] = K_tensor
        
        seq_name = f"{frame['scene_id']}_{frame['uuid']}_idx{frame['index']}"
        
        # 输出: (image_1, lidar_1, depth_1, K_1, image_2, lidar_2, depth_2, K_2,
        #        image_3, lidar_3, depth_3, K_3, seq_name, scene_id, uuid, frame_idx)
        return (images[0], lidars[0], depths[0], Ks[0],
                images[1], lidars[1], depths[1], Ks[1],
                images[2], lidars[2], depths[2], Ks[2],
                seq_name)