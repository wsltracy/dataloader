# matterport3d_dataset.py - 三目版本

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
from collections import defaultdict

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def parse_camera_intrinsics_from_conf(conf_path, view_id):
    """
    从Matterport3D的.conf文件中解析指定view的相机内参
    
    .conf文件格式:
        intrinsics_matrix 1076.45 0 631.116  0 1077.19 509.202  0 0 1
        scan ... (view 0 的数据)
        
        intrinsics_matrix 1076.01 0 635.509  0 1076.38 511.999  0 0 1
        scan ... (view 1 的数据)
        
        intrinsics_matrix 1074.56 0 639.97  0 1074.73 508.007  0 0 1
        scan ... (view 2 的数据)
    
    Args:
        conf_path: .conf文件路径
        view_id: 相机ID (0, 1, 2)
    
    Returns:
        K: 3x3内参矩阵
    """
    if not osp.exists(conf_path):
        logging.warning(f"配置文件不存在: {conf_path}")
        return None
    
    with open(conf_path, 'r') as f:
        lines = f.readlines()
    
    # 按顺序收集前三个内参矩阵
    intrinsics_list = []
    for line in lines:
        line = line.strip()
        if line.startswith('intrinsics_matrix'):
            parts = list(map(float, line.split()[1:]))
            
            if len(parts) == 9:  # 3x3矩阵格式
                K = np.array([[parts[0], parts[1], parts[2]],
                              [parts[3], parts[4], parts[5]],
                              [parts[6], parts[7], parts[8]]], dtype=np.float32)
                
                intrinsics_list.append(K)
                
                # 只需要前三个内参矩阵
                if len(intrinsics_list) >= 3:
                    break
    
    # 检查是否找到了足够的内参矩阵
    if view_id < len(intrinsics_list):
        return intrinsics_list[view_id]


class Matterport3DDataset(Dataset):
    """
    Matterport3D 三目数据集加载器
    每个样本包含3个相机(view 0,1,2)在同一位置同一方向的图像、深度图和内参
    """
    def __init__(
        self,
        root_dir: str = "/media/wsl/SANDISK E/dataset/matterport/data/v1/scans",
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

        # 构建三目帧索引
        self._build_tri_frames()

        # 数据集采样长度
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

    def _build_tri_frames(self):
        """
        构建三目帧索引
        每个条目包含同一位置(uuid)同一方向(index)的三个view (0,1,2)
        """
        self.tri_frames = []
        
        for scene_id in self.scene_ids:
            scene_path = osp.join(self.root_dir, scene_id)
            
            # 查找目录结构
            img_base = osp.join(scene_path, "undistorted_color_images", scene_id, "undistorted_color_images")
            depth_base = osp.join(scene_path, "undistorted_depth_images", scene_id, "undistorted_depth_images")
            cam_base = osp.join(scene_path, "undistorted_camera_parameters", scene_id, "undistorted_camera_parameters")
            
            if not (osp.isdir(img_base) and osp.isdir(depth_base)):
                # 尝试另一种目录结构
                img_base = osp.join(scene_path, "undistorted_color_images")
                depth_base = osp.join(scene_path, "undistorted_depth_images")
                cam_base = osp.join(scene_path, "undistorted_camera_parameters")
            
            if not osp.isdir(img_base):
                logging.warning(f"跳过 {scene_id}: 图像目录不存在")
                continue
            
            # 获取所有图像文件
            all_images = glob.glob(osp.join(img_base, "*.jpg")) + glob.glob(osp.join(img_base, "*.png"))
            
            # 按 (uuid, index) 分组
            # uuid_i{view}_{index}.jpg -> uuid, view, index
            grouped_frames = defaultdict(lambda: {'views': {}})
            
            for img_path in all_images:
                basename = osp.basename(img_path)
                # 匹配: {uuid}_i{view}_{index}.jpg
                match = re.match(r'([a-f0-9]+)_i(\d+)_(\d+)\.(jpg|png)', basename)
                if not match:
                    continue
                
                uuid = match.group(1)
                view = int(match.group(2))
                idx = int(match.group(3))
                
                key = f"{uuid}_{idx}"  # 同一位置同一方向
                grouped_frames[key]['uuid'] = uuid
                grouped_frames[key]['index'] = idx
                grouped_frames[key]['views'][view] = {
                    'image_path': img_path,
                    'view': view
                }
            
            # 查找.conf文件获取内参
            conf_files = glob.glob(osp.join(cam_base, "*.conf"))
            conf_path = conf_files[0] if conf_files else None
            
            # 为每个完整的三目组创建样本
            for key, frame_data in grouped_frames.items():
                views_data = frame_data['views']
                
                # 检查是否三个view都存在
                if len(views_data) != 3:
                    continue
                
                uuid = frame_data['uuid']
                idx = frame_data['index']
                
                tri_sample = {
                    'scene_id': scene_id,
                    'uuid': uuid,
                    'index': idx,
                    'views': {}
                }
                
                # 为每个view添加深度图和内参
                for view in [0, 1, 2]:
                    if view not in views_data:
                        break
                    
                    # 深度图路径
                    depth_filename = f"{uuid}_d{view}_{idx}.png"
                    depth_path = osp.join(depth_base, depth_filename)
                    
                    if not osp.exists(depth_path):
                        break
                    
                    # 加载内参
                    if conf_path and osp.exists(conf_path):
                        K = parse_camera_intrinsics_from_conf(conf_path, view)
                    
                    tri_sample['views'][view] = {
                        'image_path': views_data[view]['image_path'],
                        'depth_path': depth_path,
                        'K': K,
                    }
                
                # 只有三个view都完整才添加
                if len(tri_sample['views']) == 3:
                    self.tri_frames.append(tri_sample)
            
            logging.info(f"场景 {scene_id}: 找到 {len([f for f in self.tri_frames if f['scene_id'] == scene_id])} 个三目组")
        
        if not self.tri_frames:
            raise ValueError(f"没有找到任何有效的三目帧: {self.root_dir}")
        
        logging.info(f"总共找到 {len(self.tri_frames)} 个三目帧组")

    def _load_image(self, path):
        """加载RGB图像"""
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"无法读取图像: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_depth(self, path):
        """加载深度图"""
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise IOError(f"无法读取深度图: {path}")
        # 16-bit PNG，每个值 = 0.25 mm，转换为米
        depth = depth.astype(np.float32) / 4000.0
        # 裁剪深度值
        depth[depth > self.depth_max] = 0
        return depth

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        """返回三目数据"""
        if self.training:
            frame = random.choice(self.tri_frames)
        else:
            frame = self.tri_frames[idx % len(self.tri_frames)]
        
        # 加载三个相机的数据
        images = {}
        depths = {}
        Ks = {}
        
        for view in [0, 1, 2]:
            view_data = frame['views'][view]
            
            image = self._load_image(view_data['image_path'])
            depth = self._load_depth(view_data['depth_path'])
            K = view_data['K']
            
            # 转换为tensor
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)
            K_tensor = torch.from_numpy(K).float()
            
            images[view] = image_tensor
            depths[view] = depth_tensor
            Ks[view] = K_tensor
  
        # 输出格式：三个相机的数据
        return {
            'seq_name': f"{frame['scene_id']}_{frame['uuid']}_idx{frame['index']}",
            'scene_id': frame['scene_id'],
            'uuid': frame['uuid'],
            'frame_idx': frame['index'],
            # 相机1 (view 0) - 通常是左目
            'image_1': images[0],
            'depth_1': depths[0],
            'K_1': Ks[0],
            # 相机2 (view 1) - 通常是中目
            'image_2': images[1],
            'depth_2': depths[1],
            'K_2': Ks[1],
            # 相机3 (view 2) - 通常是右目
            'image_3': images[2],
            'depth_3': depths[2],
            'K_3': Ks[2],
        }


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    root_dir = "/media/wsl/SANDISK ELE/dataset/matterport/data/v1/scans"
    
    # 创建三目数据集
    try:
        dataset = Matterport3DDataset(
            root_dir=root_dir,
            split="test",
            depth_max=20.0,
            len_test=100
        )
        
        print(f"\n数据集创建成功！共 {len(dataset.tri_frames)} 个三目组")
        
        # 测试加载一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n样本键: {sample.keys()}")
            print(f"序列名: {sample['seq_name']}")
            print(f"\n相机1 (view0):")
            print(f"  图像形状: {sample['image_1'].shape}")
            print(f"  深度形状: {sample['depth_1'].shape}")
            print(f"  内参矩阵:\n{sample['K_1']}")
            print(f"\n相机2 (view1):")
            print(f"  图像形状: {sample['image_2'].shape}")
            print(f"  深度形状: {sample['depth_2'].shape}")
            print(f"  内参矩阵:\n{sample['K_2']}")
            print(f"\n相机3 (view2):")
            print(f"  图像形状: {sample['image_3'].shape}")
            print(f"  深度形状: {sample['depth_3'].shape}")
            print(f"  内参矩阵:\n{sample['K_3']}")
            
    except Exception as e:
        print(f"数据集创建失败: {e}")
        import traceback
        traceback.print_exc()