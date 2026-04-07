# ddad_multicam_dataset.py

import os
import os.path as osp
import logging
import random
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """将四元数转换为旋转矩阵 (3x3)"""
    return R.from_quat([qx, qy, qz, qw]).as_matrix()


def build_extrinsics_matrix(rot, trans):
    """
    从旋转矩阵和平移向量构建 4x4 外参矩阵。
    约定：点从车辆坐标系变换到相机坐标系：p_cam = R @ p_veh + t
    其中 rot, trans 是相机在车辆坐标系中的位姿（即车辆坐标系下的相机坐标）。
    那么 R = R_cam_to_veh^T, t = -R_cam_to_veh^T * t_cam_to_veh
    这里我们直接根据提供的旋转和平移计算从车辆到相机的变换。
    假设提供的 rot, trans 表示相机在车辆坐标系下的位姿（即 p_cam_in_vehicle = R_cam_to_veh * p_cam + t_cam_to_veh），
    则车辆到相机的变换为：R_veh_to_cam = R_cam_to_veh^T, t_veh_to_cam = -R_veh_to_cam @ t_cam_to_veh
    """
    # 输入的 rot 和 trans 表示相机相对于车辆坐标系的位姿（即相机在车辆坐标系中的位置和方向）
    # 因此，车辆坐标系中的点 p_veh 到相机坐标系 p_cam 的变换为：
    # p_cam = R_cam_to_veh^T * (p_veh - t_cam_to_veh)
    R_cam_to_veh = rot
    t_cam_to_veh = trans.reshape(3, 1)
    R_veh_to_cam = R_cam_to_veh.T
    t_veh_to_cam = -R_veh_to_cam @ t_cam_to_veh
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_veh_to_cam
    T[:3, 3] = t_veh_to_cam.flatten()
    return T


def read_calibration(calib_file, cam_names):
    """
    读取标定文件，返回每个相机的内参 K、外参矩阵 vehicle_to_camera (4x4)
    calib_file: JSON 文件路径
    cam_names: 需要读取的相机名称列表
    返回: dict {cam_name: (K, vehicle_to_camera)}
    """
    with open(calib_file, 'r') as f:
        calib = json.load(f)
    names = calib['names']
    intrinsics_list = calib['intrinsics']
    extrinsics_list = calib['extrinsics']

    # 构建名称到索引的映射
    name_to_idx = {name: i for i, name in enumerate(names)}

    result = {}
    for cam in cam_names:
        if cam not in name_to_idx:
            raise ValueError(f"Camera {cam} not found in calibration file")
        idx = name_to_idx[cam]
        intr = intrinsics_list[idx]
        extr = extrinsics_list[idx]

        # 内参矩阵 K
        fx = intr['fx']
        fy = intr['fy']
        cx = intr['cx']
        cy = intr['cy']
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float32)

        # 外参：旋转四元数和平移
        rot_q = extr['rotation']
        trans = extr['translation']
        rot_mat = quaternion_to_rotation_matrix(rot_q['qw'], rot_q['qx'], rot_q['qy'], rot_q['qz'])
        t_vec = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float32)

        # 构建从车辆坐标系到相机坐标系的变换矩阵
        vehicle_to_camera = build_extrinsics_matrix(rot_mat, t_vec)

        result[cam] = (K, vehicle_to_camera)
    return result


def load_lidar_npz(filepath):
    """加载 .npz 点云文件，返回 (N,3) 点坐标和 (N,) 强度"""
    data = np.load(filepath)
    if 'point_cloud' in data:
        points = data['point_cloud']
    elif 'arr_0' in data:
        points = data['arr_0']
    else:
        keys = list(data.keys())
        if len(keys) > 0:
            points = data[keys[0]]
        else:
            raise ValueError(f"Unknown point cloud format in {filepath}")
    if points.shape[1] >= 3:
        xyz = points[:, :3].astype(np.float32)
        intensity = points[:, 3] if points.shape[1] >= 4 else np.ones(len(xyz), dtype=np.float32)
    else:
        raise ValueError(f"Point cloud has less than 3 columns: {points.shape}")
    return xyz, intensity


def project_lidar_to_image(points_vehicle, vehicle_to_camera, K, image_shape, depth_max=250.0):
    """
    将车辆坐标系下的点云投影到图像平面，生成稀疏深度图。
    points_vehicle: (N,3)
    vehicle_to_camera: 4x4 矩阵，从车辆坐标系到相机坐标系
    K: 3x3 内参
    image_shape: (height, width)
    depth_max: 最大深度（米）
    返回: depth_map (H,W) float32
    """
    # 添加齐次坐标并变换到相机坐标系
    N = points_vehicle.shape[0]
    points_vehicle_h = np.hstack([points_vehicle, np.ones((N, 1))])  # (N,4)
    points_cam_h = points_vehicle_h @ vehicle_to_camera.T  # (N,4)
    points_cam = points_cam_h[:, :3]  # (N,3)
    eps = 1e-5
    valid = points_cam[:, 2] > eps
    # if not np.any(valid):
    #     return np.zeros(image_shape, dtype=np.float32)
    # points_cam = points_cam[valid]
    # points_cam_h = points_cam_h[valid]
    # points_cam_h[:, 2] += eps  # 避免除以零
    if not np.any(valid):
        return np.zeros(image_shape, dtype=np.float32)
    points_cam = points_cam[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = (points_cam[:, 0] * fx / points_cam[:, 2] + cx).astype(np.int32)
    v = (points_cam[:, 1] * fy / points_cam[:, 2] + cy).astype(np.int32)
    depth = points_cam[:, 2]
    H, W = image_shape
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, depth = u[mask], v[mask], depth[mask]
    depth_map = np.zeros((H, W), dtype=np.float32)
    # 每个像素保留最近的深度
    for i in range(len(u)):
        if depth[i] < depth_map[v[i], u[i]] or depth_map[v[i], u[i]] == 0:
            depth_map[v[i], u[i]] = depth[i]
    depth_map = np.clip(depth_map, 0, depth_max)
    return depth_map


class DDADDataset(Dataset):
    """
    DDAD 数据集加载器（多相机版本）
    输出：多个相机的 image, depth, K
    深度图由 LiDAR 投影生成，使用标定文件中的外参和内参
    """
    def __init__(
        self,
        root_dir: str = "/path/to/DDAD/ddad_train_val",
        ddad_json_path: str = "/path/to/ddad.json",
        split: str = "0",                       # '0' 训练, '1' 验证
        cam_names: list = None,                 # 例如 ['CAMERA_01', 'CAMERA_05', ...]
        generate_depth: bool = True,            # 是否生成深度图
        depth_max: float = 250.0,
        len_train: int = 100000,
        len_val: int = 10000,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.ddad_json_path = ddad_json_path
        self.split = split
        self.cam_names = cam_names if cam_names is not None else [
            'CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09'
        ]
        self.generate_depth = generate_depth
        self.depth_max = depth_max
        self.training = (split == "0")   # 假设 split "0" 为训练

        # 加载 ddad.json
        with open(ddad_json_path, 'r') as f:
            ddad = json.load(f)
        scene_splits = ddad['scene_splits']
        if split not in scene_splits:
            raise ValueError(f"Split {split} not found in ddad.json")
        self.scene_files = scene_splits[split]['filenames']
        logging.info(f"加载 {len(self.scene_files)} 个场景文件")

        # 构建所有帧索引
        self._build_all_frames()

        # 设置数据集长度
        if split == "0":
            self.dataset_len = len_train
        else:
            self.dataset_len = len_val

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: DDADMultiCam {split} split")
        logging.info(f"{status}: Total samples: {len(self.all_frames)}")
        logging.info(f"{status}: Dataset length: {len(self)}")

    def _build_all_frames(self):
        """构建所有帧的索引列表"""
        self.all_frames = []          # 每个元素: {image_paths: {cam: path}, lidar_path, calib_key}
        self.calib_cache = {}         # calib_key -> {cam: (K, vehicle_to_camera)}

        for scene_rel_path in self.scene_files:
            scene_json_path = osp.join(self.root_dir, scene_rel_path)
            if not osp.exists(scene_json_path):
                logging.warning(f"跳过缺失场景文件: {scene_json_path}")
                continue
            scene_dir = osp.dirname(scene_json_path)
            with open(scene_json_path, 'r') as f:
                scene_data = json.load(f)

            global_calib_key = scene_data.get('calibration_key')
            data_list = scene_data.get('data', [])
            key_to_datum = {d['key']: d for d in data_list}
            samples = scene_data.get('samples', [])
            if not samples:
                continue

            for sample in samples:
                calib_key = sample.get('calibration_key', global_calib_key)
                if calib_key is None:
                    continue
                datum_keys = sample.get('datum_keys', [])
                if not datum_keys:
                    continue

                # 收集本 sample 中所有需要的相机图像路径和 LiDAR 路径
                image_paths = {cam: None for cam in self.cam_names}
                lidar_path = None
                for dkey in datum_keys:
                    datum = key_to_datum.get(dkey)
                    if datum is None:
                        continue
                    if 'image' in datum['datum']:
                        img_info = datum['datum']['image']
                        cam_name = datum['id']['name']
                        if cam_name in self.cam_names:
                            rel_path = img_info['filename']
                            abs_path = osp.join(scene_dir, rel_path)
                            image_paths[cam_name] = abs_path
                    elif 'point_cloud' in datum['datum']:
                        pc_info = datum['datum']['point_cloud']
                        rel_path = pc_info['filename']
                        abs_path = osp.join(scene_dir, rel_path)
                        lidar_path = abs_path

                # 检查是否所有指定相机都有图像，以及是否有 LiDAR
                if any(p is None for p in image_paths.values()):
                    continue
                if self.generate_depth and lidar_path is None:
                    continue

                # 缓存标定信息
                if calib_key not in self.calib_cache:
                    calib_file = osp.join(self.root_dir, 'calibration', f'{calib_key}.json')
                    if not osp.exists(calib_file):
                        calib_file = osp.join(scene_dir, 'calibration', f'{calib_key}.json')
                    if not osp.exists(calib_file):
                        logging.warning(f"标定文件不存在: {calib_key}")
                        continue
                    try:
                        cam_data = read_calibration(calib_file, self.cam_names)
                        self.calib_cache[calib_key] = cam_data
                    except Exception as e:
                        logging.warning(f"读取标定文件失败 {calib_file}: {e}")
                        continue

                self.all_frames.append({
                    'image_paths': image_paths,
                    'lidar_path': lidar_path,
                    'calib_key': calib_key,
                    'seq_name': scene_rel_path,  # 添加场景名称
                    'frame_idx': len(self.all_frames),  # 添加帧索引
                })

        if not self.all_frames:
            raise ValueError(f"没有找到任何有效帧: {self.root_dir}")
        logging.info(f"总共找到 {len(self.all_frames)} 个有效样本")

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"无法读取图像: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.training:
            frame = random.choice(self.all_frames)
        else:
            frame = self.all_frames[idx % len(self.all_frames)]

        calib_data = self.calib_cache[frame['calib_key']]

        # 加载 LiDAR 点云（所有相机共享）
        points_vehicle = None
        if self.generate_depth and frame['lidar_path'] is not None:
            points_vehicle, _ = load_lidar_npz(frame['lidar_path'])

        batch = {}
        for cam in self.cam_names:
            img_path = frame['image_paths'][cam]
            img = self._load_image(img_path)
            H, W = img.shape[:2]

            # 获取该相机的内参和外参
            K, vehicle_to_camera = calib_data[cam]

            # 生成深度图
            if self.generate_depth and points_vehicle is not None:
                depth = project_lidar_to_image(points_vehicle, vehicle_to_camera, K, (H, W), self.depth_max)
            else:
                depth = np.zeros((H, W), dtype=np.float32)

            # 转换为 torch tensor
            img_tensor = torch.from_numpy(img).float() / 255.0
            depth_tensor = torch.from_numpy(depth).float()
            K_tensor = torch.from_numpy(K).float()
            # 提取相机编号（从 CAMERA_01 中提取 01）
            cam_id = str(int(cam.split('_')[-1]))

            batch[f'image_{cam_id}'] = img_tensor
            batch[f'sparse_{cam_id}'] = depth_tensor
            batch[f'K_{cam_id}'] = K_tensor
            # 添加序列名称和帧索引
            batch['seq_name'] = frame['seq_name']
            batch['frame_idx'] = frame['frame_idx']

        return batch