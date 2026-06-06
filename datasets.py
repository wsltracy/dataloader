import os
import warnings

import numpy as np
import json
import h5py

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import glob
import random
from PIL import Image
import cv2
import h5py
import io
from tqdm import tqdm
from torch.utils.data import DataLoader
import os.path as osp
import re
from dgp.datasets import SynchronizedSceneDataset
from streaming import MDSWriter, StreamingDataset, StreamingDataLoader
from torchvision.io import decode_jpeg, decode_png, ImageReadMode

import io
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import OpenEXR, Imath
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
# ---------- EXR 读取函数 ----------
def exr2hdr(exrpath):

    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    CNum = len(File.header()['channels'].keys())
    if CNum > 1:
        Channels = ['R', 'G', 'B']
        CNum = 3
    else:
        Channels = ['G']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [np.frombuffer(File.channel(c, PixType), dtype=np.float32) for c in Channels]
    hdr = np.zeros((Size[1], Size[0], CNum), dtype=np.float32)
    if CNum == 1:
        hdr[:, :, 0] = np.reshape(Pixels[0], (Size[1], Size[0]))
    else:
        hdr[:, :, 0] = np.reshape(Pixels[0], (Size[1], Size[0]))
        hdr[:, :, 1] = np.reshape(Pixels[1], (Size[1], Size[0]))
        hdr[:, :, 2] = np.reshape(Pixels[2], (Size[1], Size[0]))
    return hdr

def load_exr(filename):
    hdr = exr2hdr(filename)
    h, w, c = hdr.shape
    if c == 1:
        hdr = np.squeeze(hdr)
    return hdr

# ---------- RGB 图像加载函数 ----------
def load_rgb(filename):
    from skimage import io
    img = None
    if filename.find('.npy') > 0:
        img = np.load(filename)
    else:
        img = io.imread(filename)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.pad(img, ((0, 0), (0, 0), (0, 2)), 'constant')
            img[:, :, 1] = img[:, :, 0]
            img[:, :, 2] = img[:, :, 0]
        h, w, c = img.shape
        if c == 4:
            img = img[:, :, :3]
    return img  # uint8, (H,W,3)

# ---------- 深度计算辅助函数 ----------
def disparity_to_depth_left(disp, fx, baseline):
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = fx * baseline / disp
        depth[disp <= 0] = 0.0
        depth = np.nan_to_num(depth, nan=0.0)
    return depth

def left_depth_to_right_depth(depth_left, disp_left):
    H, W = depth_left.shape
    depth_right = np.zeros_like(depth_left)
    for y in range(H):
        valid = (disp_left[y] > 0) & (depth_left[y] > 0)
        if not np.any(valid):
            continue
        x_l = np.arange(W)[valid]
        d = disp_left[y][valid]
        x_r = np.round(x_l - d).astype(int)
        mask = (x_r >= 0) & (x_r < W)
        x_l = x_l[mask]
        x_r = x_r[mask]
        if len(x_r) == 0:
            continue
        unique_xr, inverse = np.unique(x_r, return_inverse=True)
        min_depth = np.full(len(unique_xr), np.inf)
        for i, idx in enumerate(inverse):
            val = depth_left[y, x_l[i]]
            if val < min_depth[idx]:
                min_depth[idx] = val
        depth_right[y, unique_xr] = min_depth
    return depth_right

def mask(depth,sigma=12.0):
    valid_depths = depth[depth > 0]

    if valid_depths.size > 0:
        global_median = np.median(valid_depths)

        lower_bound = global_median / sigma
        upper_bound = global_median * sigma

        depth[
            (depth <= 0) |
            (depth < lower_bound) |
            (depth > upper_bound)
        ] = 0

    return depth
def read_npy(filename, sigma=12.0):
    """
    读取 TartanAir 的 npy 深度图，并进行深度范围过滤

    Args:
        filename: .npy depth file
        sigma: 深度过滤系数

    Returns:
        depth: np.ndarray (H, W), float32
        scale: float
    """
    depth = np.load(filename).astype(np.float32)
    depth = mask(depth,sigma)

    return depth

# -------------------- Hypersim 相机内参计算（解析法） --------------------
def get_hypersim_intrinsics_from_csv(csv_path, scene_name):
    """
    从 metadata_camera_parameters.csv 读取场景的内参矩阵 K（OpenCV 风格）
    使用解析公式（假设 M_cam_from_uv 第三行为 [0,0,-1]）
    """
    df = pd.read_csv(csv_path, index_col="scene_name")
    row = df.loc[scene_name]

    # 原始图像尺寸
    orig_w = int(row["settings_output_img_width"])
    orig_h = int(row["settings_output_img_height"])

    # M_cam_from_uv 矩阵元素
    a = row["M_cam_from_uv_00"]
    b = row["M_cam_from_uv_11"]
    tx = row["M_cam_from_uv_02"]
    ty = row["M_cam_from_uv_12"]

    # 计算内参（解析公式）
    fx = (orig_w - 1) / (2 * a)
    fy = (orig_h - 1) / (2 * b)   # 取绝对值后为正
    cx = (orig_w - 1) / 2 - fx * tx
    cy = (orig_h - 1) / 2 + fy * ty

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    return K
class DDADStreamingSource(torch.utils.data.Dataset):
    def __init__(self, mode, scale=256):
        super().__init__()
        self.mode = mode
        self.scale = scale

        if mode != 'train' and mode != 'val':
            raise NotImplementedError

        # 相机ID列表
        # camera_ids = ['01', '05', '06', '07', '08', '09']
        json_path = "datas/DDAD/ddad_train_val/ddad.json"

        self.ddad_dataset = SynchronizedSceneDataset(
            json_path,
            datum_names=('lidar', 'CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09'),
            generate_depth_from_datum='lidar',
            split=mode
        )

    def __len__(self):
        return len(self.ddad_dataset) * 6

    def __getitem__(self, index):
        sample_index = index // 6
        camera_index = index % 6
        ddad_sample = self.ddad_dataset[sample_index][0]
        cam_data = ddad_sample[camera_index]
        # 提取数据
        rgb = cam_data['rgb']
        dep = cam_data['depth']
        Kcam = cam_data['intrinsics'].astype(np.float32)

        dep = np.clip(dep, None, 127)
        dep = (dep * self.scale).astype(np.uint16)
        # mask = (dep > 100.).astype(np.float32)
        # dep = (dep * mask * self.scale).astype(np.uint16)
        # dep = Image.fromarray(dep)

        byte_io = io.BytesIO()
        rgb.save(byte_io, format="JPEG", quality=90)
        # rgb.save(byte_io, format="PNG")
        jpeg_bytes = byte_io.getvalue()

        # byte_io = io.BytesIO()
        # dep.save(byte_io, format="PNG")
        # png_bytes = byte_io.getvalue()

        return {"rgb": jpeg_bytes, "dep": dep, "kcam": Kcam}


class TartanAirStreamingSource(torch.utils.data.Dataset):
    def __init__(self, mode, scale=256):
        super().__init__()
        self.scale = scale
        root_dir = "datas/tartanair_data/"
        # K_cam = np.array([[320.0, 0.0, 320.0],
        #                   [0.0, 320.0, 240.0],
        #                   [0.0, 0.0, 1.0]], dtype=np.float32)
        # self.K_cam = K_cam
        self.samples = []

        # 遍历所有环境文件夹
        env_dirs = [d for d in os.listdir(root_dir)
                    if osp.isdir(osp.join(root_dir, d))]

        for env in env_dirs:
            env_path = osp.join(root_dir, env)

            # 遍历所有难度文件夹 (Easy, Hard)
            difficulty_dirs = [d for d in os.listdir(env_path)
                               if osp.isdir(osp.join(env_path, d)) and d in ['Easy', 'Hard']]

            for difficulty in difficulty_dirs:
                base_path = osp.join(env_path, difficulty)

                # ========== 修改：新的目录结构 ==========
                # 构建深度图和图像的基础路径（包含 env/difficulty 子目录）
                depth_left_base = osp.join(base_path, "depth_left", env, difficulty)
                depth_right_base = osp.join(base_path, "depth_right", env, difficulty)
                image_left_base = osp.join(base_path, "image_left", env, difficulty)
                image_right_base = osp.join(base_path, "image_right", env, difficulty)
                # print(image_left_base)

                if not osp.exists(image_left_base):
                    continue

                # 获取所有轨迹 (P开头的文件夹)
                traj_dirs = [d for d in os.listdir(image_left_base)
                             if osp.isdir(osp.join(image_left_base, d)) and d.startswith('P')]

                for traj in traj_dirs:
                    # 构建各目录的完整路径
                    left_img_dir = osp.join(image_left_base, traj, "image_left")
                    right_img_dir = osp.join(image_right_base, traj, "image_right")
                    left_depth_dir = osp.join(depth_left_base, traj, "depth_left")
                    right_depth_dir = osp.join(depth_right_base, traj, "depth_right")
                    # print(left_img_dir)

                    if not osp.exists(left_img_dir):
                        continue

                    left_imgs = sorted(glob.glob(osp.join(left_img_dir, "*.png")))

                    for left_path in left_imgs:
                        basename = osp.basename(left_path)
                        match = re.search(r'(\d+)', basename)
                        frame_idx = int(match.group(1)) if match else 0

                        # 右目图像路径
                        right_filename = f"{frame_idx:06d}_right.png"
                        right_path = osp.join(right_img_dir, right_filename)

                        # 左目深度路径
                        left_depth_filename = f"{frame_idx:06d}_left_depth.npy"
                        left_depth_path = osp.join(left_depth_dir, left_depth_filename)

                        if not osp.exists(left_depth_path):
                            continue

                        # 右目深度
                        right_depth_filename = f"{frame_idx:06d}_right_depth.npy"
                        right_depth_path = osp.join(right_depth_dir, right_depth_filename)
                        if not osp.exists(right_depth_path):
                            continue

                        # 左目样本
                        if osp.exists(left_depth_path):
                            self.samples.append({
                                'image_path': left_path,
                                'depth_path': left_depth_path,
                            })

                        # 右目样本
                        if osp.exists(right_path) and osp.exists(right_depth_path):
                            self.samples.append({
                                'image_path': right_path,
                                'depth_path': right_depth_path,
                            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path = self.samples[index]['image_path']
        depth_path = self.samples[index]['depth_path']
        K_cam = np.array([[320.0, 0.0, 320.0],
                          [0.0, 320.0, 240.0],
                          [0.0, 0.0, 1.0]], dtype=np.float32)

        rgb = Image.open(image_path).convert('RGB')

        dep= read_npy(depth_path,sigma=12.0)

        dep = np.clip(dep, None, 127)
        dep = (dep * self.scale).astype(np.uint16)
        # dep = Image.fromarray(dep)
        byte_io = io.BytesIO()
        rgb.save(byte_io, format="JPEG", quality=90)
        jpeg_bytes = byte_io.getvalue()

        return {"rgb": jpeg_bytes, "dep": dep, "kcam": K_cam}


def read_cam_file_blendedmvs(filename):
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


class BlendedMVSStreamingSource(torch.utils.data.Dataset):
    def __init__(self, mode, scale=256):
        super().__init__()
        self.scale = scale
        root_dir = "datas/BlendedMVS"
        images = list(sorted(glob.glob(osp.join(root_dir, "*", "blended_images", "*.jpg"))))
        self.images = [f for f in images if osp.basename(f).split('.')[0].isdigit()]

        depths = []
        cams = []
        for img_path in self.images:
            parts = img_path.split(os.sep)
            scene = parts[-3]
            basename = osp.basename(img_path)
            frame_idx = basename.replace('.jpg', '').replace('.png', '')
            depth_path = osp.join(root_dir, scene, "rendered_depth_maps", f"{frame_idx}.pfm")
            cam_path = osp.join(root_dir, scene, "cams", f"{frame_idx}_cam.txt")
            depths.append(depth_path)
            cams.append(cam_path)
        self.cams = cams
        self.depths = depths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        depth_path = self.depths[index]
        cam_path = self.cams[index]

        Kcam = read_cam_file_blendedmvs(cam_path).astype(np.float32)

        rgb = Image.open(image_path).convert('RGB')

        dep, _ = read_pfm(depth_path)

        dep = np.clip(dep, None, 127)
        dep = (dep * self.scale).astype(np.uint16)
        # dep = Image.fromarray(dep)

        byte_io = io.BytesIO()
        rgb.save(byte_io, format="JPEG", quality=90)
        jpeg_bytes = byte_io.getvalue()

        # byte_io = io.BytesIO()
        # dep.save(byte_io, format="PNG")
        # png_bytes = byte_io.getvalue()

        return {"rgb": jpeg_bytes, "dep": dep, "kcam": Kcam}


class IRSStreamingSource(Dataset):
    def __init__(self, mode, root_dir='datas/IRS_ex', scale=256,
                 fx=480.0, fy=480.0, cx=480.0, cy=270.0, baseline=0.1):
        if mode not in ['train', 'val']:
            raise NotImplementedError(f"mode {mode} not supported, use 'train' or 'val'")

        self.mode = mode
        self.root_dir = root_dir
        self.scale = scale
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.baseline = baseline

        # 根据 mode 选择 list 文件
        if mode == 'train':
            list_file = os.path.join(root_dir, 'irs_train.list')
        else:
            list_file = os.path.join(root_dir, 'irs_test.list')

        if not os.path.exists(list_file):
            raise FileNotFoundError(f"List file not found: {list_file}")

        with open(list_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        self.samples = []  # 每个元素为 (side, left_path, right_path, disp_path)

        for line in lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            left_rel, right_rel, disp_rel = parts[0], parts[1], parts[2]
            left_abs = os.path.join(root_dir, left_rel)
            right_abs = os.path.join(root_dir, right_rel)
            disp_abs = os.path.join(root_dir, disp_rel)

            # 检查必需文件是否存在
            if not (os.path.exists(left_abs) and os.path.exists(right_abs) and os.path.exists(disp_abs)):
                # print(f"Warning: skipping line with missing files: {line}")
                continue
            print(f" line with  files: {line}")
            # 添加左目和右目样本
            self.samples.append(('left', left_rel, right_rel, disp_rel))
            self.samples.append(('right', left_rel, right_rel, disp_rel))
        print("len of samples:",len(self.samples))
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {list_file}")

        # 内参矩阵 K
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        side, left_path, right_path, disp_path = self.samples[idx]
        # 构建完整路径
        left_path = os.path.join(self.root_dir, left_path)
        right_path = os.path.join(self.root_dir, right_path)
        disp_path = os.path.join(self.root_dir, disp_path)

        # 读取视差图
        disp = load_exr(disp_path)  # shape (H, W), float32

        # 计算左深度图
        depth_left = disparity_to_depth_left(disp, self.fx, self.baseline)

        if side == 'left':
            img = load_rgb(left_path)
            depth = depth_left
        else:  # right
            img = load_rgb(right_path)
            depth = left_depth_to_right_depth(depth_left, disp)

        #mask
        depth = mask(depth, sigma=12.0)
        # 深度裁剪与缩放
        depth = np.clip(depth, 0, 127)
        depth_uint16 = (depth * self.scale).astype(np.uint16)

        # RGB 图像转为 JPEG 字节流
        img_pil = Image.fromarray(img)
        byte_io = io.BytesIO()
        img_pil.save(byte_io, format="JPEG", quality=90)
        jpeg_bytes = byte_io.getvalue()

        return {
            "rgb": jpeg_bytes,
            "dep": depth_uint16,
            "kcam": self.K
        }


# ---------- 辅助函数（仅保留深度和点云相关） ----------
def compute_point_cloud(scene_path, cam, frame, camera_params):
    """根据深度距离图与相机参数计算相机坐标系下的点云（单位：米）"""
    depth_path = os.path.join(scene_path, 'images', f'{cam}_geometry_hdf5', f'frame.{frame:04d}.depth_meters.hdf5')
    with h5py.File(depth_path, 'r') as f:
        distance_img_meters = f['dataset'][:].astype(np.float32)
    distance_img_meters = np.nan_to_num(distance_img_meters, nan=0.0)

    width_pixels = int(camera_params["settings_output_img_width"])
    height_pixels = int(camera_params["settings_output_img_height"])

    M_cam_from_uv = np.array([
        [camera_params["M_cam_from_uv_00"], camera_params["M_cam_from_uv_01"], camera_params["M_cam_from_uv_02"]],
        [camera_params["M_cam_from_uv_10"], camera_params["M_cam_from_uv_11"], camera_params["M_cam_from_uv_12"]],
        [camera_params["M_cam_from_uv_20"], camera_params["M_cam_from_uv_21"], camera_params["M_cam_from_uv_22"]]
    ])

    u_min, u_max = -1.0, 1.0
    v_min, v_max = -1.0, 1.0
    half_du = 0.5 * (u_max - u_min) / width_pixels
    half_dv = 0.5 * (v_max - v_min) / height_pixels
    u = np.linspace(u_min + half_du, u_max - half_du, width_pixels)
    v = np.linspace(v_min + half_dv, v_max - half_dv, height_pixels)[::-1]
    uu, vv = np.meshgrid(u, v)
    uvs_2d = np.dstack((uu, vv, np.ones_like(uu)))
    rays = np.dot(uvs_2d.reshape(-1, 3), M_cam_from_uv.T)
    normed_rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    points_cam = normed_rays * distance_img_meters.reshape(-1, 1)
    points_cam *= np.array([1, -1, -1])   # 方向修正
    points_cam = points_cam.reshape(height_pixels, width_pixels, 3)
    return points_cam   # 相机坐标系下的点云 (x, y, z)，单位米

def load_depth_old(scene_path, cam, frame):
    """旧版深度加载（不重投影），直接从 depth_meters.hdf5 读取深度（米）"""
    depth_path = os.path.join(scene_path, 'images', f'{cam}_geometry_hdf5', f'frame.{frame:04d}.depth_meters.hdf5')
    with h5py.File(depth_path, 'r') as f:
        depth = f['dataset'][:].astype(np.float32)
    depth = np.nan_to_num(depth, nan=0.0)
    return depth   # 单位：米
# -------------------- Hypersim 流式数据源 --------------------
class HypersimStreamingSource(Dataset):
    def __init__(self, mode, root_dir, scale=256, use_tilt_shift_conversion=True):
        """
        Args:
            mode: 占位参数，保持接口兼容
            root_dir: Hypersim 原始数据根目录（应包含 downloads 子目录）
            scale: 保留参数，不再影响深度（深度始终以米为单位）
            use_tilt_shift_conversion: True=重投影到标准针孔；False=旧方式（不重投影）
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "downloads")
        self.scale = scale
        self.use_tilt_shift_conversion = use_tilt_shift_conversion

        # 加载相机参数表（用于重投影模式）
        csv_path = os.path.join(root_dir, "metadata_camera_parameters.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"未找到相机参数文件 {csv_path}，请从 Hypersim 官方仓库下载")
        self.camera_params_df = pd.read_csv(csv_path, index_col='scene_name')

        # 收集所有有效样本
        self.samples = []
        scene_dirs = [d for d in os.listdir(self.image_dir)
                      if os.path.isdir(os.path.join(self.image_dir, d)) and d.startswith('ai_')]
        for scene in scene_dirs:
            scene_path = os.path.join(self.image_dir, scene)
            images_dir = os.path.join(scene_path, "images")
            if not os.path.isdir(images_dir):
                continue
            preview_dirs = glob.glob(os.path.join(images_dir, "*_final_preview"))
            for preview_dir in preview_dirs:
                camera_name = os.path.basename(preview_dir).replace("_final_preview", "")
                geometry_dir = os.path.join(images_dir, f"{camera_name}_geometry_hdf5")
                if not os.path.isdir(geometry_dir):
                    continue
                jpg_files = glob.glob(os.path.join(preview_dir, "frame.*.jpg"))
                for jpg_path in jpg_files:
                    frame_id_str = os.path.basename(jpg_path).split('.')[1]
                    frame_id = int(frame_id_str)
                    depth_path = os.path.join(geometry_dir, f"frame.{frame_id:04d}.depth_meters.hdf5")
                    if os.path.exists(depth_path):
                        self.samples.append((scene_path, camera_name, frame_id))
        print(f"Found {len(self.samples)} samples in {self.image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_path, camera_name, frame_id = self.samples[idx]

        # ---------- 加载 RGB（直接从官方 JPG） ----------
        jpg_path = os.path.join(scene_path, "images", f"{camera_name}_final_preview", f"frame.{frame_id:04d}.color.jpg")
        rgb_pil = Image.open(jpg_path).convert('RGB')
        rgb = np.array(rgb_pil)   # uint8, shape (H, W, 3), RGB

        if self.use_tilt_shift_conversion:
            # ---------- 重投影模式（标准针孔） ----------
            cp = self.camera_params_df.loc[os.path.basename(scene_path)]
            width = int(cp["settings_output_img_width"])
            height = int(cp["settings_output_img_height"])

            # 计算标准针孔内参（归一化 -> 像素单位）
            M = np.array([
                [cp["M_cam_from_uv_00"], cp["M_cam_from_uv_01"], cp["M_cam_from_uv_02"]],
                [cp["M_cam_from_uv_10"], cp["M_cam_from_uv_11"], cp["M_cam_from_uv_12"]],
                [cp["M_cam_from_uv_20"], cp["M_cam_from_uv_21"], cp["M_cam_from_uv_22"]]
            ])
            fx_norm = (1 / M[0, 0]) / 2
            fy_norm = (1 / M[1, 1]) / 2
            cx_norm = 0.5 - 0.5 * M[0, 2] / M[0, 0]
            cy_norm = 0.5 + 0.5 * M[1, 2] / M[1, 1]
            fx_px = fx_norm * width
            fy_px = fy_norm * height
            cx_px = cx_norm * width
            cy_px = cy_norm * height
            K = np.array([[fx_px, 0, cx_px],
                          [0, fy_px, cy_px],
                          [0, 0, 1]], dtype=np.float32)

            # 计算相机系点云
            points_cam = compute_point_cloud(scene_path, camera_name, frame_id, cp)
            # 深度 = Z 坐标（米）
            depth_meters = points_cam[:, :, 2].astype(np.float32)
            depth_meters = np.nan_to_num(depth_meters, nan=0.0)

            # 重投影到标准针孔平面
            points_flat = points_cam.reshape(-1, 3)
            valid = points_flat[:, 2] != 0
            points_flat = points_flat[valid]
            uv = points_flat[:, :2] / points_flat[:, 2:3]   # (x/z, y/z)
            uv[:, 0] = uv[:, 0] * fx_norm + cx_norm
            uv[:, 1] = uv[:, 1] * fy_norm + cy_norm
            uv[:, 0] *= (width - 0.5)
            uv[:, 1] *= (height - 0.5)
            uv = uv.astype(np.int32)
            uv[:, 0] = np.clip(uv[:, 0], 0, width-1)
            uv[:, 1] = np.clip(uv[:, 1], 0, height-1)

            unique_uv, indices = np.unique(uv, axis=0, return_index=True)
            u = unique_uv[:, 0]
            v = unique_uv[:, 1]

            rgb_mapped = np.zeros((height, width, 3), dtype=np.uint8)
            depth_mapped = np.zeros((height, width), dtype=np.float32)

            orig_indices = np.flatnonzero(valid)[indices]
            rgb_flat = rgb.reshape(-1, 3)
            depth_flat = depth_meters.reshape(-1)

            rgb_mapped[v, u] = rgb_flat[orig_indices]
            depth_mapped[v, u] = depth_flat[orig_indices]

            rgb = rgb_mapped
            depth = depth_mapped   # 单位：米

        else:
            # ---------- 旧模式（不重投影，无几何矫正） ----------
            # 内参固定（基于 1024x768 图像）
            width, height = 1024, 768
            fov_x = np.pi / 3.0
            fov_y = fov_x * (height / width)
            fx = (width / np.tan(fov_x/2)) / 2
            fy = (height / np.tan(fov_y/2)) / 2
            K = np.array([[fx, 0, width/2],
                          [0, fy, height/2],
                          [0, 0, 1]], dtype=np.float32)

            # 深度：直接读取 depth_meters.hdf5（米）
            depth = load_depth_old(scene_path, camera_name, frame_id)   # 单位：米

        # 将 RGB 转为 JPEG 字节流（兼容原有接口）
        rgb_pil = Image.fromarray(rgb)
        byte_io = io.BytesIO()
        rgb_pil.save(byte_io, format="JPEG", quality=90)
        jpeg_bytes = byte_io.getvalue()
        # 深度裁剪与缩放
        depth = np.clip(depth, 0, 127)
        depth_uint16 = (depth * self.scale).astype(np.uint16)
        return {
            "rgb": jpeg_bytes,
            "dep": depth_uint16,      # float32, 单位：米
            "kcam": K
        }



def identity_collate(batch):
    return batch


def create_ddad_streaming():
    splits = ("train", "val")
    for split in splits:
        source_ds = DDADStreamingSource(mode=split)
        # a = source_ds[0]
        dataloader = DataLoader(
            source_ds,
            batch_size=1,
            num_workers=8,
            prefetch_factor=16,
            collate_fn=identity_collate,
            shuffle=True,
        )

        with MDSWriter(out=f"datas/DDAD_streaming/{split}",
                       columns={"rgb": "bytes", "dep": "ndarray:uint16", "kcam": "ndarray:float32:3,3"},
                       size_limit='128mb') as writer:
            for batch in tqdm(dataloader):
                for sample in batch:
                    writer.write(sample)


def create_BlendedMVS_streaming():
    splits = ("train",)
    for split in splits:
        source_ds = BlendedMVSStreamingSource(mode=split)
        a = source_ds[0]
        dataloader = DataLoader(
            source_ds,
            batch_size=1,
            num_workers=8,
            prefetch_factor=16,
            collate_fn=identity_collate,
            shuffle=True,
        )

        with MDSWriter(out=f"datas/BlendedMVS_streaming/{split}",
                       columns={"rgb": "bytes", "dep": "ndarray:uint16", "kcam": "ndarray:float32:3,3"},
                       size_limit='128mb') as writer:
            for batch in tqdm(dataloader):
                for sample in batch:
                    writer.write(sample)
def create_tartanair_streaming():
    splits = ("train",)

    for split in splits:

        source_ds = TartanAirStreamingSource(
            mode=split,
        )

        print("num samples:", len(source_ds))

        dataloader = DataLoader(
            source_ds,
            batch_size=1,
            num_workers=8,
            prefetch_factor=16,
            collate_fn=identity_collate,
            shuffle=True,
        )

        with MDSWriter(
            out=f"datas/TartanAir_streaming/{split}",
            columns={
                "rgb": "bytes",
                "dep": "ndarray:uint16",
                "kcam": "ndarray:float32:3,3",
            },
            size_limit="128mb",
        ) as writer:

            for batch in tqdm(dataloader):

                for sample in batch:
                    writer.write(sample)

def create_IRS_streaming():
    """仿照 create_ddad_streaming 创建训练集和验证集的流式存储"""
    splits = ('train', 'test')
    for split in splits:
        source_ds = IRSStreamingSource(mode=split, root_dir='datas/IRS_ex', scale=256)
        dataloader = DataLoader(
            source_ds,
            batch_size=1,
            num_workers=8,
            prefetch_factor=16,
            collate_fn=identity_collate,
            shuffle=True,
        )

        output_dir = f"datas/IRS_streaming/{split}"
        with MDSWriter(
            out=output_dir,
            columns={
                "rgb": "bytes",
                "dep": "ndarray:uint16",
                "kcam": "ndarray:float32:3,3",
            },
            size_limit='128mb'
        ) as writer:
            for batch in tqdm(dataloader, desc=f"Writing IRS {split}"):
                for sample in batch:
                    writer.write(sample)


def create_hypersim_streaming():
    """
    将 Hypersim 数据集转换为 MDS 流式格式
    Args:
        root_dir: 原始数据根目录，如 '/mnt/data/Hypersim/downloads'
        output_dir: 输出 MDS 目录，如 'datas/Hypersim_streaming/train'
        split: 子集名称（仅用于标记）
        scale: 深度缩放因子
        max_frames_per_scene: 每个场景最大帧数（可选）
    """
    splits = ('train',)
    for split in splits:
        source_ds = HypersimStreamingSource(mode=split, root_dir='datas/Hypersim', scale=256, use_tilt_shift_conversion=True)

        dataloader = DataLoader(
            source_ds,
            batch_size=1,
            num_workers=8,
            prefetch_factor=16,
            collate_fn=identity_collate,
            shuffle=True,
        )

        output_dir = f"datas/Hyper_streaming/{split}"
        with MDSWriter(
            out=output_dir,
            columns={
                "rgb": "bytes",
                "dep": "ndarray:uint16",
                "kcam": "ndarray:float32:3,3",
            },
            size_limit='128mb'
        ) as writer:
            for batch in tqdm(dataloader, desc=f"Writing Hyper {split}"):
                for sample in batch:
                    writer.write(sample)

class FastStreaming(StreamingDataset):

    def __init__(self, transform=None, **kwargs):
        super().__init__(**kwargs)
        # self.transform = transform

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        raw_bytes = bytearray(sample["rgb"])
        uint8_tensor = torch.frombuffer(raw_bytes, dtype=torch.uint8)
        I = decode_jpeg(uint8_tensor, mode=ImageReadMode.RGB)
        D = torch.from_numpy(sample["dep"])[None]
        # raw_bytes = bytearray(sample["dep"])
        # uint8_tensor = torch.frombuffer(raw_bytes, dtype=torch.uint8)
        # D = decode_png(uint8_tensor, mode=ImageReadMode.UNCHANGED)
        D = D.to(torch.float32) / 256.
        # D = D.to(torch.float32)
        S = D.clone()
        K = torch.from_numpy(sample["kcam"])
        # I, S, K, D = self.transform(I, S, K, D)
        return I, S, K, D




def test():
    # import augs
    # transform = augs.Compose([
    #     augs.kaRCrop(),
    #     augs.kaRFlip(),
    #     augs.kaJitter(),
    #     augs.kaRScale(),
    #     augs.kaRSample(),
    #     augs.kaNorm(),
    # ])
    dataset = FastStreaming(
        local="datas/IRS_streaming/train",
        shuffle=False,
        transform=None,
        batch_size=1,
    )

    I, S, K, D = dataset[0]
    print(I.shape)
    print(S.shape)
    print(K.shape)
    print(D.shape)
    # I = I.permute(1, 2, 0).numpy()
    # S = S.permute(1, 2, 0).squeeze().numpy()
    # D = D.permute(1, 2, 0).squeeze().numpy()

    # transform = augs.Compose([
    #     augs.RandomResizedCrop(224),
    #     augs.RandomHorizontalFlip(),
    #     augs.ConvertRGB(),
    #     augs.ToTensor(),
    #     augs.Normalize(mean=[0.485, 0.456, 0.406],
    #                    std=[0.229, 0.224, 0.225]),
    # ])

    # def data_collate(batch):
    #     # batch is a list of {"image": ..., "label": ...}
    #     images = [transform(sample["image"]) for sample in batch]
    #     labels = [sample["label"] for sample in batch]
    #     return torch.stack(images), torch.tensor(labels)
    #
    # loader = StreamingDataLoader(
    #     dataset,
    #     batch_size=256,
    #     collate_fn=data_collate,
    #     num_workers=8,
    #     pin_memory=True,
    #     drop_last=False
    # )

    # for img, lab in tqdm(loader):
    #     print(img.shape)

def visualize_samples(num_samples=4, log_dir="runs/IRS_test2"):
    writer = SummaryWriter(log_dir)
    dataset = FastStreaming(
        local="datas/TartanAir_streaming/train",
        shuffle=False,
        transform=None,
        batch_size=1,
    )
    for idx in range(num_samples):
        I, S,K,D = dataset[idx]
        # I: (C, H, W) uint8, D: (H, W) float (深度值单位: 米 / scale? 此处是原始深度值)
        # 深度图归一化到 [0,1] 以便显示
        D=D.squeeze(0)
        # D=D.float()
        depth_min = D.min().item()
        depth_max = D.max().item()
        depth_norm = (D - depth_min) / (depth_max - depth_min + 1e-8)
        depth_norm = depth_norm.clamp(0, 1)

        # 使用 matplotlib colormap 将深度转为 RGB
        cmap = plt.cm.jet
        depth_colored = cmap(depth_norm.numpy())[:, :, :3]  # (H,W,3) 范围 [0,1]
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored = torch.from_numpy(depth_colored).permute(2,0,1)  # (3,H,W)

        # 记录到 TensorBoard
        writer.add_image(f"sample_{idx}/rgb", I, global_step=0)
        writer.add_image(f"sample_{idx}/depth_colored", depth_colored, global_step=0)


        # 打印信息
        print(f"Sample {idx}:")
        print(f"  RGB shape: {I.shape}, dtype: {I.dtype}")
        print(f"  Depth shape: {D.shape}, dtype: {D.dtype}")
        print(f"  Depth range: [{depth_min:.3f}, {depth_max:.3f}]")
        print(f"  K matrix:\n{K.numpy()}\n")

        # 可选：保存为图片文件
        # torchvision.utils.save_image(I.float()/255, f"sample_{idx}_rgb.png")
        # torchvision.utils.save_image(depth_colored.float()/255, f"sample_{idx}_depth.png")

    writer.close()
    print(f"TensorBoard logs saved to {log_dir}. Run: tensorboard --logdir {log_dir}")
if __name__ == '__main__':
    # create_ddad_streaming()
    # create_BlendedMVS_streaming()
    # create_tartanair_streaming()
    # create_IRS_streaming()
    # test()
    # visualize_samples(log_dir="runs/TartanAir_test2")
    create_hypersim_streaming()