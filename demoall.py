# unified_loader.py

import os
import os.path as osp
import logging
import random
import glob
import re
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from dgp.datasets import SynchronizedSceneDataset
import json
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 统一的目标尺寸
TARGET_HEIGHT = 1216
TARGET_WIDTH = 1936


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


def parse_camera_intrinsics_from_conf(conf_path, view_id):
    """从Matterport3D的.conf文件解析相机内参"""
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


def pad_to_target(image_or_depth, target_h=TARGET_HEIGHT, target_w=TARGET_WIDTH):
    """将图像或深度图 padding 到目标尺寸，缺失部分用 0 填充"""
    if len(image_or_depth.shape) == 2:
        h, w = image_or_depth.shape
        padded = np.zeros((target_h, target_w), dtype=image_or_depth.dtype)
        padded[:h, :w] = image_or_depth
    else:
        h, w, c = image_or_depth.shape
        padded = np.zeros((target_h, target_w, c), dtype=image_or_depth.dtype)
        padded[:h, :w, :] = image_or_depth

    return padded


class UnifiedDataset(Dataset):
    """
    统一数据集加载器，支持 BlendedMVS, TartanAir, Matterport3D

    所有图像和深度图都会被 padding 到 1216x1936，缺失部分用 0 填充
    内参矩阵保持不变（因为图像内容没有缩放）

    每个样本输出: (image, depth, K, seq_name, dataset_name, camera_id, original_h, original_w)
    """

    def __init__(
            self,
            split: str = "test",
            # BlendedMVS 配置
            blendedmvs_root: str = "./datas/BlendedMVS",
            blendedmvs_enable: bool = True,
            blendedmvs_max_samples: int = None,
            # TartanAir 配置
            tartan_root: str = "./datas/tartanair_data",
            tartan_enable: bool = True,
            tartan_max_samples: int = None,
            # Matterport3D 配置
            matterport_root: str = "./datas/matterport/data/v1/scans",
            matterport_enable: bool = True,
            matterport_max_samples: int = None,
            # DDAD 配置
            ddad_json_path: str = "./datas/DDAD/ddad_train_val/ddad.json",
            ddad_enable: bool = True,
            ddad_max_samples: int = None,
    ):
        super().__init__()

        self.split = split
        # self.depth_max = depth_max
        self.training = (split == "train")

        # 存储所有样本
        self.samples = []

        # 加载各个数据集
        if blendedmvs_enable:
            self._load_blendedmvs(blendedmvs_root, blendedmvs_max_samples)

        if tartan_enable:
            self._load_tartanair(tartan_root,  tartan_max_samples)

        if matterport_enable:
            self._load_matterport(matterport_root, matterport_max_samples)

        if ddad_enable:
            self._load_ddad(ddad_json_path, ddad_max_samples)
        # 训练模式下随机打乱
        if self.training:
            random.shuffle(self.samples)

        logging.info(f"UnifiedDataset ({split}): Total samples = {len(self.samples)}")
        logging.info(f"Target size: {TARGET_HEIGHT} x {TARGET_WIDTH} (padding with zeros)")

    def _load_blendedmvs(self, root_dir, max_samples):
        """加载 BlendedMVS 数据集"""
        logging.info(f"加载 BlendedMVS 数据集: {root_dir}")

        if not osp.exists(root_dir):
            logging.warning(f"BlendedMVS 路径不存在: {root_dir}")
            return

        images = sorted(glob.glob(osp.join(root_dir, "*", "blended_images", "*.jpg")))
        images = [f for f in images if osp.basename(f).split('.')[0].isdigit()]

        count = 0
        for img_path in images:
            if max_samples and count >= max_samples:
                break

            parts = img_path.split(os.sep)
            scene = parts[-3]
            basename = osp.basename(img_path)
            frame_idx = basename.replace('.jpg', '').replace('.png', '')

            depth_path = osp.join(root_dir, scene, "rendered_depth_maps", f"{frame_idx}.pfm")
            cam_path = osp.join(root_dir, scene, "cams", f"{frame_idx}_cam.txt")

            if osp.exists(depth_path) and osp.exists(cam_path):
                # 直接读取 K 矩阵
                K = read_cam_file_blendedmvs(cam_path)

                self.samples.append({
                    'dataset': 'BlendedMVS',
                    'seq_name': f"{scene}_{frame_idx}",
                    'image_path': img_path,
                    'depth_path': depth_path,
                    'K': K,
                    'camera_id': '0',
                })
                count += 1

        logging.info(f"  BlendedMVS 加载完成: {count} 个样本")

    def _load_tartanair(self, root_dir,  max_samples):
        """加载 TartanAir 数据集（左右目作为独立样本）"""
        logging.info(f"加载 TartanAir 数据集: {root_dir}")

        if not osp.exists(root_dir):
            logging.warning(f"TartanAir 路径不存在: {root_dir}")
            return

        # traj_dirs = [d for d in os.listdir(base_path) if d.startswith('P')]

        # 固定内参
        K = np.array([[320.0, 0.0, 320.0],
                      [0.0, 320.0, 240.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

        count = 0

        # 遍历所有环境文件夹
        env_dirs = [d for d in os.listdir(root_dir)
                    if osp.isdir(osp.join(root_dir, d))]

        for env in env_dirs:
            env_path = osp.join(root_dir, env)

            # 遍历所有难度文件夹 (Easy, Hard)
            difficulty_dirs = [d for d in os.listdir(env_path)
                               if osp.isdir(osp.join(env_path, d)) and d in [ 'Easy','Hard']]

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
                    if max_samples and count >= max_samples:
                        break
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
                        if max_samples and count >= max_samples:
                            break

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
                                'dataset': 'TartanAir',
                                'seq_name': f"{env}_{difficulty}_{traj}_{frame_idx}_left",
                                'image_path': left_path,
                                'depth_path': left_depth_path,
                                'K': K.copy(),
                                'camera_id': 'left',
                            })
                            count += 1
                        if max_samples and count >= max_samples:
                            break
                        # 右目样本
                        if osp.exists(right_path) and osp.exists(right_depth_path):
                            self.samples.append({
                                'dataset': 'TartanAir',
                                'seq_name': f"{env}_{difficulty}_{traj}_{frame_idx}_right",
                                'image_path': right_path,
                                'depth_path': right_depth_path,
                                'K': K.copy(),
                                'camera_id': 'right',
                            })
                            count += 1

                    if max_samples and count >= max_samples:
                        break

                if max_samples and count >= max_samples:
                    break

            if max_samples and count >= max_samples:
                break

        logging.info(f"  TartanAir 加载完成: {count} 个样本")

    def _load_matterport(self, root_dir, max_samples):
        """加载 Matterport3D 数据集（三个相机作为独立样本）"""
        logging.info(f"加载 Matterport3D 数据集: {root_dir}")

        if not osp.exists(root_dir):
            logging.warning(f"Matterport3D 路径不存在: {root_dir}")
            return

        scene_ids = [d for d in os.listdir(root_dir)
                     if osp.isdir(osp.join(root_dir, d))]

        count = 0
        for scene_id in scene_ids:
            if max_samples and count >= max_samples:
                break

            scene_path = osp.join(root_dir, scene_id)

            img_base = osp.join(scene_path, "undistorted_color_images", scene_id, "undistorted_color_images")
            depth_base = osp.join(scene_path, "undistorted_depth_images", scene_id, "undistorted_depth_images")
            cam_base = osp.join(scene_path, "undistorted_camera_parameters", scene_id, "undistorted_camera_parameters")

            if not osp.isdir(img_base):
                img_base = osp.join(scene_path, "undistorted_color_images")
                depth_base = osp.join(scene_path, "undistorted_depth_images")
                cam_base = osp.join(scene_path, "undistorted_camera_parameters")

            if not osp.isdir(img_base):
                continue

            conf_files = glob.glob(osp.join(cam_base, "*.conf"))
            conf_path = conf_files[0] if conf_files else None

            if conf_path is None:
                continue

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

            for key, data in grouped.items():
                # if max_samples and count >= max_samples:
                #     break

                for view in [0, 1, 2]:
                    if view not in data['views']:
                        continue
                    if max_samples and count >= max_samples:
                        break
                    view_data = data['views'][view]

                    depth_filename = f"{data['uuid']}_d{view}_{data['index']}.png"
                    depth_path = osp.join(depth_base, depth_filename)

                    if not osp.exists(depth_path):
                        continue

                    K = parse_camera_intrinsics_from_conf(conf_path, view)

                    if K is None:
                        continue

                    self.samples.append({
                        'dataset': 'Matterport3D',
                        'seq_name': f"{scene_id}_{data['uuid']}_{data['index']}_cam{view}",
                        'image_path': view_data['image_path'],
                        'depth_path': depth_path,
                        'K': K,
                        'camera_id': str(view),
                    })
                    count += 1
                    # print(count)
        logging.info(f"  Matterport3D 加载完成: {count} 个样本")

    def _load_ddad(self, json_path, max_samples):
        """加载 DDAD 数据集（6个相机作为独立单目样本）"""
        logging.info(f"加载 DDAD 数据集: {json_path}")

        if not osp.exists(json_path):
            logging.warning(f"DDAD JSON 路径不存在: {json_path}")
            return

        # 相机ID列表
        camera_ids = ['01', '05', '06', '07', '08', '09']

        try:
            # 创建 DDAD 数据集
            self.ddad_dataset = SynchronizedSceneDataset(
                json_path,
                datum_names=('lidar','CAMERA_01', 'CAMERA_05','CAMERA_06','CAMERA_07','CAMERA_08','CAMERA_09'),
                generate_depth_from_datum='lidar',
                split='train' if self.training else 'val'
            )

            count = 0
            total_frames = len(self.ddad_dataset)
            print("total:",total_frames)
            for idx in range(total_frames):
                if max_samples and count >= max_samples:
                    break

                try:
                    # sample = dataset[idx]

                    # 遍历每个相机
                    for cam_idx, cam_id in enumerate(camera_ids):
                        if max_samples and count >= max_samples:
                            break

                        # cam_data = sample[0][cam_idx]
                        #
                        # # 获取图像
                        # img_np = np.array(cam_data['rgb']).astype(np.float32) / 255.0
                        #
                        # # 获取深度图（从 lidar 投影生成）
                        # depth_pil = cam_data['depth']
                        # depth_np = np.array(depth_pil, dtype=np.float32)
                        #
                        # # 获取内参
                        # K = cam_data['intrinsics']

                        # 存储样本
                        self.samples.append({
                            'dataset': 'DDAD',
                            'seq_name': f"DDAD_frame{idx}_cam{cam_id}",
                            'ddad_idx': idx,
                            'ddad_cam_idx': cam_idx,
                            'camera_id': cam_id,
                        })
                        count += 1

                except Exception as e:
                    logging.warning(f"加载 DDAD 样本 {idx} 失败: {e}")
                    continue

            logging.info(f"  DDAD 加载完成: {count} 个样本")

        except Exception as e:
            logging.error(f"DDAD 数据集加载失败: {e}")

    def _load_image(self, path):
        """加载 RGB 图像并 padding 到目标尺寸"""
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"无法读取图像: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # original_h, original_w = img.shape[:2]
        # img = pad_to_target(img, TARGET_HEIGHT, TARGET_WIDTH)

        return img

    def _load_depth(self, depth_path):
        """加载深度图并 padding 到目标尺寸"""
        if depth_path.endswith('.pfm'):
            #blendedmvs
            depth, _ = read_pfm(depth_path)
        elif depth_path.endswith('.npy'):

            depth = np.load(depth_path)
            # tartan
            valid_depths = depth[depth > 0]
            sigma=12.0
            if valid_depths.size > 0:
                global_median = np.median(valid_depths)
                lower_bound = global_median / sigma
                upper_bound = global_median * sigma
                # 将超出范围的和无效的深度设为 0
                depth[(depth <= 0) | (depth < lower_bound) | (depth > upper_bound)] = 0
        elif depth_path.endswith('.png'):
            #matterport
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is not None:
                depth = depth.astype(np.float32) / 4000.0
        else:
            raise ValueError(f"不支持的深度图格式: {depth_path}")

        depth = depth.astype(np.float32)
        # depth[depth > self.depth_max] = 0
        depth[depth < 0] = 0

        depth = pad_to_target(depth, TARGET_HEIGHT, TARGET_WIDTH)

        return depth

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # # 加载图像和深度
        # image, original_h, original_w = self._load_image(sample['image_path'])
        # depth = self._load_depth(sample['depth_path'])
        # K = sample['K'].copy()  # 直接使用存储的 K
        #
        # # 转换为 tensor
        # image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        # depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)
        # K_tensor = torch.from_numpy(K).float()
        if 'image_path' in sample:
            # BlendedMVS, TartanAir, Matterport3D
            image = self._load_image(sample['image_path'])
            depth = self._load_depth(sample['depth_path'])
            K = sample['K'].copy()
        else:
            # DDAD：已经加载好的 numpy 数组
            ddad_sample= self.ddad_dataset[sample['ddad_idx']]
            cam_data=ddad_sample[0][sample['ddad_cam_idx']]
            # 提取数据
            image = np.array(cam_data['rgb']).astype(np.float32)
            depth = np.array(cam_data['depth'], dtype=np.float32)
            K = cam_data['intrinsics']



            # 确保图像格式正确 (H, W, C)
            if image.ndim == 3 and image.shape[2] == 3:
                pass  # 已经是 HWC 格式
            elif image.ndim == 3 and image.shape[0] == 3:
                image = image.transpose(1, 2, 0)  # CHW -> HWC

            # 确保深度是 2D
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth.squeeze(0)

        original_h, original_w = image.shape[:2]
        # 统一 padding
        image = pad_to_target(image, TARGET_HEIGHT, TARGET_WIDTH)
        depth = pad_to_target(depth, TARGET_HEIGHT, TARGET_WIDTH)

        # 转换为 tensor
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)/255.0
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)
        K_tensor = torch.from_numpy(K).float()
        return (image_tensor, depth_tensor, K_tensor,
                original_h, original_w,
                sample['seq_name'], sample['dataset'], sample['camera_id'],)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 80)
    print(f"统一数据集加载器测试 (目标尺寸: {TARGET_HEIGHT} x {TARGET_WIDTH}, padding with zeros)")
    print("=" * 80)

    dataset = UnifiedDataset(
        split="test",
        blendedmvs_enable=False,
        blendedmvs_max_samples=10,
        tartan_enable=False,
        tartan_max_samples=10,
        matterport_enable=False,
        matterport_max_samples=10,
        ddad_enable=True,
        ddad_max_samples=10,
    )

    print(f"\n总样本数: {len(dataset)}")

    # 统计
    dataset_counts = {}
    for sample in dataset.samples:
        ds_name = sample['dataset']
        dataset_counts[ds_name] = dataset_counts.get(ds_name, 0) + 1

    print("\n各数据集样本统计:")
    for ds_name, count in dataset_counts.items():
        print(f"  {ds_name}: {count} 个样本")

    # 测试加载
    print("\n" + "=" * 80)
    print("测试样本加载")
    print("=" * 80)

    for i in range(min(5, len(dataset))):
        image, depth, K, original_h, original_w,seq_name, dataset_name, camera_id = dataset[i]

        print(f"\n样本 {i}:")
        print(f"  数据集: {dataset_name}")
        print(f"  相机ID: {camera_id}")
        print(f"  序列名: {seq_name}")
        print(f"  原始尺寸: {original_h} x {original_w}")
        print(f"  Padding后: {TARGET_HEIGHT} x {TARGET_WIDTH}")
        print(f"  image shape: {image.shape}")
        print(f"  depth shape: {depth.shape}")
        print(f"  K:\n{K}")
        print(f"  ✓ 验证通过")

    # 测试 DataLoader
    print("\n" + "=" * 80)
    print("测试 DataLoader (batch_size=2)")
    print("=" * 80)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 2:
            break

        images, depths, Ks,original_hs, original_ws, seq_names, dataset_names, camera_ids= batch

        print(f"\nBatch {batch_idx}:")
        print(f"  images shape: {images.shape}")
        print(f"  depths shape: {depths.shape}")
        print(f"  Ks shape: {Ks.shape}")
        print(f"  seq_names: {seq_names}")
        print(f"  dataset_names: {dataset_names}")
        print(f"  camera_ids: {camera_ids}")
        print(f"  ✓ Batch 验证通过")

    print("\n测试完成!")