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


class tartanairStreamingSource(torch.utils.data.Dataset):
    def __init__(self, mode, scale=256):
        super().__init__()
        self.scale = scale
        root_dir = "dataset/tartanair_data/"
        K_cam = np.array([[320.0, 0.0, 320.0],
                          [0.0, 320.0, 240.0],
                          [0.0, 0.0, 1.0]], dtype=np.float32)
        self.K_cam = K_cam
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


        rgb = Image.open(image_path).convert('RGB')

        dep, _ = read_pfm(depth_path)

        dep = np.clip(dep, None, 127)
        dep = (dep * self.scale).astype(np.uint16)
        # dep = Image.fromarray(dep)


        return {"rgb": jpeg_bytes, "dep": dep, "kcam": self.K_cam}


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


class FastStreaming(StreamingDataset):

    def __init__(self, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        raw_bytes = bytearray(sample["rgb"])
        uint8_tensor = torch.frombuffer(raw_bytes, dtype=torch.uint8)
        I = decode_jpeg(uint8_tensor, mode=ImageReadMode.RGB)
        raw_bytes = bytearray(sample["dep"])
        uint8_tensor = torch.frombuffer(raw_bytes, dtype=torch.uint8)
        D = decode_png(uint8_tensor, mode=ImageReadMode.UNCHANGED)
        D = D.to(torch.float32) / 256.
        S = D.clone()
        K = torch.from_numpy(sample["kcam"])
        I, S, K, D = self.transform(I, S, K, D)
        return I, S, K, D


def test():
    import augs
    transform = augs.Compose([
        augs.kaRCrop(),
        augs.kaRFlip(),
        augs.kaJitter(),
        augs.kaRScale(),
        augs.kaRSample(),
        augs.kaNorm(),
    ])
    dataset = FastStreaming(
        local="datas/nyu_streaming/train",
        shuffle=True,
        transform=transform,
        batch_size=256,
    )

    I, S, K, D = dataset[0]
    I = I.permute(1, 2, 0).numpy()
    S = S.permute(1, 2, 0).squeeze().numpy()
    D = D.permute(1, 2, 0).squeeze().numpy()

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


if __name__ == '__main__':
    create_ddad_streaming()
    # create_BlendedMVS_streaming()
