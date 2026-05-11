import os
import glob
import re
import numpy as np
import cv2
from PIL import Image
from streaming import MDSWriter
from collections import defaultdict
import logging
from tqdm import tqdm
from dgp.datasets import SynchronizedSceneDataset

logging.basicConfig(level=logging.INFO)

# 统一的目标尺寸（和原来一样）
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
    """读取 BlendedMVS 相机参数文件"""
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
    raise ValueError(f"无法找到 'intrinsic': {filename}")


def parse_camera_intrinsics_from_conf(conf_path, view_id):
    """从Matterport3D的.conf文件解析相机内参"""
    if not os.path.exists(conf_path):
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

    if view_id < len(intrinsics_list):
        return intrinsics_list[view_id]
    return None


def pack_blendedmvs(root_dir, output_dir, max_samples=None):
    """打包 BlendedMVS 数据集"""
    logging.info(f"打包 BlendedMVS: {root_dir}")

    if not os.path.exists(root_dir):
        logging.warning(f"路径不存在: {root_dir}")
        return 0

    columns = {
        'image': 'jpeg',
        'depth': 'png',
        'K': 'json',
        'original_h': 'int',
        'original_w': 'int',
        'camera_id': 'str'
    }

    images = sorted(glob.glob(os.path.join(root_dir, "*", "blended_images", "*.jpg")))
    images = [f for f in images if os.path.basename(f).split('.')[0].isdigit()]

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    with MDSWriter(out=output_dir, columns=columns, compression='zstd', size_limit=1 << 28) as writer:
        for img_path in tqdm(images, desc="BlendedMVS"):
            if max_samples and count >= max_samples:
                break

            parts = img_path.split(os.sep)
            scene = parts[-3]
            basename = os.path.basename(img_path)
            frame_idx = basename.replace('.jpg', '').replace('.png', '')

            depth_path = os.path.join(root_dir, scene, "rendered_depth_maps", f"{frame_idx}.pfm")
            cam_path = os.path.join(root_dir, scene, "cams", f"{frame_idx}_cam.txt")

            if not (os.path.exists(depth_path) and os.path.exists(cam_path)):
                continue

            # 读取数据
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_h, original_w = image.shape[:2]

            depth, _ = read_pfm(depth_path)
            depth = depth.astype(np.float32)

            K = read_cam_file_blendedmvs(cam_path)

            writer.write({
                'image': Image.fromarray(image),
                'depth': Image.fromarray((depth * 256).astype(np.uint16)) if depth.max() > 0 else Image.fromarray(
                    np.zeros((original_h, original_w), dtype=np.uint16)),
                'K': K.tolist(),
                'original_h': original_h,
                'original_w': original_w,
                'camera_id': '0'
            })
            count += 1

    logging.info(f"  BlendedMVS 打包完成: {count} 个样本")
    return count


def pack_tartanair(root_dir, output_dir, max_samples=None):
    """打包 TartanAir 数据集"""
    logging.info(f"打包 TartanAir: {root_dir}")

    if not os.path.exists(root_dir):
        logging.warning(f"路径不存在: {root_dir}")
        return 0

    columns = {
        'image': 'jpeg',
        'depth': 'png',
        'K': 'json',
        'original_h': 'int',
        'original_w': 'int',
        'camera_id': 'str'
    }

    K = np.array([[320.0, 0.0, 320.0],
                  [0.0, 320.0, 240.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    with MDSWriter(out=output_dir, columns=columns, compression='zstd', size_limit=1 << 28) as writer:
        env_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        for env in tqdm(env_dirs, desc="TartanAir"):
            env_path = os.path.join(root_dir, env)
            difficulty_dirs = [d for d in os.listdir(env_path) if
                               os.path.isdir(os.path.join(env_path, d)) and d in ['Easy', 'Hard']]

            for difficulty in difficulty_dirs:
                base_path = os.path.join(env_path, difficulty)

                depth_left_base = os.path.join(base_path, "depth_left", env, difficulty)
                depth_right_base = os.path.join(base_path, "depth_right", env, difficulty)
                image_left_base = os.path.join(base_path, "image_left", env, difficulty)
                image_right_base = os.path.join(base_path, "image_right", env, difficulty)

                if not os.path.exists(image_left_base):
                    continue

                traj_dirs = [d for d in os.listdir(image_left_base) if
                             os.path.isdir(os.path.join(image_left_base, d)) and d.startswith('P')]

                for traj in traj_dirs:
                    if max_samples and count >= max_samples:
                        break

                    left_img_dir = os.path.join(image_left_base, traj, "image_left")
                    right_img_dir = os.path.join(image_right_base, traj, "image_right")
                    left_depth_dir = os.path.join(depth_left_base, traj, "depth_left")
                    right_depth_dir = os.path.join(depth_right_base, traj, "depth_right")

                    if not os.path.exists(left_img_dir):
                        continue

                    left_imgs = sorted(glob.glob(os.path.join(left_img_dir, "*.png")))

                    for left_path in left_imgs:
                        print(count)
                        if max_samples and count >= max_samples:
                            break

                        basename = os.path.basename(left_path)
                        match = re.search(r'(\d+)', basename)
                        frame_idx = int(match.group(1)) if match else 0

                        right_filename = f"{frame_idx:06d}_right.png"
                        right_path = os.path.join(right_img_dir, right_filename)
                        left_depth_filename = f"{frame_idx:06d}_left_depth.npy"
                        left_depth_path = os.path.join(left_depth_dir, left_depth_filename)
                        right_depth_filename = f"{frame_idx:06d}_right_depth.npy"
                        right_depth_path = os.path.join(right_depth_dir, right_depth_filename)

                        # 左目
                        if os.path.exists(left_depth_path):
                            image = cv2.imread(left_path)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            original_h, original_w = image.shape[:2]

                            depth = np.load(left_depth_path)
                            depth = depth.astype(np.float32)
                            depth[depth < 0] = 0
                            valid_depths = depth[depth > 0]
                            sigma = 12.0
                            if valid_depths.size > 0:
                                global_median = np.median(valid_depths)
                                lower_bound = global_median / sigma
                                upper_bound = global_median * sigma
                                # 将超出范围的和无效的深度设为 0
                                depth[(depth <= 0) | (depth < lower_bound) | (depth > upper_bound)] = 0
                            writer.write({
                                'image': Image.fromarray(image),
                                'depth': Image.fromarray((depth * 256).astype(np.uint16)),
                                'K': K.tolist(),
                                'original_h': original_h,
                                'original_w': original_w,
                                'camera_id': '0'
                            })
                            count += 1

                        if max_samples and count >= max_samples:
                            break

                        # 右目
                        if os.path.exists(right_path) and os.path.exists(right_depth_path):
                            image = cv2.imread(right_path)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            original_h, original_w = image.shape[:2]

                            depth = np.load(right_depth_path)
                            depth = depth.astype(np.float32)
                            depth[depth < 0] = 0
                            valid_depths = depth[depth > 0]
                            sigma = 12.0
                            if valid_depths.size > 0:
                                global_median = np.median(valid_depths)
                                lower_bound = global_median / sigma
                                upper_bound = global_median * sigma
                                # 将超出范围的和无效的深度设为 0
                                depth[(depth <= 0) | (depth < lower_bound) | (depth > upper_bound)] = 0
                            writer.write({
                                'image': Image.fromarray(image),
                                'depth': Image.fromarray((depth * 256).astype(np.uint16)),
                                'K': K.tolist(),
                                'original_h': original_h,
                                'original_w': original_w,
                                'camera_id': '1'
                            })
                            count += 1

    logging.info(f"  TartanAir 打包完成: {count} 个样本")
    return count


def pack_matterport(root_dir, output_dir, max_samples=None):
    """打包 Matterport3D 数据集"""
    logging.info(f"打包 Matterport3D: {root_dir}")

    if not os.path.exists(root_dir):
        logging.warning(f"路径不存在: {root_dir}")
        return 0

    columns = {
        'image': 'jpeg',
        'depth': 'png',
        'K': 'json',
        'original_h': 'int',
        'original_w': 'int',
        'camera_id': 'str'
    }

    os.makedirs(output_dir, exist_ok=True)

    scene_ids = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    count = 0
    with MDSWriter(out=output_dir, columns=columns, compression='zstd', size_limit=1 << 28) as writer:
        for scene_id in tqdm(scene_ids, desc="Matterport"):
            if max_samples and count >= max_samples:
                break

            scene_path = os.path.join(root_dir, scene_id)

            img_base = os.path.join(scene_path, "undistorted_color_images", scene_id, "undistorted_color_images")
            depth_base = os.path.join(scene_path, "undistorted_depth_images", scene_id, "undistorted_depth_images")
            cam_base = os.path.join(scene_path, "undistorted_camera_parameters", scene_id,
                                    "undistorted_camera_parameters")


            if not os.path.isdir(img_base):
                continue

            conf_files = glob.glob(os.path.join(cam_base, "*.conf"))
            conf_path = conf_files[0] if conf_files else None

            if conf_path is None:
                continue

            grouped = defaultdict(lambda: {'views': {}})

            for img_path in glob.glob(os.path.join(img_base, "*.jpg")):

                basename = os.path.basename(img_path)
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
                for view in [0, 1, 2]:
                    print(count)
                    if max_samples and count >= max_samples:
                        break
                    if view not in data['views']:
                        continue

                    view_data = data['views'][view]
                    depth_filename = f"{data['uuid']}_d{view}_{data['index']}.png"
                    depth_path = os.path.join(depth_base, depth_filename)

                    if not os.path.exists(depth_path):
                        continue

                    K = parse_camera_intrinsics_from_conf(conf_path, view)
                    if K is None:
                        continue

                    image = cv2.imread(view_data['image_path'])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    original_h, original_w = image.shape[:2]

                    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                    depth = depth.astype(np.float32) / 4000.0

                    writer.write({
                        'image': Image.fromarray(image),
                        'depth': Image.fromarray((depth * 256).astype(np.uint16)),
                        'K': K.tolist(),
                        'original_h': original_h,
                        'original_w': original_w,
                        'camera_id': str(view)
                    })
                    count += 1

    logging.info(f"  Matterport3D 打包完成: {count} 个样本")
    return count


def pack_ddad(json_path, output_dir, split,max_samples=None):
    """打包 DDAD 数据集（先投影深度，再打包成 MDS）"""
    logging.info(f"打包 DDAD: {json_path}")

    if not os.path.exists(json_path):
        logging.warning(f"路径不存在: {json_path}")
        return 0

    columns = {
        'image': 'jpeg',
        'depth': 'png',
        'K': 'json',
        'original_h': 'int',
        'original_w': 'int',
        'camera_id': 'str',
    }

    os.makedirs(output_dir, exist_ok=True)

    # 相机ID列表
    camera_ids = ['01', '05', '06', '07', '08', '09']

    # 创建 DDAD 数据集（生成深度投影）
    ddad_dataset = SynchronizedSceneDataset(
        json_path,
        datum_names=('lidar', 'CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09'),
        generate_depth_from_datum='lidar',
        split=split  # 或 'val'
    )

    count = 0
    total_frames = len(ddad_dataset)
    logging.info(f"DDAD 总帧数: {total_frames}")

    with MDSWriter(out=output_dir, columns=columns, compression='zstd', size_limit=1 << 28) as writer:
        for idx in tqdm(range(total_frames), desc="DDAD"):
            if max_samples and count >= max_samples:
                break

            try:
                sample = ddad_dataset[idx]

                for cam_idx, cam_id in enumerate(camera_ids):
                    print(count)
                    if max_samples and count >= max_samples:
                        break

                    cam_data = sample[0][cam_idx]

                    # 获取 RGB 图像 (H, W, 3)
                    image = np.array(cam_data['rgb'])
                    original_h, original_w = image.shape[:2]

                    # 获取投影后的深度图 (H, W)
                    depth = np.array(cam_data['depth'], dtype=np.float32)

                    # 获取内参矩阵 K
                    K = cam_data['intrinsics']

                    # 清理深度图无效值
                    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                    depth[depth < 0] = 0


                    # uint16
                    depth = (depth * 256).astype(np.uint16)

                    # 写入 MDS
                    writer.write({
                        'image': Image.fromarray(image),
                        'depth': Image.fromarray(depth),
                        'K': K.tolist(),
                        'original_h': original_h,
                        'original_w': original_w,
                        'camera_id': cam_id,

                    })
                    count += 1

            except Exception as e:
                logging.warning(f"处理 DDAD 样本 {idx} 失败: {e}")
                continue

    logging.info(f"  DDAD 打包完成: {count} 个样本")
    return count
def main():
    """主函数：打包所有数据集"""

    # 配置输出目录
    base_output_dir = "./datas/mds_datasets"

    # 打包各数据集（根据需要开启/关闭）
    # pack_blendedmvs("./datas/BlendedMVS2",
    #                 os.path.join(base_output_dir, "BlendedMVS"),
    #                 max_samples=None)

    # pack_tartanair("./datas/tartanair_data",
    #                os.path.join(base_output_dir, "Tartanair"),
    #                max_samples=None)

    # pack_matterport("./datas/matterport/data/v1/scans",
    #                 os.path.join(base_output_dir, "Matterport"),
    #                 max_samples=None)

    pack_ddad("./datas/DDAD/ddad_train_val/ddad_2.json",os.path.join(base_output_dir, "DDAD_train"),"train",max_samples=None)
    logging.info(f"\n所有数据集打包完成！输出目录: {base_output_dir}")


if __name__ == "__main__":
    main()