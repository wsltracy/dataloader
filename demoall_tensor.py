# visualize_tensorboard.py

import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from demoall import UnifiedDataset, TARGET_HEIGHT, TARGET_WIDTH

# 设置日志
logging.basicConfig(level=logging.INFO)

# TensorBoard 日志目录
LOG_DIR = "runs/unified"


def add_sample_to_tensorboard(writer, sample_data, sample_idx, global_step=0):
    """
    将单个样本添加到 TensorBoard

    Args:
        writer: SummaryWriter 对象
        sample_data: 从 dataset 返回的元组 (image, depth, K, seq_name, dataset_name, camera_id, original_h, original_w)
        sample_idx: 样本索引
        global_step: 全局步数
    """
    image, depth, K,  original_h, original_w,seq_name, dataset_name, camera_id= sample_data

    # 转换为 numpy 用于可视化
    image_np = image.permute(1, 2, 0).cpu().numpy()
    depth_np = depth.squeeze(0).cpu().numpy()
    mask = (image_np.sum(axis=2) == 0)
    image_np[mask] = [1.0, 1.0, 1.0]  # 直接赋值
    # 深度图伪彩色
    vmin, vmax = depth_np[depth_np > 0].min() if (depth_np > 0).any() else (0, 1), \
        depth_np[depth_np > 0].max() if (depth_np > 0).any() else (0, 1)
    if vmax - vmin < 1e-6:
        depth_normalized = np.zeros_like(depth_np)
    else:
        depth_normalized = (depth_np - vmin) / (vmax - vmin)

    colormap = plt.cm.jet
    depth_colored = colormap(depth_normalized)[:, :, :3]
    # 无效深度设为白色
    depth_colored[depth_np == 0] = [1.0, 1.0, 1.0]

    # ========== 创建带黑色边框的图像 ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. RGB 图像（带黑色边框）
    axes[0].imshow(image_np)
    axes[0].set_title(f"RGB - {dataset_name}\n{camera_id}", fontsize=10)
    axes[0].axis('off')
    # 添加黑色边框
    rect = patches.Rectangle(
        (0, 0), image_np.shape[1], image_np.shape[0],
        linewidth=3, edgecolor='black', facecolor='none'
    )
    axes[0].add_patch(rect)

    # 2. 深度图（带黑色边框）
    axes[1].imshow(depth_colored)
    axes[1].set_title(f"Depth - {dataset_name}\nRange: [{vmin:.2f}, {vmax:.2f}]m", fontsize=10)
    axes[1].axis('off')
    rect = patches.Rectangle(
        (0, 0), depth_colored.shape[1], depth_colored.shape[0],
        linewidth=3, edgecolor='black', facecolor='none'
    )
    axes[1].add_patch(rect)

    # 3. 内参信息
    axes[2].axis('off')
    info_text = (
        f"Dataset: {dataset_name}\n"
        f"Sequence: {seq_name}\n"
        f"Camera: {camera_id}\n"
        f"Original Size: {original_h} x {original_w}\n"
        f"Padded Size: {TARGET_HEIGHT} x {TARGET_WIDTH}\n\n"
        f"Camera Intrinsics K:\n"
        f"fx = {K[0, 0]:.2f}\n"
        f"fy = {K[1, 1]:.2f}\n"
        f"cx = {K[0, 2]:.2f}\n"
        f"cy = {K[1, 2]:.2f}\n\n"
    )
    axes[2].text(0.1, 0.5, info_text, transform=axes[2].transAxes,
                 fontsize=10, verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[2].set_title("Camera Info", fontsize=10)

    plt.suptitle(f"Sample {sample_idx}: {seq_name}", fontsize=12)
    plt.tight_layout()

    # 添加到 TensorBoard（使用滑块功能）
    writer.add_figure(f'Samples/', fig, global_step=global_step)
    plt.close(fig)

    # 单独记录内参文本
    writer.add_text(f'Sample_{sample_idx}/Info',
                    f"Dataset: {dataset_name}\nSequence: {seq_name}\nCamera: {camera_id}\n"
                    f"Original Size: {original_h}x{original_w}\n"
                    f"K: fx={K[0, 0]:.2f}, fy={K[1, 1]:.2f}, cx={K[0, 2]:.2f}, cy={K[1, 2]:.2f}",
                    global_step=global_step)


def main():
    """主函数：加载数据集并可视化到 TensorBoard"""
    print("\n" + "=" * 80)
    print("TensorBoard 可视化 - 统一数据集加载器")
    print("=" * 80)

    # 创建数据集（加载所有样本，用于可视化）
    print("\n正在加载数据集...")
    dataset = UnifiedDataset(
        split="train",
        blendedmvs_enable=True,
        blendedmvs_max_samples=10,  # 限制样本数量，避免过多
        tartan_enable=True,
        tartan_max_samples=10,
        matterport_enable=True,
        matterport_max_samples=10,
        ddad_enable=True,
        ddad_max_samples=10,
    )

    print(f"\n总样本数: {len(dataset)}")

    # 统计各数据集样本数
    dataset_counts = {}
    for sample in dataset.samples:
        ds_name = sample['dataset']
        dataset_counts[ds_name] = dataset_counts.get(ds_name, 0) + 1

    print("\n各数据集样本统计:")
    for ds_name, count in dataset_counts.items():
        print(f"  {ds_name}: {count} 个样本")

    # 创建 TensorBoard writer
    writer = SummaryWriter(log_dir=LOG_DIR)
    print(f"\nTensorBoard 日志目录: {LOG_DIR}")

    # 记录数据集总体信息
    writer.add_text("Dataset/Info",
                    f"Total samples: {len(dataset)}\n"
                    f"Target size: {TARGET_HEIGHT} x {TARGET_WIDTH}\n"
                    f"Datasets: {list(dataset_counts.keys())}\n"
                    f"Sample counts: {dataset_counts}",
                    global_step=0)

    # 逐个添加样本到 TensorBoard
    # print("\n正在添加样本到 TensorBoard...")
    for idx in range(len(dataset)):
        sample = dataset[idx]
        add_sample_to_tensorboard(writer, sample, idx, global_step=idx)

        if (idx + 1) % 10 == 0:
            print(f"  已处理 {idx + 1}/{len(dataset)} 个样本")

    writer.close()

    print("\n" + "=" * 80)
    print(f"完成！运行以下命令启动 TensorBoard:")
    print(f"  tensorboard --logdir={LOG_DIR} --samples_per_plugin=images=100")
    print("=" * 80)


if __name__ == "__main__":
    main()