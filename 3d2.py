import streamlit as st
import numpy as np
import plotly.graph_objects as go
from demoall import UnifiedDataset
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="DDAD 点云可视化 - 所有相机")

st.title("DDAD 数据集点云可视化 - 所有 6 个相机")


@st.cache_resource
def load_data():
    """加载数据集"""
    dataset = UnifiedDataset(
        split="train",
        blendedmvs_enable=False,
        tartan_enable=False,
        matterport_enable=False,
        ddad_enable=True,
        ddad_max_samples=20,
    )

    # 收集所有相机的样本
    samples = {
        '01': [],
        '05': [],
        '06': [],
        '07': [],
        '08': [],
        '09': []
    }

    for i in range(len(dataset)):
        image, depth, K, original_h, original_w, seq_name, dataset_name, camera_id = dataset[i]
        if camera_id in samples:
            samples[camera_id].append((image, depth, K, seq_name, original_h, original_w))

    return samples


def depth_to_points(depth, K, rgb_image=None, max_depth=100.0):
    """将深度图转换为点云"""
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    valid_mask = (z > 0) & (z < max_depth) & np.isfinite(z)

    points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=-1)

    if rgb_image is not None and valid_mask.any():
        if rgb_image.max() <= 1.0:
            colors = rgb_image[valid_mask]
        else:
            colors = rgb_image[valid_mask].astype(np.float32) / 255.0
    else:
        colors = None

    return points, colors, valid_mask


def depth_to_colored(depth_np, valid_depth):
    """将深度图转换为伪彩色图像"""
    if len(valid_depth) > 0:
        vmin, vmax = valid_depth.min(), valid_depth.max()

        if vmax - vmin > 1e-6:
            depth_normalized = (depth_np - vmin) / (vmax - vmin)
        else:
            depth_normalized = np.zeros_like(depth_np)

        colormap = plt.cm.jet
        depth_colored = colormap(depth_normalized)[:, :, :3]
        depth_colored[depth_np == 0] = [1.0, 1.0, 1.0]  # 无效深度设为白色

        return depth_colored, vmin, vmax
    else:
        return np.zeros((*depth_np.shape, 3)), 0, 0


def main():
    st.sidebar.title("控制面板")

    # 加载数据
    with st.spinner("正在加载数据集..."):
        samples = load_data()

    # 显示各相机样本数
    st.sidebar.subheader("相机样本统计")
    for cam_id, cam_samples in samples.items():
        st.sidebar.write(f"相机 {cam_id}: {len(cam_samples)} 个样本")

    # 选择相机
    camera_id = st.sidebar.selectbox("选择相机", ['01', '05', '06', '07', '08', '09'])

    # 选择样本
    cam_samples = samples[camera_id]
    if len(cam_samples) == 0:
        st.warning(f"相机 {camera_id} 没有样本")
        return

    sample_names = [s[3] for s in cam_samples]
    selected_idx = st.sidebar.selectbox("选择样本", range(len(cam_samples)),
                                        format_func=lambda i: sample_names[i])

    # 获取选中的样本
    image, depth, K, seq_name, original_h, original_w = cam_samples[selected_idx]

    # 转换为 numpy
    image_np = image.permute(1, 2, 0).cpu().numpy()
    depth_np = depth.squeeze(0).cpu().numpy()
    K_np = K.cpu().numpy()

    # 裁剪到原始尺寸（去除 padding）
    image_np_cropped = image_np[:original_h, :original_w, :]
    depth_np_cropped = depth_np[:original_h, :original_w]

    # 统计深度信息
    valid_depth = depth_np_cropped[depth_np_cropped > 0]
    depth_min = valid_depth.min() if len(valid_depth) > 0 else 0
    depth_max = valid_depth.max() if len(valid_depth) > 0 else 0

    # 显示信息
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("序列名", seq_name)
    col2.metric("原始尺寸", f"{original_h} x {original_w}")
    col3.metric("深度有效点数", f"{len(valid_depth)} / {original_h * original_w}")
    col4.metric("深度范围", f"{depth_min:.2f} - {depth_max:.2f} m")

    # 显示 RGB 图像和深度图（伪彩色）
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"RGB 图像 - 相机 {camera_id}")
        st.image(image_np_cropped, use_container_width=True)

    with col2:
        st.subheader(f"深度图 - 伪彩色 (相机 {camera_id})")
        depth_colored, vmin, vmax = depth_to_colored(depth_np_cropped, valid_depth)
        st.image(depth_colored, use_container_width=True)
        if len(valid_depth) > 0:
            st.caption(f"深度范围: {vmin:.2f}m (蓝色) → {vmax:.2f}m (红色) | 白色区域: 无效深度")

    # 点云参数
    st.sidebar.subheader("点云参数")
    max_depth = st.sidebar.slider("最大深度 (米)", 10.0, 200.0, 100.0, 10.0)
    point_size = st.sidebar.slider("点大小", 1, 5, 2)
    use_color = st.sidebar.checkbox("使用 RGB 颜色", value=True)

    # 生成点云
    with st.spinner("正在生成点云..."):
        points, colors, valid_mask = depth_to_points(
            depth_np_cropped, K_np,
            rgb_image=image_np_cropped if use_color else None,
            max_depth=max_depth
        )

    st.subheader(f"3D 点云 - 相机 {camera_id} ({len(points)} 个点)")

    # 创建并显示点云
    if len(points) > 0:
        if colors is not None:
            colors_plotly = [f'rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})' for r, g, b in colors]
        else:
            colors_plotly = 'red'

        fig = go.Figure(data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=colors_plotly if colors is not None else 'red',
                    opacity=0.8
                ),
                name='Point Cloud'
            )
        ])

        fig.update_layout(
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            height=600,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        # 统计信息
        st.write(f"点云范围: X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] m")
        st.write(f"           Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] m")
        st.write(f"           Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}] m")
    else:
        st.warning("没有有效点云数据")


if __name__ == "__main__":
    main()