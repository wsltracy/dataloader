dataset_util.py:
图像和相机参数的底层操作函数:
crop_image_depth_and_intrinsic_by_pp()	根据主点位置裁剪图像、深度图和相机内参
resize_image_depth_and_intrinsic()	缩放图像、深度图，并同步更新相机内参
rotate_90_degrees()	将图像、深度图、内外参旋转90度
depth_to_world_coords_points()	从深度图生成3D点云（相机坐标系和世界坐标系）
threshold_depth_map()	深度图阈值处理

base_dataset.py:
class BaseDataset(Dataset):
    - __init__()         # 继承配置（img_size, patch_size, aug_scale等）
    - __getitem__()      # 调用 get_data()
    - get_data()         # 抽象方法，子类必须实现
    - get_target_shape() # 计算目标尺寸（对齐ViT patch size）
    - process_one_image() # 核心方法：缩放、裁剪、旋转、生成3D点云
    - get_nearby_ids()   # 扩展帧索引（用于时序任务）

定义了数据加载的通用流程

提供了 process_one_image() 这个核心预处理方法，统一处理图像缩放、裁剪、相机参数更新、3D点云生成

子类只需要实现 get_data() 方法（负责读取原始文件），然后调用父类的 process_one_image() 处理每一帧

帧在self.all_frames中的顺序是：先按轨迹排序，再在每个轨迹内按文件名排序
训练模式下，每次随机选择一个帧
测试模式下，按顺序选择帧，如果索引超出范围则循环
DataLoader的shuffle参数会影响批次的顺序