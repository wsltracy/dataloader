1.利用self.all_frames构建所有帧中的顺序是：先按轨迹排序，再在每个轨迹内按文件名排序
训练模式下，每次随机选择一个帧
测试模式下，按顺序选择帧，如果索引超出范围则循环
DataLoader的shuffle参数会影响批次的顺序。

2.blendedmvs：相机内参在每个场景里不同。
3.blendedmvs输出：(image_tensor, depth_tensor, K_tensor, seq_name, scene, frame_idx)
matterport输出：元组格式: (image_1, depth_1, K_1, image_2, depth_2, K_2, image_3, depth_3, K_3, seq_name, scene_id, uuid, frame_idx)
tartanair: (left_img_tensor, left_depth_tensor, K_tensor,right_img_tensor, right_depth_tensor, K_tensor,seq_name, frame_idx)
ddad:每个sample是一个列表,列表里包含几个OrderedDict的列表:
#sample[0] = [ OrderedDict (CAMERA_01),  # index 0    
            #OrderedDict (CAMERA_05),  # index 1    
            #OrderedDict (CAMERA_06),  # index 2    
            #OrderedDict (CAMERA_07),  # index 3    
            #OrderedDict (CAMERA_08),  # index 4    
            #OrderedDict (CAMERA_09),  # index 5    
            #OrderedDict (LIDAR),      # index 6 (注意：LiDAR在这里！)]
