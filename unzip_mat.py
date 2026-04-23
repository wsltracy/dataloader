#!/usr/bin/env python3
"""
解压 Matterport3D 数据集中每个场景下的三个 zip 文件，
确保解压后的文件夹直接位于场景文件夹下，不会多出一层。
"""

import os
import zipfile
import shutil
import tempfile

ROOT_DIR = "./datas/matterport/data/v1/scans"
DELETE_ZIP_AFTER_EXTRACT = False   # 是否解压后删除原 zip 文件

ZIP_NAMES = [
    "undistorted_color_images.zip",
    "undistorted_depth_images.zip",
    "undistorted_camera_parameters.zip"
]

def extract_zip_clean(zip_path, extract_to):
    """
    解压 zip 文件，并自动消除可能的多余顶层文件夹。
    如果 zip 内只有一个顶层目录，且该目录名等于 zip 文件名（不含扩展名）或等于场景名，
    则将该目录内的内容提升到 extract_to 目录下。
    使用临时目录避免路径拼接错误。
    """
    if not os.path.exists(zip_path):
        print(f"  跳过（不存在）: {zip_path}")
        return False

    zip_basename = os.path.splitext(os.path.basename(zip_path))[0]
    scene_name = os.path.basename(extract_to)

    print(f"  正在解压: {os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        namelist = zf.namelist()
        # 找出所有顶层条目
        top_levels = set()
        for name in namelist:
            parts = name.split('/')
            if parts[0]:
                top_levels.add(parts[0])
        
        should_flatten = False
        if len(top_levels) == 1:
            top_dir = top_levels.pop()
            if top_dir == zip_basename or top_dir == scene_name:
                should_flatten = True
                print(f"    检测到多余顶层文件夹 '{top_dir}'，将提升其内容")
        
        if should_flatten:
            # 解压到临时目录
            with tempfile.TemporaryDirectory() as tmpdir:
                zf.extractall(tmpdir)
                src_dir = os.path.join(tmpdir, top_dir)
                # 将 src_dir 下的所有内容移动到 extract_to
                for item in os.listdir(src_dir):
                    src_path = os.path.join(src_dir, item)
                    dst_path = os.path.join(extract_to, item)
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_path, dst_path)
        else:
            # 正常解压到目标目录
            zf.extractall(extract_to)
    
    print(f"    解压完成 -> {extract_to}")
    if DELETE_ZIP_AFTER_EXTRACT:
        os.remove(zip_path)
        print(f"    已删除原 zip 文件")
    return True

def main():
    if not os.path.isdir(ROOT_DIR):
        print(f"错误: 根目录不存在 {ROOT_DIR}")
        return

    for scene_name in os.listdir(ROOT_DIR):
        scene_path = os.path.join(ROOT_DIR, scene_name)
        if not os.path.isdir(scene_path):
            continue
        print(f"\n处理场景: {scene_name}")
        any_extracted = False
        for zip_name in ZIP_NAMES:
            zip_path = os.path.join(scene_path, zip_name)
            if os.path.exists(zip_path):
                extract_zip_clean(zip_path, scene_path)
                any_extracted = True
        if not any_extracted:
            print("  未找到任何需要解压的 zip 文件")

    print("\n所有场景处理完成。")

if __name__ == "__main__":
    main()