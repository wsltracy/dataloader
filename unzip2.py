#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量解压 IRS 数据集
将 ./datas/IRS 下的所有 .tar.gz 文件解压到 ./datas/IRS_e
每个压缩包对应一个单独的文件夹，并打印解压出的所有文件名
"""

import tarfile
from pathlib import Path

# 源目录
src_dir = Path("datas/IRS")
# 目标目录
dst_dir = Path("datas/IRS_ext")
dst_dir.mkdir(exist_ok=True)

# 遍历所有 tar.gz 文件
for tar_path in src_dir.glob("*.tar.gz"):
    # 去掉 .tar.gz 后的名字作为文件夹
    scene_name = tar_path.stem.replace(".tar", "")
    scene_dir = dst_dir/scene_name
    scene_dir.mkdir(exist_ok=True)

    # 打开并解压
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(scene_dir)

        # 打印压缩包内所有文件名
        print(f"\n[√] 解压完成: {tar_path.name} -> {scene_dir}")
        print("解压出的文件列表:")
        for member in tar.getnames():
            print("  ", member)

print("\n所有压缩包解压完成！")