import os
from pathlib import Path
from PIL import Image
import concurrent.futures
from tqdm import tqdm


def check_image(img_path):
    """
    尝试打开并加载图像。
    如果图像损坏，返回该图像的绝对路径；如果正常，返回 None。
    """
    try:
        # 1. verify() 会检查文件的 header 是否损坏
        with Image.open(img_path) as img:
            img.verify()

            # 2. verify() 运行后，文件指针会改变。为了检查图像数据块(chunk)是否截断，
        # 需要重新 open 并调用 load() 强制读取全部像素数据。
        with Image.open(img_path) as img:
            img.load()

        return None
    except Exception as e:
        # 捕获到任何异常（如 OSError: image file is truncated）都视为损坏
        return str(img_path)


def find_corrupted_pngs(directory, num_workers=8):
    directory = Path(directory)
    print(f"正在扫描目录: {directory} 下的所有 .png 文件...")

    # 递归获取所有 png 文件路径 (rglob 相当于深度遍历)
    png_files = list(directory.rglob('*.png'))
    total_files = len(png_files)

    if total_files == 0:
        print("未找到任何 .png 文件，请检查路径是否正确。")
        return []

    print(f"共找到 {total_files} 个 .png 文件。开始多进程校验...")

    # 使用多进程池加速检查
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 使用 tqdm 包装 executor.map 以显示进度条
        results = list(tqdm(executor.map(check_image, png_files), total=total_files, desc="Checking Images"))

    # 过滤出所有损坏的文件路径 (非 None 的返回值)
    corrupted_files = [res for res in results if res is not None]

    return corrupted_files


if __name__ == '__main__':
    # 将这里替换为你的 TartanAir 根目录
    target_dir = "datas/tartanair_data/"

    # max_workers 可以根据你的 CPU 核心数进行调整，通常设置为 8 或 16 速度最快
    bad_files = find_corrupted_pngs(target_dir, num_workers=8)

    if bad_files:
        print(f"\n[警告] 检查完毕！发现 {len(bad_files)} 个损坏的文件:")

        # 将损坏的文件路径写入 txt，方便后续处理
        output_txt = "corrupted_png_list.txt"
        with open(output_txt, "w") as f:
            for bad_file in bad_files:
                f.write(bad_file + "\n")

        print(f"损坏的文件列表已完整保存至: {output_txt}")
        print("你可以根据这个 txt 文件编写一个批量删除或重新下载的脚本。")
    else:
        print("\n[成功] 检查完毕！所有文件均可正常读取，没有发现损坏的图像。")