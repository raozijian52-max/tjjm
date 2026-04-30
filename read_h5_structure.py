import h5py
from pathlib import Path
import json
import io
from PIL import Image
import matplotlib.pyplot as plt
import os

def identify_h5_by_content(filepath):
    """通过读取 HDF5 内部属性来识别文件"""

    with h5py.File(filepath, 'r') as f:
        print(f"文件名: {filepath}")
        print(f"文件中的 UUID: {Path(filepath).stem}")

        # 1. 查看根目录的属性（如果有）
        print("\n=== 文件属性 ===")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")

        # 2. 查找可能的标识信息
        identifying_keys = ['product_id', 'product_name', 'title', 'name', 'description', 'asin', 'sku']

        def search_identifiers(obj, path=""):
            """递归搜索标识信息"""
            for key in obj.keys():
                item = obj[key]
                current_path = f"{path}/{key}"

                # 检查 dataset 名称是否包含标识词
                for id_key in identifying_keys:
                    if id_key in key.lower():
                        try:
                            if isinstance(item, h5py.Dataset):
                                data = item[()]
                                print(f"\n找到可能的标识: {current_path}")
                                print(f"  值: {data}")
                        except:
                            pass

                # 检查属性
                if isinstance(item, h5py.Group) or isinstance(item, h5py.Dataset):
                    for attr_name, attr_value in item.attrs.items():
                        for id_key in identifying_keys:
                            if id_key in attr_name.lower():
                                print(f"\n找到属性标识: {current_path}.attrs[{attr_name}]")
                                print(f"  值: {attr_value}")

                # 递归
                if isinstance(item, h5py.Group):
                    search_identifiers(item, current_path)

        search_identifiers(f)

        # 3. 显示数据集概览
        print("\n=== 数据集概览 ===")

        def show_structure(obj, prefix=""):
            for key in obj.keys():
                item = obj[key]
                if isinstance(item, h5py.Dataset):
                    print(f"  {prefix}/{key}: shape={item.shape}, dtype={item.dtype}")
                elif isinstance(item, h5py.Group):
                    show_structure(item, f"{prefix}/{key}")

        show_structure(f)

def inspect_metadata_raw(directory):
    """查看 metadata.json 的原始内容"""
    directory = Path(directory)

    for i, h5_file in enumerate(directory.glob("*.h5")):
        if i >= 2:  # 只看前2个文件
            break

        print(f"\n{'=' * 60}")
        print(f"文件: {h5_file.name}")
        print('=' * 60)

        with h5py.File(h5_file, 'r') as f:
            if '/metadata.json' in f:
                meta = f['/metadata.json'][()]
                if isinstance(meta, bytes):
                    meta = meta.decode('utf-8')

                print(f"类型: {type(meta)}")
                print(f"长度: {len(meta)} 字符")
                print(f"\n原始内容:\n{meta[:2000]}")  # 显示前2000字符

                # 尝试解析 JSON
                try:
                    data = json.loads(meta)
                    print(f"\n解析后的键: {list(data.keys())}")

                    # 递归打印所有内容（限制深度）
                    def print_dict(d, indent=0):
                        for k, v in d.items():
                            if isinstance(v, dict):
                                print(f"{'  ' * indent}{k}:")
                                print_dict(v, indent + 1)
                            elif isinstance(v, list):
                                print(f"{'  ' * indent}{k}: [列表，长度{len(v)}]")
                                if len(v) > 0 and isinstance(v[0], dict):
                                    print(f"{'  ' * (indent + 1)}前1项: {v[0]}")
                            else:
                                val_str = str(v)[:100]
                                print(f"{'  ' * indent}{k}: {val_str}")

                    print_dict(data)

                except json.JSONDecodeError as e:
                    print(f"JSON 解析失败: {e}")
                    print(f"可能是纯文本或其他格式")
            else:
                print("没有 metadata.json")

        # 同时看看根目录的属性
        with h5py.File(h5_file, 'r') as f:
            if f.attrs:
                print(f"\n根目录属性:")
                for k, v in f.attrs.items():
                    print(f"  {k}: {v}")


def show_camera_image(h5_path, idx):
    """查看指定摄像头的其中一张图"""
    with h5py.File(h5_path, 'r') as f:
        images = f['/cameras/head/color/data'][:]
        timestamps = f['/cameras/head/depth/timestamp'][:]

        raw = images[idx]
        if hasattr(raw, 'tobytes'):
            raw = raw.tobytes()

        img = Image.open(io.BytesIO(raw))

        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"timestamp: {timestamps[idx]}")
        plt.show()

def save_images(h5_path, idx1, idx2, output_dir="image_output"):
    """保存指定摄像头数据的连续图片"""
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        images = f['/cameras/head/color/data']
        timestamps = f['/cameras/head/color/timestamp']

        total = len(images)

        # 边界处理
        idx1 = max(0, idx1)
        idx2 = min(idx2, total - 1)

        if idx1 > idx2:
            raise ValueError(f"无效范围: idx1={idx1}, idx2={idx2}, 总图片数={total}")

        for i in range(idx1, idx2 + 1):
            raw = images[i]

            # 兼容 numpy 数组 / bytes
            if hasattr(raw, "tobytes"):
                raw = raw.tobytes()

            img = Image.open(io.BytesIO(raw))

            ts = timestamps[i]
            save_path = os.path.join(output_dir, f"img_{i:06d}_{ts}.png")
            img.save(save_path)

            print(f"已保存: {save_path}")

# base_dir = os.path.abspath(r"D:\小yy\统计建模大赛")

# directory = os.path.join(base_dir, "LET数据集-单个商品抓取")
filepath = os.path.join('.', "s4h5.h5")
# filepath = r"..\LET数据集-单个商品抓取\a448952b49034faa851c86dec5e75926.h5"


# inspect_metadata_raw(directory)
identify_h5_by_content(filepath)
# show_camera_image(filepath, 120)
# save_images(filepath, 50, 150)