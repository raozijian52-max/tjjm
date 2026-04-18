import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def clean_sheet_name(name, used_names):
    """清理 Excel sheet 名称，避免非法字符和重名"""
    invalid_chars = ['\\', '/', '*', '?', ':', '[', ']']
    for ch in invalid_chars:
        name = name.replace(ch, '_')

    name = name.strip()
    if not name:
        name = "Sheet"

    # Excel sheet 名最大 31 个字符
    base = name[:31]
    new_name = base
    idx = 1

    while new_name in used_names:
        suffix = f"_{idx}"
        new_name = base[:31 - len(suffix)] + suffix
        idx += 1

    used_names.add(new_name)
    return new_name


def decode_bytes_array(arr):
    """把 bytes 类型尽量转成字符串"""
    if isinstance(arr, bytes):
        try:
            return arr.decode("utf-8")
        except Exception:
            return str(arr)

    if isinstance(arr, np.ndarray):
        if arr.dtype.kind == "S":  # bytes
            return np.vectorize(lambda x: x.decode("utf-8", errors="ignore"))(arr)
        elif arr.dtype.kind == "O":  # object
            def convert(x):
                if isinstance(x, bytes):
                    try:
                        return x.decode("utf-8")
                    except Exception:
                        return str(x)
                return x
            return np.vectorize(convert, otypes=[object])(arr)

    return arr


def dataset_to_dataframe(data):
    """
    把 dataset 转成 DataFrame：
    - 标量 -> 1行1列
    - 1维 -> 1列
    - 2维 -> 正常表格
    - 3维及以上 -> 展平后输出
    """
    data = decode_bytes_array(data)
    arr = np.array(data)

    # 标量
    if arr.ndim == 0:
        return pd.DataFrame({"value": [arr.item()]})

    # 1维
    if arr.ndim == 1:
        return pd.DataFrame({"value": arr})

    # 2维
    if arr.ndim == 2:
        return pd.DataFrame(arr)

    # 3维及以上：展平成二维
    reshaped = arr.reshape(arr.shape[0], -1)
    df = pd.DataFrame(reshaped)
    df.insert(0, "__original_shape__", str(arr.shape))
    return df


def collect_datasets(h5obj, prefix=""):
    """递归收集所有 dataset，返回 [(路径, 数据), ...]"""
    results = []

    for key in h5obj.keys():
        item = h5obj[key]
        current_path = f"{prefix}/{key}" if prefix else key

        if isinstance(item, h5py.Dataset):
            try:
                data = item[()]
                results.append((current_path, data))
            except Exception as e:
                results.append((current_path, f"读取失败: {e}"))
        elif isinstance(item, h5py.Group):
            results.extend(collect_datasets(item, current_path))

    return results


def try_read_with_pandas(h5_path):
    """
    如果是 pandas 写的 HDF 文件，优先尝试用 pandas 读取。
    返回 dict: {sheet_name: DataFrame}
    """
    tables = {}
    try:
        with pd.HDFStore(h5_path, mode="r") as store:
            keys = store.keys()
            if not keys:
                return tables

            used_names = set()
            for key in keys:
                try:
                    obj = pd.read_hdf(h5_path, key=key)
                    if isinstance(obj, pd.Series):
                        obj = obj.to_frame(name="value")
                    elif not isinstance(obj, pd.DataFrame):
                        obj = pd.DataFrame(obj)

                    sheet_name = clean_sheet_name(key.strip("/"), used_names)
                    tables[sheet_name] = obj
                except Exception:
                    pass
    except Exception:
        pass

    return tables


def h5_to_excel(h5_path):
    h5_path = Path(h5_path)
    xlsx_path = h5_path.with_suffix(".xlsx")

    # 先尝试按 pandas HDF 读取
    pandas_tables = try_read_with_pandas(h5_path)

    used_names = set()
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        wrote_anything = False

        if pandas_tables:
            for sheet_name, df in pandas_tables.items():
                final_name = clean_sheet_name(sheet_name, used_names)
                df.to_excel(writer, sheet_name=final_name, index=False)
                wrote_anything = True
        else:
            # 普通 HDF5 文件读取
            with h5py.File(h5_path, "r") as f:
                datasets = collect_datasets(f)

                if not datasets:
                    pd.DataFrame({"info": ["这个 .h5 文件里没有可读取的 dataset"]}).to_excel(
                        writer, sheet_name="info", index=False
                    )
                    wrote_anything = True
                else:
                    for ds_path, data in datasets:
                        sheet_name = clean_sheet_name(ds_path.replace("/", "_"), used_names)

                        if isinstance(data, str):
                            df = pd.DataFrame({"info": [data]})
                        else:
                            df = dataset_to_dataframe(data)

                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        wrote_anything = True

        if not wrote_anything:
            pd.DataFrame({"info": ["没有成功写入任何数据"]}).to_excel(
                writer, sheet_name="info", index=False
            )

    print(f"已生成: {xlsx_path.name}")


def main():
    current_dir = Path(".")
    h5_files = list(current_dir.glob("*.h5")) + list(current_dir.glob("*.hdf5"))

    if not h5_files:
        print("当前目录下没有找到 .h5 或 .hdf5 文件。")
        return

    for h5_file in h5_files:
        try:
            h5_to_excel(h5_file)
        except Exception as e:
            print(f"处理失败: {h5_file.name}，错误: {e}")


if __name__ == "__main__":
    main()