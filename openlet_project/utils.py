import json
import os
import re

import numpy as np
import pandas as pd


# 判断文件名是否近似 UUID 风格
# 输入：不带扩展名的文件名
# 输出：True / False
def is_uuid_like(file_stem):
    pattern = r"^[0-9a-fA-F]{32}$"
    return re.fullmatch(pattern, file_stem) is not None


# 安全地把 numpy 类型转换成普通 Python 类型
# 输入：任意对象
# 输出：可 JSON 序列化的普通 Python 对象
def to_python(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


# 保存 DataFrame 为 CSV
# 输入：DataFrame、保存路径
# 输出：无
def save_csv(df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


# 保存字典为 JSON
# 输入：字典、保存路径
# 输出：无
def save_json(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)