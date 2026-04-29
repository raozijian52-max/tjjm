from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


def _safe_ratio(numerator, denominator):
    """安全计算比例，并把结果限制在 0 到 1 之间。"""
    if denominator is None or denominator <= 0:
        return np.nan
    return float(np.clip(numerator / denominator, 0.0, 1.0))

def _safe_mean(values):
    """计算列表均值，自动跳过 NaN 或空值。"""
    values = [v for v in values if pd.notna(v)]
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))

def _finite_array(seq):
    """把输入序列统一转成二维浮点数组，方便后续按时间维处理。"""
    arr = np.asarray(seq, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def uniform_sample_indices(n_total, max_samples):
    """从总帧数里均匀抽取最多 max_samples 个索引。"""
    if n_total <= 0:
        return np.array([], dtype=int)
    if n_total <= max_samples:
        return np.arange(n_total, dtype=int)
    return np.linspace(0, n_total - 1, num=max_samples, dtype=int)

def decode_image_bytes(img_bytes, mode):
    """把 H5 里保存的图片字节流解码成 numpy 图像数组。"""
    if img_bytes is None or len(img_bytes) == 0:
        return None
    try:
        with Image.open(BytesIO(img_bytes)) as img:
            if mode == "color":
                img = img.convert("RGB")
            return np.array(img)
    except (UnidentifiedImageError, OSError, ValueError):
        return None

def simple_kmeans(X, n_clusters, random_state=42, max_iter=100):
    """用 numpy 实现一个轻量版 KMeans，避免额外依赖 sklearn。"""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(random_state)
    if len(X) <= n_clusters:
        return np.arange(len(X), dtype=int)

    centers = X[rng.choice(len(X), size=n_clusters, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centers[k] = X[mask].mean(axis=0)

    return labels

def normalized_mutual_information(labels_a, labels_b):
    """计算两个离散序列的归一化互信息，用于衡量视觉变化和动作变化的关联。"""
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if len(labels_a) == 0 or len(labels_a) != len(labels_b):
        return np.nan

    a_values, a_inverse = np.unique(labels_a, return_inverse=True)
    b_values, b_inverse = np.unique(labels_b, return_inverse=True)
    contingency = np.zeros((len(a_values), len(b_values)), dtype=float)

    for a, b in zip(a_inverse, b_inverse):
        contingency[a, b] += 1.0

    total = contingency.sum()
    if total <= 0:
        return np.nan

    p_ab = contingency / total
    p_a = p_ab.sum(axis=1)
    p_b = p_ab.sum(axis=0)

    expected = p_a[:, None] * p_b[None, :]
    valid = p_ab > 0
    mi = float(np.sum(p_ab[valid] * np.log(p_ab[valid] / expected[valid])))

    h_a = float(-np.sum(p_a[p_a > 0] * np.log(p_a[p_a > 0])))
    h_b = float(-np.sum(p_b[p_b > 0] * np.log(p_b[p_b > 0])))
    denom = (h_a + h_b) / 2.0
    if denom <= 1e-12:
        return 0.0
    return float(mi / denom)

def _robust_zscore(values):
    """计算稳健 z-score，用中位数和 MAD 降低极端值影响。"""
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad <= 1e-12:
        std = np.nanstd(values)
        if not np.isfinite(std) or std <= 1e-12:
            return np.zeros_like(values, dtype=float)
        return (values - np.nanmean(values)) / std
    return 0.6745 * (values - median) / mad
