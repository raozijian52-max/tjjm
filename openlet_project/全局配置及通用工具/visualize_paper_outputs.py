import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


QUALITY_DIMS = [
    ("Q_completeness", "完整性"),
    ("Q_accuracy", "准确性"),
    ("Q_diversity", "多样性"),
    ("Q_consistency", "一致性"),
    ("Q_usability", "可用性"),
]

SCENE_IDS = [f"S{i}" for i in range(1, 11)]

SOURCE_EMPIRICAL = "empirical"
SOURCE_PREDICTED = "ridge_predicted"

TASK_GROUPS = {
    "S1": "视觉主导型",
    "S2": "力控主导型",
    "S3": "视觉主导型",
    "S4": "力控主导型",
    "S5": "力控主导型",
    "S6": "视觉主导型",
    "S7": "视觉主导型",
    "S8": "视觉主导型",
    "S9": "力控主导型",
    "S10": "力控主导型",
}

COLORS = {
    "quality": "#2F80ED",
    "value": "#EB5757",
    "dual": "#7B61FF",
    "random": "#8A8F98",
    "ascending": "#F2994A",
    "high": "#2D9CDB",
    "mid": "#F2C94C",
    "low": "#EB5757",
}


def setup_matplotlib():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def find_files_by_name(file_name):
    matches = []
    for path in Path(".").rglob(file_name):
        parts = set(path.parts)
        if ".git" in parts or "visualizations" in parts:
            continue
        matches.append(path)

    def priority(path):
        text = str(path).replace("\\", "/")
        if "new_stage3-stage5/十场景阶段三" in text:
            return 0
        if "new_stage3-stage5/阶段四" in text:
            return 1
        if "/data/interim/" in text or "/data/processed/" in text:
            return 2
        if "stage3-stage5/processed" in text:
            return 3
        if "stage3-stage5" in text:
            return 4
        if "stage1-stage2" in text or "pca_check" in text:
            return 5
        return 6

    return sorted(matches, key=lambda p: (priority(p), len(str(p))))


def find_file(file_name):
    matches = find_files_by_name(file_name)
    return matches[0] if matches else None


def read_csv(file_name, missing, required=True):
    path = find_file(file_name)
    if path is None:
        if required:
            missing.append(f"缺少文件：{file_name}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        missing.append(f"读取失败：{path} ({exc})")
        return None


def normalize_series(values):
    values = pd.Series(values, dtype=float)
    min_v = values.min(skipna=True)
    max_v = values.max(skipna=True)
    if pd.isna(min_v) or pd.isna(max_v) or abs(max_v - min_v) < 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - min_v) / (max_v - min_v)


def load_quality_all(missing):
    dfs = []
    for scene_id in SCENE_IDS:
        prefix = scene_id.lower()
        df = read_csv(f"{prefix}_stage2_quality_dataset.csv", [], required=False)
        if df is None:
            df = read_csv(f"{prefix}_quality_scores.csv", [], required=False)
        if df is None:
            missing.append(f"缺少质量分数：{prefix}_stage2_quality_dataset.csv 或 {prefix}_quality_scores.csv")
            continue
        df = df.copy()
        df["scene_id"] = scene_id
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_scene_quality_value(missing):
    df = read_csv("stage3_scene_value_model_table.csv", missing)
    if df is None:
        return pd.DataFrame()
    return df[df["scene_id"].isin(SCENE_IDS)].copy()


def load_delta_summary(missing):
    df = read_csv("stage3_repeat_delta_summary.csv", missing)
    if df is None:
        return pd.DataFrame()
    return df[df["scene_id"].isin(SCENE_IDS)].copy()


def aggregate_scene_quality(quality_df):
    if quality_df.empty:
        return pd.DataFrame()
    quality_cols = [
        "Q_score",
        "Q_completeness",
        "Q_accuracy",
        "Q_diversity",
        "Q_consistency",
        "Q_usability",
    ]
    existing_cols = [col for col in quality_cols if col in quality_df.columns]
    if not existing_cols:
        return pd.DataFrame()

    df = quality_df.copy()
    df = df[df["scene_id"].isin(SCENE_IDS)].copy()
    for col in existing_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    count_col = "trajectory_id" if "trajectory_id" in df.columns else "scene_id"
    grouped = df.groupby("scene_id", as_index=False).agg(
        n_trajectories=(count_col, "count"),
        **{f"{col}_mean": (col, "mean") for col in existing_cols},
    )
    order = {scene_id: idx for idx, scene_id in enumerate(SCENE_IDS)}
    grouped["_order"] = grouped["scene_id"].map(order)
    grouped = grouped.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return grouped


def ridge_predict_from_empirical(train_df, predict_df, feature_cols, target_col):
    train = train_df.dropna(subset=feature_cols + [target_col]).copy()
    if len(train) < 2:
        fallback = float(train[target_col].mean()) if len(train) else 0.0
        return pd.Series([fallback] * len(predict_df), index=predict_df.index)

    train_x = train[feature_cols].astype(float).values
    train_y = train[target_col].astype(float).values
    full_x = predict_df[feature_cols].astype(float).values

    means = np.nanmean(train_x, axis=0)
    stds = np.nanstd(train_x, axis=0)
    stds[stds < 1e-12] = 1.0
    train_x = np.nan_to_num((train_x - means) / stds, nan=0.0)
    full_x = np.nan_to_num((full_x - means) / stds, nan=0.0)

    y_mean = float(np.nanmean(train_y))
    centered_y = train_y - y_mean
    alpha = 1.0
    reg = alpha * np.eye(train_x.shape[1])
    coef = np.linalg.pinv(train_x.T @ train_x + reg) @ train_x.T @ centered_y
    pred = y_mean + full_x @ coef
    return pd.Series(pred, index=predict_df.index)


def complete_scene_value_inputs(quality_df, scene_df, delta_df, missing):
    scene_quality = aggregate_scene_quality(quality_df)
    if scene_quality.empty:
        return scene_df, delta_df

    empirical_cols = [
        "scene_id",
        "delta_score_mean",
        "delta_score_std",
        "delta_score_positive_rate",
    ]
    empirical = pd.DataFrame(columns=empirical_cols)
    if not delta_df.empty and "delta_score_mean" in delta_df.columns:
        available_cols = [col for col in empirical_cols if col in delta_df.columns]
        empirical = delta_df[available_cols].copy()
    elif not scene_df.empty and "delta_score_mean" in scene_df.columns:
        available_cols = [col for col in empirical_cols if col in scene_df.columns]
        empirical = scene_df[available_cols].copy()

    full_scene = pd.merge(scene_quality, empirical, on="scene_id", how="left")
    full_scene["value_source"] = np.where(
        full_scene["delta_score_mean"].notna(),
        SOURCE_EMPIRICAL,
        SOURCE_PREDICTED,
    )

    feature_cols = [
        "n_trajectories",
        "Q_score_mean",
        "Q_completeness_mean",
        "Q_accuracy_mean",
        "Q_diversity_mean",
        "Q_consistency_mean",
    ]
    if "Q_usability_mean" in full_scene.columns:
        feature_cols.append("Q_usability_mean")
    feature_cols = [col for col in feature_cols if col in full_scene.columns]

    train_df = full_scene[full_scene["delta_score_mean"].notna()].copy()
    missing_value_mask = full_scene["delta_score_mean"].isna()
    if missing_value_mask.any():
        if not feature_cols:
            missing.append("图3-5无法补齐十场景：缺少可用于元模型预测的质量特征")
        else:
            pred = ridge_predict_from_empirical(train_df, full_scene.loc[missing_value_mask], feature_cols, "delta_score_mean")
            full_scene.loc[missing_value_mask, "delta_score_mean"] = pred
            full_scene.loc[missing_value_mask, "delta_score_std"] = np.nan
            full_scene.loc[missing_value_mask, "delta_score_positive_rate"] = np.nan

    full_scene["delta_score_pred_for_plot"] = np.where(
        full_scene["value_source"].eq(SOURCE_PREDICTED),
        full_scene["delta_score_mean"],
        np.nan,
    )

    order = {scene_id: idx for idx, scene_id in enumerate(SCENE_IDS)}
    full_scene["_order"] = full_scene["scene_id"].map(order)
    full_scene = full_scene.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    delta_completed = full_scene[[
        "scene_id",
        "delta_score_mean",
        "delta_score_std",
        "delta_score_positive_rate",
        "value_source",
    ]].copy()
    return full_scene, delta_completed


def load_trajectory_outputs(missing):
    final_df = read_csv("stage3_final_value_outputs.csv", missing)
    label_df = read_csv("stage4_bc_trajectory_labels.csv", missing)
    if final_df is None or label_df is None:
        return pd.DataFrame()

    final_df = final_df.copy()
    label_df = label_df.copy()

    if "global_id" in final_df.columns and "global_id" in label_df.columns:
        df = pd.merge(final_df, label_df, on=["global_id", "trajectory_id", "scene_id"], how="inner")
    else:
        df = pd.merge(final_df, label_df, on=["trajectory_id", "scene_id"], how="inner")

    if "trajectory_imitation_score" not in df.columns:
        missing.append("stage4_bc_trajectory_labels.csv 缺少 trajectory_imitation_score")
        return pd.DataFrame()

    df = df[df["scene_id"].isin(SCENE_IDS)].copy()
    if "delta_score_pred" not in df.columns and "delta_score_emp" in df.columns:
        df["delta_score_pred"] = df["delta_score_emp"]
    if "delta_score_pred" not in df.columns:
        missing.append("stage3_final_value_outputs.csv 缺少 delta_score_pred / delta_score_emp")
        return pd.DataFrame()

    df["quality_norm"] = normalize_series(df["Q_score"])
    df["value_norm"] = normalize_series(df["delta_score_pred"])
    df["dual_score"] = 0.5 * df["quality_norm"] + 0.5 * df["value_norm"]
    df["task_group"] = df["scene_id"].map(TASK_GROUPS).fillna("其他")
    return df


def load_stage4_curation_summary():
    df = read_csv("stage4_curation_summary.csv", [], required=False)
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    numeric_cols = [
        "ratio",
        "test_imitation_score_mean",
        "test_imitation_score_std",
        "test_nmse_mean",
        "test_nmse_std",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_stage4_per_scene_metrics():
    df = read_csv("stage4_curation_per_scene_metrics.csv", [], required=False)
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    numeric_cols = [
        "ratio",
        "scene_imitation_score",
        "scene_nmse",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "test_scene" in df.columns:
        df["task_group"] = df["test_scene"].map(TASK_GROUPS).fillna("其他")
    return df


def draw_radar(df, scene_ids, out_path):
    labels = [label for _, label in QUALITY_DIMS]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)

    cmap = plt.get_cmap("tab10", len(scene_ids))
    for i, scene_id in enumerate(scene_ids):
        row = df[df["scene_id"] == scene_id]
        if row.empty:
            continue
        row = row.iloc[0]
        values = [float(row[col]) for col, _ in QUALITY_DIMS]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, color=cmap(i), label=scene_id.lower())
        ax.fill(angles, values, color=cmap(i), alpha=0.06)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=9)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig1_quality_radar(quality_df, out_dir, missing):
    required = ["Q_score"] + [col for col, _ in QUALITY_DIMS]
    if quality_df.empty or any(col not in quality_df.columns for col in required):
        missing.append("图1需要十场景质量分数及五维分量")
        return

    grouped = quality_df.groupby("scene_id")[required].mean().reset_index()
    grouped = grouped[grouped["scene_id"].isin(SCENE_IDS)].copy()
    grouped = grouped.sort_values("Q_score", ascending=False)
    top5 = grouped.head(5)["scene_id"].tolist()
    bottom5 = grouped.tail(5).sort_values("Q_score", ascending=False)["scene_id"].tolist()

    draw_radar(grouped, top5, out_dir / "fig1a_quality_radar_top5.png")
    draw_radar(grouped, bottom5, out_dir / "fig1b_quality_radar_bottom5.png")


def fig2_qscore_boxplot(quality_df, out_path, missing):
    if quality_df.empty or "Q_score" not in quality_df.columns:
        missing.append("图2需要 Q_score")
        return

    df = quality_df.dropna(subset=["Q_score"]).copy()
    order = [s for s in SCENE_IDS if s in set(df["scene_id"])]
    data = [df.loc[df["scene_id"] == s, "Q_score"].values for s in order]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bp = ax.boxplot(data, patch_artist=True, tick_labels=[s.lower() for s in order], showfliers=True)
    cmap = plt.get_cmap("tab10", len(order))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i))
        patch.set_alpha(0.45)
    rng = np.random.default_rng(42)
    for i, values in enumerate(data, start=1):
        x = rng.normal(i, 0.035, size=len(values))
        ax.scatter(x, values, s=14, alpha=0.42, color=cmap(i - 1), edgecolors="none")
    ax.set_xlabel("场景编号")
    ax.set_ylabel("综合质量得分")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.25)
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=cmap(i), label=s.lower(), markersize=8) for i, s in enumerate(order)]
    ax.legend(handles=handles, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def tier_colors(values):
    q1 = np.nanquantile(values, 1 / 3)
    q2 = np.nanquantile(values, 2 / 3)
    colors = []
    labels = []
    for v in values:
        if v >= q2:
            colors.append(COLORS["high"])
            labels.append("高价值")
        elif v >= q1:
            colors.append(COLORS["mid"])
            labels.append("中价值")
        else:
            colors.append(COLORS["low"])
            labels.append("低价值")
    return colors, labels


def fig3_scene_value_bar(delta_df, out_path, missing):
    if delta_df.empty:
        missing.append("图3需要 stage3_repeat_delta_summary.csv")
        return
    df = delta_df.sort_values("delta_score_mean", ascending=False).copy()
    colors, labels = tier_colors(df["delta_score_mean"].values)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(df))
    if "delta_score_std" in df.columns:
        yerr = pd.to_numeric(df["delta_score_std"], errors="coerce").fillna(0.0).values
    else:
        yerr = None
    bars = ax.bar(x, df["delta_score_mean"], yerr=yerr, capsize=4, color=colors, alpha=0.88)
    if "value_source" in df.columns:
        for bar, (_, row) in zip(bars, df.iterrows()):
            if row.get("value_source") == SOURCE_PREDICTED:
                bar.set_hatch("//")
                bar.set_edgecolor("#344054")
                bar.set_linewidth(0.8)
    ax.axhline(0, color="#344054", linestyle="--", linewidth=1)
    y_values = pd.to_numeric(df["delta_score_mean"], errors="coerce").fillna(0.0).values
    for xi, value in zip(x, y_values):
        ax.annotate(
            f"{value:.4f}",
            xy=(xi, value),
            xytext=(0, 3 if value >= 0 else -3),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.6),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(df["scene_id"].astype(str).str.lower())
    ax.set_xlabel("场景编号")
    ax.set_ylabel("边际效能价值")
    ax.grid(axis="y", alpha=0.25)
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["high"], label="高价值", markersize=9),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["mid"], label="中价值", markersize=9),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["low"], label="低价值", markersize=9),
    ]
    if "value_source" in df.columns and df["value_source"].eq(SOURCE_PREDICTED).any():
        handles.extend([
            Patch(facecolor="#D0D5DD", edgecolor="#344054", label="已有经验值"),
            Patch(facecolor="#D0D5DD", edgecolor="#344054", hatch="//", label="元模型预测补齐"),
        ])
    ax.legend(handles=handles)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig4_quality_value_scatter(scene_df, out_path, missing):
    if scene_df.empty or "Q_accuracy_mean" not in scene_df.columns or "delta_score_mean" not in scene_df.columns:
        missing.append("图4需要 stage3_scene_value_model_table.csv 中的 Q_accuracy_mean 和 delta_score_mean")
        return
    df = scene_df.dropna(subset=["Q_accuracy_mean", "delta_score_mean"]).copy()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    if "value_source" in df.columns:
        empirical = df[df["value_source"].eq(SOURCE_EMPIRICAL)]
        predicted = df[df["value_source"].eq(SOURCE_PREDICTED)]
        if not empirical.empty:
            ax.scatter(empirical["Q_accuracy_mean"], empirical["delta_score_mean"], s=70, color=COLORS["quality"], alpha=0.82, label="已有经验值")
        if not predicted.empty:
            ax.scatter(predicted["Q_accuracy_mean"], predicted["delta_score_mean"], s=78, color=COLORS["value"], marker="^", alpha=0.82, label="元模型预测补齐")
    else:
        ax.scatter(df["Q_accuracy_mean"], df["delta_score_mean"], s=70, color=COLORS["quality"], alpha=0.82, label="场景")
    scene_label_offsets = {
        "S2": (-2, -18),
        "S6": (12, -18),
        "S8": (12, 4),
    }
    label_offsets = [(4, 4), (4, -12), (-18, 5), (-18, -12), (7, 8), (7, -14)]
    for idx, (_, row) in enumerate(df.iterrows()):
        scene_id = str(row["scene_id"])
        offset = scene_label_offsets.get(scene_id, label_offsets[idx % len(label_offsets)])
        ax.annotate(
            scene_id.lower(),
            xy=(row["Q_accuracy_mean"], row["delta_score_mean"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("准确性维度得分均值")
    ax.set_ylabel("边际效能价值")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_quality_features_value_scatter(scene_df, out_path, missing):
    feature_specs = [
        ("Q_score_mean", "综合质量得分均值"),
        ("Q_completeness_mean", "完整性得分均值"),
        ("Q_accuracy_mean", "准确性得分均值"),
        ("Q_diversity_mean", "多样性得分均值"),
        ("Q_consistency_mean", "一致性得分均值"),
        ("Q_usability_mean", "可用性得分均值"),
    ]
    required_cols = ["scene_id", "delta_score_mean"] + [col for col, _ in feature_specs]
    missing_cols = [col for col in required_cols if col not in scene_df.columns]
    if scene_df.empty or missing_cols:
        missing.append(f"质量特征与边际效能价值散点图缺少字段：{missing_cols}")
        return

    df = scene_df.dropna(subset=required_cols).copy()
    if df.empty:
        missing.append("质量特征与边际效能价值散点图没有可用数据")
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    axes = axes.ravel()
    cmap = plt.get_cmap("tab10", len(SCENE_IDS))
    scene_colors = {scene_id: cmap(i) for i, scene_id in enumerate(SCENE_IDS)}

    for ax, (feature_col, feature_label) in zip(axes, feature_specs):
        for _, row in df.iterrows():
            scene_id = str(row["scene_id"])
            ax.scatter(
                row[feature_col],
                row["delta_score_mean"],
                s=42,
                color=scene_colors.get(scene_id, "#2F80ED"),
                alpha=0.85,
            )
            ax.annotate(
                scene_id.lower(),
                xy=(row[feature_col], row["delta_score_mean"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xlabel(feature_label)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("边际效能价值")
    axes[3].set_ylabel("边际效能价值")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=scene_colors[scene_id], label=scene_id.lower(), markersize=7)
        for scene_id in SCENE_IDS
        if scene_id in set(df["scene_id"].astype(str))
    ]
    fig.legend(handles=handles, ncol=10, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def stage_base_score():
    base = read_csv("stage3_repeat_base_all.csv", [], required=False)
    if base is not None and "imitation_score" in base.columns:
        return float(base["imitation_score"].mean())
    labels = read_csv("stage4_bc_trajectory_labels.csv", [], required=False)
    if labels is not None and "trajectory_imitation_score" in labels.columns:
        return float(labels["trajectory_imitation_score"].mean())
    return 1.0


def fig5_scene_curation_curve(delta_df, out_path, missing):
    if delta_df.empty:
        missing.append("图5需要 stage3_repeat_delta_summary.csv")
        return
    df_desc = delta_df.sort_values("delta_score_mean", ascending=False).reset_index(drop=True)
    df_asc = delta_df.sort_values("delta_score_mean", ascending=True).reset_index(drop=True)
    base = stage_base_score()

    def curve(df):
        values = []
        xs = []
        for k in range(1, len(df) + 1):
            xs.append(k)
            values.append(base + float(df.iloc[:k]["delta_score_mean"].mean()))
        return np.array(xs), np.array(values)

    x1, y1 = curve(df_desc)
    x2, y2 = curve(df_asc)
    fig, ax = plt.subplots(figsize=(8, 5.3))
    ax.plot(x1, y1, marker="o", linewidth=2, color=COLORS["value"], label="场景价值降序纳入")
    ax.plot(x2, y2, marker="o", linewidth=2, color=COLORS["ascending"], label="场景价值升序纳入")
    ax.axhline(base, color="#344054", linestyle="--", linewidth=1.2, label="全量数据效能基线")
    ax.set_xlabel("纳入场景数量")
    ax.set_ylabel("测试集离线模仿效能")
    ax.set_xticks(x1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def curation_curve(df, score_col, ratios, ascending=False, random_runs=100):
    perf_col = "trajectory_imitation_score"
    rows = []
    if score_col == "__random__":
        work = df.dropna(subset=[perf_col]).copy()
        if work.empty:
            return pd.DataFrame()
        rng = np.random.default_rng(42)
        values = work[perf_col].values
        n_total = len(work)
        for ratio in ratios:
            n = max(1, int(math.ceil(ratio * n_total)))
            samples = []
            for _ in range(random_runs):
                idx = rng.choice(n_total, size=n, replace=False)
                samples.append(float(np.mean(values[idx])))
            rows.append({"ratio": ratio, "mean": float(np.mean(samples)), "std": float(np.std(samples, ddof=1))})
        return pd.DataFrame(rows)

    work = df.dropna(subset=[score_col, perf_col]).copy()
    if work.empty:
        return pd.DataFrame()
    work = work.sort_values(score_col, ascending=ascending)
    n_total = len(work)
    for ratio in ratios:
        n = max(1, int(math.ceil(ratio * n_total)))
        selected = work.head(n)
        rows.append({"ratio": ratio, "mean": float(selected[perf_col].mean()), "std": float(selected[perf_col].std(ddof=1) / math.sqrt(len(selected)))})
    return pd.DataFrame(rows)


def stage4_curve(summary_df, strategy):
    required = {"strategy", "ratio", "test_imitation_score_mean"}
    if summary_df.empty or not required.issubset(summary_df.columns):
        return pd.DataFrame()
    df = summary_df[summary_df["strategy"].eq(strategy)].copy()
    if df.empty:
        return pd.DataFrame()
    if "test_imitation_score_std" not in df.columns:
        df["test_imitation_score_std"] = 0.0
    df = df.sort_values("ratio")
    return df.rename(columns={
        "test_imitation_score_mean": "mean",
        "test_imitation_score_std": "std",
    })[["ratio", "mean", "std"]]


def stage4_base_score(summary_df):
    full = stage4_curve(summary_df, "full")
    if not full.empty:
        return float(full["mean"].mean())
    one = summary_df[pd.to_numeric(summary_df.get("ratio"), errors="coerce").eq(1.0)]
    if not one.empty and "test_imitation_score_mean" in one.columns:
        return float(one["test_imitation_score_mean"].mean())
    return np.nan


def fig6_trajectory_curation(traj_df, stage4_summary_df, out_path, missing):
    if not stage4_summary_df.empty:
        curves = {
            "质量得分降序筛选": (stage4_curve(stage4_summary_df, "high_Q_global"), COLORS["quality"]),
            "随机筛选": (stage4_curve(stage4_summary_df, "random"), COLORS["random"]),
            "质量得分升序筛选": (stage4_curve(stage4_summary_df, "low_Q_score"), COLORS["ascending"]),
        }
        curves = {label: item for label, item in curves.items() if not item[0].empty}
        if curves:
            base = stage4_base_score(stage4_summary_df)
            fig, ax = plt.subplots(figsize=(8, 5.3))
            for label, (curve_df, color) in curves.items():
                ax.plot(curve_df["ratio"], curve_df["mean"], marker="o", linewidth=2, label=label, color=color)
                ax.fill_between(curve_df["ratio"], curve_df["mean"] - curve_df["std"], curve_df["mean"] + curve_df["std"], color=color, alpha=0.12)
            if pd.notna(base):
                ax.axhline(base, color="#344054", linestyle="--", linewidth=1.2, label="全量数据效能基线")
            ax.set_xlabel("训练数据保留比例")
            ax.set_ylabel("测试集离线模仿效能")
            ax.set_xlim(0.08, 1.02)
            ax.grid(alpha=0.25)
            ax.legend()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            return

    if traj_df.empty:
        missing.append("图6需要 stage3_final_value_outputs.csv 和 stage4_bc_trajectory_labels.csv")
        return
    ratios = np.linspace(0.1, 1.0, 10)
    curves = {
        "质量得分降序筛选": (curation_curve(traj_df, "Q_score", ratios), COLORS["quality"]),
        "随机筛选": (curation_curve(traj_df, "__random__", ratios), COLORS["random"]),
        "质量得分升序筛选": (curation_curve(traj_df, "Q_score", ratios, ascending=True), COLORS["ascending"]),
    }
    base = float(traj_df["trajectory_imitation_score"].mean())
    fig, ax = plt.subplots(figsize=(8, 5.3))
    for label, (curve_df, color) in curves.items():
        ax.plot(curve_df["ratio"], curve_df["mean"], marker="o", linewidth=2, label=label, color=color)
        ax.fill_between(curve_df["ratio"], curve_df["mean"] - curve_df["std"], curve_df["mean"] + curve_df["std"], color=color, alpha=0.12)
    ax.axhline(base, color="#344054", linestyle="--", linewidth=1.2, label="全量数据效能基线")
    ax.set_xlabel("训练数据保留比例")
    ax.set_ylabel("测试集离线模仿效能")
    ax.set_xlim(0.08, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def min_ratio_for_target(curve_df, target):
    ok = curve_df[curve_df["mean"] >= target]
    if ok.empty:
        return np.nan
    return float(ok.iloc[0]["ratio"])


def fig7_dual_track_curve(traj_df, stage4_summary_df, out_path, missing):
    if not stage4_summary_df.empty:
        curves = {
            "双轨联合策展": (stage4_curve(stage4_summary_df, "hybrid_Q_delta"), COLORS["dual"]),
            "单轨轨迹质量策展": (stage4_curve(stage4_summary_df, "high_Q_global"), COLORS["quality"]),
            "单轨场景价值策展": (stage4_curve(stage4_summary_df, "high_delta"), COLORS["value"]),
            "随机筛选": (stage4_curve(stage4_summary_df, "random"), COLORS["random"]),
        }
        curves = {label: item for label, item in curves.items() if not item[0].empty}
        if curves:
            base = stage4_base_score(stage4_summary_df)
            target = 0.95 * base if pd.notna(base) else np.nan
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            min_ratios = {}
            for label, (curve_df, color) in curves.items():
                ax.plot(curve_df["ratio"], curve_df["mean"], marker="o", linewidth=2, label=label, color=color)
                min_ratios[label] = min_ratio_for_target(curve_df, target) if pd.notna(target) else np.nan
            if pd.notna(base):
                ax.axhline(base, color="#344054", linestyle="--", linewidth=1.2, label="全量数据效能基线")
            if pd.notna(target):
                ax.axhline(target, color="#98A2B3", linestyle=":", linewidth=1.2, label="全量95%效能")
            best = min_ratios.get("双轨联合策展")
            if pd.notna(best):
                ax.text(0.50, 0.04, f"双轨约{best:.0%}数据达到全量95%效能", transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#D0D5DD"))
            ax.set_xlabel("训练数据保留比例")
            ax.set_ylabel("测试集离线模仿效能")
            ax.set_xlim(0.08, 1.02)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=9)
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            return min_ratios

    if traj_df.empty:
        missing.append("图7需要轨迹级质量、价值和效能标签")
        return None
    ratios = np.linspace(0.1, 1.0, 10)
    curves = {
        "双轨联合策展": (curation_curve(traj_df, "dual_score", ratios), COLORS["dual"]),
        "单轨轨迹质量策展": (curation_curve(traj_df, "Q_score", ratios), COLORS["quality"]),
        "单轨场景价值策展": (curation_curve(traj_df, "delta_score_pred", ratios), COLORS["value"]),
        "随机筛选": (curation_curve(traj_df, "__random__", ratios), COLORS["random"]),
    }
    base = float(traj_df["trajectory_imitation_score"].mean())
    target = 0.95 * base
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    min_ratios = {}
    for label, (curve_df, color) in curves.items():
        ax.plot(curve_df["ratio"], curve_df["mean"], marker="o", linewidth=2, label=label, color=color)
        min_ratios[label] = min_ratio_for_target(curve_df, target)
    ax.axhline(base, color="#344054", linestyle="--", linewidth=1.2, label="全量数据效能基线")
    ax.axhline(target, color="#98A2B3", linestyle=":", linewidth=1.2, label="全量95%效能")
    best = min_ratios.get("双轨联合策展")
    if pd.notna(best):
        ax.text(0.50, 0.04, f"双轨约{best:.0%}数据达到全量95%效能", transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#D0D5DD"))
    ax.set_xlabel("训练数据保留比例")
    ax.set_ylabel("测试集离线模仿效能")
    ax.set_xlim(0.08, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return min_ratios


def fig8_min_ratio(min_ratios, out_path, missing):
    if not min_ratios:
        missing.append("图8需要图7曲线计算得到的最小数据比例")
        return
    df = pd.DataFrame([
        {"策略": k, "最小数据比例": v}
        for k, v in min_ratios.items()
        if pd.notna(v)
    ])
    if df.empty:
        missing.append("图8没有任何策略达到全量95%效能")
        return
    df = df.sort_values("最小数据比例")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    palette = [COLORS["dual"] if "双轨" in s else COLORS["quality"] if "质量" in s else COLORS["value"] if "价值" in s else COLORS["random"] for s in df["策略"]]
    ax.bar(df["策略"], df["最小数据比例"], color=palette)
    for i, v in enumerate(df["最小数据比例"]):
        ax.text(i, v, f"{v:.0%}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("达到全量95%效能所需最小数据比例")
    ax.set_ylim(0, min(1.05, max(df["最小数据比例"]) + 0.15))
    ax.tick_params(axis="x", rotation=15)
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, label=s, markersize=8) for s, c in zip(df["策略"], palette)]
    ax.legend(handles=handles, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def performance_at_ratio(df, score_col, ratio):
    curve = curation_curve(df, score_col, np.array([ratio]))
    if curve.empty:
        return np.nan
    return float(curve.iloc[0]["mean"])


def stage4_performance_at_ratio(summary_df, strategy, ratio):
    curve_df = stage4_curve(summary_df, strategy)
    if curve_df.empty:
        return np.nan
    exact = curve_df[np.isclose(curve_df["ratio"], ratio)]
    if exact.empty:
        return np.nan
    return float(exact.iloc[0]["mean"])


def fig9_ablation(traj_df, stage4_summary_df, out_path, missing):
    if not stage4_summary_df.empty:
        ratio = 0.5
        dual = stage4_performance_at_ratio(stage4_summary_df, "hybrid_Q_delta", ratio)
        value_only = stage4_performance_at_ratio(stage4_summary_df, "high_delta", ratio)
        quality_only = stage4_performance_at_ratio(stage4_summary_df, "high_Q_global", ratio)
        rows = [
            ("移除质量维度", max(0.0, dual - value_only), COLORS["value"]),
            ("移除价值维度", max(0.0, dual - quality_only), COLORS["quality"]),
        ]
        rows = [(label, loss, color) for label, loss, color in rows if pd.notna(loss)]
        if rows:
            rows = sorted(rows, key=lambda x: x[1], reverse=True)
            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            x = np.arange(len(rows))
            values = [r[1] for r in rows]
            ax.bar(x, values, color=[r[2] for r in rows], width=0.55)
            ax.axhline(0, color="#344054", linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels([r[0] for r in rows])
            ax.set_ylabel("性能损失 ΔPerf")
            for i, value in enumerate(values):
                ax.text(i, value, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
            ax.legend(handles=[
                plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["value"], label="仅保留场景价值轨", markersize=9),
                plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["quality"], label="仅保留轨迹质量轨", markersize=9),
            ])
            ax.grid(axis="y", alpha=0.25)
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            return

    if traj_df.empty:
        missing.append("图9需要轨迹级质量、价值和效能标签")
        return
    ratio = 0.5
    rows = [
        ("双轨完整策展", performance_at_ratio(traj_df, "dual_score", ratio), COLORS["dual"]),
        ("移除质量维度", performance_at_ratio(traj_df, "delta_score_pred", ratio), COLORS["value"]),
        ("移除价值维度", performance_at_ratio(traj_df, "Q_score", ratio), COLORS["quality"]),
    ]
    base = rows[0][1]
    rows = sorted(rows, key=lambda x: base - x[1], reverse=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], color=[r[2] for r in rows])
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=10)
    ax.set_ylabel("测试集离线模仿效能")
    for i, (label, value, _) in enumerate(rows):
        loss = base - value
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
        if label != "双轨完整策展":
            ax.annotate(f"损失 {loss:.3f}", xy=(i, value), xytext=(i, base), ha="center", arrowprops=dict(arrowstyle="<->", color="#344054", lw=1))
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["dual"], label="双轨完整策展", markersize=9),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["value"], label="单轨场景价值", markersize=9),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["quality"], label="单轨轨迹质量", markersize=9),
    ])
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig10_heterogeneity(traj_df, stage4_per_scene_df, out_path, missing):
    if not stage4_per_scene_df.empty and {"strategy", "ratio", "test_scene", "scene_imitation_score"}.issubset(stage4_per_scene_df.columns):
        ratio = 0.5
        df = stage4_per_scene_df[np.isclose(stage4_per_scene_df["ratio"], ratio)].copy()
        df = df[df["strategy"].isin(["hybrid_Q_delta", "random"])]
        if not df.empty:
            scene_perf = (
                df.groupby(["test_scene", "strategy"], as_index=False)["scene_imitation_score"]
                .mean()
                .pivot(index="test_scene", columns="strategy", values="scene_imitation_score")
                .reset_index()
            )
            if {"hybrid_Q_delta", "random"}.issubset(scene_perf.columns):
                scene_perf["task_group"] = scene_perf["test_scene"].map(TASK_GROUPS).fillna("其他")
                scene_perf["效能改善幅度"] = scene_perf["hybrid_Q_delta"] - scene_perf["random"]
                out_df = scene_perf.groupby("task_group", as_index=False)["效能改善幅度"].mean().sort_values("task_group")
                fig, ax = plt.subplots(figsize=(6.8, 5.2))
                colors = [COLORS["quality"] if "视觉" in g else COLORS["value"] for g in out_df["task_group"]]
                ax.bar(out_df["task_group"], out_df["效能改善幅度"], color=colors, width=0.55)
                ax.axhline(0, color="#344054", linewidth=1)
                for i, value in enumerate(out_df["效能改善幅度"]):
                    ax.text(i, value, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
                ax.set_ylabel("双轨策展相对随机筛选的效能改善幅度")
                ax.legend(handles=[
                    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["quality"], label="视觉主导型", markersize=9),
                    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["value"], label="力控主导型", markersize=9),
                ])
                ax.grid(axis="y", alpha=0.25)
                fig.savefig(out_path, bbox_inches="tight")
                plt.close(fig)
                return

    if traj_df.empty:
        missing.append("图10需要轨迹级质量、价值和效能标签")
        return
    ratio = 0.5
    rows = []
    for group, group_df in traj_df.groupby("task_group"):
        dual = performance_at_ratio(group_df, "dual_score", ratio)
        random_curve = curation_curve(group_df, "__random__", np.array([ratio]), random_runs=200)
        random_perf = float(random_curve.iloc[0]["mean"]) if not random_curve.empty else np.nan
        rows.append({"任务分组": group, "效能改善幅度": dual - random_perf})
    df = pd.DataFrame(rows).sort_values("任务分组")
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    colors = [COLORS["quality"] if "视觉" in g else COLORS["value"] for g in df["任务分组"]]
    ax.bar(df["任务分组"], df["效能改善幅度"], color=colors, width=0.55)
    ax.axhline(0, color="#344054", linewidth=1)
    for i, v in enumerate(df["效能改善幅度"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    ax.set_ylabel("双轨策展相对随机筛选的效能改善幅度")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["quality"], label="视觉主导型", markersize=9),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["value"], label="力控主导型", markersize=9),
    ])
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    setup_matplotlib()
    out_root = Path("data") / "visualizations" / "new_checklist"
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    missing = []

    quality_df = load_quality_all(missing)
    scene_df = load_scene_quality_value(missing)
    delta_df = load_delta_summary(missing)
    stage4_summary_df = load_stage4_curation_summary()
    stage4_per_scene_df = load_stage4_per_scene_metrics()
    need_trajectory_fallback = stage4_summary_df.empty or stage4_per_scene_df.empty
    traj_df = load_trajectory_outputs(missing if need_trajectory_fallback else [])
    scene_df, delta_df = complete_scene_value_inputs(quality_df, scene_df, delta_df, missing)
    if not scene_df.empty:
        scene_df.to_csv(out_root / "scene_value_completed_for_fig3_5.csv", index=False, encoding="utf-8-sig")

    fig1_quality_radar(quality_df, fig_dir, missing)
    fig2_qscore_boxplot(quality_df, fig_dir / "fig2_qscore_boxplot.png", missing)
    fig3_scene_value_bar(delta_df, fig_dir / "fig3_empirical_marginal_value_bar.png", missing)
    fig4_quality_value_scatter(scene_df, fig_dir / "fig4_quality_value_correlation_scatter.png", missing)
    fig_quality_features_value_scatter(scene_df, fig_dir / "quality_features_marginal_value_scatter.png", missing)
    fig5_scene_curation_curve(delta_df, fig_dir / "fig5_scene_curation_curve.png", missing)
    fig6_trajectory_curation(traj_df, stage4_summary_df, fig_dir / "fig6_trajectory_curation_curve.png", missing)
    min_ratios = fig7_dual_track_curve(traj_df, stage4_summary_df, fig_dir / "fig7_dual_track_curation_curve.png", missing)
    fig8_min_ratio(min_ratios, fig_dir / "fig8_min_data_ratio_bar.png", missing)
    fig9_ablation(traj_df, stage4_summary_df, fig_dir / "fig9_ablation_performance_loss.png", missing)
    fig10_heterogeneity(traj_df, stage4_per_scene_df, fig_dir / "fig10_heterogeneity_group_bar.png", missing)

    report_path = out_root / "missing_inputs_report.txt"
    report_path.write_text("\n".join(dict.fromkeys(missing)) if missing else "All requested figures were generated.\n", encoding="utf-8")
    print(f"[Visualization] new checklist figures saved to: {fig_dir}")
    print(f"[Visualization] notes: {report_path}")


if __name__ == "__main__":
    main()
