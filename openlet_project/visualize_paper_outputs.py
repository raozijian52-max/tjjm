import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MplPath

from config import CONFIG, ensure_dirs


SCENE_METADATA = {
    "S1": {"category": "基础抓取场景", "task": "single_item_grasp", "feature": "单物品恒定目标，基础抓取轨迹"},
    "S2": {"category": "工业物料场景", "task": "material_issuance_card_grabbing", "feature": "物料发放卡抓取，薄片目标"},
    "S3": {"category": "商超商品场景", "task": "FMCG_placing", "feature": "快消品放置，视觉主导"},
    "S4": {"category": "零件装配场景", "task": "parts_offline_into_box", "feature": "零件下线入箱，空间约束更强"},
    "S5": {"category": "工具包装场景", "task": "small_tool_packing", "feature": "小工具包装，多物体/多姿态"},
    "S6": {"category": "扩展场景", "task": "openlet_task_s6", "feature": "按论文五类场景定义补充"},
    "S7": {"category": "扩展场景", "task": "openlet_task_s7", "feature": "按论文五类场景定义补充"},
    "S8": {"category": "扩展场景", "task": "openlet_task_s8", "feature": "按论文五类场景定义补充"},
    "S9": {"category": "扩展场景", "task": "openlet_task_s9", "feature": "按论文五类场景定义补充"},
    "S10": {"category": "扩展场景", "task": "openlet_task_s10", "feature": "按论文五类场景定义补充"},
}


QUALITY_DIMS = [
    ("Q_completeness", "完整性"),
    ("Q_accuracy", "准确性"),
    ("Q_diversity", "多样性"),
    ("Q_consistency", "一致性"),
    ("Q_usability", "可用性"),
]


COLORS = {
    "raw": "#8A8F98",
    "quality": "#2F80ED",
    "causal": "#EB5757",
    "purple": "#7B61FF",
    "blue": "#2D9CDB",
    "red": "#EB5757",
    "green": "#27AE60",
    "orange": "#F2994A",
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
    plt.rcParams["figure.dpi"] = 140


def path_interim(name):
    return Path(CONFIG["interim_dir"]) / name


def path_processed(name):
    return Path(CONFIG["processed_dir"]) / name


def find_files_by_name(file_name):
    matches = []
    for path in Path(".").rglob(file_name):
        parts = set(path.parts)
        if ".git" in parts or "visualizations" in parts:
            continue
        matches.append(path)

    def priority(path):
        text = str(path).replace("\\", "/")
        if "/data/interim/" in text or "/data/processed/" in text:
            return 0
        if "stage3-stage5/processed" in text:
            return 1
        if "stage3-stage5" in text:
            return 2
        if "stage1-stage2" in text or "pca_check" in text:
            return 3
        return 4

    return sorted(matches, key=lambda p: (priority(p), len(str(p))))


def find_file_by_name(file_name):
    matches = find_files_by_name(file_name)
    return matches[0] if matches else None


def read_csv_if_exists(path, missing):
    path = Path(path)
    if not path.exists():
        found = find_file_by_name(path.name)
        if found is None:
            missing.append(str(path))
            return None
        path = found
    try:
        return pd.read_csv(path)
    except Exception as exc:
        missing.append(f"{path} (read failed: {exc})")
        return None


def read_json_if_exists(path, missing):
    path = Path(path)
    if not path.exists():
        found = find_file_by_name(path.name)
        if found is None:
            missing.append(str(path))
            return None
        path = found
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        missing.append(f"{path} (read failed: {exc})")
        return None


def discover_scene_ids():
    # 论文规划使用 S1-S10；仓库里若有 S11 等额外调试场景，这里默认不纳入图表。
    return [f"S{i}" for i in range(1, 11)]


def load_quality_all(scene_ids, missing):
    dfs = []
    for scene_id in scene_ids:
        prefix = scene_id.lower()
        candidates = [
            path_processed(f"{prefix}_stage2_quality_dataset.csv"),
            path_interim(f"{prefix}_quality_scores.csv"),
        ]
        df = None
        for path in candidates:
            found = path if path.exists() else find_file_by_name(path.name)
            if found is not None:
                df = read_csv_if_exists(found, missing)
                break
        if df is None:
            continue
        df = df.copy()
        if "scene_id" not in df.columns:
            df["scene_id"] = scene_id
        df["scene_id"] = scene_id
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def save_table_csv_png(df, csv_path, png_path, title=None, max_rows=30):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    show_df = df.head(max_rows).copy()
    fig_h = max(2.4, 0.38 * (len(show_df) + 2))
    fig_w = max(8, 1.7 * len(show_df.columns))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    table = ax.table(
        cellText=show_df.values,
        colLabels=show_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#EAF2FF")
            cell.set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, xy, wh, text, fc="#F7F9FC", ec="#344054", fontsize=10, weight="normal"):
    x, y = xy
    w, h = wh
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight)
    return rect


def arrow(ax, start, end, color="#344054", lw=1.5):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=4, shrinkB=4),
    )


def fig1_hdf5_structure(out_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("图1 OpenLET单条轨迹HDF5文件结构示意图", fontsize=16, weight="bold")

    nodes = [
        (0.05, 0.82, "机器人数据.h5", 0),
        (0.23, 0.90, "/cameras\n视觉传感器组", 1),
        (0.23, 0.72, "/joints\n关节运动数据组", 1),
        (0.23, 0.54, "/metadata.json\n任务元数据", 1),
        (0.23, 0.36, "/parameters\n相机参数", 1),
        (0.50, 0.94, "hand_left/color, depth\nN帧 + timestamp", 2),
        (0.50, 0.84, "hand_right/color, depth\nN帧 + timestamp", 2),
        (0.50, 0.74, "head/color, depth\nN帧 + timestamp", 2),
        (0.50, 0.62, "action/arm, head, leg, effector\nposition/velocity/timestamp", 2),
        (0.50, 0.50, "state/arm, head, leg, effector\n自由度D + timestamp", 2),
        (0.50, 0.36, "intrinsic/extrinsic\n分辨率、内外参", 2),
    ]
    colors = ["#EAF2FF", "#F2F4F7", "#FFFFFF"]
    boxes = {}
    for x, y, text, level in nodes:
        boxes[text] = draw_box(ax, (x, y), (0.22, 0.075), text, fc=colors[min(level, 2)], fontsize=9)

    root_end = (0.27, 0.86)
    for y in [0.94, 0.76, 0.58, 0.40]:
        arrow(ax, root_end, (0.23, y))
    for y in [0.975, 0.875, 0.775]:
        arrow(ax, (0.45, 0.94), (0.50, y))
    for y in [0.66, 0.54]:
        arrow(ax, (0.45, 0.76), (0.50, y))
    arrow(ax, (0.45, 0.40), (0.50, 0.40))

    ax.text(
        0.50,
        0.17,
        "末端节点标注：数组维度、帧数/自由度数量、纳秒时间戳；视觉帧保持原始频率，关节序列在阶段一对齐至100Hz。",
        ha="center",
        fontsize=10,
        color="#475467",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig2_dual_track(out_path):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("图2 双轨并行框架与研究闭环示意图", fontsize=16, weight="bold")

    draw_box(ax, (0.08, 0.68), (0.36, 0.20), "轨道一：通用描述性质量评估\n完整性 / 准确性 / 多样性 / 一致性 / 可用性", "#EAF2FF", "#2F80ED", 11, "bold")
    draw_box(ax, (0.56, 0.68), (0.36, 0.20), "轨道二：因果效能价值量化\n留一场景法 → 元模型预测", "#FFF1F1", "#EB5757", 11, "bold")
    draw_box(ax, (0.32, 0.46), (0.36, 0.12), "双轨融合特征\n描述性质量特征 + 因果价值特征", "#F7F9FC", "#344054", 11, "bold")
    draw_box(ax, (0.10, 0.22), (0.34, 0.12), "阶段四：连续效能预测建模\nXGBoost", "#F2F4F7", "#667085", 10, "bold")
    draw_box(ax, (0.56, 0.22), (0.34, 0.12), "阶段五：因果推断与验证\nPSM + 消融实验 + 异质性分析", "#F2F4F7", "#667085", 10, "bold")

    arrow(ax, (0.26, 0.68), (0.42, 0.58))
    arrow(ax, (0.74, 0.68), (0.58, 0.58))
    arrow(ax, (0.43, 0.46), (0.28, 0.34))
    arrow(ax, (0.57, 0.46), (0.73, 0.34))
    arrow(ax, (0.44, 0.28), (0.56, 0.28))
    ax.text(0.5, 0.10, "{ 描述统计 → 预测建模 → 因果推断 }", ha="center", fontsize=15, weight="bold")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig3_configs(out_path):
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("图3 三组对照配置（Config A/B/C）特征构成示意图", fontsize=16, weight="bold")

    ys = [0.74, 0.50, 0.26]
    configs = [
        ("Config A", [("原始特征\n234维", 0.42, COLORS["raw"])]),
        ("Config B", [("原始特征\n234维", 0.42, COLORS["raw"]), ("描述性质量特征\nQ_score + 五维，6维", 0.20, COLORS["quality"])]),
        ("Config C", [("原始特征\n234维", 0.42, COLORS["raw"]), ("描述性质量特征\n6维", 0.20, COLORS["quality"]), ("因果效能价值特征\nΔŶ_pred，1维", 0.18, COLORS["causal"])]),
    ]
    for y, (name, parts) in zip(ys, configs):
        ax.text(0.06, y + 0.04, name, fontsize=12, weight="bold", ha="left")
        x = 0.20
        for label, width, color in parts:
            draw_box(ax, (x, y), (width, 0.10), label, fc=color, ec="white", fontsize=9, weight="bold")
            x += width + 0.012
        arrow(ax, (x, y + 0.05), (0.88, y + 0.05))
    draw_box(ax, (0.88, 0.30), (0.10, 0.52), "XGBoost\n连续效能预测", "#F7F9FC", "#344054", 10, "bold")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_table1(scene_ids, quality_df, out_csv, out_png):
    rows = []
    for scene_id in scene_ids:
        meta = SCENE_METADATA.get(scene_id, {})
        n = 0
        if not quality_df.empty and "scene_id" in quality_df.columns:
            n = int((quality_df["scene_id"].astype(str).str.upper() == scene_id).sum())
        rows.append({
            "场景编号": scene_id.lower(),
            "场景类别": meta.get("category", "待补充"),
            "任务名称": meta.get("task", "待补充"),
            "轨迹数量": n,
            "核心特性": meta.get("feature", "待补充"),
        })
    df = pd.DataFrame(rows)
    save_table_csv_png(df, out_csv, out_png, title="表1 本研究所用数据集基本信息")


def fig4_quality_box(quality_df, out_path, missing):
    if quality_df.empty or "Q_score" not in quality_df.columns:
        missing.append("图4需要 s*_stage2_quality_dataset.csv 或 s*_quality_scores.csv，且包含 Q_score")
        return
    df = quality_df.dropna(subset=["Q_score"]).copy()
    order = sorted(df["scene_id"].unique(), key=lambda x: int(str(x)[1:]) if str(x)[1:].isdigit() else 999)
    data = [df.loc[df["scene_id"] == s, "Q_score"].values for s in order]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data, patch_artist=True, tick_labels=[s.lower() for s in order], showfliers=True)
    cmap = plt.get_cmap("tab10", len(order))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i))
        patch.set_alpha(0.45)
    rng = np.random.default_rng(42)
    for i, values in enumerate(data, start=1):
        x = rng.normal(i, 0.035, size=len(values))
        ax.scatter(x, values, s=18, alpha=0.45, color=cmap(i - 1), edgecolors="none")
    ax.set_xlabel("场景")
    ax.set_ylabel("Q_score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("图4 十场景综合质量指数（Q_score）对比箱线图", fontsize=15, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig5_quality_radar(quality_df, out_path, missing):
    if quality_df.empty or not all(col in quality_df.columns for col, _ in QUALITY_DIMS):
        missing.append("图5需要质量分数表包含五维度列 Q_completeness/Q_accuracy/Q_diversity/Q_consistency/Q_usability")
        return
    grouped = quality_df.groupby("scene_id")[[col for col, _ in QUALITY_DIMS]].mean().reset_index()
    grouped = grouped.sort_values("scene_id")
    labels = [label for _, label in QUALITY_DIMS]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, subplot_kw=dict(polar=True), figsize=(13, 6))
    chunks = [grouped.iloc[:5], grouped.iloc[5:10]]
    for ax, chunk, title in zip(axes, chunks, ["S1-S5", "S6-S10"]):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        for _, row in chunk.iterrows():
            values = [float(row[col]) for col, _ in QUALITY_DIMS]
            values += values[:1]
            ax.plot(angles, values, linewidth=1.8, label=str(row["scene_id"]).lower())
            ax.fill(angles, values, alpha=0.06)
        ax.set_title(title, y=1.08, weight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=8)
    fig.suptitle("图5 十场景五维度质量雷达图", fontsize=15, weight="bold")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def stars(p):
    if p is None or pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def make_table2(scene_ids, out_csv, out_png, missing):
    rows = []
    found = False
    for scene_id in scene_ids:
        prefix = scene_id.lower()
        info = read_json_if_exists(path_interim(f"{prefix}_quality_pca_check.json"), [])
        if info is None:
            rows.append({"场景编号": prefix, "第一主成分相关系数": "NA", "累计主成分相关系数": "NA"})
            continue
        found = True
        pc1_corr = info.get("spearman_corr_with_entropy_q")
        pc1_p = info.get("spearman_pvalue_with_entropy_q")
        combined_corr = info.get("combined_selected_components_spearman_corr_with_entropy_q")
        combined_p = info.get("combined_selected_components_spearman_pvalue_with_entropy_q")
        rows.append({
            "场景编号": prefix,
            "第一主成分相关系数": f"{pc1_corr:.3f}{stars(pc1_p)}" if pc1_corr is not None else "NA",
            "累计主成分相关系数": f"{combined_corr:.3f}{stars(combined_p)}" if combined_corr is not None else "NA",
        })
    if not found:
        missing.append("表2需要 s*_quality_pca_check.json")
    df = pd.DataFrame(rows)
    save_table_csv_png(df, out_csv, out_png, title="表2 PCA得分与熵权法得分的Spearman相关性")


def fig6_value_comparison(out_path, missing):
    pred = read_csv_if_exists(path_interim("stage3_scene_value_prediction.csv"), missing)
    if pred is None:
        missing.append("图6需要 stage3_scene_value_prediction.csv")
        return
    scene_col = "scene_id"
    emp_col = "delta_score_mean" if "delta_score_mean" in pred.columns else "delta_emp"
    pred_col = "delta_score_pred" if "delta_score_pred" in pred.columns else "delta_pred"
    if emp_col not in pred.columns or pred_col not in pred.columns:
        missing.append("图6需要经验值列 delta_score_mean 和预测值列 delta_score_pred")
        return
    pred = pred.sort_values(scene_col)
    x = np.arange(len(pred))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, pred[emp_col], width, label="ΔY_emp", color="#2D9CDB")
    ax.bar(x + width / 2, pred[pred_col], width, label="ΔŶ_pred", color="#F2994A")
    ax.axhline(0, color="#344054", lw=1)
    for xi, v in zip(x - width / 2, pred[emp_col]):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    for xi, v in zip(x + width / 2, pred[pred_col]):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(pred[scene_col].astype(str).str.lower())
    ax.set_ylabel("效能影响 / 预测价值")
    ax.set_title("图6 留一场景法经验效能影响与元模型预测效能价值对比图", fontsize=15, weight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def select_stage4_rows(df):
    work = df.copy()
    if "model" in work.columns and (work["model"] == "xgboost").any():
        work = work[work["model"] == "xgboost"]
    if "protocol" in work.columns and (work["protocol"] == "random_kfold").any():
        work = work[work["protocol"] == "random_kfold"]
    return work


def make_table3_and_fig7(table_csv, table_png, fig_png, missing):
    comp = read_csv_if_exists(path_interim("stage4_config_comparison.csv"), missing)
    if comp is None:
        missing.append("表3/图7需要 stage4_config_comparison.csv")
        return
    comp = select_stage4_rows(comp)
    name_map = {"config_A": "Config A", "config_B": "Config B", "config_C": "Config C"}
    rows = []
    for cfg in ["config_A", "config_B", "config_C"]:
        row = comp[comp["config"] == cfg]
        if row.empty:
            continue
        row = row.iloc[0]
        rows.append({
            "配置编号": name_map[cfg],
            "RMSE": row.get("rmse_mean", np.nan),
            "MAE": row.get("mae_mean", np.nan),
            "R2": row.get("r2_mean", np.nan),
        })
    table_df = pd.DataFrame(rows)
    save_table_csv_png(table_df, table_csv, table_png, title="表3 三组对照配置效能预测性能对比")

    if table_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [COLORS["raw"], COLORS["quality"], COLORS["purple"]][:len(table_df)]
    ax.bar(table_df["配置编号"], table_df["RMSE"], color=colors)
    for i, v in enumerate(table_df["RMSE"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("RMSE")
    ax.set_title("图7 三组配置效能预测精度对比条形图", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(fig_png, bbox_inches="tight")
    plt.close(fig)


def feature_type(feature):
    name = str(feature)
    if name in {"Q_score", "Q_completeness", "Q_accuracy", "Q_diversity", "Q_consistency", "Q_usability"}:
        return "描述性质量特征"
    if "delta" in name.lower() or "value" in name.lower() or "pred" in name.lower():
        return "因果价值特征"
    return "原始特征"


def fig8_importance(out_path, missing):
    imp = read_csv_if_exists(path_interim("stage4_feature_importance.csv"), missing)
    if imp is None or imp.empty:
        missing.append("图8需要 stage4_feature_importance.csv")
        return
    work = imp.copy()
    if "config" in work.columns and (work["config"] == "config_C").any():
        work = work[work["config"] == "config_C"]
    if "model" in work.columns and (work["model"] == "xgboost").any():
        work = work[work["model"] == "xgboost"]
    work = work.groupby("feature", as_index=False)["importance"].mean()
    work = work.sort_values("importance", ascending=False).head(20)
    work["type"] = work["feature"].map(feature_type)
    palette = {"原始特征": COLORS["raw"], "描述性质量特征": COLORS["quality"], "因果价值特征": COLORS["causal"]}
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(work))[::-1]
    ax.barh(y, work["importance"], color=[palette[t] for t in work["type"]])
    ax.set_yticks(y)
    ax.set_yticklabels(work["feature"])
    ax.set_xlabel("平均特征重要性 / |SHAP|替代值")
    ax.set_title("图8 SHAP特征重要性排序图", fontsize=15, weight="bold")
    handles = [patches.Patch(color=c, label=k) for k, c in palette.items()]
    ax.legend(handles=handles, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def simple_kde(values, xs):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.zeros_like(xs)
    std = np.std(values)
    bw = 1.06 * std * (len(values) ** (-1 / 5)) if std > 1e-12 else 0.05
    bw = max(bw, 0.03)
    density = np.exp(-0.5 * ((xs[:, None] - values[None, :]) / bw) ** 2).sum(axis=1)
    density /= len(values) * bw * math.sqrt(2 * math.pi)
    return density


def fig9_psm(out_path, missing):
    causal = read_csv_if_exists(path_interim("stage5_causal_dataset.csv"), missing)
    balance = read_csv_if_exists(path_interim("stage5_balance_summary.csv"), missing)
    weights = read_csv_if_exists(path_interim("stage5_ipw_weights.csv"), [])
    pairs = read_csv_if_exists(path_interim("stage5_matched_pairs.csv"), [])
    if balance is None:
        missing.append("图9需要 stage5_balance_summary.csv")
        return

    ps_df = None
    tcol = None
    ps_col = None
    if weights is not None and {"treat", "propensity"}.issubset(weights.columns):
        ps_df = weights.copy()
        if "treatment" in ps_df.columns and (ps_df["treatment"] == "treat_high_quality").any():
            ps_df = ps_df[ps_df["treatment"] == "treat_high_quality"].copy()
        if "analysis_variant" in ps_df.columns and (ps_df["analysis_variant"] == "no_scene").any():
            ps_df = ps_df[ps_df["analysis_variant"] == "no_scene"].copy()
        tcol = "treat"
        ps_col = "propensity"
    elif causal is not None:
        treatment_cols = [c for c in causal.columns if c.startswith("treat_") or c in ["high_quality_treatment", "treatment"]]
        ps_col = "_propensity" if "_propensity" in causal.columns else None
        if ps_col is None:
            ps_candidates = [c for c in causal.columns if "propensity" in c.lower()]
            ps_col = ps_candidates[0] if ps_candidates else None
        if treatment_cols and ps_col is not None:
            ps_df = causal.copy()
            tcol = treatment_cols[0]

    if ps_df is None or tcol is None or ps_col is None:
        missing.append("图9需要 stage5_ipw_weights.csv 中的 treat/propensity，或 causal_dataset 中的 treatment/propensity")
        return

    xs = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw={"width_ratios": [1.2, 1]})

    matched_ps_df = ps_df
    if pairs is not None and {
        "treated_propensity",
        "control_propensity",
    }.issubset(pairs.columns):
        if "treatment" in pairs.columns and (pairs["treatment"] == "treat_high_quality").any():
            pairs = pairs[pairs["treatment"] == "treat_high_quality"].copy()
        if "analysis_variant" in pairs.columns and (pairs["analysis_variant"] == "no_scene").any():
            pairs = pairs[pairs["analysis_variant"] == "no_scene"].copy()
        matched_ps_df = pd.concat([
            pd.DataFrame({tcol: 1, ps_col: pairs["treated_propensity"]}),
            pd.DataFrame({tcol: 0, ps_col: pairs["control_propensity"]}),
        ], ignore_index=True)

    for ax, title, df in [
        (axes[0, 0], "匹配前", ps_df),
        (axes[1, 0], "匹配后", matched_ps_df),
    ]:
        for val, label, color in [(1, "高质量组", COLORS["quality"]), (0, "低质量组", COLORS["raw"])]:
            arr = df.loc[df[tcol] == val, ps_col].values
            ax.plot(xs, simple_kde(arr, xs), label=label, color=color, lw=2)
        ax.set_title(title, weight="bold")
        ax.set_xlabel("倾向得分")
        ax.set_ylabel("密度")
        ax.legend()
        ax.grid(alpha=0.2)

    bal = balance.copy()
    if "method" in bal.columns and (bal["method"] == "psm_att").any():
        bal = bal[bal["method"] == "psm_att"]
    bal = bal.dropna(subset=["abs_smd_before", "abs_smd_after"]).head(20)
    y = np.arange(len(bal))
    axes[0, 1].hlines(y, bal["abs_smd_after"], bal["abs_smd_before"], color="#D0D5DD")
    axes[0, 1].scatter(bal["abs_smd_before"], y, color=COLORS["red"], label="匹配前")
    axes[0, 1].scatter(bal["abs_smd_after"], y, color=COLORS["blue"], label="匹配后")
    axes[0, 1].axvline(0.1, color="#344054", linestyle="--", lw=1)
    axes[0, 1].set_yticks(y)
    axes[0, 1].set_yticklabels(bal["covariate"] if "covariate" in bal.columns else y, fontsize=8)
    axes[0, 1].set_title("SMD平衡性", weight="bold")
    axes[0, 1].legend()
    axes[1, 1].axis("off")
    fig.suptitle("图9 PSM匹配前后倾向得分分布对比图", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_table4(out_csv, out_png, missing):
    eff = read_csv_if_exists(path_interim("stage5_effect_estimates.csv"), missing)
    if eff is None:
        missing.append("表4需要 stage5_effect_estimates.csv")
        return
    work = eff.copy()
    if "effect_type" in work.columns and (work["effect_type"] == "ATT").any():
        work = work[work["effect_type"] == "ATT"]
    rows = []
    for _, row in work.iterrows():
        rows.append({
            "方法": row.get("method", "最近邻匹配"),
            "ATT估计值": row.get("estimate", np.nan),
            "Bootstrap标准误": row.get("bootstrap_se", row.get("std_error", np.nan)),
            "95%置信区间下限": row.get("ci_lower", row.get("bootstrap_ci_lower", np.nan)),
            "95%置信区间上限": row.get("ci_upper", row.get("bootstrap_ci_upper", np.nan)),
        })
    save_table_csv_png(pd.DataFrame(rows), out_csv, out_png, title="表4 PSM平均处理效应（ATT）估计结果")


def fig10_ablation(out_path, missing):
    comp = read_csv_if_exists(path_interim("stage4_config_comparison.csv"), missing)
    work = pd.DataFrame()
    x_label = "性能损失 ΔPerf（RMSE增加）"

    if comp is not None:
        # Prefer a real ablation table if the config names already encode dropped dimensions.
        work = comp[comp["config"].astype(str).str.contains("drop|remove|ablation", case=False, regex=True)]
        if not work.empty:
            base = comp[comp["config"].astype(str).str.lower().isin(["config_c", "config c"])]
            base_rmse = float(base["rmse_mean"].iloc[0]) if not base.empty else float(work["rmse_mean"].min())
            work = work.copy()
            work["dimension"] = work["config"]
            work["loss"] = work["rmse_mean"] - base_rmse

    if work.empty:
        # Current repository does not include a dedicated ablation table. As a data-backed fallback,
        # use Stage5 dimension-level ATT on outcome_mse as the dimension performance-change estimate.
        effects = read_csv_if_exists(path_interim("stage5_effect_estimates.csv"), [])
        if effects is None:
            missing.append("图10需要消融实验结果，或 stage5_effect_estimates.csv 中的维度处理效应")
            return

        dimension_map = {
            "treat_high_completeness": "完整性",
            "treat_high_accuracy": "准确性",
            "treat_high_diversity": "多样性",
            "treat_high_consistency": "一致性",
            "treat_high_usability": "可用性",
        }
        work = effects[
            effects["treatment"].isin(dimension_map.keys())
            & (effects["method"] == "psm_att")
            & (effects["analysis_variant"] == "no_scene")
            & (effects["outcome"] == "outcome_mse")
        ].copy()
        if work.empty:
            missing.append("图10需要维度级 psm_att / outcome_mse 结果")
            return
        work["dimension"] = work["treatment"].map(dimension_map)
        work["loss"] = work["estimate"]
        x_label = "维度处理效应 ΔPerf（ATT on outcome_mse）"

        missing_dims = sorted(set(dimension_map.values()) - set(work["dimension"]))
        if missing_dims:
            missing.append("图10维度处理效应缺少：" + "、".join(missing_dims))

    work = work.sort_values("loss", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [COLORS["red"] if v > 0 else COLORS["blue"] for v in work["loss"]]
    ax.barh(work["dimension"], work["loss"], color=colors)
    ax.axvline(0, color="#344054", lw=1)
    for i, v in enumerate(work["loss"]):
        ax.text(v, i, f"{v:.4f}", ha="left" if v >= 0 else "right", va="center", fontsize=9)
    ax.set_xlabel(x_label)
    ax.set_title("图10 消融实验性能损失柱状图", fontsize=15, weight="bold")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig11_heterogeneity(out_path, missing):
    master = read_csv_if_exists(path_interim("stage4_modeling_master_table.csv"), missing)
    causal = read_csv_if_exists(path_interim("stage5_causal_dataset.csv"), [])
    if master is None:
        master = causal
    elif causal is not None and ("Q_score" not in master.columns or "outcome_score" not in master.columns):
        master = causal

    if master is None:
        missing.append("图11需要 stage4_modeling_master_table.csv 或 stage5_causal_dataset.csv")
        return
    q_col = "Q_score"
    y_col = "outcome_score" if "outcome_score" in master.columns else None
    if y_col is None:
        candidates = [c for c in master.columns if c.lower() in ["y", "target", "continuous_y"]]
        y_col = candidates[0] if candidates else None
    if q_col not in master.columns or y_col is None:
        missing.append("图11需要 Q_score 和连续效能得分列")
        return
    df = master.dropna(subset=[q_col, y_col]).copy()
    df["task_group"] = np.where(df["scene_id"].astype(str).isin(["S2", "S4", "S5"]), "力/触觉主导型", "视觉主导型")
    fig, ax = plt.subplots(figsize=(8, 6))
    for group, color in [("力/触觉主导型", COLORS["red"]), ("视觉主导型", COLORS["blue"])]:
        part = df[df["task_group"] == group]
        ax.scatter(part[q_col], part[y_col], label=group, alpha=0.65, color=color)
        if len(part) >= 2:
            coef = np.polyfit(part[q_col], part[y_col], 1)
            xs = np.linspace(part[q_col].min(), part[q_col].max(), 50)
            ax.plot(xs, coef[0] * xs + coef[1], color=color, lw=2)
    ax.set_xlabel("Q_score")
    ax.set_ylabel("连续效能得分 Y")
    ax.set_title("图11 异质性分析：分任务类型质量-效能关系散点图", fontsize=14, weight="bold")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig12_application_flow(out_path):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("图12 双轨框架应用价值流程图", fontsize=16, weight="bold")
    steps = [
        ("新采集的原始轨迹", "#FFFFFF"),
        ("多模态特征提取", "#F2F4F7"),
        ("轨道一：描述性质量评估\n输出 Q_score 及五维分量", "#EAF2FF"),
        ("轨道二：效能价值预测\n元模型输出 ΔŶ_pred", "#FFF1F1"),
        ("阶段四：连续效能预测\nXGBoost主模型输出 Ŷ", "#F7F9FC"),
        ("甄别决策\nŶ ≥ τ：高价值数据，纳入训练\nŶ < τ：低价值数据，建议剔除", "#ECFDF3"),
    ]
    y = 0.84
    prev = None
    for text, color in steps:
        draw_box(ax, (0.28, y), (0.44, 0.095), text, fc=color, ec="#344054", fontsize=10, weight="bold")
        if prev is not None:
            arrow(ax, (0.50, prev), (0.50, y + 0.095))
        prev = y
        y -= 0.14
    ax.text(0.78, 0.61, "阶段二", fontsize=11, weight="bold", color=COLORS["quality"])
    ax.text(0.78, 0.47, "阶段三", fontsize=11, weight="bold", color=COLORS["causal"])
    ax.text(0.78, 0.33, "阶段四", fontsize=11, weight="bold", color="#344054")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    setup_matplotlib()
    ensure_dirs()

    out_root = Path("data") / "visualizations"
    fig_dir = out_root / "figures"
    table_dir = out_root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    missing = []

    scene_ids = discover_scene_ids()
    quality_df = load_quality_all(scene_ids, missing)

    make_table1(scene_ids, quality_df, table_dir / "table1_dataset_basic_info.csv", table_dir / "table1_dataset_basic_info.png")
    fig1_hdf5_structure(fig_dir / "fig1_hdf5_structure.png")
    fig2_dual_track(fig_dir / "fig2_dual_track_framework.png")
    fig3_configs(fig_dir / "fig3_config_features.png")
    fig4_quality_box(quality_df, fig_dir / "fig4_qscore_boxplot.png", missing)
    fig5_quality_radar(quality_df, fig_dir / "fig5_quality_radar.png", missing)
    make_table2(scene_ids, table_dir / "table2_pca_spearman.csv", table_dir / "table2_pca_spearman.png", missing)
    fig6_value_comparison(fig_dir / "fig6_empirical_vs_predicted_value.png", missing)
    make_table3_and_fig7(
        table_dir / "table3_prediction_performance.csv",
        table_dir / "table3_prediction_performance.png",
        fig_dir / "fig7_prediction_performance_bar.png",
        missing,
    )
    fig8_importance(fig_dir / "fig8_feature_importance.png", missing)
    fig9_psm(fig_dir / "fig9_psm_balance.png", missing)
    make_table4(table_dir / "table4_psm_att.csv", table_dir / "table4_psm_att.png", missing)
    fig10_ablation(fig_dir / "fig10_ablation_loss.png", missing)
    fig11_heterogeneity(fig_dir / "fig11_heterogeneity_scatter.png", missing)
    fig12_application_flow(fig_dir / "fig12_application_flow.png")

    report_path = out_root / "missing_inputs_report.txt"
    if missing:
        report_path.write_text("\n".join(dict.fromkeys(missing)), encoding="utf-8")
    else:
        report_path.write_text("All requested figures/tables were generated from available inputs.\n", encoding="utf-8")

    print(f"[Visualization] outputs saved to: {out_root}")
    print(f"[Visualization] missing/input notes: {report_path}")


if __name__ == "__main__":
    main()
