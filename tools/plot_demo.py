"""绘制论文 demo（demo/nav_demo，图像在环闭环仿真）的图表与汇总表。

    python3 tools/plot_demo.py [results/demo/demo] [data/ahr_0.50m.map]
输出 <prefix>_summary.md、fig_demo_track.png、fig_demo_error.png。
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from experiments import C, INK2, style  # noqa: E402

VARIANTS = ("original", "original_recovery_sep", "original_window", "original_recovery_sep_window", "matched",
            "matched_recovery", "matched_recovery_sep", "matched_recovery_sep_window", "original_floor",
            "original_recovery_sep_floor", "matched_recovery_sep_floor", "original_afloor",
            "original_recovery_sep_afloor", "matched_recovery_sep_afloor", "original_sh", "original_fading",
            "original_huber", "original_studentt", "matched_sh", "matched_fading", "matched_huber", "matched_studentt")
LABEL = {"original": "原模型", "original_recovery_sep": "原模型 + 恢复（间隔 3 s）",
         "original_window": "原模型 + 窗口下限", "original_recovery_sep_window": "原模型 + 恢复 + 窗口下限",
         "matched": "匹配模型", "matched_recovery": "匹配模型 + 恢复（K=2）",
         "matched_recovery_sep": "匹配模型 + 恢复（间隔 3 s）", "matched_recovery_sep_window": "匹配模型 + 恢复 + 窗口下限",
         "original_floor": "原模型 + 协方差下限", "original_recovery_sep_floor": "原模型 + 恢复 + 协方差下限",
         "matched_recovery_sep_floor": "匹配模型 + 恢复 + 协方差下限",
         "original_afloor": "原模型 + 自适应下限", "original_recovery_sep_afloor": "原模型 + 恢复 + 自适应下限",
         "matched_recovery_sep_afloor": "匹配模型 + 恢复 + 自适应下限",
         "original_sh": "原模型 + Sage–Husa R", "original_fading": "原模型 + 自适应渐消",
         "original_huber": "原模型 + Huber", "original_studentt": "原模型 + Student-t VB",
         "matched_sh": "匹配模型 + Sage–Husa R", "matched_fading": "匹配模型 + 自适应渐消",
         "matched_huber": "匹配模型 + Huber", "matched_studentt": "匹配模型 + Student-t VB"}
COLOR = {"original": C[1], "original_recovery_sep": C[4], "original_window": C[4], "original_recovery_sep_window": C[2],
         "matched": C[3], "matched_recovery": C[2], "matched_recovery_sep": C[0], "matched_recovery_sep_window": C[3],
         "original_floor": C[5], "original_recovery_sep_floor": C[2], "matched_recovery_sep_floor": C[3],
         **{v: C[5] for v in ("original_afloor", "original_recovery_sep_afloor", "matched_recovery_sep_afloor",
                               "original_sh", "original_fading", "original_huber", "original_studentt", "matched_sh",
                               "matched_fading", "matched_huber", "matched_studentt")}}


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k not in ("variant", "stage"):
                try:
                    r[k] = float(v)
                except ValueError:
                    r[k] = np.nan
    return rows


def load_map(path, step=4):
    """读取 vision::GeoMap 二进制文件中的 PRE 影像（降采样）。"""
    with open(path, "rb") as f:
        hdr = f.read(48)
        w, h = np.frombuffer(hdr[8:16], np.int32)
        gsd = np.frombuffer(hdr[40:48], np.float64)[0]
        pre = np.frombuffer(f.read(int(w) * int(h)), np.uint8).reshape(h, w)
    return pre[::step, ::step], gsd * step


def summary(runs, fixes, prefix):
    by = defaultdict(list)
    for r in runs:
        by[r["variant"]].append(r)
    fx = defaultdict(list)
    for f in fixes:
        fx[f["variant"]].append(f)
    lines = ["| 配置 | RMS 中位数 | RMS 95% | 终值误差中位数 | 锁死运行比例 | 正确匹配被拒 | 错误匹配被接受 | 恢复次数 | 位置 ANEES |",
             "|---|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        if v not in by:
            continue
        rs = by[v]
        rms = np.array([r["rms_h"] for r in rs])
        fin = np.array([r["final_h"] for r in rs])
        ta = sum(r["true_accepted"] for r in rs)
        tr = sum(r["true_rejected"] for r in rs)
        oa = sum(r["outlier_accepted"] for r in rs)
        orj = sum(r["outlier_rejected"] for r in rs)
        lock = np.mean([r["lockout_episodes"] > 0 for r in rs]) * 100
        rec = sum(r["recoveries"] for r in rs)
        anees = np.mean([r["nees_pos_mean"] for r in rs]) / 3
        lines.append(f"| {LABEL[v]} | {np.median(rms):.1f} m | {np.percentile(rms, 95):.1f} m | {np.median(fin):.1f} m | "
                     f"{lock:.0f}% | {tr:.0f}/{ta + tr:.0f} | {oa:.0f}/{oa + orj:.0f} | {rec:.0f} | {anees:.1f} |")
    att = defaultdict(int)
    for r in runs:
        att[r["variant"]] += r["attempts"]
    lines.append("")
    lines.append("| 配置 | 景象匹配尝试帧 | 匹配器报告 | 报告率 |")
    lines.append("|---|---|---|---|")
    for v in VARIANTS:
        if v in by:
            n = len(fx[v])
            lines.append(f"| {LABEL[v]} | {att[v]:.0f} | {n} | {100 * n / max(att[v], 1):.0f}% |")
    text = "\n".join(lines) + "\n"
    with open(prefix + "_summary.md", "w") as f:
        f.write(text)
    print(text)


def plot_track(series, fixes, out, map_path, seed):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 7), gridspec_kw={"width_ratios": [1.25, 1]})
    vs = [v for v in ("original", "original_floor", "matched_recovery_sep")
          if any(r["variant"] == v for r in series)]
    truth = [r for r in series if r["variant"] == vs[0] and r["seed"] == seed]
    for ax in (ax0, ax1):
        ax.plot([r["e_true"] / 1000 for r in truth], [r["n_true"] / 1000 for r in truth], color=INK2, lw=1.0,
                label="真值航迹")
    for v in vs:
        s = [r for r in series if r["variant"] == v and r["seed"] == seed]
        for ax in (ax0, ax1):
            ax.plot([r["e_est"] / 1000 for r in s], [r["n_est"] / 1000 for r in s], color=COLOR[v], lw=1.2,
                    label=LABEL[v] + " 估计")
        f = [x for x in fixes if x["variant"] == v and x["seed"] == seed]
        acc = [x for x in f if x["accepted"]]
        rej = [x for x in f if not x["accepted"]]
        ax1.scatter([x["e_fix"] / 1000 for x in acc], [x["n_fix"] / 1000 for x in acc], s=14, color=COLOR[v],
                    edgecolors="white", linewidths=0.4, zorder=3, label=LABEL[v] + " 接受的修正")
        ax1.scatter([x["e_fix"] / 1000 for x in rej], [x["n_fix"] / 1000 for x in rej], s=22, color=COLOR[v],
                    marker="x", linewidths=1.2, zorder=3, label=LABEL[v] + " 拒绝的修正")
    if map_path and os.path.exists(map_path):
        img, g = load_map(map_path)
        ext = [0, img.shape[1] * g / 1000, -img.shape[0] * g / 1000, 0]
        for ax in (ax0, ax1):
            ax.imshow(img, cmap="gray", extent=ext, origin="upper", alpha=0.85, zorder=0)
        ax1.set_xlim(-1.2, ext[1] + 1.2)
        ax1.set_ylim(ext[2] - 0.3, 0.3)
    es = [r["e_true"] / 1000 for r in truth] + [r["e_est"] / 1000 for r in series if r["seed"] == seed]
    ns = [r["n_true"] / 1000 for r in truth] + [r["n_est"] / 1000 for r in series if r["seed"] == seed]
    ax0.set_xlim(min(es) - 0.5, max(es) + 0.5)
    ax0.set_ylim(min(ns) - 0.5, max(ns) + 0.5)
    ax0.set_aspect("equal")
    ax1.set_aspect("equal")
    ax0.set_xlabel("东 / km")
    ax0.set_ylabel("北 / km")
    ax1.set_xlabel("东 / km")
    ax0.set_title("全程航迹（种子 %d）" % seed, fontsize=10)
    ax1.set_title("影像条带（洪水前参考库）", fontsize=10)
    h, lab = ax1.get_legend_handles_labels()
    fig.legend(h, lab, fontsize=7, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(out, "fig_demo_track.png"))
    plt.close(fig)


def plot_error(series, fixes, out):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    # “匹配模型”与“匹配模型 + 恢复（间隔 3 s）”结果相同（恢复从未触发），只画后者
    if any(r["variant"].endswith("_window") for r in series):
        pick = ("original", "original_window", "original_floor", "matched_recovery_sep")
    elif any(r["variant"].endswith("_floor") for r in series):
        pick = ("original", "original_floor", "matched_recovery_sep", "matched_recovery_sep_floor")
    else:
        pick = ("original", "matched_recovery", "matched_recovery_sep")
    vs = [v for v in pick if any(r["variant"] == v for r in series)]
    for v in vs:
        rows = [r for r in series if r["variant"] == v]
        ts = sorted({r["t"] for r in rows})
        err = defaultdict(list)
        sig = defaultdict(list)
        for r in rows:
            err[r["t"]].append(r["err_h"])
            sig[r["t"]].append(r["sigma_ne"])
        med = np.array([np.median(err[t]) for t in ts])
        p90 = np.array([np.percentile(err[t], 90) for t in ts])
        s3 = np.array([3 * np.median(sig[t]) for t in ts])
        ax.plot(ts, med, color=COLOR[v], lw=1.5, label=LABEL[v] + " 误差中位数")
        ax.fill_between(ts, med, p90, color=COLOR[v], alpha=0.15, lw=0)
        ax.plot(ts, s3, color=COLOR[v], lw=0.9, ls="--", label=LABEL[v] + " 估计 3σ")
    # 飞越条带的时间段（有景象匹配尝试的时刻）
    tf = sorted({round(x["t"]) for x in fixes})
    if tf:
        start = prev = tf[0]
        for t in tf[1:] + [None]:
            if t is None or t - prev > 5:
                ax.axvspan(start, prev, color=INK2, alpha=0.08, lw=0)
                start = t
            prev = t if t is not None else prev
    ax.set_yscale("log")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("水平位置误差 / m")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_demo_error.png"))
    plt.close(fig)


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "results/demo/demo"
    map_path = sys.argv[2] if len(sys.argv) > 2 else "data/ahr_0.50m.map"
    out = os.path.dirname(prefix)
    style(plt)
    runs = load(prefix + "_runs.csv")
    series = load(prefix + "_series.csv")
    fixes = load(prefix + "_fixes.csv")
    summary(runs, fixes, prefix)
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else int(min(r["seed"] for r in runs))
    plot_track(series, fixes, out, map_path, seed)
    plot_error(series, fixes, out)
    print("图已写入", out)
