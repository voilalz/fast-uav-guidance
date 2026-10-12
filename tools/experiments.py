#!/usr/bin/env python3
"""门限锁死（gate lockout）对比实验：批量运行 nav_mc，汇总指标并绘图。

用法（先构建 nav_mc）：
    cmake -S . -B build && cmake --build build -j
    python3 tools/experiments.py --bin build/nav/nav_mc --runs 100 --out results

输出：
    <out>/summary.csv、<out>/summary.md   每组配置一行汇总
    <out>/fig_*.png                       论文用图
    <out>/raw/                            nav_mc 原始 CSV（不入库）
依赖：numpy、matplotlib
"""
import argparse
import csv
import os
import subprocess
import sys

import numpy as np

REAL = "profile=realistic"
GM = "filter_vo_scale_tau=120 filter_vo_scale_sigma=0.005"

# (实验组, 名称, 参数)
EXPERIMENTS = [
    # E1：理想误差模型下，时间相关误差未建模 → 锁死；锁死恢复的作用
    ("E1", "完整建模", ""),
    ("E1", "磁偏差未建模", "est_mag_bias=0"),
    ("E1", "VO 尺度未建模", "est_vo_scale=0"),
    ("E1", "磁偏差+尺度未建模", "est_mag_bias=0 est_vo_scale=0"),
    ("E1", "磁偏差未建模+恢复", "est_mag_bias=0 recovery=1"),
    ("E1", "VO 尺度未建模+恢复", "est_vo_scale=0 recovery=1"),
    ("E1", "磁偏差+尺度未建模+恢复", "est_mag_bias=0 est_vo_scale=0 recovery=1"),
    # E2：更真实误差模型（时变 VO 尺度、零偏温漂、近距离错误匹配、真值子步）
    ("E2", "随机游走尺度模型", REAL),
    ("E2", "随机游走+恢复", f"{REAL} recovery=1"),
    ("E2", "马尔可夫尺度模型", f"{REAL} {GM}"),
    ("E2", "马尔可夫+恢复", f"{REAL} {GM} recovery=1"),
    ("E2", "马尔可夫+恢复+匹配航向", f"{REAL} {GM} recovery=1 fix_yaw=1"),
    # E3：VO 尺度时变幅度（滤波器按随机游走建模）对锁死的影响
    *[("E3", f"尺度GM σ={s}{'+恢复' if r else ''}", f"vo_scale_gm_sigma={s}{' recovery=1' if r else ''}")
      for s in (0.001, 0.002, 0.003, 0.005, 0.01) for r in (0, 1)],
    # E4：时延量测参考状态（原实现 vs 叠加之后的修正量），时延 0.25 / 1.0 s
    *[("E4", f"时延{d}s {'叠加修正' if c else '原实现'}", f"{REAL} {GM} fix_delay={d} delay_corr={c}")
      for d in (0.25, 1.0) for c in (0, 1)],
    # E5：近距离错误匹配比例 —— 恢复机制的误接风险
    *[("E5", f"错误匹配 {int(p*100)}%{'+恢复' if r else ''}",
       f"{REAL} {GM} outlier_rate={p}{' recovery=1' if r else ''}")
      for p in (0.05, 0.1, 0.2, 0.3) for r in (0, 1)],
    # E6：景象匹配间隔
    *[("E6", f"间隔 {int(i)} s", f"{REAL} {GM} recovery=1 fix_interval={i}") for i in (15, 30, 60, 90)],
]


def slug(group, idx):
    return f"{group}_{idx:02d}"


def run_all(bin_path, runs, out):
    raw = os.path.join(out, "raw")
    os.makedirs(raw, exist_ok=True)
    rows = []
    counters = {}
    for group, name, params in EXPERIMENTS:
        counters[group] = counters.get(group, 0) + 1
        tag = slug(group, counters[group])
        prefix = os.path.join(raw, tag)
        cmd = [bin_path, f"runs={runs}", f"out={prefix}"] + params.split()
        print("运行", tag, name, flush=True)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        rows.append(summarize(group, tag, name, params, prefix))
    return rows


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize(group, tag, name, params, prefix):
    r = read_csv(prefix + "_runs.csv")
    rms = np.array([float(x["rms_h"]) for x in r])
    mx = np.array([float(x["max_h"]) for x in r])
    ta = sum(int(x["true_accepted"]) for x in r)
    tr = sum(int(x["true_rejected"]) for x in r)
    oa = sum(int(x["outlier_accepted"]) for x in r)
    orj = sum(int(x["outlier_rejected"]) for x in r)
    lock = np.mean([int(x["lockout_episodes"]) > 0 for x in r])
    longest = np.array([float(x["lockout_longest_s"]) for x in r])
    rec = np.mean([int(x["recoveries"]) for x in r])
    s = read_csv(prefix + "_series.csv")
    anees = np.array([float(x["anees_pos"]) for x in s])
    return {
        "group": group, "tag": tag, "name": name, "params": params, "runs": len(r),
        "rms_median": np.median(rms), "rms_p95": np.percentile(rms, 95), "max_p95": np.percentile(mx, 95),
        "false_reject_pct": 100.0 * tr / max(1, ta + tr),
        "false_accept_pct": 100.0 * oa / max(1, oa + orj),
        "lockout_run_pct": 100.0 * lock,
        "lockout_longest_p95_s": np.percentile(longest, 95),
        "recoveries_mean": rec,
        # 时间平均 ANEES 的中位数（对少数尖峰稳健）；理想值 3
        "anees_pos_median": np.median(anees),
    }


def write_summary(rows, out):
    keys = list(rows[0].keys())
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("| 组 | 配置 | RMS 中位数 m | RMS 95% m | 最大误差 95% m | 正确修正错拒 % | 错误匹配误接 % "
                "| 发生锁死的运行 % | 锁死最长 95% s | 恢复次数 | 位置 ANEES 中位数 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['group']} | {r['name']} | {r['rms_median']:.1f} | {r['rms_p95']:.1f} | {r['max_p95']:.0f} "
                    f"| {r['false_reject_pct']:.1f} | {r['false_accept_pct']:.1f} | {r['lockout_run_pct']:.0f} "
                    f"| {r['lockout_longest_p95_s']:.0f} | {r['recoveries_mean']:.1f} | {r['anees_pos_median']:.1f} |\n")


# 参考调色板（categorical 固定顺序）
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(plt):
    for fam in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "SimHei", "Source Han Sans SC"):
        from matplotlib import font_manager
        if any(fam in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams.update({
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2, "figure.dpi": 150, "axes.unicode_minus": False,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False,
    })


def series(out, tag):
    s = read_csv(os.path.join(out, "raw", tag + "_series.csv"))
    t = np.array([float(x["t"]) for x in s])
    return t, {k: np.array([float(x[k]) for x in s]) for k in s[0] if k != "t"}


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    by = {(r["group"], r["name"]): r for r in rows}

    # 图 1：E1 平均水平误差（实线）与滤波器估计 σ（虚线）
    sel = ["完整建模", "磁偏差未建模", "磁偏差未建模+恢复"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, n in enumerate(sel):
        t, d = series(out, by[("E1", n)]["tag"])
        ax.plot(t / 60, d["err_mean"], color=C[i], label=f"{n}：实际误差")
        ax.plot(t / 60, d["sigma_mean"], color=C[i], linestyle=(0, (3, 2)), linewidth=1.5, label=f"{n}：估计 σ")
    ax.set_yscale("log")
    ax.set_xlabel("时间 / min")
    ax.set_ylabel("水平位置误差 / m（Monte-Carlo 均值）")
    ax.axvspan(10, 12, color=GRID, alpha=0.6, linewidth=0)
    ax.text(11, ax.get_ylim()[1] * 0.6, "无景象匹配区段", ha="center", color=INK2, fontsize=8)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_e1_lockout_timeseries.png"))
    plt.close(fig)

    # 图 2：E1/E2 位置 ANEES 时间序列
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, (g, n) in enumerate([("E1", "完整建模"), ("E1", "磁偏差未建模"), ("E2", "随机游走尺度模型"),
                                ("E2", "马尔可夫+恢复")]):
        t, d = series(out, by[(g, n)]["tag"])
        ax.plot(t / 60, d["anees_pos"], color=C[i], label=f"{g} {n}")
    ax.axhline(3, color=INK2, linewidth=1, linestyle=(0, (1, 2)))
    ax.text(0.5, 3.4, "理想值 3", color=INK2, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("时间 / min")
    ax.set_ylabel("位置 ANEES")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_anees.png"))
    plt.close(fig)

    # 图 3：E3 尺度时变幅度 vs 发生锁死比例
    sig = [0.001, 0.002, 0.003, 0.005, 0.01]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for i, rec in enumerate(("", "+恢复")):
        y = [by[("E3", f"尺度GM σ={s}{rec}")]["lockout_run_pct"] for s in sig]
        ax.plot([s * 100 for s in sig], y, color=C[i], marker="o", markersize=5,
                label="固定门限" if not rec else "固定门限 + 锁死恢复")
    ax.set_xlabel("VO 尺度时变分量 1σ / %（τ = 120 s，滤波器按随机游走建模）")
    ax.set_ylabel("发生锁死的运行 / %")
    ax.set_ylim(-3, 103)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_e3_scale_mismatch.png"))
    plt.close(fig)

    # 图 4：E5 错误匹配比例 vs 误接率 / RMS
    ps = [5, 10, 20, 30]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    for i, rec in enumerate(("", "+恢复")):
        rr = [by[("E5", f"错误匹配 {p}%{rec}")] for p in ps]
        lab = "无恢复" if not rec else "锁死恢复"
        axs[0].plot(ps, [r["false_accept_pct"] for r in rr], color=C[i], marker="o", markersize=5, label=lab)
        axs[1].plot(ps, [r["rms_median"] for r in rr], color=C[i], marker="o", markersize=5, label=lab)
    axs[0].set_xlabel("近距离错误匹配比例 / %")
    axs[0].set_ylabel("错误匹配被接受 / %")
    axs[1].set_xlabel("近距离错误匹配比例 / %")
    axs[1].set_ylabel("水平 RMS 中位数 / m")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_e5_outliers.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/nav/nav_mc")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--out", default="results")
    ap.add_argument("--plot-only", action="store_true", help="只用已有 summary.csv 与 raw/ 重画")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.plot_only:
        rows = read_csv(os.path.join(a.out, "summary.csv"))
        for r in rows:
            for k, v in r.items():
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    else:
        rows = run_all(a.bin, a.runs, a.out)
        write_summary(rows, a.out)
    plot(rows, a.out)
    print(open(os.path.join(a.out, "summary.md")).read())


if __name__ == "__main__":
    sys.exit(main())
