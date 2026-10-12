"""绘制视觉管线评估图（数据由 C++ 程序 vision_eval_scene_match / vision_eval_vo 生成）。

    python3 tools/plot_vision.py [results/vision]
输出 fig_scene_match_errors.png、fig_vo_scale.png。
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from experiments import C, INK2, style  # noqa: E402

RATE = 30


def load(path):
    with open(path) as f:
        return [{k: (v if k == "stage" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def plot_scene_match(out):
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for i, (name, lab) in enumerate([("same_source", "同源（上界）"), ("cross_time", "跨时相")]):
        rows = load(os.path.join(out, f"scene_match_{name}.csv"))
        ok = [r for r in rows if r["ok"]]
        en = np.array([r["err_n"] for r in ok])
        ee = np.array([r["err_e"] for r in ok])
        axs[0].scatter(ee, en, s=10, color=C[i], alpha=0.7, label=lab, linewidths=0)
        err = np.sort(np.hypot(en, ee))
        axs[1].plot(err, np.arange(1, len(err) + 1) / len(rows) * 100, color=C[i], label=lab)
    axs[0].set_xlim(-60, 60)
    axs[0].set_ylim(-60, 60)
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("东向误差 / m")
    axs[0].set_ylabel("北向误差 / m")
    axs[0].legend(fontsize=8)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("水平误差 / m")
    axs[1].set_ylabel("累计比例（占全部帧）/ %")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_scene_match_errors.png"))
    plt.close(fig)


def plot_vo(out):
    modes = ("radar", "dem_mean", "dem_ray")
    lab = {"radar": "雷达高度表（天底点离地高）", "dem_mean": "气压高度 - 足迹平均 DEM", "dem_ray": "逐点 DEM 射线求交"}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    rows = {m: load(os.path.join(out, f"vo_{m}.csv")) for m in modes}
    for i, m in enumerate(modes):
        d = np.array([r["dist"] for r in rows[m]]) / 1000
        s = np.array([r["scale_err"] for r in rows[m]]) * 100
        ds = [d[j:j + RATE].mean() for j in range(0, len(d) - RATE + 1, RATE)]
        ss = [s[j:j + RATE].mean() for j in range(0, len(s) - RATE + 1, RATE)]
        ax.plot(ds, ss, color=C[i], label=lab[m])
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_ylabel("VO 尺度误差 / %（1 s 平均）")
    ax.legend(fontsize=8)
    d = np.array([r["dist"] for r in rows["radar"]]) / 1000
    ax2.plot(d, [r["terrain"] for r in rows["radar"]], color=INK2, linewidth=1.5)
    ax2.set_ylabel("足迹平均地形高 / m")
    ax2.set_xlabel("沿航迹距离 / km")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_vo_scale.png"))
    plt.close(fig)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/vision"
    style(plt)
    plot_scene_match(out)
    plot_vo(out)
    print("图已写入", out)
