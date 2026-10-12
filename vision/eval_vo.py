"""在真实影像 + DEM 上评估下视 VO，重点量化不同尺度来源在山地/林区引入的 VO 尺度误差及其时间相关性。

轨迹：沿 Ahr 河谷条带由北向南定高直线飞行，600 km/h，30 Hz，叠加小幅滚转/俯仰振荡；
导航给出的姿态含 0.05° 水平 / 0.3° 航向偏差，位置含 30 m 误差，高度含 5 m 偏差。

    python3 -m vision.eval_vo --out results/vision
输出 vo_<mode>.csv、vo_summary.json、fig_vo_scale.png。
结论用于设定仿真中的 vo_scale / vo_scale_gm_sigma / vo_scale_gm_tau（见 docs/experiments.md）。
"""
import argparse
import csv
import json
import os

import numpy as np

from .camera import Camera, Pose, Terrain, euler_to_dcm, ray_ground, render
from .mapdata import build
from .vo import DownwardVo

V = 600 / 3.6
RATE = 30.0
MODES = ("radar", "dem_mean", "dem_ray")


def corr_length(x, step_m):
    """自相关降到 1/e 的距离（m），用于估计一阶马尔可夫相关时间 τ = 距离 / V。"""
    x = np.asarray(x) - np.mean(x)
    if x.std() < 1e-9:
        return 0.0
    ac = np.correlate(x, x, "full")[len(x) - 1:] / (x.var() * len(x))
    below = np.where(ac < np.exp(-1))[0]
    return float(below[0] * step_m) if len(below) else float(len(x) * step_m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/vision")
    ap.add_argument("--src", default="pre", choices=("pre", "post"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pre, post, mask, dem, grid = build(a.data)
    src = pre if a.src == "pre" else post
    T = Terrain(dem, grid.gsd)
    cam = Camera()
    rng = np.random.default_rng(a.seed)
    H, W = mask.shape
    e0 = W * grid.gsd / 2
    n_start, n_end = -260.0, -(H * grid.gsd - 260)
    h = float(np.nanmean(dem)) + 1000.0  # 定高飞行，离地高随地形变化
    dt = 1 / RATE
    att_bias = np.radians([0.05, -0.05, 0.3])
    pos_err = rng.normal(0, 30, 2)
    # 导航姿态误差为慢变过程（陀螺积分，帧间相对姿态非常准确），用一阶马尔可夫 σ 0.02°、τ 10 s
    att_gm = np.zeros(3)
    a_gm = np.exp(-dt / 10.0)
    vos = {m: DownwardVo(cam) for m in MODES}
    rows = {m: [] for m in MODES}
    k = 0
    while True:
        t = k * dt
        n = n_start - V * t
        if n < n_end:
            break
        roll = np.radians(1.5) * np.sin(2 * np.pi * 0.3 * t)
        pitch = np.radians(2.0) + np.radians(1.0) * np.sin(2 * np.pi * 0.2 * t)
        truth = Pose(n, e0, h, roll, pitch, np.pi)
        img, valid = render(src, grid.gsd, T, truth, cam, rng)
        att_gm = a_gm * att_gm + rng.normal(0, np.radians(0.02) * np.sqrt(1 - a_gm ** 2), 3)
        noise = att_gm
        est = Pose(n + pos_err[0], e0 + pos_err[1], h + 5.0, roll + att_bias[0] + noise[0],
                   pitch + att_bias[1] + noise[1], np.pi + att_bias[2] + noise[2])
        # 雷达高度表：沿机体 z 轴到真值地形的距离 + 1 m 噪声
        # （相机光轴 = 机体 z 轴，即相机系射线 (0, 0, 1)）
        _, _, gz = ray_ground(truth, np.array([[0.0, 0.0, 1.0]]), T)
        agl_radar = h - float(gz[0]) + rng.normal(0, 1.0)
        agl_dem = est.h - T.mean_height(est.n, est.e, 190)
        v_true = euler_to_dcm(roll, pitch, np.pi).T @ np.array([-V, 0, 0])
        for m in MODES:
            agl = agl_radar if m == "radar" else agl_dem
            r = vos[m].process(img, est, dt, agl=agl, terrain=T, scale_mode=m, valid=valid)
            if r.ok:
                scale = np.linalg.norm(r.vn[:2]) / V - 1
                rows[m].append(dict(t=t, dist=V * t, n=n, terrain=T.mean_height(n, e0, 190),
                                    agl_true=h - T.mean_height(n, e0, 190), scale_err=scale,
                                    vb_err_x=r.vb[0] - v_true[0], vb_err_y=r.vb[1] - v_true[1],
                                    vb_err_z=r.vb[2] - v_true[2], sigma=r.sigma, inliers=r.inliers))
        k += 1

    summary = {}
    for m in MODES:
        with open(os.path.join(a.out, f"vo_{m}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[m][0]))
            w.writeheader()
            w.writerows(rows[m])
        s = np.array([r["scale_err"] for r in rows[m]])
        ex = np.array([r["vb_err_x"] for r in rows[m]])
        ey = np.array([r["vb_err_y"] for r in rows[m]])
        ez = np.array([r["vb_err_z"] for r in rows[m]])
        # 1 s 平均后的尺度误差（滤除帧间噪声，保留时间相关分量）
        kk = int(RATE)
        s1 = np.array([s[i:i + kk].mean() for i in range(0, len(s) - kk + 1, kk)])
        summary[m] = {
            "frames_ok_pct": 100 * len(s) / k,
            "scale_err_mean_pct": 100 * float(s.mean()),
            "scale_err_std_pct": 100 * float(s.std()),
            "scale_err_1s_std_pct": 100 * float(s1.std()),
            # 相关长度在 1 s 滑动平均序列上计算（滤除帧间白噪声）
            "scale_corr_length_m": corr_length(np.convolve(s, np.ones(kk) / kk, "valid"), V / RATE),
            "scale_corr_time_s": corr_length(np.convolve(s, np.ones(kk) / kk, "valid"), V / RATE) / V,
            "vb_err_rms": [float(np.sqrt(np.mean(x ** 2))) for x in (ex, ey, ez)],
            "reported_sigma_median": float(np.median([r["sigma"] for r in rows[m]])),
        }
        print(m, json.dumps(summary[m], ensure_ascii=False))
    summary["terrain_relief_m"] = float(np.ptp([r["terrain"] for r in rows["radar"]]))
    with open(os.path.join(a.out, "vo_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    plot(rows, a.out)


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from experiments import C, INK2, style
    style(plt)
    lab = {"radar": "雷达高度表（天底点离地高）", "dem_mean": "气压高度 - 足迹平均 DEM", "dem_ray": "逐点 DEM 射线求交"}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for i, m in enumerate(MODES):
        d = np.array([r["dist"] for r in rows[m]]) / 1000
        s = np.array([r["scale_err"] for r in rows[m]]) * 100
        kk = int(RATE)
        ds = [d[j:j + kk].mean() for j in range(0, len(d) - kk + 1, kk)]
        ss = [s[j:j + kk].mean() for j in range(0, len(s) - kk + 1, kk)]
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
    main()
