"""在真实多时相影像上评估景象匹配：精度、成功率、错误匹配率与形态，并拟合仿真用的量测模型。

实时图由 POST-event 影像（洪水后、不同卫星与时相、含薄云）按随机位姿渲染，参考图为 PRE-event 影像。
另跑一组“同源”（实时图也由 PRE 渲染）作为上界。

    python3 -m vision.eval_scene_match --trials 300 --out results/vision
输出 <out>/scene_match_<case>.csv、scene_match_summary.json、fig_scene_match_*.png
"""
import argparse
import csv
import json
import os

import numpy as np

from .camera import Camera, Pose, Terrain, render
from .mapdata import build
from .scene_match import MatchConfig, SceneMatcher

CASES = {
    # 名称: (实时图来源, MatchConfig 覆盖, 先验位置 1σ m)
    "same_source": ("pre", {}, 50.0),
    "cross_time": ("post", {}, 50.0),
    "cross_time_grad": ("post", {"preprocess": "grad"}, 50.0),
    "cross_time_flat": ("post", {"use_dem": False}, 50.0),
    "cross_time_prior150": ("post", {}, 150.0),
}


def sample_pose(rng, terrain, mask, gsd, margin_m=230):
    """随机真值位姿：足迹在有效数据内，航向沿条带（南北向）±20°。"""
    H, W = mask.shape
    for _ in range(1000):
        n = -rng.uniform(margin_m, H * gsd - margin_m)
        e = rng.uniform(W * gsd / 2 - 60, W * gsd / 2 + 60)
        r, c = int(-n / gsd), int(e / gsd)
        k = int(150 / gsd)
        if mask[r - k:r + k, c - k:c + k].mean() > 0.97:
            h = terrain.mean_height(n, e, 300) + 1000 + rng.normal(0, 30)
            yaw = (0 if rng.uniform() < 0.5 else np.pi) + rng.normal(0, np.radians(10))
            return Pose(n, e, h, rng.normal(0, np.radians(2)), rng.normal(0, np.radians(2)), yaw)
    raise RuntimeError("找不到有效位姿")


def perturb(rng, p, sig_pos):
    """导航先验：位置/高度/姿态误差（GNSS 拒止若干分钟后的典型量级）。"""
    return Pose(p.n + rng.normal(0, sig_pos), p.e + rng.normal(0, sig_pos), p.h + rng.normal(0, 10),
                p.roll + rng.normal(0, np.radians(0.1)), p.pitch + rng.normal(0, np.radians(0.1)),
                p.yaw + rng.normal(0, np.radians(0.5)))


def run_case(name, src_key, overrides, prior_sigma, data, trials, seed, out):
    pre, post, mask, dem, grid = data
    terrain = Terrain(dem, grid.gsd)
    cam = Camera()
    cfg = MatchConfig(**overrides)
    m = SceneMatcher(pre, grid.gsd, cam, terrain, cfg)
    src = pre if src_key == "pre" else post
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(trials):
        truth = sample_pose(rng, terrain, mask, grid.gsd)
        est = perturb(rng, truth, prior_sigma)
        img, valid = render(src, grid.gsd, terrain, truth, cam, rng)
        r = m.match(img, valid, est, prior_sigma)
        err_n, err_e = (r.n - truth.n, r.e - truth.e) if r.ok else (np.nan, np.nan)
        yaw_err = np.degrees(np.angle(np.exp(1j * (est.yaw + r.dyaw - truth.yaw)))) if r.ok and not np.isnan(r.dyaw) \
            else np.nan
        rows.append(dict(i=i, ok=int(r.ok), stage=r.stage, err_n=err_n, err_e=err_e, err=np.hypot(err_n, err_e),
                         sigma=r.sigma, ncc=r.ncc, psr=r.psr, amb=r.ambiguity, inliers=r.inliers,
                         prior_err=np.hypot(est.n - truth.n, est.e - truth.e),
                         yaw_prior_err=np.degrees(est.yaw - truth.yaw), yaw_err=yaw_err, ms=r.ms))
    with open(os.path.join(out, f"scene_match_{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def stats(rows, outlier_m=30.0):
    ok = [r for r in rows if r["ok"]]
    err = np.array([r["err"] for r in ok])
    en = np.array([r["err_n"] for r in ok])
    ee = np.array([r["err_e"] for r in ok])
    good = err < outlier_m
    sig = np.array([r["sigma"] for r in ok])
    nis = (en ** 2 + ee ** 2) / sig ** 2
    out = {
        "trials": len(rows),
        "reported_pct": 100 * len(ok) / len(rows),
        "fine_pct": 100 * np.mean([r["stage"] == "fine" for r in ok]) if ok else 0,
        "outlier_pct_of_reported": 100 * (1 - good.mean()) if ok else 0,
        "bias_n": float(en[good].mean()) if good.any() else np.nan,
        "bias_e": float(ee[good].mean()) if good.any() else np.nan,
        "std_n": float(en[good].std()) if good.any() else np.nan,
        "std_e": float(ee[good].std()) if good.any() else np.nan,
        "err_median": float(np.median(err[good])) if good.any() else np.nan,
        "err_p95_inliers": float(np.percentile(err[good], 95)) if good.any() else np.nan,
        "outlier_err_min": float(err[~good].min()) if (~good).any() else np.nan,
        "outlier_err_median": float(np.median(err[~good])) if (~good).any() else np.nan,
        "outlier_err_max": float(err[~good].max()) if (~good).any() else np.nan,
        "nis_mean_inliers": float(nis[good].mean()) if good.any() else np.nan,  # 理想 2
        "yaw_err_std_deg": float(np.nanstd([r["yaw_err"] for r in ok])) if ok else np.nan,
        "yaw_prior_std_deg": float(np.std([r["yaw_prior_err"] for r in rows])),
        "ms_median": float(np.median([r["ms"] for r in rows])),
        "fail_reasons": {s: sum(1 for r in rows if not r["ok"] and r["stage"] == s)
                         for s in {r["stage"] for r in rows if not r["ok"]}},
    }
    return out


def plot(all_rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from experiments import C, style
    style(plt)
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for i, (name, lab) in enumerate([("same_source", "同源（上界）"), ("cross_time", "跨时相")]):
        ok = [r for r in all_rows[name] if r["ok"]]
        en = np.array([r["err_n"] for r in ok])
        ee = np.array([r["err_e"] for r in ok])
        axs[0].scatter(ee, en, s=10, color=C[i], alpha=0.7, label=lab, linewidths=0)
        err = np.sort(np.hypot(en, ee))
        axs[1].plot(err, np.arange(1, len(err) + 1) / len(all_rows[name]) * 100, color=C[i], label=lab)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/vision")
    ap.add_argument("--cases", default=",".join(CASES))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = build(a.data)
    summary, all_rows = {}, {}
    for name in a.cases.split(","):
        src, ov, ps = CASES[name]
        rows = run_case(name, src, ov, ps, data, a.trials, a.seed, a.out)
        all_rows[name] = rows
        summary[name] = stats(rows)
        print(name, json.dumps(summary[name], ensure_ascii=False, default=float))
    with open(os.path.join(a.out, "scene_match_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    if "same_source" in all_rows and "cross_time" in all_rows:
        plot(all_rows, a.out)


if __name__ == "__main__":
    main()
