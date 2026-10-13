#!/usr/bin/env python3
"""图像闭环演示的过顶重捕获统计（失锁恢复时间）。

每次运行按真值是否在参考图覆盖内切分为若干“过顶”（连续覆盖 ≥ 20 s）。对每次过顶统计：
  入图误差      进入覆盖时的水平误差
  首个正确修正  进入覆盖到第一次被接受的正确修正（真误差 < 30 m）的时间
  恢复时间      入图误差 > 50 m 的过顶中，误差首次降到 50 m 以下所用时间（失锁恢复时间）
过顶内始终未恢复的记为删失（recovery_* 按过顶时长计入，recovered_* 只统计已恢复的过顶，未恢复比例单独报告）。

    python3 tools/recovery_time.py --map <la_1.00m.map> --dir results/demo_la [--route-offset 0]
输出 <dir>/recovery_time.csv（每个变体一行）与 <dir>/recovery_passes.csv（每次过顶一行）
"""
import argparse
import csv
import os
import struct
from collections import defaultdict

import numpy as np


def map_extent(path):
    with open(path, "rb") as f:
        h = f.read(48)
    magic, w, hgt, epsg, _, x0, y0, gsd = struct.unpack("<8s4i3d", h)
    assert magic.startswith(b"FUGMAP1"), path
    return hgt * gsd, w * gsd


def passes(t, n, e, H, W, min_len=20.0):
    inside = (n <= 0) & (n >= -H) & (e >= 0) & (e <= W)
    out, start = [], None
    for i, v in enumerate(inside):
        if v and start is None:
            start = i
        if (not v or i == len(inside) - 1) and start is not None:
            end = i if v else i - 1
            if t[end] - t[start] >= min_len:
                out.append((start, end))
            start = None
    return out


def analyse(dir_, H, W, thr=50.0):
    series = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(dir_, "demo_series.csv")) as f:
        for r in csv.DictReader(f):
            series[(r["variant"], r["seed"])]["t"].append(float(r["t"]))
            series[(r["variant"], r["seed"])]["n"].append(float(r["n_true"]))
            series[(r["variant"], r["seed"])]["e"].append(float(r["e_true"]))
            series[(r["variant"], r["seed"])]["err"].append(float(r["err_h"]))
    fixes = defaultdict(list)
    with open(os.path.join(dir_, "demo_fixes.csv")) as f:
        for r in csv.DictReader(f):
            if r["accepted"] == "1" and r["outlier"] == "0" and float(r["err"]) < 30:
                fixes[(r["variant"], r["seed"])].append(float(r["t"]))
    res = defaultdict(lambda: {"entry": [], "first_fix": [], "first_cens": 0, "rec": [], "rec_ok": [], "rec_cens": 0,
                               "n": 0})
    detail = []
    for (v, s), d in series.items():
        t, n, e, err = (np.array(d[k]) for k in ("t", "n", "e", "err"))
        fx = np.array(sorted(fixes[(v, s)]))
        R = res[v]
        for a, b in passes(t, n, e, H, W):
            R["n"] += 1
            row = {"variant": v, "seed": s, "t_entry": t[a], "duration_s": t[b] - t[a], "entry_err": err[a],
                   "lost": int(err[a] > thr), "recovered": 0, "recovery_s": ""}
            R["entry"].append(err[a])
            f_in = fx[(fx >= t[a]) & (fx <= t[b])]
            if len(f_in):
                R["first_fix"].append(f_in[0] - t[a])
            else:
                R["first_fix"].append(t[b] - t[a])
                R["first_cens"] += 1
            if err[a] > thr:
                below = np.nonzero(err[a:b + 1] < thr)[0]
                if len(below):
                    R["rec"].append(t[a + below[0]] - t[a])
                    R["rec_ok"].append(t[a + below[0]] - t[a])
                    row["recovered"], row["recovery_s"] = 1, t[a + below[0]] - t[a]
                else:
                    R["rec"].append(t[b] - t[a])
                    R["rec_cens"] += 1
            detail.append(row)
    rows = []
    for v, R in res.items():
        rec = np.array(R["rec"]) if R["rec"] else np.array([np.nan])
        ok = np.array(R["rec_ok"]) if R["rec_ok"] else np.array([np.nan])
        rows.append({
            "variant": v, "passes": R["n"],
            "entry_err_median": np.median(R["entry"]),
            "first_fix_median_s": np.median(R["first_fix"]), "first_fix_p95_s": np.percentile(R["first_fix"], 95),
            "no_fix_pass_pct": 100 * R["first_cens"] / R["n"],
            "lost_passes": len(R["rec"]), "lost_pass_pct": 100 * len(R["rec"]) / R["n"],
            "recovery_median_s": np.nanmedian(rec), "recovery_p95_s": np.nanpercentile(rec, 95),
            "unrecovered_pct": 100 * R["rec_cens"] / max(1, len(R["rec"])),
            "recovered_median_s": np.nanmedian(ok), "recovered_p95_s": np.nanpercentile(ok, 95),
        })
    return rows, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    H, W = map_extent(a.map)
    rows, detail = analyse(a.dir, H, W)
    with open(os.path.join(a.dir, "recovery_passes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0]))
        w.writeheader()
        w.writerows(detail)
    keys = list(rows[0])
    with open(os.path.join(a.dir, "recovery_time.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{x:.1f}" if isinstance(x, float) else x) for k, x in r.items()})
    print(f"map {H:.0f} x {W:.0f} m")
    print(f"{'variant':34s} {'pass':>4s} {'entry':>6s} {'fix50':>6s} {'fix95':>6s} {'nofix%':>6s} {'lost%':>6s} {'rec50':>6s} {'rec95':>6s} {'unrec%':>6s}")
    for r in rows:
        print(f"{r['variant']:34s} {r['passes']:4d} {r['entry_err_median']:6.0f} {r['first_fix_median_s']:6.0f} "
              f"{r['first_fix_p95_s']:6.0f} {r['no_fix_pass_pct']:6.0f} {r['lost_pass_pct']:6.0f} "
              f"{r['recovery_median_s']:6.0f} {r['recovery_p95_s']:6.0f} {r['unrecovered_pct']:6.0f}")


if __name__ == "__main__":
    main()
