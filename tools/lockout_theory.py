"""门限锁死的降阶解析模型：次优滤波灵敏度分析 + χ² 门限拒绝概率，预测锁死概率并与 E3 Monte-Carlo 对照。

模型（沿航迹单轴 + 横向一致轴，见 docs/lockout-theory.md）：
  真实系统  ṗ = V·s + w_v，s 为一阶马尔可夫（σ_s, τ_s）——VO 尺度的时变分量
  滤波模型  ṗ = V·ŝ + w_v，s 按随机游走（q_s）建模——与 E3 中滤波器的默认设定一致
在“误差 + 真实尺度”的增广状态上传播真实误差协方差 E 与滤波器自认的协方差 P，
每次绝对修正时：
  滤波器新息协方差 S_f = P_pp + R，真实新息协方差 S_t = E_pp + R（沿航迹）；横向轴两者相同；
  NIS = (ν_a² + ν_c²) / S_f 超过门限 γ 即拒绝，拒绝概率由广义 χ² 分布给出；
  接受则两者都按滤波器增益 K = P H'/S_f 更新（Joseph 形式），拒绝则都不更新。
拒绝/接受序列按上述概率随机抽样，统计“连续 ≥ 2 次正确修正被拒”（与 E3 的锁死定义一致）的运行比例。

    python3 tools/lockout_theory.py [--runs 4000] [--plot results/fig_lockout_theory.png]

修订（第二版）：
  1. 真实系统的 Q_t 原为 diag(q_v Δ, 0, σ²(1−φ²))，漏掉了驱动 s 的白噪声同样驱动 e_s = s − ŝ
     （以及 e_s–s 交叉项、e_p–e_s 积分交叉项）；现由连续模型经 Van Loan 精确离散化（exact=True）。
  2. 协方差版把各次拒绝当作独立抽样，忽略了相邻新息经慢变 s 的相关；默认预测器改为对降阶系统的
     实现抽样（lockout_prob_sampled）。
  3. 修正后的降阶模型低估全状态滤波器的超额新息方差约 2 倍，用 σ_s = 0.1 %、30 s 下锁死前的
     NIS 做单参数标定（calibrate_kappa），其余 σ_s 与修正间隔为预测。
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

V = 600 / 3.6
DT = 1.0
DUR = 1200.0


def reject_prob(S_f, S_t_along, S_t_cross, gate, extra_dof=0):
    """P(NIS > gate)，NIS = Σ ν_i² / S_f，ν_along ~ N(0, S_t_along)，ν_cross ~ N(0, S_t_cross)，
    另有 extra_dof 个一致维（方差 = S_f，如三维修正的高度）。用 Imhof 型数值积分的蒙特卡罗替代：维数少，直接解析卷积。"""
    a, c = S_t_along / S_f, S_t_cross / S_f
    # 广义 χ²：a·χ²₁ + c·χ²₁ + χ²_extra，用固定样本的高斯积分（Gauss–Hermite 外层 + χ² 尾概率内层）
    x, w = _GH
    # 对沿航迹分量取条件，剩余 c·χ²₁ + χ²_k 的尾概率
    rem = np.maximum(gate - a * x * x, 0.0)
    if extra_dof == 0:
        tail = 2 * (1 - _ncdf(np.sqrt(rem / c)))
    else:  # c·χ²₁ + χ²₁（三维修正：横向 + 高度），c≈1 时近似为 χ²₂ 尾概率
        tail = np.exp(-rem / (1 + c))
    return float(np.sum(w * tail))


_GH = np.polynomial.hermite_e.hermegauss(80)
_GH = (_GH[0], _GH[1] / _GH[1].sum())


def _ncdf(z):
    from scipy.special import ndtr
    return ndtr(z)


def van_loan(A, G, Qc, dt):
    """连续模型 ẋ = A x + G w（E[w wᵀ] = Qc δ）的精确离散化：返回 (Φ, Q_d)，Q_d 含全部交叉项。"""
    from scipy.linalg import expm
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = G @ Qc @ G.T
    M[n:, n:] = A.T
    E = expm(M * dt)
    Phi = E[n:, n:].T
    return Phi, Phi @ E[:n, n:]


class Model:
    """增广状态 [e_p, e_s, s]（沿航迹）；横向轴只有 e_p（无尺度效应）。

    e_s = s − ŝ 为尺度估计误差。ŝ 在两次修正之间不变，所以 ė_s = ṡ = −s/τ + w_s：
    驱动 s 的白噪声 w_s 同样驱动 e_s，两者的离散过程噪声完全相关。
    exact=True（默认）：真实系统与滤波模型都由连续模型经 Van Loan 精确离散化，
      Q_t 含 e_s 噪声、e_s–s 交叉项以及 e_p–e_s 的积分交叉项；s 的初值与仿真一致取 0。
    exact=False：旧版（docs/lockout-theory.md 初版）的对角 Q_t = diag(q_v Δ, 0, σ²(1−φ²))，
      漏掉了 e_s 上的噪声及其交叉项，仅为复现旧结果保留。"""

    def __init__(self, sigma_s, tau_s, q_s, q_v, p_s0, sigma_s_const=0.0, exact=True):
        self.sig, self.tau, self.q_s, self.q_v = sigma_s, tau_s, q_s, q_v
        phi = math.exp(-DT / tau_s) if tau_s > 0 else 1.0
        if exact:
            a = 1.0 / tau_s
            At = np.array([[0, V, 0], [0, 0, -a], [0, 0, -a]])
            Gt = np.array([[1.0, 0], [0, 1], [0, 1]])
            self.Ft, self.Qt = van_loan(At, Gt, np.diag([q_v, 2 * sigma_s ** 2 * a]), DT)
            self.Ff, self.Qf = van_loan(np.array([[0, V], [0, 0.0]]), np.eye(2), np.diag([q_v, q_s]), DT)
        else:
            # 旧版：e_s' = e_s + (phi − 1)·s 只有确定性部分
            self.Ft = np.array([[1, V * DT, 0], [0, 1, phi - 1], [0, 0, phi]])
            self.Qt = np.diag([q_v * DT, 0.0, sigma_s ** 2 * (1 - phi ** 2)])
            self.Ff = np.array([[1, V * DT], [0, 1]])
            self.Qf = np.diag([q_v * DT, q_s * DT])
        self.P = np.diag([5.0 ** 2, p_s0])                    # 滤波器自认（沿航迹）
        self.E = np.zeros((3, 3))
        self.E[:2, :2] = self.P
        if not exact:
            self.E[2, 2] = sigma_s ** 2                        # 旧版：s 取稳态方差且与 e_s 无相关
        self.Pc = 5.0 ** 2                                     # 横向（滤波器与真实一致）
        self._cache = {}

    def predict(self, n=1):
        """推进 n 步（n 步的转移与累积过程噪声按需缓存）。"""
        if n not in self._cache:
            Fn_f, Qn_f, Fn_t, Qn_t = np.eye(2), np.zeros((2, 2)), np.eye(3), np.zeros((3, 3))
            for _ in range(n):
                Fn_f, Qn_f = self.Ff @ Fn_f, self.Ff @ Qn_f @ self.Ff.T + self.Qf
                Fn_t, Qn_t = self.Ft @ Fn_t, self.Ft @ Qn_t @ self.Ft.T + self.Qt
            self._cache[n] = (Fn_f, Qn_f, Fn_t, Qn_t)
        Fn_f, Qn_f, Fn_t, Qn_t = self._cache[n]
        self.P = Fn_f @ self.P @ Fn_f.T + Qn_f
        self.E = Fn_t @ self.E @ Fn_t.T + Qn_t
        self.Pc += self.q_v * DT * n

    def innov(self, R):
        return self.P[0, 0] + R, self.E[0, 0] + R, self.Pc + R

    def update(self, R):
        S = self.P[0, 0] + R
        K = self.P[:, 0] / S
        A = np.eye(2) - np.outer(K, [1, 0])
        self.P = A @ self.P @ A.T + np.outer(K, K) * R
        Aa = np.eye(3)
        Aa[:2, :2] = A
        Ka = np.r_[K, 0]
        self.E = Aa @ self.E @ Aa.T + np.outer(Ka, Ka) * R
        self.Pc = self.Pc * R / (self.Pc + R)


def _psd_sqrt(Q):
    w, U = np.linalg.eigh(Q)
    return U * np.sqrt(np.maximum(w, 0.0))


def lockout_prob_sampled(sigma_s, runs, rng, tau_s=120.0, q_s=1e-10, q_v=0.5, p_s0=4e-6, outlier_rate=0.05,
                         gate2=13.8, gate3=16.3, interval=30.0, kappa=1.0):
    """按实现抽样的锁死概率（默认预测器）。

    协方差版 lockout_prob 把每次修正的拒绝当作独立事件（只依赖 E、P）；但相邻修正的新息通过
    慢变的 s 强相关，一次因 s 偏大而被拒，下一次多半也会被拒。这里对降阶系统本身抽样：
    真实误差 x = [e_p, e_s, s] 与横向误差按精确离散模型传播，滤波器 P 只随接受/拒绝历史变化，
    NIS 用实际新息计算后与门限比较。各次运行向量化并行。
    kappa：真实时变尺度过程噪声的方差倍数（σ_s → √kappa·σ_s），用于单参数标定，见 calibrate_kappa。"""
    ev = schedule(interval)
    m = Model(sigma_s * math.sqrt(kappa), tau_s, q_s, q_v, p_s0, exact=True)
    P = np.repeat(m.P[None], runs, 0)
    Pc = np.full(runs, m.Pc)
    x = rng.standard_normal((runs, 3)) @ _psd_sqrt(m.E).T
    ec = rng.normal(0.0, math.sqrt(m.Pc), runs)
    consec = np.zeros(runs, int)
    locked = np.zeros(runs, bool)
    t_now = 0.0
    for t, sig, kind in ev:
        n = int(round((t - t_now) / DT))
        if n > 0:
            m.predict(n)
            Ff, Qf, Ft, Qt = m._cache[n]
            x = x @ Ft.T + rng.standard_normal((runs, 3)) @ _psd_sqrt(Qt).T
            ec += rng.normal(0.0, math.sqrt(q_v * DT * n), runs)
            P = Ff @ P @ Ff.T + Qf
            Pc += q_v * DT * n
            t_now += n * DT
        use = rng.uniform(size=runs) >= outlier_rate if kind == 2 else np.ones(runs, bool)  # 错误匹配被拒，不计
        R = sig * sig
        Sf, Sc = P[:, 0, 0] + R, Pc + R
        na, nc = x[:, 0] + rng.normal(0.0, sig, runs), ec + rng.normal(0.0, sig, runs)
        nis = na ** 2 / Sf + nc ** 2 / Sc
        if kind == 3:
            nis += rng.standard_normal(runs) ** 2                                         # 高度轴：一致
        rej = use & (nis > (gate2 if kind == 2 else gate3))
        acc = use & ~rej
        consec[rej] += 1
        locked |= consec >= 2
        consec[acc] = 0
        K = P[:, :, 0] / Sf[:, None]
        x[acc, :2] -= K[acc] * na[acc, None]
        ec[acc] -= (Pc / Sc * nc)[acc]
        A = np.repeat(np.eye(2)[None], runs, 0)
        A[:, :, 0] -= K
        Pn = A @ P @ A.transpose(0, 2, 1) + np.einsum("ri,rj->rij", K, K) * R
        P[acc] = Pn[acc]
        Pc[acc] = (Pc * R / (Pc + R))[acc]
    return float(locked.mean())


def steady_rho(sigma_s, tau_s=120.0, q_s=1e-10, q_v=0.5, p_s0=4e-6, exact=True, interval=30.0):
    """无拒绝时 300–600 s 内景象匹配修正前的平均 S_t/S_f 与 S_f。"""
    m = Model(sigma_s, tau_s, q_s, q_v, p_s0, exact=exact)
    t_now, rows = 0.0, []
    for t, sg, kind in schedule(interval):
        m.predict(int(round(t - t_now)))
        t_now = t
        Sf, St, _ = m.innov(sg * sg)
        if kind == 2 and 300 < t < 600:
            rows.append((Sf, St))
        m.update(sg * sg)
    Sf, St = np.mean(rows, axis=0)
    return St / Sf, Sf


def empirical_rho(fixes_csv):
    """全状态 Monte-Carlo 中 300–600 s、该次运行尚无任何拒绝之前的正确景象匹配修正的平均 NIS − 1。
    二维修正 E[NIS] = ρ + 1（沿航迹 ρ、横向 1），只用锁死前的样本，不使用任何锁死结果。"""
    from collections import defaultdict
    by = defaultdict(list)
    for r in csv.DictReader(open(fixes_csv)):
        by[r["seed"]].append(r)
    nis = []
    for rr in by.values():
        clean = True
        for r in rr:
            if r["outlier"] == "1":
                continue
            if r["source"] == "1" and 300 < float(r["t"]) < 600 and clean:
                nis.append(float(r["nis"]))
            if r["accepted"] == "0":
                clean = False
    return float(np.mean(nis)) - 1.0, len(nis)


def calibrate_kappa(fixes_csv, sigma_s):
    """单参数标定：κ = (ρ_MC − 1)/(ρ_model − 1)，即全状态滤波器中未建模尺度引起的超额新息方差
    相对降阶模型的倍数。只用一个 σ_s 下锁死前的 NIS。"""
    rho_mc, n = empirical_rho(fixes_csv)
    rho_m, _ = steady_rho(sigma_s)
    return (rho_mc - 1) / (rho_m - 1), rho_mc, rho_m, n


def schedule(interval=30.0):
    """与 nav_sim 默认场景一致：景象匹配 fix_first = 30 s 起每 interval 一次（600–720 s 无），
    三维地形 45 s 起每 60 s 一次。"""
    ev = []
    t = 30.0
    while t < DUR:
        if not (600 < t < 720):
            ev.append((t, 10.0, 2))
        t += interval
    t = 45.0
    while t < DUR:
        ev.append((t, 20.0, 3))
        t += 60
    return sorted(ev)


def lockout_prob(sigma_s, runs, rng, tau_s=120.0, q_s=1e-10, q_v=0.5, p_s0=4e-6, outlier_rate=0.05,
                 gate2=13.8, gate3=16.3, return_trace=False, interval=30.0, exact=True):
    ev = schedule(interval)
    hits = 0
    trace = None
    for k in range(runs):
        m = Model(sigma_s, tau_s, q_s, q_v, p_s0, exact=exact)
        t_now, consec, locked = 0.0, 0, False
        tr = []
        for t, sig, kind in ev:
            n = int(round((t - t_now) / DT))
            if n > 0:
                m.predict(n)
                t_now += n * DT
            if rng.uniform() < outlier_rate and kind == 2:  # 错误匹配：被拒，不计入“正确修正被拒”
                continue
            R = sig * sig
            Sf, St, Sc = m.innov(R)
            p = reject_prob(Sf, St, Sc, gate2 if kind == 2 else gate3, 0 if kind == 2 else 1)
            rej = rng.uniform() < p
            if return_trace and k == 0:
                tr.append((t, Sf, St, p, rej))
            if rej:
                consec += 1
                if consec >= 2:
                    locked = True
            else:
                consec = 0
                m.update(R)
        hits += locked
        if return_trace and k == 0:
            trace = tr
    return (hits / runs, trace) if return_trace else hits / runs


def rho_closed_form(sigma_s, S_f, R, T=30.0, tau_s=120.0):
    """闭式近似：滤波器未跟踪的速度偏差 b = V·s（一阶马尔可夫，相邻修正间相关系数 φ = e^(−T/τ)）
    在每次修正后只被消去 K 倍，跨周期累积：
        Δ = E⁻ − P⁻ ≈ (V σ_s T)² · G(K, φ)，G = (1 + (1−K)φ) / ((1 − (1−K)²)(1 − (1−K)φ))
        ρ = S_t / S_f = 1 + Δ / S_f，K = (S_f − R) / S_f"""
    K = (S_f - R) / S_f
    phi = math.exp(-T / tau_s)
    G = (1 + (1 - K) * phi) / ((1 - (1 - K) ** 2) * (1 - (1 - K) * phi))
    return 1 + (V * sigma_s * T) ** 2 * G / S_f, G, K


def e3_reference(path="results/summary.csv"):
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        if r["group"] == "E3" and "恢复" not in r["name"]:
            s = float(r["params"].split("=")[1])
            out[s] = float(r["lockout_run_pct"]) / 100
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=4000)
    ap.add_argument("--plot", default="results/fig_lockout_theory.png")
    ap.add_argument("--csv", default="results/lockout_theory_predictions.csv")
    ap.add_argument("--calib", default="results/raw/E3_01_fixes.csv", help="σ_s = 0.1 %、30 s 的全状态修正记录")
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    ref = e3_reference()
    val = {30.0: ref}
    if os.path.exists("results/lockout_theory_validation.csv"):
        for r in csv.DictReader(open("results/lockout_theory_validation.csv")):
            val.setdefault(float(r["fix_interval"]), {})[float(r["vo_scale_gm_sigma"])] = \
                int(r["lockout_runs"]) / int(r["runs"])
    kappa, rho_mc, rho_m, n = calibrate_kappa(a.calib, 0.001)
    print(f"标定：σ_s 0.1 %、30 s 锁死前 {n} 次修正，ρ_MC = {rho_mc:.2f}，降阶模型 ρ = {rho_m:.2f}，κ = {kappa:.2f}")
    print("\nσ_s     ρ旧版   ρ精确   ρ闭式   ρ_MC（锁死前 NIS）")
    for sg, f in ((0.001, "results/raw/E3_01_fixes.csv"), (0.002, "results/raw/E3_03_fixes.csv"),
                  (0.003, "results/raw/E3_05_fixes.csv")):
        r_old, _ = steady_rho(sg, exact=False)
        r_new, Sf = steady_rho(sg)
        rc, _, _ = rho_closed_form(sg, Sf, 100.0)
        emp = empirical_rho(f)[0] if os.path.exists(f) else float("nan")
        print(f"{sg:6.4f}  {r_old:6.2f}  {r_new:6.2f}  {rc:6.2f}  {emp:6.2f}")
    sig = {30.0: [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005],
           15.0: [0.0005, 0.001, 0.0015, 0.002, 0.003], 60.0: [0.0005, 0.001, 0.0015, 0.002, 0.003]}
    rows = []
    print("\n间隔  σ_s     旧版   精确   精确+κ   全状态 MC")
    for iv, ss in sig.items():
        for sg in ss:
            old = lockout_prob(sg, 2000, rng, interval=iv, exact=False)
            ex = lockout_prob_sampled(sg, a.runs, rng, interval=iv)
            cal = lockout_prob_sampled(sg, a.runs, rng, interval=iv, kappa=kappa)
            mc = val.get(iv, {}).get(sg)
            rows.append((iv, sg, old, ex, cal, mc))
            print(f"{iv:4.0f}  {sg:6.4f}  {old * 100:5.1f}  {ex * 100:5.1f}  {cal * 100:6.1f}   "
                  + ("--" if mc is None else f"{mc * 100:.0f}"))
    with open(a.csv, "w") as f:
        f.write(f"# kappa={kappa:.4f} calibrated from {a.calib} (sigma_s=0.001, T=30 s)\n")
        f.write("interval_s,sigma_s,legacy_pct,exact_pct,calibrated_pct,mc_pct\n")
        for iv, sg, old, ex, cal, mc in rows:
            f.write(f"{iv:g},{sg:g},{old * 100:.1f},{ex * 100:.1f},{cal * 100:.1f},{'' if mc is None else mc * 100:}\n")
    print("预测已写入", a.csv)
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, os.path.dirname(__file__))
        from experiments import C, style
        style(plt)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for i, iv in enumerate((30.0, 15.0, 60.0)):
            rr = [r for r in rows if r[0] == iv]
            ax.plot([r[1] * 100 for r in rr], [r[4] * 100 for r in rr], color=C[i], lw=1.2,
                    label=f"降阶模型 + κ（{iv:.0f} s）")
            ax.plot([r[1] * 100 for r in rr], [r[3] * 100 for r in rr], color=C[i], lw=0.8, ls=":",
                    label=f"降阶模型，未标定（{iv:.0f} s）")
            ks = sorted(val.get(iv, {}))
            ax.plot([k * 100 for k in ks], [val[iv][k] * 100 for k in ks], color=C[i], marker="os^"[i], ms=4,
                    ls="none", label=f"全状态 MC（{iv:.0f} s）")
        ax.set_xlabel("VO 尺度时变分量 σ / %（τ 120 s，滤波器按随机游走建模）")
        ax.set_ylabel("锁死运行比例 / %")
        ax.legend(fontsize=6.5)
        fig.tight_layout()
        fig.savefig(a.plot)
        print("图已写入", a.plot)


if __name__ == "__main__":
    main()
