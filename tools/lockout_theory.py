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


class Model:
    """增广状态 [e_p, e_s, s]（沿航迹）；横向轴只有 e_p（无尺度效应）。"""

    def __init__(self, sigma_s, tau_s, q_s, q_v, p_s0, sigma_s_const=0.0):
        self.sig, self.tau, self.q_s, self.q_v = sigma_s, tau_s, q_s, q_v
        phi = math.exp(-DT / tau_s) if tau_s > 0 else 1.0
        # 真实转移：e_p' = e_p + V·DT·e_s ；e_s' = e_s + (phi − 1)·s（滤波器以为 s 不变）；s' = phi·s
        self.Ft = np.array([[1, V * DT, 0], [0, 1, phi - 1], [0, 0, phi]])
        self.Qt = np.diag([q_v * DT, 0.0, sigma_s ** 2 * (1 - phi ** 2)])  # 真实尺度无随机游走
        self.Ff = np.array([[1, V * DT], [0, 1]])
        self.Qf = np.diag([q_v * DT, q_s * DT])
        self.P = np.diag([5.0 ** 2, p_s0])                    # 滤波器自认（沿航迹）
        self.E = np.zeros((3, 3))
        self.E[:2, :2] = self.P
        self.E[2, 2] = sigma_s ** 2                            # 时变尺度的稳态方差
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
                 gate2=13.8, gate3=16.3, return_trace=False, interval=30.0):
    ev = schedule(interval)
    hits = 0
    trace = None
    for k in range(runs):
        m = Model(sigma_s, tau_s, q_s, q_v, p_s0)
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
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--plot", default="results/fig_lockout_theory.png")
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    ref = e3_reference()
    sig = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01]
    pred = [lockout_prob(s, a.runs, rng) for s in sig]
    print("σ_s     预测锁死概率   E3 Monte-Carlo")
    for s, p in zip(sig, pred):
        print(f"{s:7.4f}  {p * 100:8.1f}%      {ref[s] * 100 if s in ref else float('nan'):6.1f}%")
    # 解析指标：单次修正前沿航迹真实/滤波新息方差之比 ρ（无拒绝稳态）
    # 无拒绝时的稳态（强制全部接受）：S_t/S_f 与闭式近似对照
    print("\nσ_s     S_t/S_f（模型）  S_t/S_f（闭式）  G     K     单次拒绝概率")
    for s in sig:
        m = Model(s, 120.0, 1e-10, 0.5, 4e-6)
        t_now = 0.0
        rows = []
        for t, sg, kind in schedule():
            m.predict(int(round(t - t_now)))
            t_now = t
            Sf, St, Sc = m.innov(sg * sg)
            if kind == 2 and 300 < t < 600:
                rows.append((Sf, St, reject_prob(Sf, St, Sc, 13.8)))
            m.update(sg * sg)
        Sf, St, p = np.mean(rows, axis=0)
        rc, G, K = rho_closed_form(s, Sf, 100.0)
        print(f"{s:7.4f}  {St / Sf:12.2f}  {rc:14.2f}  {G:5.1f} {K:5.2f}  {p * 100:8.2f}%")
    # 样本外验证：修正间隔 15 / 60 s 的预测（全状态结果见 results/lockout_theory_validation.csv）
    val = {}
    if os.path.exists("results/lockout_theory_validation.csv"):
        for r in csv.DictReader(open("results/lockout_theory_validation.csv")):
            val.setdefault(float(r["fix_interval"]), {})[float(r["vo_scale_gm_sigma"])] = \
                int(r["lockout_runs"]) / int(r["runs"])
    sig_v = [0.0005, 0.001, 0.0015, 0.002, 0.003]
    pred_v = {iv: [lockout_prob(s, a.runs, rng, interval=iv) for s in sig_v] for iv in val}
    for iv in pred_v:
        print(f"修正间隔 {iv:.0f} s：" + "，".join(f"σ {s * 100:.2f}% 预测 {p * 100:.0f}%"
                                              + (f" / 全状态 {val[iv][s] * 100:.0f}%" if s in val[iv] else "")
                                              for s, p in zip(sig_v, pred_v[iv])))
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, os.path.dirname(__file__))
        from experiments import C, style
        style(plt)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot([s * 100 for s in sig], [p * 100 for p in pred], color=C[0], marker="o", ms=3, label="解析模型（修正间隔 30 s）")
        ks = sorted(ref)
        ax.plot([s * 100 for s in ks], [ref[s] * 100 for s in ks], color=C[1], marker="s", ms=4, ls="none",
                label="E3 全状态（30 s）")
        for i, iv in enumerate(sorted(pred_v)):
            col = C[2 + i]
            ax.plot([s * 100 for s in sig_v], [p * 100 for p in pred_v[iv]], color=col, lw=1, ls="--",
                    label=f"解析模型（修正间隔 {iv:.0f} s）")
            ks = sorted(val[iv])
            ax.plot([s * 100 for s in ks], [val[iv][s] * 100 for s in ks], color=col, marker="^", ms=5, ls="none",
                    label=f"全状态（{iv:.0f} s，样本外）")
        ax.set_xlabel("VO 尺度时变分量 σ / %（τ 120 s，滤波器按随机游走建模）")
        ax.set_ylabel("锁死运行比例 / %")
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(a.plot)
        print("图已写入", a.plot)


if __name__ == "__main__":
    main()
