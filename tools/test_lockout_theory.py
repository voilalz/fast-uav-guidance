"""降阶锁死模型的回归检查（CI 运行，约 1 s）：数值与论文第 IV 节、docs/lockout-theory.md 第 3 节一致。

    python3 tools/test_lockout_theory.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lockout_theory as L  # noqa: E402


def test_van_loan_matches_fine_integration():
    V, tau, qv, sig = 50.0, 120.0, 0.5, 0.002
    A = np.array([[0, V, 0], [0, 0, -1 / tau], [0, 0, -1 / tau]])
    G = np.array([[1, 0], [0, 1], [0, 1]], float)
    Qc = np.diag([qv, 2 * sig * sig / tau])
    dt = 1.0
    _, Qd = L.van_loan(A, G, Qc, dt)
    # 细步长的 Q(t) 积分：dQ/dt = A Q + Q A' + G Qc G'
    n = 20000
    h = dt / n
    Q = np.zeros((3, 3))
    GQG = G @ Qc @ G.T
    for _ in range(n):
        Q = Q + h * (A @ Q + Q @ A.T + GQG)
    assert np.allclose(Qd, Q, rtol=1e-3, atol=1e-12), (Qd, Q)
    # e_s 与 s 的噪声完全相关（第二版修正的交叉项）
    assert abs(Qd[1, 2] - Qd[1, 1]) < 1e-9 * max(1.0, Qd[1, 1])


def test_steady_rho_values():
    exact = [L.steady_rho(s)[0] for s in (0.001, 0.002, 0.003)]
    legacy = [L.steady_rho(s, exact=False)[0] for s in (0.001, 0.002, 0.003)]
    assert np.allclose(exact, [1.48, 2.97, 5.45], atol=0.01), exact
    assert np.allclose(legacy, [2.31, 6.25, 12.8], atol=0.05), legacy


def test_calibrated_lockout_at_0p2_percent():
    p = L.lockout_prob_sampled(0.002, 1000, np.random.default_rng(1), kappa=2.106)
    assert 0.70 < p < 0.85, p  # 论文：标定预测 77%，全状态 MC 76%


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
