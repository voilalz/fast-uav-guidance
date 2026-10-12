#!/usr/bin/env python3
"""高速固定翼无人机导航误差预算估算。

用单通道 Schuler 误差模型估算纯惯导水平位置漂移，并给出
视觉/地形匹配修正间隔与先验搜索窗口、成像运动模糊等关键约束。
只依赖 Python 标准库：python3 tools/error_budget.py
"""
import math

G = 9.80665          # m/s^2
R = 6.371e6          # 地球半径 m
WS = math.sqrt(G / R)  # Schuler 角频率 rad/s
DEG_H = math.pi / 180 / 3600  # 1 deg/h -> rad/s

V = 600 / 3.6        # 飞行速度 m/s
H = 1000.0           # 飞行高度 m

# 名称: (陀螺零偏 deg/h, 加计零偏 mg, 初始水平姿态误差 mrad, 初始速度误差 m/s, ARW deg/sqrt(h))
IMU_GRADES = {
    "消费级 MEMS":   (10.0, 2.0, 2.0, 0.5, 0.3),
    "高端 MEMS":     (1.0, 0.5, 1.0, 0.2, 0.1),
    "战术级 FOG":    (0.1, 0.1, 0.5, 0.1, 0.02),
    "导航级 RLG/FOG": (0.01, 0.025, 0.1, 0.05, 0.002),
}


def free_inertial_error(t, gyro_dph, acc_mg, att_mrad, vel0, arw):
    """单通道 Schuler 误差模型，各项取 RSS（忽略耦合与地球自转项）。"""
    s, c = math.sin(WS * t), math.cos(WS * t)
    eps = gyro_dph * DEG_H
    ba = acc_mg * 1e-3 * G
    e_gyro = R * eps * (t - s / WS)
    e_acc = ba / WS**2 * (1 - c)
    e_att = R * att_mrad * 1e-3 * (1 - c)
    e_vel = vel0 * s / WS
    # 角度随机游走：姿态误差 ~ ARW*sqrt(t)，近似按 g*theta*t^2/... 取其等效
    theta_rw = arw * math.pi / 180 / 60 * math.sqrt(t)  # rad
    e_arw = G * theta_rw * t**2 / (2 * math.sqrt(5))
    return math.sqrt(e_gyro**2 + e_acc**2 + e_att**2 + e_vel**2 + e_arw**2)


def main():
    times = [60, 300, 600, 1800, 3600]
    print("一、纯惯导水平位置漂移（单轴，m）")
    print(f"{'等级':<14}" + "".join(f"{t//60:>8d}min" for t in times))
    for name, p in IMU_GRADES.items():
        row = "".join(f"{free_inertial_error(t, *p):>11.0f}" for t in times)
        print(f"{name:<14}{row}")

    print("\n二、视觉/地形匹配修正间隔 vs. 先验 3σ 搜索半径（组合导航，m）")
    print("  假设两次绝对修正之间由 INS+视觉里程计/空速推算，相对速度误差 1σ 为 0.5%~2% V")
    for frac in (0.005, 0.01, 0.02):
        cells = []
        for gap_km in (5, 10, 20, 50):
            t = gap_km * 1000 / V
            sig = math.hypot(frac * V * t, 15.0)  # 15 m 为上一次修正残差
            cells.append(f"{gap_km}km→{3*sig:>5.0f}")
        print(f"  速度误差 {frac*100:.1f}%: " + "  ".join(cells))

    print("\n三、成像约束（下视，H=1km，V=600km/h）")
    for f_mm, pix_um in ((8, 3.45), (16, 3.45), (25, 3.45), (25, 12.0)):
        gsd = H * pix_um * 1e-6 / (f_mm * 1e-3)
        for exp_us in (100, 500, 1000):
            blur = V * exp_us * 1e-6 / gsd
            print(f"  f={f_mm:>2}mm 像元{pix_um:>5}µm GSD={gsd:5.2f}m 曝光{exp_us:>4}µs 运动模糊={blur:4.2f}px")
    print(f"  光流角速率 V/H = {V/H:.3f} rad/s；地面每帧位移 @10Hz = {V/10:.1f} m")
    print("\n四、消费级 MEMS + 视觉里程计/空速辅助：两次绝对修正之间的水平误差 3σ（m）")
    print("  误差 = 速度比例误差 + 航向误差(初值 + 陀螺 z 轴残余零偏积分) 投影到航程，再叠加上次修正残差")
    for bz_dph in (10.0, 3.0):
        for frac in (0.01, 0.02):
            cells = []
            for gap_km in (2, 5, 10, 20):
                d = gap_km * 1000.0
                t = d / V
                dpsi = math.radians(0.3) + bz_dph * DEG_H * t  # 修正后航向残差 0.3°
                along = frac * d
                cross = dpsi * d / 2  # 航向误差线性增长时平均作用一半航程
                sig = math.sqrt(along**2 + cross**2 + 15.0**2)
                cells.append(f"{gap_km:>2}km→{3*sig:>5.0f}")
            print(f"  z陀螺残余零偏 {bz_dph:>4.1f}°/h, VO 误差 {frac*100:.0f}%: " + "  ".join(cells))

    print("\n五、运动恢复结构测高（下视相机 f=16mm，像元 3.45µm，视差精度 0.5px）")
    fpx = 16e-3 / 3.45e-6
    for dt in (0.25, 0.5, 1.0):
        B = V * dt
        print(f"  帧间隔 {dt:>4}s 基线 {B:5.0f}m B/H={B/H:4.2f} 地表高度精度≈{H**2/(B*fpx)*0.5:4.2f} m")

    print("\n六、时间同步误差 -> 位置误差")
    for dt_ms in (0.1, 1, 5, 10, 50):
        print(f"  {dt_ms:>5} ms -> {V*dt_ms/1000:6.2f} m")


if __name__ == "__main__":
    main()
