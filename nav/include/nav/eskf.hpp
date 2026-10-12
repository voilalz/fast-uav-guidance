// 误差状态卡尔曼滤波器（ESKF）。
//
// 误差状态（20 维，真值 = 名义值 ⊕ 误差）：
//   δp  [0..2]   NED 位置误差 m
//   δv  [3..5]   NED 速度误差 m/s
//   φ   [6..8]   导航系姿态误差 rad，C_true = (I + [φ×]) C_hat
//   δbg [9..11]  陀螺零偏误差 rad/s
//   δba [12..14] 加计零偏误差 m/s^2
//   δw  [15..16] 水平风误差 m/s
//   δbb [17]     气压高度偏差误差 m
//   δk  [18]     视觉里程计尺度误差
//   δbm [19]     磁航向偏差（残余硬/软铁、磁偏角误差）rad
//
// 时延量测：调用者传入量测时刻的历史名义状态 ref（来自 StateHistory），
// 新息与雅可比在 ref 处计算，修正量施加到当前状态。
#pragma once

#include "nav/types.hpp"

namespace nav {

constexpr std::size_t kN = 20;
namespace idx {
constexpr std::size_t P = 0, V = 3, TH = 6, BG = 9, BA = 12, W = 15, BB = 17, KV = 18, MB = 19;
}

using StateVec = Mat<kN, 1>;
using StateCov = Mat<kN, kN>;

struct EskfConfig {
  // IMU 噪声（按商用级 MEMS 偏保守取值，实际以 Allan 方差标定结果替换）
  real gyro_arw = 0.3 * kDeg / 60.0;        // rad/sqrt(s)  (0.3 deg/sqrt(h))
  real accel_vrw = 0.05 / 60.0;             // m/s/sqrt(s)  (0.05 m/s/sqrt(h))
  real gyro_bias_rw = 0.5 * kDeg / 3600.0 / 60.0;  // rad/s/sqrt(s)
  real accel_bias_rw = 2e-5;                // m/s^2/sqrt(s)
  real wind_sigma = 5.0, wind_tau = 600.0;  // 风：一阶马尔可夫
  real baro_sigma = 30.0, baro_tau = 1800.0;
  real vo_scale_rw = 1e-5;                  // 1/sqrt(s)
  real mag_bias_sigma = 2.0 * kDeg, mag_bias_tau = 3600.0;  // 磁航向偏差（时间相关，不能当白噪声）
  real cov_period = 0.01;                   // 协方差传播周期 s（100 Hz）
  real gate_pos = 13.8;                     // χ²(2) 99.9%
  real gate_1d = 10.8;                      // χ²(1) 99.9%
  real gate_3d = 16.3;                      // χ²(3) 99.9%
};

struct InitSigma {
  real pos = 5, vel = 0.2, roll_pitch = 0.2 * kDeg, yaw = 1.0 * kDeg;
  real bg = 10.0 * kDeg / 3600.0, ba = 2e-3 * 9.80665;
  real wind = 5.0, baro = 30.0, vo_scale = 0.02, mag_bias = 2.0 * kDeg;
};

struct UpdateResult {
  bool accepted = false;
  real nis = 0;  // 归一化新息平方（用于一致性检验与完好性监测）
};

class Eskf {
 public:
  explicit Eskf(const EskfConfig& cfg = EskfConfig{}) : cfg_(cfg) {}

  void init(const NavState& x0, const InitSigma& s);

  // IMU 时间更新（原始量测，内部扣除零偏）
  void predict(const ImuSample& imu, real dt);

  // ---- 量测更新 ----
  UpdateResult updatePosition(const PositionFix& z, const NavState& ref);
  UpdateResult updateVoVelocity(const VoVelocity& z, const NavState& ref);
  UpdateResult updateAirspeed(const Airspeed& z, const NavState& ref);
  UpdateResult updateBaro(const BaroAlt& z, const NavState& ref);
  UpdateResult updateHeading(const MagHeading& z, const NavState& ref);
  // 雷达高度 + DEM（SITAN 形式）：dem_h 为 ref 位置处地形高，grad_n/grad_e 为地形梯度
  UpdateResult updateRadarAlt(const RadarAlt& z, const NavState& ref, real dem_h, real grad_n, real grad_e);

  const NavState& state() const { return x_; }
  const StateCov& cov() const { return P_; }
  real sigmaNe() const { return std::sqrt(0.5 * (P_(0, 0) + P_(1, 1))); }
  real sigmaD() const { return std::sqrt(P_(2, 2)); }
  Vec3 lastGyroCorrected() const { return last_gyro_; }

 private:
  template <std::size_t M>
  UpdateResult update(const Mat<M, 1>& r, const Mat<M, kN>& H, const Mat<M, M>& R, real gate);
  void inject(const StateVec& dx);
  void propagateCov(real dt);

  EskfConfig cfg_;
  NavState x_;
  StateCov P_;
  real cov_dt_acc_ = 0;
  Vec3 f_acc_, w_acc_;  // 协方差周期内的平均比力/角速度
  int acc_n_ = 0;
  Vec3 last_gyro_;
};

}  // namespace nav
