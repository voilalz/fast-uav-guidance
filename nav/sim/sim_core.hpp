// 闭环数字仿真核心：真值轨迹、带误差的传感器、NavCore 闭环、误差与一致性统计。
// 由 nav_sim（单次运行）和 nav_mc（Monte-Carlo 批量运行）共用。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nav/nav_core.hpp"

namespace sim {

using nav::real;

// 全部参数可由命令行 key=value 覆盖（见 setParam）。默认值即仓库原有仿真场景。
struct SimConfig {
  real duration = 1200;  // s
  unsigned seed = 1;

  // ---- 真值与 IMU ----
  int truth_substeps = 1;      // 真值积分子步数；>1 时 IMU 输出为子步平均（真值与滤波器编排不再同步）
  real gyro_bias_dph = 10, accel_bias_mg = 2;
  real gyro_arw_dpsh = 0.3, accel_vrw_mpsh = 0.05;  // deg/sqrt(h), m/s/sqrt(h)
  real gyro_sf_ppm = 500, accel_sf_ppm = 0;
  real gyro_drift_dph = 0, gyro_drift_tau = 300;    // 零偏一阶马尔可夫漂移（温漂近似）
  real accel_drift_mg = 0, accel_drift_tau = 300;

  // ---- 视觉里程计 ----
  bool vo = true;
  real vo_rate = 30, vo_delay = 0.03;
  real vo_scale = 0.008;                       // 常值尺度误差
  real vo_scale_gm_sigma = 0, vo_scale_gm_tau = 120;  // 时变尺度误差（林区雷达测冠层等）
  real vo_noise = 0.5, vo_sigma = 0.6;         // 实际噪声 / 滤波器假设
  real vo_gap_start = 600, vo_gap_end = 610;

  // ---- 景象匹配 ----
  bool fixes = true;
  real fix_interval = 30, fix_first = 30, fix_delay = 0.25;
  real fix_noise = 10, fix_sigma = 10;
  real outlier_rate = 0.05;
  int outlier_mode = 0;             // 0：远距离固定 (+400, −300) m；1：近距离 50–150 m 随机方向（重复纹理）
  real near_min = 50, near_max = 150;
  real gap_start = 600, gap_end = 720;  // 无景象匹配区段（森林弱纹理）
  bool fix_yaw = false;
  real fix_yaw_noise_deg = 0.5, fix_yaw_sigma_deg = 0.7;

  // ---- 三维地形匹配 ----
  bool t3d = true;
  real t3d_interval = 60, t3d_first = 45, t3d_delay = 0.4;
  real t3d_noise_ne = 20, t3d_noise_d = 10;

  // ---- 大气数据与磁 ----
  bool airspeed = true, heading = true;
  real wind_n = 6, wind_e = -4;
  real baro_bias = 25;
  real mag_bias_deg = 0.5, mag_noise_deg = 2.0, mag_sigma_deg = 2.5;

  // ---- 滤波器 ----
  bool est_baro_bias = true, est_vo_scale = true, est_mag_bias = true;
  real filter_vo_scale_rw = 1e-5;                     // 滤波器假设的 VO 尺度随机游走
  real filter_vo_scale_sigma = 0, filter_vo_scale_tau = 0;  // >0 时滤波器按一阶马尔可夫建模 VO 尺度
  bool delay_corr = true;  // 时延量测参考状态叠加之后的修正量（false 为原实现）
  bool recovery = false;
  int recovery_k = 2;
  real recovery_drift_rate = 2.0;
};

// 设置参数，未知 key 返回 false。profile=realistic 一次性启用更真实的误差模型。
bool setParam(SimConfig& c, const std::string& key, const std::string& val);
bool parseArgs(SimConfig& c, int argc, char** argv, int first, std::string* err);
std::string describe(const SimConfig& c);

struct Sample {  // 每秒一个
  real t, err_n, err_e, err_d, err_h, sigma_ne, nees_pos, nees_kin;
  int mode;
};

struct FixRecord {
  real t;
  int source;     // 1 景象匹配 2 三维地形
  bool outlier;   // 仿真注入的错误匹配
  bool accepted, recovered;
  real nis, innov;  // NIS 与水平误差大小（真值 − 量测，m）
};

struct Result {
  real rms_h = 0, max_h = 0, final_h = 0;
  real err_at[5] = {};  // 60, 300, 600, 900, 1200 s
  int outputs = 0;
  nav::MeasurementStats fix;
  int true_accepted = 0, true_rejected = 0, outlier_accepted = 0, outlier_rejected = 0;
  int lockout_episodes = 0;   // 连续 ≥2 次正确修正被拒
  real lockout_longest_s = 0;
  std::uint32_t recoveries = 0;
  real nees_pos_mean = 0, nees_kin_mean = 0;
  real nees_pos_in95 = 0;     // 位置 NEES 落在 χ²(3) 95% 区间内的比例
  std::vector<Sample> samples;
  std::vector<FixRecord> fixes;
};

struct Scenario {
  bool vo = true, fixes = true, airspeed = true, heading = true;
};

Result run(const SimConfig& cfg, const Scenario& sc, bool record, bool verbose);

}  // namespace sim
