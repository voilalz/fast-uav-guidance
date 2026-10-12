// 导航核心：在导航 MCU 上运行。
// 职责：IMU 时间更新、异步/时延量测更新、模式管理、按固定频率（默认 30 Hz）输出导航解。
// 所有接口均为非阻塞、无动态内存，可在 RTOS 任务或裸机主循环中调用。
#pragma once

#include "nav/eskf.hpp"
#include "nav/state_history.hpp"
#include "nav/terrain.hpp"

namespace nav {

enum class NavMode : std::uint8_t {
  kGnssIns = 0,     // M0：GNSS 可用
  kVisionAided = 1, // M1：景象匹配有效
  kTerrainAided = 2,// M2：地形匹配有效
  kDeadReckon = 3,  // M3：无绝对修正，仅 VO/空速推算
  kReacquire = 4,   // M4：协方差超限，等待大窗口重捕获
};

enum HealthBits : std::uint8_t {
  kHealthImuDegraded = 1 << 0,
  kHealthVoLost = 1 << 1,
  kHealthFixRejected = 1 << 2,  // 最近一次绝对修正被门限拒绝
  kHealthCovLarge = 1 << 3,
};

struct NavCoreConfig {
  real output_rate_hz = 30.0;
  real history_rate_hz = 100.0;
  real fix_timeout_s = 30.0;      // 超过此时间无绝对修正 -> M3
  real reacquire_sigma_m = 300.0; // 水平 1σ 超过此值 -> M4
  real vo_timeout_s = 1.0;
  EskfConfig eskf;
};

struct MeasurementStats {
  std::uint32_t accepted = 0, rejected = 0, too_old = 0;
};

class NavCore {
 public:
  explicit NavCore(const NavCoreConfig& cfg = NavCoreConfig{}) : cfg_(cfg), eskf_(cfg.eskf) {}

  void init(const NavState& x0, const InitSigma& s);

  // IMU 到达（≥200 Hz）。若到了输出时刻，填写 out 并返回 true。
  bool onImu(const ImuSample& imu, NavOutput& out);

  void onPositionFix(const PositionFix& z);
  void onVoVelocity(const VoVelocity& z);
  void onAirspeed(const Airspeed& z);
  void onBaro(const BaroAlt& z);
  void onHeading(const MagHeading& z);
  void onRadarAlt(const RadarAlt& z);  // 需先 setDem()

  // DEM 局部网格原点（西南角）经纬度，由瓦片管理器在换瓦片时调用
  void setDem(const Dem& dem, real origin_lat, real origin_lon);
  void setGnssAvailable(bool v) { gnss_ok_ = v; }

  NavMode mode() const { return mode_; }
  const Eskf& filter() const { return eskf_; }
  const MeasurementStats& fixStats() const { return fix_stats_; }

 private:
  bool ref(TimeUs t, NavState& out, MeasurementStats& st) const;
  void updateMode(TimeUs now);
  NavOutput makeOutput() const;

  NavCoreConfig cfg_;
  Eskf eskf_;
  StateHistory<256> hist_;  // 100 Hz × 2.56 s
  TimeUs last_imu_ = 0, next_out_ = 0, next_hist_ = 0;
  TimeUs last_fix_ = 0, last_scene_fix_ = 0, last_vo_ = 0;
  NavMode mode_ = NavMode::kDeadReckon;
  std::uint8_t health_ = 0;
  bool gnss_ok_ = false;
  bool have_dem_ = false;
  Dem dem_;
  real dem_lat0_ = 0, dem_lon0_ = 0;
  MeasurementStats fix_stats_, aux_stats_;
};

}  // namespace nav
