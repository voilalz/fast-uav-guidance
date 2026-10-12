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

// 门限锁死（gate lockout）检测与恢复。
// 现象：时间相关误差未建模或建模不足时，滤波器协方差在修正间隙中被高频辅助量测压小，
// 下一次正确的绝对修正因新息超出 χ² 门限被拒，此后协方差继续偏小，形成自锁发散。
// 判据：连续 K 次被拒的绝对修正，其新息彼此一致（差值在量测噪声 + 漂移率上界内），
// 说明被拒的是滤波器自身而不是量测（独立的错误匹配之间通常不一致）。
// 恢复：按最新新息把运动学误差协方差放大到 NIS ≈ 量测维数，再重新执行该次更新。
struct LockoutRecoveryConfig {
  bool enable = false;
  int consecutive = 2;          // 连续被拒且相互一致的次数 K
  real window_s = 150.0;        // 两次被拒修正的最大间隔
  real drift_rate = 2.0;        // 修正间隙内导航误差变化率上界（1σ）m/s
  real consistency_gate = 9.21; // χ²(2) 99%
  // 两次被拒修正的最小时间间隔：间隔更短（视场重叠）的被拒修正不作为独立证据。
  // 真实景象匹配的错误匹配与地物绑定，相邻关键帧会在同一位置重复出现、彼此“一致”；
  // 取 ≥ 视场幅宽 / 地速（如 384 m / 167 m/s ≈ 2.3 s）可避免把它们误判为锁死。0 表示不限制。
  real min_separation_s = 0.0;
  // 协方差下限：水平 1σ ≥ sigma_floor_rate × 距上次被接受的绝对修正的时间（m/s），不足时抬高水平位置方差。
  // 搜索窗口与 χ² 门限都由同一个协方差决定；只放大匹配器的搜索窗口而不放大协方差时，
  // 找到的正确修正仍会被门限拒绝（见 docs/demo.md 第 7 节）。0 表示不启用。
  real sigma_floor_rate = 0.0;
  // 自适应协方差下限：只有在“有参考覆盖、尝试了匹配却没有被接受的修正”的关键帧连续达到该次数后才启用下限
  // （noteFixAttempt 计数，被接受的绝对修正清零）。滤波器自信且匹配正常时下限不介入，避免无谓地放大窗口与门限。
  // 0 表示始终启用（原实现）。
  int sigma_floor_trigger = 0;
  // 下限的上界 m（由参考库在航线附近的覆盖范围决定：3σ 窗口不应超出覆盖）；0 表示不限制
  real sigma_floor_max = 0.0;
};

struct FixEvent {
  TimeUs t_us = 0;
  FixSource source = FixSource::kSceneMatch;
  real nis = 0;
  bool accepted = false;
  bool recovered = false;  // 经锁死恢复后被接受
};

struct NavCoreConfig {
  real output_rate_hz = 30.0;
  real history_rate_hz = 100.0;
  real fix_timeout_s = 30.0;      // 超过此时间无绝对修正 -> M3
  real reacquire_sigma_m = 300.0; // 水平 1σ 超过此值 -> M4
  real vo_timeout_s = 1.0;
  // 时延量测参考状态：false 为量测时刻的历史名义状态（原实现）；
  // true 为历史名义状态叠加此后已注入的修正量（避免重复修正，推荐）
  bool delay_apply_corrections = true;
  EskfConfig eskf;
  LockoutRecoveryConfig recovery;
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
  // 按 recovery.sigma_floor_rate 施加协方差下限。在向视觉计算机发布 σ（决定搜索窗口）之前调用；
  // onPositionFix 也会先调用一次。
  void applySigmaFloor(TimeUs now);
  // 一次有参考覆盖的景象匹配尝试（无论匹配器是否报告），用于自适应协方差下限的触发计数
  void noteFixAttempt() { ++attempts_since_fix_; }
  bool sigmaFloorArmed() const;
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
  const FixEvent& lastFixEvent() const { return last_event_; }
  std::uint32_t recoveries() const { return recoveries_; }

 private:
  bool ref(TimeUs t, NavState& out, MeasurementStats& st) const;
  bool tryRecover(const PositionFix& z, const NavState& r, UpdateResult& u);
  void updateMode(TimeUs now);
  NavOutput makeOutput() const;

  NavCoreConfig cfg_;
  Eskf eskf_;
  struct HistEntry {
    TimeUs t_us = 0;
    NavState x;
    StateVec cum;  // 该时刻的累计修正量
  };
  StateHistory<256, HistEntry> hist_;  // 100 Hz × 2.56 s
  TimeUs last_imu_ = 0, next_out_ = 0, next_hist_ = 0;
  TimeUs last_fix_ = 0, last_scene_fix_ = 0, last_vo_ = 0;
  NavMode mode_ = NavMode::kDeadReckon;
  std::uint8_t health_ = 0;
  bool gnss_ok_ = false;
  bool have_dem_ = false;
  Dem dem_;
  real dem_lat0_ = 0, dem_lon0_ = 0;
  MeasurementStats fix_stats_, aux_stats_;
  FixEvent last_event_;
  // 最近一串连续被拒的修正（用于一致性判据）
  struct Rejected { TimeUs t_us; Vec2 r; real sigma; };
  static constexpr int kMaxRejected = 8;
  Rejected rej_[kMaxRejected];
  int n_rej_ = 0;
  std::uint32_t recoveries_ = 0;
  int attempts_since_fix_ = 0;
};

}  // namespace nav
