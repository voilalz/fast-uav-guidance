// 导航 MCU 固件主程序框架。
//
// 目标板（STM32H743 + FreeRTOS）任务划分：
//   ISR  IMU_DR (2 kHz)  : 读 IMU 阵列 -> 5 点平均降采样 -> 推入 imu_q（400 Hz）
//   T1   NavTask  (最高) : 取 imu_q -> NavCore::onImu -> 30 Hz 输出到飞控 UART + Jetson UDP
//   T2   LinkTask        : 解码 Jetson 发来的 VO / 位置修正，推入 meas_q
//   T3   SensorTask      : 空速/气压 50 Hz、磁 10 Hz、雷达高 50 Hz，推入 meas_q
//   T4   HealthTask      : 1 Hz 自检、看门狗、日志
// NavTask 是唯一调用 NavCore 的任务，其余任务只通过无锁队列投递，避免加锁。
//
// 宿主机构建时使用 Host* 桩实现，验证调度与协议打通。
#include <cstdio>
#include <cstring>

#include "mcu_hal.hpp"
#include "nav/nav_core.hpp"
#include "nav/protocol.h"
#include "nav/spsc_queue.hpp"

namespace app {

using namespace nav;

// 统一的量测消息，所有非 IMU 数据经 meas_q 进入 NavTask
struct Meas {
  enum Kind : std::uint8_t { kVo, kFix, kAirspeed, kBaro, kHeading, kRadar } kind;
  union {
    VoVelocity vo;
    PositionFix fix;
    Airspeed as;
    BaroAlt baro;
    MagHeading mag;
    RadarAlt radar;
  };
  Meas() : kind(kBaro), baro() {}
};

class NavApp {
 public:
  NavApp(hal::ITimebase& tb, hal::IImuArrayDriver& imu, hal::IAirData& air, hal::IMagnetometer& mag,
         hal::IRadarAltimeter& ra, hal::ICameraTrigger& cam, hal::ILink& companion, hal::ILink& autopilot)
      : tb_(tb), imu_(imu), air_(air), mag_(mag), ra_(ra), cam_(cam), comp_(companion), ap_(autopilot) {
    nav_decoder_init(&dec_);
  }

  void init(const NavState& x0) { core_.init(x0, InitSigma{}); }

  // ---- ISR 上下文 ----
  void onImuDataReady() {
    RawImu raw[hal::kImuCount];
    TimeUs t;
    if (!imu_.read(raw, t)) return;
    ImuSample s;
    if (array_.fuse(raw, t, s) == 0) return;
    // 2 kHz -> 400 Hz：5 点平均（抗混叠）
    acc_g_ += s.gyro;
    acc_a_ += s.accel;
    if (++acc_n_ == 5) {
      s.gyro = acc_g_ * 0.2;
      s.accel = acc_a_ * 0.2;
      imu_q_.push(s);
      acc_g_ = Vec3{};
      acc_a_ = Vec3{};
      acc_n_ = 0;
    }
  }

  // ---- 任务上下文 ----
  void sensorTask() {
    Meas m;
    if (air_.readAirspeed(m.as)) { m.kind = Meas::kAirspeed; meas_q_.push(m); }
    if (air_.readBaro(m.baro)) { m.kind = Meas::kBaro; meas_q_.push(m); }
    if (mag_.readHeading(m.mag)) { m.kind = Meas::kHeading; meas_q_.push(m); }
    if (ra_.read(m.radar)) { m.kind = Meas::kRadar; meas_q_.push(m); }
  }

  void linkTask() {
    // 相机触发时间戳 -> Jetson（Jetson 用它给图像打时间戳）
    nav_cam_trigger_t trig{};
    std::uint32_t fid;
    std::uint8_t cam;
    TimeUs t;
    while (cam_.poll(t, fid, cam)) {
      trig.t_mid_exposure_us = t;
      trig.frame_id = fid;
      trig.camera = cam;
      trig.exposure_us = 400;
      send(comp_, NAV_MSG_CAM_TRIGGER, &trig, sizeof trig);
    }
    // Jetson -> VO / 位置修正
    std::uint8_t buf[512];
    const std::size_t n = comp_.read(buf, sizeof buf);
    for (std::size_t i = 0; i < n; ++i) {
      if (!nav_decoder_push(&dec_, buf[i])) continue;
      handleFrame(nav_frame_id(&dec_), nav_frame_payload(&dec_), nav_frame_len(&dec_));
    }
  }

  void navTask() {
    Meas m;
    while (meas_q_.pop(m)) dispatch(m);
    ImuSample s;
    while (imu_q_.pop(s)) {
      NavOutput o;
      if (core_.onImu(s, o)) publish(o);
    }
  }

  std::uint32_t outputs() const { return outputs_; }
  const NavCore& core() const { return core_; }

 private:
  void handleFrame(std::uint8_t id, const std::uint8_t* p, std::uint8_t len) {
    Meas m;
    if (id == NAV_MSG_VO_VELOCITY && len == sizeof(nav_vo_velocity_t)) {
      nav_vo_velocity_t v;
      std::memcpy(&v, p, sizeof v);
      m.kind = Meas::kVo;
      m.vo.t_us = v.t_us;
      m.vo.vb = vec3(v.vb[0], v.vb[1], v.vb[2]);
      m.vo.sigma = v.sigma;
      meas_q_.push(m);
    } else if (id == NAV_MSG_POSITION_FIX && len == sizeof(nav_position_fix_t)) {
      nav_position_fix_t f;
      std::memcpy(&f, p, sizeof f);
      m.kind = Meas::kFix;
      m.fix.t_us = f.t_us;
      m.fix.lat = f.lat_rad;
      m.fix.lon = f.lon_rad;
      m.fix.h = f.alt_m;
      m.fix.sigma_ne = f.sigma_ne_m;
      m.fix.sigma_d = f.sigma_d_m;
      m.fix.yaw = f.yaw_rad;
      m.fix.source = static_cast<FixSource>(f.source);
      meas_q_.push(m);
    }
  }

  void dispatch(const Meas& m) {
    switch (m.kind) {
      case Meas::kVo: core_.onVoVelocity(m.vo); break;
      case Meas::kFix: core_.onPositionFix(m.fix); break;
      case Meas::kAirspeed: core_.onAirspeed(m.as); break;
      case Meas::kBaro: core_.onBaro(m.baro); break;
      case Meas::kHeading: core_.onHeading(m.mag); break;
      case Meas::kRadar: core_.onRadarAlt(m.radar); break;
    }
  }

  void publish(const NavOutput& o) {
    nav_output_t msg{};
    msg.t_us = o.t_us;
    msg.lat_rad = o.lat;
    msg.lon_rad = o.lon;
    msg.alt_m = static_cast<float>(o.h);
    for (int i = 0; i < 3; ++i) {
      msg.vel_ned[i] = static_cast<float>(o.vn[i]);
      msg.euler_rad[i] = static_cast<float>(o.euler[i]);
      msg.gyro_rad_s[i] = static_cast<float>(o.gyro[i]);
    }
    msg.sigma_ne_m = static_cast<float>(o.sigma_ne);
    msg.sigma_d_m = static_cast<float>(o.sigma_d);
    msg.mode = o.mode;
    msg.health = o.health;
    send(ap_, NAV_MSG_NAV_OUTPUT, &msg, sizeof msg);
    send(comp_, NAV_MSG_NAV_OUTPUT, &msg, sizeof msg);
    ++outputs_;
  }

  void send(hal::ILink& l, std::uint8_t id, const void* p, std::size_t len) {
    std::uint8_t frame[NAV_MAX_FRAME];
    const std::size_t n = nav_encode(id, seq_++, p, static_cast<std::uint8_t>(len), frame, sizeof frame);
    if (n) l.write(frame, n);
  }

  hal::ITimebase& tb_;
  hal::IImuArrayDriver& imu_;
  hal::IAirData& air_;
  hal::IMagnetometer& mag_;
  hal::IRadarAltimeter& ra_;
  hal::ICameraTrigger& cam_;
  hal::ILink& comp_;
  hal::ILink& ap_;

  NavCore core_;
  ImuArray<hal::kImuCount> array_;
  SpscQueue<ImuSample, 64> imu_q_;
  SpscQueue<Meas, 64> meas_q_;
  Vec3 acc_g_, acc_a_;
  int acc_n_ = 0;
  nav_decoder_t dec_;
  std::uint16_t seq_ = 0;
  std::uint32_t outputs_ = 0;
};

}  // namespace app

// ======================= 宿主机桩实现 =======================
namespace {

using nav::TimeUs;

struct HostClock : hal::ITimebase {
  TimeUs t = 0;
  TimeUs nowUs() const override { return t; }
  bool lastPps(TimeUs& p) const override { p = t - t % 1000000; return true; }
};

// 静止水平放置的 IMU：比力 = -g（FRD 机体系 z 向下）
struct HostImu : hal::IImuArrayDriver {
  HostClock& clk;
  explicit HostImu(HostClock& c) : clk(c) {}
  bool read(nav::RawImu (&out)[hal::kImuCount], TimeUs& t) override {
    for (auto& r : out) {
      r.gyro = nav::Vec3{};
      r.accel = nav::vec3(0, 0, -9.7936);
      r.temp_c = 30;
    }
    t = clk.t;
    return true;
  }
};

struct HostAir : hal::IAirData {
  bool readAirspeed(nav::Airspeed&) override { return false; }
  bool readBaro(nav::BaroAlt& z) override { z = {0, 1500.0, 2.0}; return false; }
};
struct HostMag : hal::IMagnetometer { bool readHeading(nav::MagHeading&) override { return false; } };
struct HostRadar : hal::IRadarAltimeter { bool read(nav::RadarAlt&) override { return false; } };
struct HostCam : hal::ICameraTrigger {
  bool poll(TimeUs&, std::uint32_t&, std::uint8_t&) override { return false; }
};
struct HostLink : hal::ILink {
  std::size_t bytes = 0;
  std::size_t write(const std::uint8_t*, std::size_t n) override { bytes += n; return n; }
  std::size_t read(std::uint8_t*, std::size_t) override { return 0; }
};

}  // namespace

int main() {
  HostClock clk;
  HostImu imu(clk);
  HostAir air;
  HostMag mag;
  HostRadar ra;
  HostCam cam;
  HostLink comp, ap;
  static app::NavApp nav_app(clk, imu, air, mag, ra, cam, comp, ap);

  nav::NavState x0;
  x0.lat = 30 * nav::kDeg;
  x0.lon = 110 * nav::kDeg;
  x0.h = 1500;
  nav_app.init(x0);

  // 模拟 2 s：2 kHz IMU 中断，1 kHz 调度 NavTask，100 Hz 调度其余任务
  for (int k = 1; k <= 4000; ++k) {
    clk.t = static_cast<TimeUs>(k) * 500;
    nav_app.onImuDataReady();
    if (k % 2 == 0) nav_app.navTask();
    if (k % 20 == 0) { nav_app.sensorTask(); nav_app.linkTask(); }
  }
  const auto& s = nav_app.core().filter().state();
  std::printf("2 s 输出帧数 %u（期望约 60），飞控链路 %zu 字节，静止漂移 %.3f m/s\n", nav_app.outputs(), ap.bytes,
              nav::norm(s.vn));
  return nav_app.outputs() >= 58 ? 0 : 1;
}
