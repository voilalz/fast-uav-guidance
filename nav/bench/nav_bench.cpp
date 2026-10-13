// 导航核心计算耗时与内存基准（宿主机）。
//
// 按目标板的调度喂数：IMU 400 Hz（协方差 100 Hz 传播）、VO 30 Hz、空速/气压 50 Hz、磁航向 10 Hz、
// 景象匹配位置修正 1 Hz（含 250 ms 时延，经历史缓冲回溯），统计每类调用的耗时分布与单位飞行时间的
// 总计算量。量测值取真值附近的常值，耗时只取决于固定维数的矩阵运算，与数值无关。
//
//   nav_bench [seconds=600] [notime] [mask=15]
// mask：启用的辅助量测（1 VO，2 空速+气压，4 磁航向，8 位置修正），用于按类型分摊指令数
// notime：不计时、不统计，只跑导航核心（在指令级模拟器下按指令数估算目标处理器负载时用）
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "nav/nav_core.hpp"

using namespace nav;
using Clock = std::chrono::steady_clock;

namespace {

bool g_time = true;

Clock::time_point tick() { return g_time ? Clock::now() : Clock::time_point{}; }

struct Stat {
  const char* name;
  std::vector<double> us;
  void add(Clock::time_point a, Clock::time_point b) {
    if (!g_time) return;
    us.push_back(std::chrono::duration<double, std::micro>(b - a).count());
  }
  double pct(double p) const {
    std::vector<double> v = us;
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[std::min(v.size() - 1, static_cast<std::size_t>(p * v.size()))];
  }
  double mean() const {
    double s = 0;
    for (double x : us) s += x;
    return us.empty() ? 0 : s / us.size();
  }
  double sum() const { return mean() * us.size(); }
};

}  // namespace

int main(int argc, char** argv) {
  const double secs = argc > 1 ? std::atof(argv[1]) : 600.0;
  g_time = !(argc > 2 && !std::strcmp(argv[2], "notime"));
  const int mask = argc > 3 ? std::atoi(argv[3]) : 15;
  NavCoreConfig cfg;
  cfg.recovery.enable = true;
  cfg.recovery.min_separation_s = 3.0;
  cfg.recovery.sigma_floor_rate = 2.0;
  cfg.recovery.sigma_floor_trigger = 10;
  NavCore core(cfg);
  NavState x0;
  x0.lat = 50.0 * kDeg;
  x0.lon = 0.1222;
  x0.h = 1500;
  x0.vn = Vec3{166.7, 0, 0};
  core.init(x0, InitSigma{});

  Stat s_imu{"IMU predict (400 Hz)", {}}, s_imu_cov{"  of which with covariance step (100 Hz)", {}},
      s_vo{"VO velocity update (30 Hz)", {}}, s_as{"Airspeed update (50 Hz)", {}}, s_baro{"Baro update (50 Hz)", {}},
      s_mag{"Mag heading update (10 Hz)", {}}, s_fix{"Delayed position fix (1 Hz)", {}};
  const TimeUs dt_imu = 2500;  // 400 Hz
  TimeUs t = 0;
  int k = 0;
  const double g = 9.80665;
  NavOutput out;
  while (t * 1e-6 < secs) {
    t += dt_imu;
    ++k;
    ImuSample imu;
    imu.t_us = t;
    imu.gyro = Vec3{0, 0, 0};
    imu.accel = Vec3{0, 0, -g};
    auto a = tick();
    core.onImu(imu, out);
    auto b = tick();
    s_imu.add(a, b);
    if (k % 4 == 0) s_imu_cov.add(a, b);
    if ((mask & 1) && k % 13 == 0) {  // ≈30 Hz
      VoVelocity z;
      z.t_us = t;
      z.vb = Vec3{166.7, 0, 0};
      a = tick();
      core.onVoVelocity(z);
      s_vo.add(a, tick());
    }
    if ((mask & 2) && k % 8 == 0) {  // 50 Hz
      Airspeed as;
      as.t_us = t;
      as.tas = 166.7;
      a = tick();
      core.onAirspeed(as);
      s_as.add(a, tick());
      BaroAlt ba;
      ba.t_us = t;
      ba.alt = 1500;
      a = tick();
      core.onBaro(ba);
      s_baro.add(a, tick());
    }
    if ((mask & 4) && k % 40 == 0) {  // 10 Hz
      MagHeading mh;
      mh.t_us = t;
      a = tick();
      core.onHeading(mh);
      s_mag.add(a, tick());
    }
    if ((mask & 8) && k % 400 == 0 && t > 1000000) {  // 1 Hz，250 ms 时延
      const NavState& st = core.filter().state();
      PositionFix z;
      z.t_us = t - 250000;
      z.lat = st.lat;
      z.lon = st.lon;
      z.sigma_ne = 10;
      core.noteFixAttempt();
      a = tick();
      core.onPositionFix(z);
      s_fix.add(a, tick());
    }
  }
  if (!g_time) {
    std::printf("done %.0f s\n", secs);
    return 0;
  }
  std::printf("NavCore sizeof = %zu bytes (Eskf %zu bytes), no heap allocation\n", sizeof(NavCore), sizeof(Eskf));
  std::printf("%-42s %8s %8s %8s %8s %8s\n", "call", "n", "mean_us", "p50_us", "p99_us", "max_us");
  double total = 0;
  for (Stat* s : {&s_imu, &s_imu_cov, &s_vo, &s_as, &s_baro, &s_mag, &s_fix}) {
    std::printf("%-42s %8zu %8.2f %8.2f %8.2f %8.2f\n", s->name, s->us.size(), s->mean(), s->pct(0.5), s->pct(0.99),
                s->pct(1.0));
    if (s != &s_imu_cov) total += s->sum();
  }
  std::printf("total CPU per second of flight: %.1f us (%.3f %% of one core)\n", total / secs, total / secs / 1e4);
  return 0;
}
