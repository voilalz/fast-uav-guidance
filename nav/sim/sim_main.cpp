// 闭环数字仿真：600 km/h、1 km 高度、商用级 MEMS + 视觉里程计 + 景象/地形匹配 + 空速 + 气压 + 磁航向。
// 用于验证 NavCore/ESKF 的实现与方案的精度预算。宿主机运行：./nav_sim [seconds] [seed]
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "nav/earth.hpp"
#include "nav/nav_core.hpp"
#include "nav/strapdown.hpp"

using namespace nav;

namespace {

constexpr real kV = 600.0 / 3.6;
constexpr real kImuHz = 400.0;

struct Scenario {
  const char* name;
  bool vo = true, fixes = true, airspeed = true, heading = true;
};

struct Rng {
  std::mt19937 g;
  explicit Rng(unsigned s) : g(s) {}
  real n(real sigma) { return std::normal_distribution<real>(0.0, sigma)(g); }
  real u() { return std::uniform_real_distribution<real>(0.0, 1.0)(g); }
};

// 航向角速度剖面（含 5 s 平滑过渡）：直线 + 三次转弯
real yawRate(real t) {
  struct Turn { real t0, t1, rate; };
  static const Turn turns[] = {{200, 230, 3.0 * kDeg}, {500, 530, -3.0 * kDeg}, {800, 845, 2.0 * kDeg}};
  for (const Turn& tr : turns) {
    if (t < tr.t0 || t > tr.t1) continue;
    const real ramp = 5.0;
    real k = 1.0;
    if (t < tr.t0 + ramp) k = (t - tr.t0) / ramp;
    if (t > tr.t1 - ramp) k = (tr.t1 - t) / ramp;
    return tr.rate * k;
  }
  return 0.0;
}

// 带交付时延的量测队列
template <class T>
struct Delayed { TimeUs deliver; T z; };

struct Result {
  real rms_h = 0, max_h = 0, final_h = 0;
  real err_at[6] = {};  // 60, 300, 600, 900, 1200, end
  int outputs = 0;
  MeasurementStats fix;
};

Result run(const Scenario& sc, real duration, unsigned seed, bool verbose) {
  Rng rng(seed);
  const real dt = 1.0 / kImuHz;
  const TimeUs dt_us = static_cast<TimeUs>(1e6 / kImuHz);

  // ---- 真值初始状态 ----
  NavState truth;
  truth.lat = 30.0 * kDeg;
  truth.lon = 110.0 * kDeg;
  truth.h = 1500.0;
  real yaw = 45.0 * kDeg;
  const real pitch = 2.0 * kDeg;  // 巡航攻角
  truth.vn = vec3(kV * std::cos(yaw), kV * std::sin(yaw), 0);
  truth.q = Quat::fromEuler(0, pitch, yaw);
  const real wind_n = 6.0, wind_e = -4.0, baro_bias = 25.0, mag_bias = 0.5 * kDeg, vo_scale = 0.008;

  // ---- 商用级 MEMS 误差（偏保守）----
  Vec3 bg, ba;
  for (int i = 0; i < 3; ++i) {
    bg[i] = rng.n(10.0 * kDeg / 3600.0);
    ba[i] = rng.n(2e-3 * 9.80665);
  }
  const real arw = 0.3 * kDeg / 60.0, vrw = 0.05 / 60.0, gyro_sf = 500e-6;

  // ---- 导航初值：GNSS 拒止发生时刻的组合导航残差 ----
  NavState x0 = truth;
  const earth::Radii r0 = earth::radii(truth.lat);
  x0.lat += rng.n(5.0) / r0.rn;
  x0.lon += rng.n(5.0) / (r0.re * std::cos(truth.lat));
  x0.h += rng.n(5.0);
  for (int i = 0; i < 3; ++i) x0.vn[i] += rng.n(0.1);
  x0.q = Quat::exp(vec3(rng.n(0.1 * kDeg), rng.n(0.1 * kDeg), rng.n(0.5 * kDeg))) * truth.q;
  x0.bg = Vec3{};
  x0.ba = Vec3{};

  NavCoreConfig cfg;
  NavCore core(cfg);
  core.init(x0, InitSigma{});

  std::vector<Delayed<VoVelocity>> q_vo;
  std::vector<Delayed<PositionFix>> q_fix;

  TimeUs t_us = 0;
  real sum2 = 0;
  long n_err = 0;
  Result res;
  int next_report = 60;
  real next_vo = 0, next_fix = 30, next_t3d = 45, next_50hz = 0, next_10hz = 0;

  const long steps = static_cast<long>(duration * kImuHz);
  for (long k = 0; k < steps; ++k) {
    const real t = k * dt;
    // ---- 真值推进，并反算理想 IMU ----
    yaw += yawRate(t + dt) * dt;
    const real bank = std::atan(kV * yawRate(t + dt) / 9.8);
    const Vec3 v1 = vec3(kV * std::cos(yaw), kV * std::sin(yaw), 0);
    const Quat q1 = Quat::fromEuler(bank, pitch, yaw);
    const Mat3 C = truth.q.toDcm();
    const Vec3 w_ie = earth::omegaIe(truth.lat), w_en = earth::omegaEn(truth.lat, truth.h, truth.vn);
    const Vec3 g = vec3(0, 0, earth::gravity(truth.lat, truth.h));
    const Vec3 f_n = (v1 - truth.vn) * (1.0 / dt) - g + cross(w_ie * 2.0 + w_en, truth.vn);
    const Vec3 f_b = C.t() * f_n;
    const Vec3 w_b = (truth.q.conj() * q1).log() * (1.0 / dt) + C.t() * (w_ie + w_en);
    Strapdown::propagate(truth, w_b, f_b, dt);
    t_us += dt_us;
    truth.t_us = t_us;

    // ---- 带误差的 IMU ----
    ImuSample imu;
    imu.t_us = t_us;
    for (int i = 0; i < 3; ++i) {
      imu.gyro[i] = w_b[i] * (1 + gyro_sf) + bg[i] + rng.n(arw / std::sqrt(dt));
      imu.accel[i] = f_b[i] + ba[i] + rng.n(vrw / std::sqrt(dt));
    }
    NavOutput out;
    if (core.onImu(imu, out)) ++res.outputs;

    const Mat3 Ct = truth.q.toDcm().t();
    // ---- 视觉里程计 30 Hz，交付时延 30 ms；600–610 s 云遮挡丢失 ----
    if (sc.vo && t >= next_vo) {
      next_vo += 1.0 / 30.0;
      if (!(t > 600 && t < 610)) {
        VoVelocity z;
        z.t_us = t_us;
        z.vb = Ct * truth.vn * (1 + vo_scale);
        for (int i = 0; i < 3; ++i) z.vb[i] += rng.n(0.5);
        z.sigma = 0.6;
        q_vo.push_back({t_us + 30000, z});
      }
    }
    // ---- 景象匹配：每 30 s（5 km）一次，σ 10 m，时延 250 ms，5% 错误匹配；
    //      600–720 s 模拟森林弱纹理区，无修正 ----
    if (sc.fixes && t >= next_fix) {
      next_fix += 30.0;
      if (!(t > 600 && t < 720)) {
        const bool outlier = rng.u() < 0.05;
        const real e_n = outlier ? 400.0 : rng.n(10.0), e_e = outlier ? -300.0 : rng.n(10.0);
        const earth::Radii r = earth::radii(truth.lat);
        PositionFix z;
        z.t_us = t_us;
        z.lat = truth.lat + e_n / r.rn;
        z.lon = truth.lon + e_e / (r.re * std::cos(truth.lat));
        z.sigma_ne = 10.0;
        z.source = FixSource::kSceneMatch;
        q_fix.push_back({t_us + 250000, z});
      }
    }
    // ---- 三维地形匹配：每 60 s 一次，含高度，σ 水平 20 m / 高度 10 m，时延 400 ms ----
    if (sc.fixes && t >= next_t3d) {
      next_t3d += 60.0;
      const earth::Radii r = earth::radii(truth.lat);
      PositionFix z;
      z.t_us = t_us;
      z.lat = truth.lat + rng.n(20.0) / r.rn;
      z.lon = truth.lon + rng.n(20.0) / (r.re * std::cos(truth.lat));
      z.h = truth.h + rng.n(10.0);
      z.sigma_ne = 20.0;
      z.sigma_d = 10.0;
      z.source = FixSource::kTerrain3D;
      q_fix.push_back({t_us + 400000, z});
    }
    // ---- 50 Hz：空速、气压 ----
    if (t >= next_50hz) {
      next_50hz += 0.02;
      if (sc.airspeed) {
        const Vec3 va = truth.vn - vec3(wind_n, wind_e, 0);
        core.onAirspeed({t_us, norm(va) + rng.n(1.0), 1.0});
      }
      core.onBaro({t_us, truth.h + baro_bias + rng.n(1.5), 2.0});
    }
    // ---- 10 Hz：磁航向 ----
    if (sc.heading && t >= next_10hz) {
      next_10hz += 0.1;
      const real yaw_t = truth.q.toEuler()[2];
      core.onHeading({t_us, wrapPi(yaw_t + mag_bias + rng.n(2.0 * kDeg)), 2.5 * kDeg});
    }
    // ---- 交付到期的时延量测 ----
    for (std::size_t i = 0; i < q_vo.size();) {
      if (q_vo[i].deliver <= t_us) { core.onVoVelocity(q_vo[i].z); q_vo.erase(q_vo.begin() + i); }
      else ++i;
    }
    for (std::size_t i = 0; i < q_fix.size();) {
      if (q_fix[i].deliver <= t_us) { core.onPositionFix(q_fix[i].z); q_fix.erase(q_fix.begin() + i); }
      else ++i;
    }

    // ---- 误差统计 ----
    const NavState& s = core.filter().state();
    const Vec3 e = earth::llaDiffToNed(truth.lat, truth.h, s.lat - truth.lat, s.lon - truth.lon, s.h - truth.h);
    const real eh = std::hypot(e[0], e[1]);
    if (k % 40 == 0) { sum2 += eh * eh; ++n_err; }
    if (eh > res.max_h) res.max_h = eh;
    res.final_h = eh;
    if (t + dt >= next_report) {
      const int slot = next_report == 60 ? 0 : next_report == 300 ? 1 : next_report == 600 ? 2
                     : next_report == 900 ? 3 : next_report == 1200 ? 4 : -1;
      if (slot >= 0) res.err_at[slot] = eh;
      if (verbose) {
        const real yaw_err = wrapPi(s.q.toEuler()[2] - truth.q.toEuler()[2]) / kDeg;
        std::printf("  t=%5d s  模式=M%d  水平误差=%8.1f m  估计σ=%7.1f m  高度误差=%6.1f m  航向误差=%6.2f°\n",
                    next_report, static_cast<int>(core.mode()), eh, core.filter().sigmaNe(), -e[2], yaw_err);
      }
      next_report += 60;
    }
  }
  res.rms_h = std::sqrt(sum2 / n_err);
  res.fix = core.fixStats();
  return res;
}

}  // namespace

int main(int argc, char** argv) {
  const real duration = argc > 1 ? std::atof(argv[1]) : 1200.0;
  const unsigned seed = argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : 1u;

  std::printf("场景 A：组合导航（VO + 景象/三维地形匹配 + 空速 + 气压 + 磁航向），航程 %.0f km\n", duration * kV / 1000);
  const Result a = run({"aided"}, duration, seed, true);
  std::printf("  => 水平 RMS %.1f m，最大 %.1f m，输出帧数 %d（%.1f Hz），绝对修正 接受 %u / 拒绝 %u / 超时 %u\n\n",
              a.rms_h, a.max_h, a.outputs, a.outputs / duration, a.fix.accepted, a.fix.rejected, a.fix.too_old);

  std::printf("场景 B：仅视觉里程计 + 空速 + 气压 + 磁航向（无绝对修正）\n");
  Scenario b{"vo_only"};
  b.fixes = false;
  const Result rb = run(b, duration, seed, false);
  std::printf("  => 1/5/10 min 水平误差 %.0f / %.0f / %.0f m，终点 %.0f m\n\n", rb.err_at[0], rb.err_at[1], rb.err_at[2],
              rb.final_h);

  std::printf("场景 C：纯惯导（仅气压定高）\n");
  Scenario c{"ins_only"};
  c.vo = c.fixes = c.airspeed = c.heading = false;
  const Result rc = run(c, duration, seed, false);
  std::printf("  => 1/5/10 min 水平误差 %.0f / %.0f / %.0f m\n", rc.err_at[0], rc.err_at[1], rc.err_at[2]);

  // 作为回归测试：组合导航 RMS 应在设计指标量级内，输出频率不低于 30 Hz
  const bool ok = a.rms_h < 50.0 && a.outputs / duration >= 29.5;
  std::printf("\n%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
