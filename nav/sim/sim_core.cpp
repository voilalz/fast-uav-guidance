#include "sim_core.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>

#include "nav/earth.hpp"
#include "nav/strapdown.hpp"

namespace sim {

using namespace nav;

namespace {

constexpr real kV = 600.0 / 3.6;
constexpr real kImuHz = 400.0;

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

template <class T>
struct Delayed { TimeUs deliver; T z; bool outlier; };

// 一阶马尔可夫过程离散化
struct GaussMarkov {
  real x = 0, sigma = 0, tau = 1;
  void step(Rng& r, real dt) {
    if (sigma <= 0) return;
    const real a = std::exp(-dt / tau);
    x = a * x + r.n(sigma * std::sqrt(1 - a * a));
  }
};

real chi2Lo3 = 0.216, chi2Hi3 = 9.348;  // χ²(3) 2.5% / 97.5%

}  // namespace

bool setParam(SimConfig& c, const std::string& k, const std::string& v) {
  const real x = std::atof(v.c_str());
  const bool b = x != 0;
  if (k == "profile") {
    if (v == "realistic") {
      // 更真实的误差模型：真值与滤波器编排不同步、零偏温漂、加计比例因子、时变 VO 尺度、近距离错误匹配
      c.truth_substeps = 5;
      c.gyro_drift_dph = 3;
      c.accel_drift_mg = 0.5;
      c.accel_sf_ppm = 300;
      c.vo_scale_gm_sigma = 0.005;
      c.outlier_mode = 1;
      c.outlier_rate = 0.1;
      return true;
    }
    return v == "default";
  }
#define P(name) if (k == #name) { c.name = x; return true; }
#define PB(name) if (k == #name) { c.name = b; return true; }
#define PI(name) if (k == #name) { c.name = static_cast<int>(x); return true; }
  P(duration) if (k == "seed") { c.seed = static_cast<unsigned>(x); return true; }
  PI(truth_substeps) P(gyro_bias_dph) P(accel_bias_mg) P(gyro_arw_dpsh) P(accel_vrw_mpsh) P(gyro_sf_ppm)
  P(accel_sf_ppm) P(gyro_drift_dph) P(gyro_drift_tau) P(accel_drift_mg) P(accel_drift_tau)
  PB(vo) P(vo_rate) P(vo_delay) P(vo_scale) P(vo_scale_gm_sigma) P(vo_scale_gm_tau) P(vo_noise) P(vo_sigma)
  P(vo_gap_start) P(vo_gap_end)
  PB(fixes) P(fix_interval) P(fix_first) P(fix_delay) P(fix_noise) P(fix_sigma) P(outlier_rate) PI(outlier_mode)
  P(near_min) P(near_max) P(gap_start) P(gap_end) PB(fix_yaw) P(fix_yaw_noise_deg) P(fix_yaw_sigma_deg)
  PB(t3d) P(t3d_interval) P(t3d_first) P(t3d_delay) P(t3d_noise_ne) P(t3d_noise_d)
  PB(airspeed) PB(heading) P(wind_n) P(wind_e) P(baro_bias) P(mag_bias_deg) P(mag_noise_deg) P(mag_sigma_deg)
  PB(est_baro_bias) PB(est_vo_scale) PB(est_mag_bias) P(filter_vo_scale_rw) P(filter_vo_scale_sigma) P(filter_vo_scale_tau) PB(recovery) PB(delay_corr) PI(recovery_k) P(recovery_drift_rate)
#undef P
#undef PB
#undef PI
  return false;
}

bool parseArgs(SimConfig& c, int argc, char** argv, int first, std::string* err) {
  // 先应用 profile，使其余 key=value 可以覆盖 profile 中的取值
  for (int pass = 0; pass < 2; ++pass)
    for (int i = first; i < argc; ++i) {
      const std::string a = argv[i];
      const auto eq = a.find('=');
      if (eq == std::string::npos) continue;
      const std::string k = a.substr(0, eq), v = a.substr(eq + 1);
      if ((k == "profile") != (pass == 0)) continue;
      if (!setParam(c, k, v)) {
        if (err) *err = "未知参数: " + k;
        return false;
      }
    }
  return true;
}

std::string describe(const SimConfig& c) {
  std::ostringstream o;
  o << "substeps=" << c.truth_substeps << " gyro_drift=" << c.gyro_drift_dph << "dph vo_scale=" << c.vo_scale
    << " vo_gm=" << c.vo_scale_gm_sigma << " outlier=" << c.outlier_rate << "(mode " << c.outlier_mode << ")"
    << " fix_interval=" << c.fix_interval << " gap=[" << c.gap_start << "," << c.gap_end << "]"
    << " fix_yaw=" << c.fix_yaw << " mag_bias=" << c.mag_bias_deg << "deg"
    << " filter_vo(rw,sigma,tau)=" << c.filter_vo_scale_rw << "," << c.filter_vo_scale_sigma << ","
    << c.filter_vo_scale_tau << " est(baro,vo,mag)=" << c.est_baro_bias << c.est_vo_scale << c.est_mag_bias << " delay_corr=" << c.delay_corr << " recovery=" << c.recovery;
  return o.str();
}

Result run(const SimConfig& cfg, const Scenario& sc, bool record, bool verbose) {
  Rng rng(cfg.seed);
  Rng rng2(cfg.seed * 7919u + 17u);  // 新增误差项使用独立随机流，保证默认场景与历史结果逐位一致
  const real dt = 1.0 / kImuHz;
  const TimeUs dt_us = static_cast<TimeUs>(1e6 / kImuHz);
  const int nsub = cfg.truth_substeps < 1 ? 1 : cfg.truth_substeps;
  const real dts = dt / nsub;

  // ---- 真值初始状态 ----
  NavState truth;
  truth.lat = 30.0 * kDeg;
  truth.lon = 110.0 * kDeg;
  truth.h = 1500.0;
  real yaw = 45.0 * kDeg;
  const real pitch = 2.0 * kDeg;  // 巡航攻角
  truth.vn = vec3(kV * std::cos(yaw), kV * std::sin(yaw), 0);
  truth.q = Quat::fromEuler(0, pitch, yaw);
  const real mag_bias = cfg.mag_bias_deg * kDeg;

  // ---- 商用级 MEMS 误差 ----
  Vec3 bg, ba;
  for (int i = 0; i < 3; ++i) {
    bg[i] = rng.n(cfg.gyro_bias_dph * kDeg / 3600.0);
    ba[i] = rng.n(cfg.accel_bias_mg * 1e-3 * 9.80665);
  }
  const real arw = cfg.gyro_arw_dpsh * kDeg / 60.0, vrw = cfg.accel_vrw_mpsh / 60.0;
  const real gyro_sf = cfg.gyro_sf_ppm * 1e-6, accel_sf = cfg.accel_sf_ppm * 1e-6;
  GaussMarkov bg_drift[3], ba_drift[3], vo_gm;
  for (int i = 0; i < 3; ++i) {
    bg_drift[i] = {0, cfg.gyro_drift_dph * kDeg / 3600.0, cfg.gyro_drift_tau};
    ba_drift[i] = {0, cfg.accel_drift_mg * 1e-3 * 9.80665, cfg.accel_drift_tau};
  }
  vo_gm = {0, cfg.vo_scale_gm_sigma, cfg.vo_scale_gm_tau};

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

  NavCoreConfig ncfg;
  ncfg.eskf.est_baro_bias = cfg.est_baro_bias;
  ncfg.eskf.est_vo_scale = cfg.est_vo_scale;
  ncfg.eskf.est_mag_bias = cfg.est_mag_bias;
  ncfg.eskf.vo_scale_rw = cfg.filter_vo_scale_rw;
  if (cfg.filter_vo_scale_tau > 0) {
    ncfg.eskf.vo_scale_tau = cfg.filter_vo_scale_tau;
    ncfg.eskf.vo_scale_sigma = cfg.filter_vo_scale_sigma;
  }
  ncfg.delay_apply_corrections = cfg.delay_corr;
  ncfg.recovery.enable = cfg.recovery;
  ncfg.recovery.consecutive = cfg.recovery_k;
  ncfg.recovery.drift_rate = cfg.recovery_drift_rate;
  NavCore core(ncfg);
  core.init(x0, InitSigma{});

  std::vector<Delayed<VoVelocity>> q_vo;
  std::vector<Delayed<PositionFix>> q_fix;

  TimeUs t_us = 0;
  real sum2 = 0;
  long n_err = 0;
  Result res;
  int next_report = 60;
  real next_vo = 0, next_fix = cfg.fix_first, next_t3d = cfg.t3d_first, next_50hz = 0, next_10hz = 0;
  real next_sample = 1.0;
  real nees_pos_sum = 0, nees_kin_sum = 0;
  int nees_n = 0, nees_in = 0;
  int consec_false_rej = 0;
  real lockout_start = 0;

  const long steps = static_cast<long>(cfg.duration * kImuHz);
  for (long k = 0; k < steps; ++k) {
    const real t = k * dt;
    // ---- 真值推进（可选子步），并反算理想 IMU ----
    Vec3 w_sum, f_sum;
    for (int s = 0; s < nsub; ++s) {
      const real ts = t + (s + 1) * dts;
      yaw += yawRate(ts) * dts;
      const real bank = std::atan(kV * yawRate(ts) / 9.8);
      const Vec3 v1 = vec3(kV * std::cos(yaw), kV * std::sin(yaw), 0);
      const Quat q1 = Quat::fromEuler(bank, pitch, yaw);
      const Mat3 C = truth.q.toDcm();
      const Vec3 w_ie = earth::omegaIe(truth.lat), w_en = earth::omegaEn(truth.lat, truth.h, truth.vn);
      const Vec3 g = vec3(0, 0, earth::gravity(truth.lat, truth.h));
      const Vec3 f_n = (v1 - truth.vn) * (1.0 / dts) - g + cross(w_ie * 2.0 + w_en, truth.vn);
      const Vec3 f_b = C.t() * f_n;
      const Vec3 w_b = (truth.q.conj() * q1).log() * (1.0 / dts) + C.t() * (w_ie + w_en);
      Strapdown::propagate(truth, w_b, f_b, dts);
      w_sum += w_b;
      f_sum += f_b;
    }
    const Vec3 w_b = w_sum * (1.0 / nsub), f_b = f_sum * (1.0 / nsub);
    t_us += dt_us;
    truth.t_us = t_us;

    // ---- 带误差的 IMU ----
    for (int i = 0; i < 3; ++i) {
      bg_drift[i].step(rng2, dt);
      ba_drift[i].step(rng2, dt);
    }
    ImuSample imu;
    imu.t_us = t_us;
    for (int i = 0; i < 3; ++i) {
      imu.gyro[i] = w_b[i] * (1 + gyro_sf) + bg[i] + bg_drift[i].x + rng.n(arw / std::sqrt(dt));
      imu.accel[i] = f_b[i] * (1 + accel_sf) + ba[i] + ba_drift[i].x + rng.n(vrw / std::sqrt(dt));
    }
    NavOutput out;
    if (core.onImu(imu, out)) ++res.outputs;

    const Mat3 Ct = truth.q.toDcm().t();
    // ---- 视觉里程计 ----
    vo_gm.step(rng2, dt);
    if (sc.vo && cfg.vo && t >= next_vo) {
      next_vo += 1.0 / cfg.vo_rate;
      if (!(t > cfg.vo_gap_start && t < cfg.vo_gap_end)) {
        VoVelocity z;
        z.t_us = t_us;
        z.vb = Ct * truth.vn * (1 + cfg.vo_scale + vo_gm.x);
        for (int i = 0; i < 3; ++i) z.vb[i] += rng.n(cfg.vo_noise);
        z.sigma = cfg.vo_sigma;
        q_vo.push_back({t_us + static_cast<TimeUs>(cfg.vo_delay * 1e6), z, false});
      }
    }
    // ---- 景象匹配 ----
    if (sc.fixes && cfg.fixes && t >= next_fix) {
      next_fix += cfg.fix_interval;
      if (!(t > cfg.gap_start && t < cfg.gap_end)) {
        const bool outlier = rng.u() < cfg.outlier_rate;
        real e_n, e_e;
        if (!outlier) {
          e_n = rng.n(cfg.fix_noise);
          e_e = rng.n(cfg.fix_noise);
        } else if (cfg.outlier_mode == 0) {
          e_n = 400.0;
          e_e = -300.0;
        } else {
          const real m = cfg.near_min + (cfg.near_max - cfg.near_min) * rng2.u();
          const real a = 2 * kPi * rng2.u();
          e_n = m * std::cos(a);
          e_e = m * std::sin(a);
        }
        const earth::Radii r = earth::radii(truth.lat);
        PositionFix z;
        z.t_us = t_us;
        z.lat = truth.lat + e_n / r.rn;
        z.lon = truth.lon + e_e / (r.re * std::cos(truth.lat));
        z.sigma_ne = cfg.fix_sigma;
        z.source = FixSource::kSceneMatch;
        if (cfg.fix_yaw) {
          const real yt = truth.q.toEuler()[2];
          z.yaw = wrapPi(yt + (outlier ? 3.0 * kDeg : 0.0) + rng2.n(cfg.fix_yaw_noise_deg * kDeg));
          z.sigma_yaw = cfg.fix_yaw_sigma_deg * kDeg;
        }
        q_fix.push_back({t_us + static_cast<TimeUs>(cfg.fix_delay * 1e6), z, outlier});
      }
    }
    // ---- 三维地形匹配（含高度）----
    if (sc.fixes && cfg.t3d && t >= next_t3d) {
      next_t3d += cfg.t3d_interval;
      const earth::Radii r = earth::radii(truth.lat);
      PositionFix z;
      z.t_us = t_us;
      z.lat = truth.lat + rng.n(cfg.t3d_noise_ne) / r.rn;
      z.lon = truth.lon + rng.n(cfg.t3d_noise_ne) / (r.re * std::cos(truth.lat));
      z.h = truth.h + rng.n(cfg.t3d_noise_d);
      z.sigma_ne = cfg.t3d_noise_ne;
      z.sigma_d = cfg.t3d_noise_d;
      z.source = FixSource::kTerrain3D;
      q_fix.push_back({t_us + static_cast<TimeUs>(cfg.t3d_delay * 1e6), z, false});
    }
    // ---- 50 Hz：空速、气压 ----
    if (t >= next_50hz) {
      next_50hz += 0.02;
      if (sc.airspeed && cfg.airspeed) {
        const Vec3 va = truth.vn - vec3(cfg.wind_n, cfg.wind_e, 0);
        core.onAirspeed({t_us, norm(va) + rng.n(1.0), 1.0});
      }
      core.onBaro({t_us, truth.h + cfg.baro_bias + rng.n(1.5), 2.0});
    }
    // ---- 10 Hz：磁航向 ----
    if (sc.heading && cfg.heading && t >= next_10hz) {
      next_10hz += 0.1;
      const real yaw_t = truth.q.toEuler()[2];
      core.onHeading({t_us, wrapPi(yaw_t + mag_bias + rng.n(cfg.mag_noise_deg * kDeg)), cfg.mag_sigma_deg * kDeg});
    }
    // ---- 交付到期的时延量测 ----
    for (std::size_t i = 0; i < q_vo.size();) {
      if (q_vo[i].deliver <= t_us) { core.onVoVelocity(q_vo[i].z); q_vo.erase(q_vo.begin() + i); }
      else ++i;
    }
    for (std::size_t i = 0; i < q_fix.size();) {
      if (q_fix[i].deliver > t_us) { ++i; continue; }
      const Delayed<PositionFix>& d = q_fix[i];
      const MeasurementStats before = core.fixStats();
      core.onPositionFix(d.z);
      const MeasurementStats& after = core.fixStats();
      if (after.too_old == before.too_old) {
        const FixEvent& ev = core.lastFixEvent();
        if (d.outlier) (ev.accepted ? res.outlier_accepted : res.outlier_rejected)++;
        else (ev.accepted ? res.true_accepted : res.true_rejected)++;
        if (!d.outlier && !ev.accepted) {
          if (consec_false_rej == 0) lockout_start = t;
          ++consec_false_rej;
          if (consec_false_rej == 2) ++res.lockout_episodes;
          if (consec_false_rej >= 2 && t - lockout_start > res.lockout_longest_s) res.lockout_longest_s = t - lockout_start;
        } else if (!d.outlier) {
          consec_false_rej = 0;
        }
        if (record) {
          const earth::Radii r = earth::radii(d.z.lat);
          const real mn = (d.z.lat - truth.lat) * r.rn, me = (d.z.lon - truth.lon) * r.re * std::cos(truth.lat);
          res.fixes.push_back({t, static_cast<int>(d.z.source), d.outlier, ev.accepted, ev.recovered, ev.nis,
                               std::hypot(mn, me)});
        }
      }
      q_fix.erase(q_fix.begin() + i);
    }

    // ---- 误差统计 ----
    const NavState& s = core.filter().state();
    const Vec3 e = earth::llaDiffToNed(s.lat, s.h, truth.lat - s.lat, truth.lon - s.lon, truth.h - s.h);
    const real eh = std::hypot(e[0], e[1]);
    if (k % 40 == 0) { sum2 += eh * eh; ++n_err; }
    if (eh > res.max_h) res.max_h = eh;
    res.final_h = eh;
    if (t + dt >= next_sample) {
      next_sample += 1.0;
      const StateCov& P = core.filter().cov();
      Mat<9, 1> ek;
      const Vec3 phi = (truth.q * s.q.conj()).log();
      for (int i = 0; i < 3; ++i) {
        ek[i] = e[i];
        ek[3 + i] = truth.vn[i] - s.vn[i];
        ek[6 + i] = phi[i];
      }
      Mat<3, 3> Pp = P.block<3, 3>(0, 0), Ppi;
      Mat<9, 9> Pk = P.block<9, 9>(0, 0), Pki;
      real nees_p = 0, nees_k = 0;
      if (inverse(Pp, Ppi)) nees_p = (e.t() * Ppi * e)(0, 0);
      if (inverse(Pk, Pki)) nees_k = (ek.t() * Pki * ek)(0, 0);
      nees_pos_sum += nees_p;
      nees_kin_sum += nees_k;
      ++nees_n;
      if (nees_p >= chi2Lo3 && nees_p <= chi2Hi3) ++nees_in;
      if (record)
        res.samples.push_back({t + dt, e[0], e[1], e[2], eh, core.filter().sigmaNe(), nees_p, nees_k,
                               static_cast<int>(core.mode())});
    }
    if (t + dt >= next_report) {
      const int slot = next_report == 60 ? 0 : next_report == 300 ? 1 : next_report == 600 ? 2
                     : next_report == 900 ? 3 : next_report == 1200 ? 4 : -1;
      if (slot >= 0) res.err_at[slot] = eh;
      if (verbose) {
        const real yaw_err = wrapPi(s.q.toEuler()[2] - truth.q.toEuler()[2]) / kDeg;
        std::printf("  t=%5d s  模式=M%d  水平误差=%8.1f m  估计σ=%7.1f m  高度误差=%6.1f m  航向误差=%6.2f°\n",
                    next_report, static_cast<int>(core.mode()), eh, core.filter().sigmaNe(), e[2], yaw_err);
      }
      next_report += 60;
    }
  }
  res.rms_h = std::sqrt(sum2 / n_err);
  res.fix = core.fixStats();
  res.recoveries = core.recoveries();
  if (nees_n > 0) {
    res.nees_pos_mean = nees_pos_sum / nees_n;
    res.nees_kin_mean = nees_kin_sum / nees_n;
    res.nees_pos_in95 = static_cast<real>(nees_in) / nees_n;
  }
  return res;
}

}  // namespace sim
