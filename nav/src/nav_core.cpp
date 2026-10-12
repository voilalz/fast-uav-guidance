#include "nav/nav_core.hpp"

#include "nav/earth.hpp"

namespace nav {

namespace {
TimeUs periodUs(real hz) { return static_cast<TimeUs>(1e6 / hz); }
}  // namespace

void NavCore::init(const NavState& x0, const InitSigma& s) {
  eskf_.init(x0, s);
  last_imu_ = x0.t_us;
  next_out_ = x0.t_us;
  next_hist_ = x0.t_us;
  last_fix_ = x0.t_us;
  last_scene_fix_ = x0.t_us;
  last_vo_ = x0.t_us;
  hist_.push({x0.t_us, x0, eskf_.cumulativeCorrection()});
}

bool NavCore::onImu(const ImuSample& imu, NavOutput& out) {
  if (imu.t_us <= last_imu_) return false;  // 时间戳回退：丢弃
  const real dt = (imu.t_us - last_imu_) * 1e-6;
  last_imu_ = imu.t_us;
  if (!imu.valid || dt > 0.05) {  // 丢帧过多：标记并跳过
    health_ |= kHealthImuDegraded;
    return false;
  }
  health_ &= ~kHealthImuDegraded;
  eskf_.predict(imu, dt);

  if (imu.t_us >= next_hist_) {
    hist_.push({eskf_.state().t_us, eskf_.state(), eskf_.cumulativeCorrection()});
    next_hist_ += periodUs(cfg_.history_rate_hz);
  }
  updateMode(imu.t_us);
  if (imu.t_us >= next_out_) {
    next_out_ += periodUs(cfg_.output_rate_hz);
    if (next_out_ <= imu.t_us) next_out_ = imu.t_us + periodUs(cfg_.output_rate_hz);
    out = makeOutput();
    return true;
  }
  return false;
}

bool NavCore::ref(TimeUs t, NavState& out, MeasurementStats& st) const {
  HistEntry e;
  if (!hist_.lookup(t, e)) {
    ++st.too_old;
    return false;
  }
  out = e.x;
  if (cfg_.delay_apply_corrections) Eskf::applyCorrection(out, eskf_.cumulativeCorrection() - e.cum);
  return true;
}

void NavCore::onPositionFix(const PositionFix& z) {
  NavState r;
  if (!ref(z.t_us, r, fix_stats_)) return;
  UpdateResult u = eskf_.updatePosition(z, r);
  bool recovered = false;
  if (!u.accepted && cfg_.recovery.enable) recovered = tryRecover(z, r, u);
  last_event_ = {z.t_us, z.source, u.nis, u.accepted, recovered};
  if (u.accepted) {
    ++fix_stats_.accepted;
    n_rej_ = 0;
    last_fix_ = last_imu_;
    if (z.source == FixSource::kSceneMatch) last_scene_fix_ = last_imu_;
    health_ &= ~kHealthFixRejected;
    if (!std::isnan(z.yaw) && ref(z.t_us, r, fix_stats_)) eskf_.updateFixHeading(z.yaw, z.sigma_yaw, r);
  } else {
    ++fix_stats_.rejected;
    health_ |= kHealthFixRejected;
  }
}

bool NavCore::tryRecover(const PositionFix& z, const NavState& r, UpdateResult& u) {
  Vec2 inn;
  Mat<2, 2> S;
  eskf_.positionInnovation(z, r, inn, S);
  const LockoutRecoveryConfig& rc = cfg_.recovery;
  // 超出时间窗口的旧记录作废
  if (n_rej_ > 0 && (z.t_us - rej_[n_rej_ - 1].t_us) * 1e-6 > rc.window_s) n_rej_ = 0;
  if (n_rej_ == kMaxRejected) {
    for (int i = 1; i < kMaxRejected; ++i) rej_[i - 1] = rej_[i];
    --n_rej_;
  }
  rej_[n_rej_++] = {z.t_us, inn, z.sigma_ne};
  if (n_rej_ < rc.consecutive) return false;

  // 最近 K 次被拒新息两两相邻一致
  for (int i = n_rej_ - rc.consecutive + 1; i < n_rej_; ++i) {
    const Rejected &a = rej_[i - 1], &b = rej_[i];
    const real dt = (b.t_us - a.t_us) * 1e-6;
    const real var = a.sigma * a.sigma + b.sigma * b.sigma + rc.drift_rate * rc.drift_rate * dt * dt;
    const real dn = b.r[0] - a.r[0], de = b.r[1] - a.r[1];
    if ((dn * dn + de * de) / var > rc.consistency_gate) return false;
  }

  // 放大运动学协方差，使该新息的 NIS ≈ 2（量测维数）
  const real p_mean = 0.5 * (S(0, 0) + S(1, 1)) - z.sigma_ne * z.sigma_ne;
  const real need = 0.5 * (inn[0] * inn[0] + inn[1] * inn[1]) - z.sigma_ne * z.sigma_ne;
  const real lambda = p_mean > 0 ? need / p_mean : 1.0;
  if (lambda <= 1.0) return false;
  eskf_.inflateKinematic(lambda);
  u = eskf_.updatePosition(z, r);
  if (u.accepted) ++recoveries_;
  return u.accepted;
}

void NavCore::onVoVelocity(const VoVelocity& z) {
  NavState r;
  if (!ref(z.t_us, r, aux_stats_)) return;
  if (eskf_.updateVoVelocity(z, r).accepted) {
    last_vo_ = last_imu_;
    ++aux_stats_.accepted;
  } else {
    ++aux_stats_.rejected;
  }
}

void NavCore::onAirspeed(const Airspeed& z) {
  NavState r;
  if (ref(z.t_us, r, aux_stats_)) eskf_.updateAirspeed(z, r);
}

void NavCore::onBaro(const BaroAlt& z) {
  NavState r;
  if (ref(z.t_us, r, aux_stats_)) eskf_.updateBaro(z, r);
}

void NavCore::onHeading(const MagHeading& z) {
  NavState r;
  if (ref(z.t_us, r, aux_stats_)) eskf_.updateHeading(z, r);
}

void NavCore::setDem(const Dem& dem, real origin_lat, real origin_lon) {
  dem_ = dem;
  dem_lat0_ = origin_lat;
  dem_lon0_ = origin_lon;
  have_dem_ = true;
}

void NavCore::onRadarAlt(const RadarAlt& z) {
  if (!have_dem_) return;
  NavState r;
  if (!ref(z.t_us, r, aux_stats_)) return;
  const Vec3 d = earth::llaDiffToNed(dem_lat0_, 0, r.lat - dem_lat0_, r.lon - dem_lon0_, 0);
  real h, gn, ge;
  if (!dem_.height(d[0], d[1], h) || !dem_.gradient(d[0], d[1], gn, ge)) return;
  eskf_.updateRadarAlt(z, r, h, gn, ge);
}

void NavCore::updateMode(TimeUs now) {
  if (now - last_vo_ > static_cast<TimeUs>(cfg_.vo_timeout_s * 1e6)) health_ |= kHealthVoLost;
  else health_ &= ~kHealthVoLost;

  const real sig = eskf_.sigmaNe();
  if (sig > cfg_.reacquire_sigma_m) health_ |= kHealthCovLarge;
  else health_ &= ~kHealthCovLarge;

  if (gnss_ok_) { mode_ = NavMode::kGnssIns; return; }
  if (sig > cfg_.reacquire_sigma_m) { mode_ = NavMode::kReacquire; return; }
  const TimeUs timeout = static_cast<TimeUs>(cfg_.fix_timeout_s * 1e6);
  if (now - last_fix_ > timeout) { mode_ = NavMode::kDeadReckon; return; }
  // 景象匹配在超时窗口内有效 -> M1，否则仅地形类修正有效 -> M2
  mode_ = (now - last_scene_fix_ <= timeout) ? NavMode::kVisionAided : NavMode::kTerrainAided;
}

NavOutput NavCore::makeOutput() const {
  const NavState& s = eskf_.state();
  NavOutput o;
  o.t_us = s.t_us;
  o.lat = s.lat;
  o.lon = s.lon;
  o.h = s.h;
  o.vn = s.vn;
  o.euler = s.q.toEuler();
  o.gyro = eskf_.lastGyroCorrected();
  o.sigma_ne = eskf_.sigmaNe();
  o.sigma_d = eskf_.sigmaD();
  o.mode = static_cast<std::uint8_t>(mode_);
  o.health = health_;
  return o;
}

}  // namespace nav
