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
  hist_.push(x0);
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
    hist_.push(eskf_.state());
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
  if (!hist_.lookup(t, out)) {
    ++st.too_old;
    return false;
  }
  return true;
}

void NavCore::onPositionFix(const PositionFix& z) {
  NavState r;
  if (!ref(z.t_us, r, fix_stats_)) return;
  const UpdateResult u = eskf_.updatePosition(z, r);
  if (u.accepted) {
    ++fix_stats_.accepted;
    last_fix_ = last_imu_;
    if (z.source == FixSource::kSceneMatch) last_scene_fix_ = last_imu_;
    health_ &= ~kHealthFixRejected;
  } else {
    ++fix_stats_.rejected;
    health_ |= kHealthFixRejected;
  }
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
