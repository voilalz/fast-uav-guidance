#include "nav/eskf.hpp"

#include "nav/earth.hpp"
#include "nav/strapdown.hpp"

namespace nav {

namespace {
template <std::size_t R, std::size_t C>
void addBlock(StateCov& m, std::size_t r0, std::size_t c0, const Mat<R, C>& b) {
  for (std::size_t r = 0; r < R; ++r)
    for (std::size_t c = 0; c < C; ++c) m(r0 + r, c0 + c) += b(r, c);
}
}  // namespace

void Eskf::init(const NavState& x0, const InitSigma& s) {
  x_ = x0;
  P_ = StateCov::zeros();
  auto set3 = [&](std::size_t i, real a, real b, real c) {
    P_(i, i) = a * a; P_(i + 1, i + 1) = b * b; P_(i + 2, i + 2) = c * c;
  };
  set3(idx::P, s.pos, s.pos, s.pos);
  set3(idx::V, s.vel, s.vel, s.vel);
  set3(idx::TH, s.roll_pitch, s.roll_pitch, s.yaw);
  set3(idx::BG, s.bg, s.bg, s.bg);
  set3(idx::BA, s.ba, s.ba, s.ba);
  P_(idx::W, idx::W) = P_(idx::W + 1, idx::W + 1) = s.wind * s.wind;
  P_(idx::BB, idx::BB) = s.baro * s.baro;
  P_(idx::KV, idx::KV) = s.vo_scale * s.vo_scale;
  P_(idx::MB, idx::MB) = s.mag_bias * s.mag_bias;
  if (!cfg_.est_baro_bias) P_(idx::BB, idx::BB) = 0;
  if (!cfg_.est_vo_scale) P_(idx::KV, idx::KV) = 0;
  if (!cfg_.est_mag_bias) P_(idx::MB, idx::MB) = 0;
  cov_dt_acc_ = 0;
  cum_dx_ = StateVec{};
  acc_n_ = 0;
  f_acc_ = Vec3{};
  w_acc_ = Vec3{};
}

void Eskf::predict(const ImuSample& imu, real dt) {
  const Vec3 w = imu.gyro - x_.bg;
  const Vec3 f = imu.accel - x_.ba;
  last_gyro_ = w;
  Strapdown::propagate(x_, w, f, dt);
  x_.t_us = imu.t_us;

  // 名义状态中的马尔可夫项衰减
  const real kw = std::exp(-dt / cfg_.wind_tau);
  x_.wind *= kw;
  x_.baro_bias *= std::exp(-dt / cfg_.baro_tau);
  x_.mag_bias *= std::exp(-dt / cfg_.mag_bias_tau);

  f_acc_ += f;
  w_acc_ += w;
  ++acc_n_;
  cov_dt_acc_ += dt;
  if (cov_dt_acc_ >= cfg_.cov_period) {
    propagateCov(cov_dt_acc_);
    cov_dt_acc_ = 0;
  }
}

void Eskf::propagateCov(real dt) {
  const Vec3 f = f_acc_ * (1.0 / acc_n_);
  f_acc_ = Vec3{};
  w_acc_ = Vec3{};
  acc_n_ = 0;

  const Mat3 C = x_.q.toDcm();
  const Vec3 w_ie = earth::omegaIe(x_.lat);
  const Vec3 w_en = earth::omegaEn(x_.lat, x_.h, x_.vn);
  const Mat3 I3 = Mat3::identity();

  // 连续时间误差动力学 A，离散 Φ ≈ I + A dt
  StateCov Phi = StateCov::identity();
  addBlock(Phi, idx::P, idx::V, I3 * dt);
  addBlock(Phi, idx::V, idx::V, -skew(w_ie * 2.0 + w_en) * dt);
  addBlock(Phi, idx::V, idx::TH, -skew(C * f) * dt);
  addBlock(Phi, idx::V, idx::BA, -C * dt);
  addBlock(Phi, idx::TH, idx::TH, -skew(w_ie + w_en) * dt);
  addBlock(Phi, idx::TH, idx::BG, -C * dt);
  Phi(idx::W, idx::W) = Phi(idx::W + 1, idx::W + 1) = std::exp(-dt / cfg_.wind_tau);
  Phi(idx::BB, idx::BB) = std::exp(-dt / cfg_.baro_tau);
  Phi(idx::MB, idx::MB) = std::exp(-dt / cfg_.mag_bias_tau);
  if (cfg_.vo_scale_tau > 0) Phi(idx::KV, idx::KV) = std::exp(-dt / cfg_.vo_scale_tau);

  P_ = Phi * P_ * Phi.t();

  auto addDiag3 = [&](std::size_t i, real q) { for (int k = 0; k < 3; ++k) P_(i + k, i + k) += q; };
  addDiag3(idx::V, cfg_.accel_vrw * cfg_.accel_vrw * dt);
  addDiag3(idx::TH, cfg_.gyro_arw * cfg_.gyro_arw * dt);
  addDiag3(idx::BG, cfg_.gyro_bias_rw * cfg_.gyro_bias_rw * dt);
  addDiag3(idx::BA, cfg_.accel_bias_rw * cfg_.accel_bias_rw * dt);
  const real qw = 2 * cfg_.wind_sigma * cfg_.wind_sigma / cfg_.wind_tau * dt;
  P_(idx::W, idx::W) += qw;
  P_(idx::W + 1, idx::W + 1) += qw;
  if (cfg_.est_baro_bias) P_(idx::BB, idx::BB) += 2 * cfg_.baro_sigma * cfg_.baro_sigma / cfg_.baro_tau * dt;
  if (cfg_.est_vo_scale)
    P_(idx::KV, idx::KV) += cfg_.vo_scale_tau > 0
                                ? 2 * cfg_.vo_scale_sigma * cfg_.vo_scale_sigma / cfg_.vo_scale_tau * dt
                                : cfg_.vo_scale_rw * cfg_.vo_scale_rw * dt;
  if (cfg_.est_mag_bias)
    P_(idx::MB, idx::MB) += 2 * cfg_.mag_bias_sigma * cfg_.mag_bias_sigma / cfg_.mag_bias_tau * dt;

  // 数值对称化
  for (std::size_t r = 0; r < kN; ++r)
    for (std::size_t c = r + 1; c < kN; ++c) P_(r, c) = P_(c, r) = 0.5 * (P_(r, c) + P_(c, r));
}

template <std::size_t M>
UpdateResult Eskf::update(const Mat<M, 1>& r, const Mat<M, kN>& H, const Mat<M, M>& R, real gate) {
  UpdateResult res;
  const Mat<kN, M> PHt = P_ * H.t();
  const Mat<M, M> S = H * PHt + R;
  Mat<M, M> Si;
  if (!inverse(S, Si)) return res;
  res.nis = (r.t() * Si * r)(0, 0);
  if (res.nis > gate) return res;  // χ² 门限拒绝（错误匹配/野值）

  const Mat<kN, M> K = PHt * Si;
  const StateVec dx = K * r;
  // Joseph 形式，保证 P 对称正定
  const StateCov IKH = StateCov::identity() - K * H;
  P_ = IKH * P_ * IKH.t() + K * R * K.t();
  inject(dx);
  res.accepted = true;
  return res;
}

void Eskf::inject(const StateVec& dx) {
  applyCorrection(x_, dx);
  cum_dx_ += dx;
}

void Eskf::applyCorrection(NavState& x, const StateVec& dx) {
  const earth::Radii r = earth::radii(x.lat);
  x.lat += dx[idx::P] / (r.rn + x.h);
  x.lon += dx[idx::P + 1] / ((r.re + x.h) * std::cos(x.lat));
  x.h -= dx[idx::P + 2];
  for (int i = 0; i < 3; ++i) {
    x.vn[i] += dx[idx::V + i];
    x.bg[i] += dx[idx::BG + i];
    x.ba[i] += dx[idx::BA + i];
  }
  x.q = Quat::exp(vec3(dx[idx::TH], dx[idx::TH + 1], dx[idx::TH + 2])) * x.q;
  x.q.normalize();
  x.wind[0] += dx[idx::W];
  x.wind[1] += dx[idx::W + 1];
  x.baro_bias += dx[idx::BB];
  x.vo_scale += dx[idx::KV];
  x.mag_bias += dx[idx::MB];
  // 注：姿态误差复位雅可比 G = I - 0.5[δθ×] 在小修正量下近似为 I，此处省略。
}

UpdateResult Eskf::updatePosition(const PositionFix& z, const NavState& ref) {
  const Vec3 d = earth::llaDiffToNed(ref.lat, ref.h, z.lat - ref.lat, z.lon - ref.lon, z.h - ref.h);
  if (z.sigma_d > 0) {
    Mat<3, kN> H;
    for (int i = 0; i < 3; ++i) H(i, idx::P + i) = 1;
    Mat<3, 3> R;
    R(0, 0) = R(1, 1) = z.sigma_ne * z.sigma_ne;
    R(2, 2) = z.sigma_d * z.sigma_d;
    return update<3>(d, H, R, cfg_.gate_3d);
  }
  Mat<2, kN> H;
  H(0, idx::P) = H(1, idx::P + 1) = 1;
  Mat<2, 1> r;
  r[0] = d[0];
  r[1] = d[1];
  Mat<2, 2> R;
  R(0, 0) = R(1, 1) = z.sigma_ne * z.sigma_ne;
  return update<2>(r, H, R, cfg_.gate_pos);
}

bool Eskf::positionInnovation(const PositionFix& z, const NavState& ref, Vec2& r, Mat<2, 2>& S) const {
  const Vec3 d = earth::llaDiffToNed(ref.lat, ref.h, z.lat - ref.lat, z.lon - ref.lon, z.h - ref.h);
  r[0] = d[0];
  r[1] = d[1];
  S(0, 0) = P_(0, 0) + z.sigma_ne * z.sigma_ne;
  S(1, 1) = P_(1, 1) + z.sigma_ne * z.sigma_ne;
  S(0, 1) = S(1, 0) = P_(0, 1);
  return true;
}

void Eskf::inflateKinematic(real lambda) {
  if (lambda <= 1) return;
  const real s = std::sqrt(lambda);
  for (std::size_t r = 0; r < kN; ++r)
    for (std::size_t c = 0; c < kN; ++c) {
      const real dr = r < idx::BG ? s : 1.0, dc = c < idx::BG ? s : 1.0;
      P_(r, c) *= dr * dc;
    }
}

UpdateResult Eskf::updateVoVelocity(const VoVelocity& z, const NavState& ref) {
  const Mat3 Ct = ref.q.toDcm().t();
  const real s = 1.0 + ref.vo_scale;
  const Vec3 vb = Ct * ref.vn;
  const Vec3 r = z.vb - vb * s;
  Mat<3, kN> H;
  H.setBlock(0, idx::V, Ct * s);
  H.setBlock(0, idx::TH, Ct * skew(ref.vn) * s);
  H.setBlock(0, idx::KV, vb);
  Mat<3, 3> R = Mat<3, 3>::identity() * (z.sigma * z.sigma);
  return update<3>(r, H, R, cfg_.gate_3d);
}

UpdateResult Eskf::updateAirspeed(const Airspeed& z, const NavState& ref) {
  const Vec3 va = ref.vn - vec3(ref.wind[0], ref.wind[1], 0);
  const real n = norm(va);
  if (n < 10.0) return {};
  const Vec3 u = va * (1.0 / n);
  Mat<1, 1> r;
  r[0] = z.tas - n;
  Mat<1, kN> H;
  for (int i = 0; i < 3; ++i) H(0, idx::V + i) = u[i];
  H(0, idx::W) = -u[0];
  H(0, idx::W + 1) = -u[1];
  Mat<1, 1> R;
  R[0] = z.sigma * z.sigma;
  return update<1>(r, H, R, cfg_.gate_1d);
}

UpdateResult Eskf::updateBaro(const BaroAlt& z, const NavState& ref) {
  Mat<1, 1> r;
  r[0] = z.alt - (ref.h + ref.baro_bias);
  Mat<1, kN> H;
  H(0, idx::P + 2) = -1;
  H(0, idx::BB) = 1;
  Mat<1, 1> R;
  R[0] = z.sigma * z.sigma;
  return update<1>(r, H, R, cfg_.gate_1d);
}

UpdateResult Eskf::updateHeading(const MagHeading& z, const NavState& ref) {
  const Mat3 C = ref.q.toDcm();
  const real d = C(0, 0) * C(0, 0) + C(1, 0) * C(1, 0);
  if (d < 1e-6) return {};
  Mat<1, 1> r;
  r[0] = wrapPi(z.yaw - std::atan2(C(1, 0), C(0, 0)) - ref.mag_bias);
  Mat<1, kN> H;
  H(0, idx::TH) = -C(0, 0) * C(2, 0) / d;
  H(0, idx::TH + 1) = -C(1, 0) * C(2, 0) / d;
  H(0, idx::TH + 2) = 1;
  H(0, idx::MB) = 1;
  Mat<1, 1> R;
  R[0] = z.sigma * z.sigma;
  return update<1>(r, H, R, cfg_.gate_1d);
}

UpdateResult Eskf::updateFixHeading(real yaw, real sigma, const NavState& ref) {
  const Mat3 C = ref.q.toDcm();
  const real d = C(0, 0) * C(0, 0) + C(1, 0) * C(1, 0);
  if (d < 1e-6) return {};
  Mat<1, 1> r;
  r[0] = wrapPi(yaw - std::atan2(C(1, 0), C(0, 0)));
  Mat<1, kN> H;
  H(0, idx::TH) = -C(0, 0) * C(2, 0) / d;
  H(0, idx::TH + 1) = -C(1, 0) * C(2, 0) / d;
  H(0, idx::TH + 2) = 1;
  Mat<1, 1> R;
  R[0] = sigma * sigma;
  return update<1>(r, H, R, cfg_.gate_1d);
}

UpdateResult Eskf::updateRadarAlt(const RadarAlt& z, const NavState& ref, real dem_h, real grad_n,
                                  real grad_e) {
  Mat<1, 1> r;
  r[0] = z.agl - (ref.h - dem_h);
  Mat<1, kN> H;
  H(0, idx::P) = -grad_n;
  H(0, idx::P + 1) = -grad_e;
  H(0, idx::P + 2) = -1;
  Mat<1, 1> R;
  R[0] = z.sigma * z.sigma;
  return update<1>(r, H, R, cfg_.gate_1d);
}

}  // namespace nav
