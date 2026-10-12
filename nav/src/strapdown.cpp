#include "nav/strapdown.hpp"

#include "nav/earth.hpp"

namespace nav {

void Strapdown::propagate(NavState& s, const Vec3& gyro, const Vec3& accel, real dt) {
  const Vec3 w_ie = earth::omegaIe(s.lat);
  const Vec3 w_en = earth::omegaEn(s.lat, s.h, s.vn);
  const Mat3 c_bn = s.q.toDcm();

  // 速度：v' = C f + g - (2 w_ie + w_en) x v
  Vec3 g = vec3(0, 0, earth::gravity(s.lat, s.h));
  Vec3 dv = c_bn * accel + g - cross(w_ie * 2.0 + w_en, s.vn);
  const Vec3 v_old = s.vn;
  s.vn += dv * dt;

  // 位置：梯形积分
  const earth::Radii r = earth::radii(s.lat);
  const Vec3 v_mid = (v_old + s.vn) * 0.5;
  s.lat += v_mid[0] / (r.rn + s.h) * dt;
  s.lon += v_mid[1] / ((r.re + s.h) * std::cos(s.lat)) * dt;
  s.h -= v_mid[2] * dt;

  // 姿态：q <- q ⊗ Exp((ω_ib - C^T ω_in) dt)
  const Vec3 w_nb = gyro - c_bn.t() * (w_ie + w_en);
  s.q = s.q * Quat::exp(w_nb * dt);
  s.q.normalize();
}

}  // namespace nav
