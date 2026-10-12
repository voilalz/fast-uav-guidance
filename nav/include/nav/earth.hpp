// WGS-84 地球模型：曲率半径、地球自转、正常重力。
#pragma once

#include "nav/math.hpp"

namespace nav::earth {

constexpr real kA = 6378137.0;              // 长半轴 m
constexpr real kE2 = 6.69437999014e-3;      // 第一偏心率平方
constexpr real kOmega = 7.292115e-5;        // 地球自转角速率 rad/s

struct Radii {
  real rn;  // 子午圈曲率半径 R_N（北向）
  real re;  // 卯酉圈曲率半径 R_E（东向）
};

inline Radii radii(real lat) {
  const real s = std::sin(lat);
  const real d = 1.0 - kE2 * s * s;
  const real re = kA / std::sqrt(d);
  return {re * (1.0 - kE2) / d, re};
}

// 正常重力（Somigliana + 高度修正），NED 下 g = [0, 0, g]
inline real gravity(real lat, real h) {
  const real s2 = std::sin(lat) * std::sin(lat);
  const real g0 = 9.7803253359 * (1 + 0.00193185265241 * s2) / std::sqrt(1 - kE2 * s2);
  return g0 * (1 - 2 * h / kA);
}

// 地球自转角速度在导航系(NED)的投影 ω_ie^n
inline Vec3 omegaIe(real lat) { return vec3(kOmega * std::cos(lat), 0.0, -kOmega * std::sin(lat)); }

// 牵连角速度 ω_en^n
inline Vec3 omegaEn(real lat, real h, const Vec3& vn) {
  const Radii r = radii(lat);
  return vec3(vn[1] / (r.re + h), -vn[0] / (r.rn + h), -vn[1] * std::tan(lat) / (r.re + h));
}

// 经纬高差 -> NED 米（小范围线性化）
inline Vec3 llaDiffToNed(real lat0, real h0, real dlat, real dlon, real dh) {
  const Radii r = radii(lat0);
  return vec3(dlat * (r.rn + h0), dlon * (r.re + h0) * std::cos(lat0), -dh);
}

}  // namespace nav::earth
