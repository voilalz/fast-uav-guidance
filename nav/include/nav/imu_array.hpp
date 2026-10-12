// 多片 MEMS IMU 阵列：逐片标定补偿 + 野值剔除 + 平均。
#pragma once

#include <cstddef>

#include "nav/types.hpp"

namespace nav {

struct ImuCalib {
  Mat3 gyro_m = Mat3::identity();   // 比例因子 + 轴间失准 + 安装矩阵（到机体系）
  Mat3 accel_m = Mat3::identity();
  Vec3 gyro_b0, accel_b0;           // 参考温度下零偏
  Vec3 gyro_bt, accel_bt;           // 零偏温度系数（每 °C）
  real t_ref = 25.0;
};

struct RawImu {
  Vec3 gyro, accel;
  real temp_c = 25.0;
  bool valid = true;
};

template <std::size_t N>
class ImuArray {
 public:
  void setCalib(std::size_t i, const ImuCalib& c) { cal_[i] = c; }

  // 融合 N 片 IMU 同一时刻的采样。返回参与融合的片数（0 表示无可用数据）。
  std::size_t fuse(const RawImu (&raw)[N], TimeUs t_us, ImuSample& out) {
    Vec3 g[N], a[N];
    bool ok[N];
    for (std::size_t i = 0; i < N; ++i) {
      ok[i] = raw[i].valid;
      const real dT = raw[i].temp_c - cal_[i].t_ref;
      g[i] = cal_[i].gyro_m * (raw[i].gyro - cal_[i].gyro_b0 - cal_[i].gyro_bt * dT);
      a[i] = cal_[i].accel_m * (raw[i].accel - cal_[i].accel_b0 - cal_[i].accel_bt * dT);
    }
    // 以逐轴中位数为参考剔除故障片，再对剩余片求平均（单片故障不会污染参考值）
    if (count(ok) >= 3) {
      const Vec3 gmed = median(g, ok), amed = median(a, ok);
      for (std::size_t i = 0; i < N; ++i) {
        if (!ok[i]) continue;
        if (norm(g[i] - gmed) > gyro_tol_ || norm(a[i] - amed) > accel_tol_) ok[i] = false;
      }
    }
    Vec3 gm, am;
    const std::size_t n = mean(g, a, ok, gm, am);
    if (n == 0) return 0;
    out.t_us = t_us;
    out.gyro = gm;
    out.accel = am;
    out.valid = true;
    return n;
  }

  void setTolerance(real gyro_rad_s, real accel_m_s2) { gyro_tol_ = gyro_rad_s; accel_tol_ = accel_m_s2; }

 private:
  static std::size_t count(const bool (&ok)[N]) {
    std::size_t n = 0;
    for (bool b : ok) n += b;
    return n;
  }
  static Vec3 median(const Vec3 (&v)[N], const bool (&ok)[N]) {
    Vec3 m;
    for (int ax = 0; ax < 3; ++ax) {
      real tmp[N];
      std::size_t n = 0;
      for (std::size_t i = 0; i < N; ++i)
        if (ok[i]) tmp[n++] = v[i][ax];
      for (std::size_t i = 1; i < n; ++i)  // 插入排序，N 很小
        for (std::size_t j = i; j > 0 && tmp[j - 1] > tmp[j]; --j) { real t = tmp[j]; tmp[j] = tmp[j - 1]; tmp[j - 1] = t; }
      m[ax] = (n % 2) ? tmp[n / 2] : 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
    }
    return m;
  }
  static std::size_t mean(const Vec3 (&g)[N], const Vec3 (&a)[N], const bool (&ok)[N], Vec3& gm, Vec3& am) {
    gm = Vec3{};
    am = Vec3{};
    std::size_t n = 0;
    for (std::size_t i = 0; i < N; ++i)
      if (ok[i]) { gm += g[i]; am += a[i]; ++n; }
    if (n) { gm *= 1.0 / n; am *= 1.0 / n; }
    return n;
  }

  ImuCalib cal_[N];
  real gyro_tol_ = 5.0 * kDeg;  // rad/s
  real accel_tol_ = 2.0;        // m/s^2
};

}  // namespace nav
