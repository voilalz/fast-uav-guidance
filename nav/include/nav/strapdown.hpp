// 捷联惯导机械编排（当地水平 NED，经纬高位置）。
#pragma once

#include "nav/types.hpp"

namespace nav {

class Strapdown {
 public:
  // 用去零偏后的角速度/比力推进 dt 秒。
  static void propagate(NavState& s, const Vec3& gyro, const Vec3& accel, real dt);
};

}  // namespace nav
