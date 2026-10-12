// 传感器数据与导航输出的公共类型。时间戳统一为导航 MCU 硬件时基（微秒）。
#pragma once

#include <cstdint>

#include "nav/math.hpp"

namespace nav {

using TimeUs = std::uint64_t;

struct ImuSample {
  TimeUs t_us = 0;
  Vec3 gyro;   // rad/s，机体系 FRD
  Vec3 accel;  // m/s^2，比力
  bool valid = true;
};

// 导航状态（名义状态）
struct NavState {
  TimeUs t_us = 0;
  real lat = 0, lon = 0, h = 0;  // rad, rad, m（椭球高）
  Vec3 vn;                        // NED 速度 m/s
  Quat q;                         // 机体 -> NED
  Vec3 bg;                        // 陀螺零偏 rad/s
  Vec3 ba;                        // 加计零偏 m/s^2
  Vec2 wind;                      // 水平风（北、东）m/s
  real baro_bias = 0;             // 气压高度偏差 m
  real vo_scale = 0;              // 视觉里程计尺度误差（无量纲）
  real mag_bias = 0;              // 磁航向偏差 rad
};

enum class FixSource : std::uint8_t { kSceneMatch = 1, kTerrain3D = 2, kTercom = 3, kGnss = 4 };

// 绝对位置量测（景象匹配 / 三维地形匹配 / TERCOM）
struct PositionFix {
  TimeUs t_us = 0;  // 量测对应时刻（相机曝光中点），不是结果产生时刻
  real lat = 0, lon = 0, h = 0;
  real sigma_ne = 10;   // 水平 1σ m
  real sigma_d = -1;    // 高度 1σ m，<0 表示不含高度
  FixSource source = FixSource::kSceneMatch;
};

struct VoVelocity {  // 机体系速度
  TimeUs t_us = 0;
  Vec3 vb;
  real sigma = 1.0;
};

struct Airspeed { TimeUs t_us = 0; real tas = 0; real sigma = 1.0; };
struct BaroAlt { TimeUs t_us = 0; real alt = 0; real sigma = 2.0; };
struct MagHeading { TimeUs t_us = 0; real yaw = 0; real sigma = 2.0 * kDeg; };
struct RadarAlt { TimeUs t_us = 0; real agl = 0; real sigma = 2.0; };

// 30 Hz 输出给飞控的导航解
struct NavOutput {
  TimeUs t_us = 0;
  real lat = 0, lon = 0, h = 0;
  Vec3 vn;
  Vec3 euler;   // roll pitch yaw
  Vec3 gyro;    // 去零偏角速度
  real sigma_ne = 0, sigma_d = 0;
  std::uint8_t mode = 0;
  std::uint8_t health = 0;
};

}  // namespace nav
