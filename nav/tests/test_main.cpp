// 单元测试（无第三方框架，便于在目标板上同样运行）
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "nav/earth.hpp"
#include "nav/eskf.hpp"
#include "nav/imu_array.hpp"
#include "nav/protocol.h"
#include "nav/strapdown.hpp"
#include "nav/terrain.hpp"

using namespace nav;

static int g_fail = 0;
#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);    \
      ++g_fail;                                                        \
    }                                                                  \
  } while (0)

static void testMath() {
  Mat3 a = Mat3::identity() * 2.0, inv;
  CHECK(inverse(a, inv) && std::fabs(inv(1, 1) - 0.5) < 1e-12);
  const Quat q = Quat::fromEuler(0.1, -0.2, 2.5);
  const Vec3 e = q.toEuler();
  CHECK(std::fabs(e[0] - 0.1) < 1e-12 && std::fabs(e[1] + 0.2) < 1e-12 && std::fabs(e[2] - 2.5) < 1e-12);
  const Vec3 r = vec3(0.01, -0.02, 0.03);
  const Vec3 r2 = Quat::exp(r).log();
  CHECK(norm(r - r2) < 1e-12);
}

// 静止对准状态下，理想 IMU 推算 60 s 应几乎不漂
static void testStrapdownStatic() {
  NavState s;
  s.lat = 30 * kDeg;
  s.h = 1000;
  const real g = earth::gravity(s.lat, s.h);
  const Vec3 wie = earth::omegaIe(s.lat);
  for (int k = 0; k < 400 * 60; ++k) Strapdown::propagate(s, wie, vec3(0, 0, -g), 1.0 / 400);
  CHECK(norm(s.vn) < 0.01);
  CHECK(std::fabs(s.h - 1000) < 0.5);
}

// 位置量测更新应把误差拉回，且协方差减小
static void testEskfPositionUpdate() {
  Eskf f;
  NavState x;
  x.lat = 30 * kDeg;
  x.h = 1000;
  f.init(x, InitSigma{});
  const real p0 = f.cov()(0, 0);
  PositionFix z;
  z.lat = x.lat + 20.0 / earth::radii(x.lat).rn;  // 北偏 20 m
  z.lon = x.lon;
  z.sigma_ne = 5;
  const UpdateResult u = f.updatePosition(z, f.state());
  CHECK(u.accepted);
  const real dn = (f.state().lat - x.lat) * earth::radii(x.lat).rn;
  CHECK(dn > 9 && dn < 11);  // P0=R 时修正一半
  CHECK(f.cov()(0, 0) < p0);

  // 远超门限的错误匹配应被拒绝
  PositionFix bad = z;
  bad.lat = x.lat + 2000.0 / earth::radii(x.lat).rn;
  CHECK(!f.updatePosition(bad, f.state()).accepted);
}

// 航向量测雅可比数值校验
static void testHeadingJacobian() {
  NavState x;
  x.q = Quat::fromEuler(0.3, -0.2, 1.0);
  const auto yawOf = [](const Quat& q) { const Mat3 c = q.toDcm(); return std::atan2(c(1, 0), c(0, 0)); };
  const Mat3 C = x.q.toDcm();
  const real d = C(0, 0) * C(0, 0) + C(1, 0) * C(1, 0);
  const real H[3] = {-C(0, 0) * C(2, 0) / d, -C(1, 0) * C(2, 0) / d, 1.0};
  for (int i = 0; i < 3; ++i) {
    Vec3 phi;
    phi[i] = 1e-6;
    const real num = (yawOf(Quat::exp(phi) * x.q) - yawOf(x.q)) / 1e-6;
    CHECK(std::fabs(num - H[i]) < 1e-4);
  }
}

static void testImuArrayOutlier() {
  ImuArray<4> arr;
  RawImu raw[4];
  for (auto& r : raw) { r.gyro = vec3(0.01, 0, 0); r.accel = vec3(0, 0, -9.8); }
  raw[2].gyro = vec3(1.0, 0, 0);  // 故障片
  ImuSample s;
  CHECK(arr.fuse(raw, 0, s) == 3);
  CHECK(std::fabs(s.gyro[0] - 0.01) < 1e-9);
}

static void testTercom() {
  // 合成山地 DEM：200×200，30 m 网格
  const std::size_t R = 200, C = 200;
  std::vector<float> dem(R * C);
  for (std::size_t r = 0; r < R; ++r)
    for (std::size_t c = 0; c < C; ++c)
      dem[r * C + c] = static_cast<float>(500 + 80 * std::sin(r * 0.11) * std::cos(c * 0.07) +
                                          40 * std::sin(r * 0.031 + c * 0.053) + 15 * std::cos(r * 0.27 - c * 0.19));
  Dem d(dem.data(), R, C, 30.0);
  // 真航迹：东北方向直线；惯导航迹相对真值偏 (+120, -90) m
  Tercom<256> t;
  const real off_n = 120, off_e = -90;
  for (int i = 0; i < 150; ++i) {
    const real n = 1500 + i * 20.0, e = 1500 + i * 15.0;
    real h;
    d.height(n, e, h);
    t.add(n + off_n, e + off_e, h + 7.0 + ((i * 37) % 11 - 5) * 0.3);  // 7 m 气压偏差 + 噪声
  }
  const TercomResult r = t.match(d, 300, 10, 60);
  CHECK(r.valid);
  CHECK(std::fabs(r.dn + off_n) <= 10 && std::fabs(r.de + off_e) <= 10);
  CHECK(r.ambiguity > 2);
}

static void testProtocol() {
  nav_vo_velocity_t m{};
  m.t_us = 123456789;
  m.vb[0] = 166.7f;
  std::uint8_t buf[NAV_MAX_FRAME];
  const size_t n = nav_encode(NAV_MSG_VO_VELOCITY, 7, &m, sizeof m, buf, sizeof buf);
  CHECK(n == NAV_HEADER_LEN + sizeof m + NAV_CRC_LEN);
  nav_decoder_t d;
  nav_decoder_init(&d);
  // 前置噪声字节 + 完整帧
  nav_decoder_push(&d, 0x00);
  nav_decoder_push(&d, 0xA5);
  int got = 0;
  for (size_t i = 0; i < n; ++i) got += nav_decoder_push(&d, buf[i]);
  CHECK(got == 1);
  nav_vo_velocity_t back;
  std::memcpy(&back, nav_frame_payload(&d), sizeof back);
  CHECK(back.t_us == m.t_us && back.vb[0] == m.vb[0]);
  // 篡改一个字节应校验失败
  buf[10] ^= 0xFF;
  got = 0;
  for (size_t i = 0; i < n; ++i) got += nav_decoder_push(&d, buf[i]);
  CHECK(got == 0 && d.crc_errors == 1);
}

int main() {
  testMath();
  testStrapdownStatic();
  testEskfPositionUpdate();
  testHeadingJacobian();
  testImuArrayOutlier();
  testTercom();
  testProtocol();
  std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
