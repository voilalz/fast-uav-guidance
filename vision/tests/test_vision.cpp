// 视觉管线单元测试：在合成纹理 + 合成地形上检查几何一致性、景象匹配与 VO 精度（不依赖外部数据）。
#include <cmath>
#include <cstdio>
#include <random>

#include <opencv2/imgproc.hpp>

#include "vision/scene_match.hpp"
#include "vision/vo.hpp"

using namespace vision;

static int g_fail = 0;
#define CHECK(c)                                                    \
  do {                                                              \
    if (!(c)) {                                                     \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c);      \
      ++g_fail;                                                     \
    }                                                               \
  } while (0)

namespace {

constexpr double kGsd = 0.5;

// 多尺度随机纹理（类似地物），2000×6000 像素 = 1 km × 3 km
cv::Mat texture(int rows, int cols, unsigned seed) {
  cv::RNG r(seed);
  cv::Mat acc = cv::Mat::zeros(rows, cols, CV_32F);
  for (int s : {4, 16, 64}) {
    cv::Mat n(rows / s + 2, cols / s + 2, CV_32F), up;
    r.fill(n, cv::RNG::NORMAL, 0, 1);
    cv::resize(n, up, cv::Size(cols, rows), 0, 0, cv::INTER_CUBIC);
    acc += up * (s / 16.0 + 0.5);
  }
  cv::Mat out;
  cv::normalize(acc, out, 20, 235, cv::NORM_MINMAX, CV_8U);
  return out;
}

cv::Mat hills(int rows, int cols) {
  cv::Mat d(rows, cols, CV_32F);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      d.at<float>(r, c) = static_cast<float>(250 + 60 * std::sin(r * kGsd / 700) * std::cos(c * kGsd / 400));
  return d;
}

void testGeometry() {
  const Terrain T(hills(400, 400), kGsd);
  const Camera cam;
  Pose p;
  p.n = -100;
  p.e = 100;
  p.h = 1200;
  p.roll = 0.03;
  p.pitch = -0.02;
  p.yaw = 2.0;
  const Mat3 C = p.C_nc();
  for (double u : {10.0, 320.0, 600.0})
    for (double v : {30.0, 320.0, 630.0}) {
      const Vec3 g = rayGround(p, C, cam.ray(u, v), &T, 0, 10);
      CHECK(std::fabs(g[2] - T.height(g[0], g[1])) < 0.05);
      double uu, vv;
      CHECK(project(p, C, cam, g[0], g[1], g[2], uu, vv));
      CHECK(std::fabs(uu - u) < 1e-6 && std::fabs(vv - v) < 1e-6);
    }
  // 图像上方指向机头：机头朝北时，图像上方的地面点在北侧
  Pose q;
  q.h = 1000;
  const Vec3 top = rayGround(q, q.C_nc(), cam.ray(320, 0), nullptr, 0);
  CHECK(top[0] > 150 && std::fabs(top[1]) < 1e-6);
}

void testSceneMatch() {
  const cv::Mat img_map = texture(6000, 2000, 7);
  const cv::Mat dem = hills(6000, 2000);
  const Terrain T(dem, kGsd);
  const Camera cam;
  const Renderer rd(img_map, kGsd, T);
  SceneMatcher m(img_map, kGsd, cam, T);
  std::mt19937_64 rng(3);
  std::normal_distribution<double> N(0, 1);
  int ok = 0, good = 0, yaw_ok = 0;
  for (int i = 0; i < 10; ++i) {
    Pose truth;
    truth.n = -600 - 180 * i;
    truth.e = 500;
    truth.h = T.height(truth.n, truth.e) + 1000;
    truth.yaw = i % 2 ? M_PI : 0.2;
    truth.roll = 0.02 * N(rng);
    truth.pitch = 0.02 * N(rng);
    Pose est = truth;
    est.n += 40 * N(rng);
    est.e += 40 * N(rng);
    est.yaw += 0.5 * M_PI / 180 * N(rng);
    cv::Mat img, valid;
    rd.render(truth, cam, img, valid, &rng);
    const MatchResult r = m.match(img, valid, est, 50);
    if (!r.ok) continue;
    ++ok;
    good += std::hypot(r.n - truth.n, r.e - truth.e) < 5;
    yaw_ok += r.stage == "fine" && std::fabs(std::remainder(est.yaw + r.dyaw - truth.yaw, 2 * M_PI)) < 0.3 * M_PI / 180;
  }
  std::printf("scene match: ok %d/10, <5 m %d, yaw<0.3° %d\n", ok, good, yaw_ok);
  CHECK(ok >= 9 && good == ok && yaw_ok >= 8);
}

void testVo() {
  const cv::Mat img_map = texture(6000, 2000, 11);
  const Terrain T(hills(6000, 2000), kGsd);
  const Camera cam;
  const Renderer rd(img_map, kGsd, T);
  DownwardVo vo(cam);
  const double V = 600 / 3.6, dt = 1 / 30.0;
  double sum = 0;
  int n_ok = 0;
  for (int k = 0; k < 60; ++k) {
    Pose p;
    p.n = -400 - V * k * dt;
    p.e = 500;
    p.h = 1300;
    p.yaw = M_PI;
    p.roll = 0.02 * std::sin(k * 0.1);
    cv::Mat img, valid;
    rd.render(p, cam, img, valid, nullptr);
    const VoResult r = vo.process(img, p, dt, 0, &T, ScaleMode::DemRay, valid);
    if (!r.ok) continue;
    ++n_ok;
    sum += std::fabs(std::hypot(r.vn[0], r.vn[1]) / V - 1);
    CHECK(r.vn[0] < 0);  // 向南飞
  }
  std::printf("vo dem_ray: ok %d/59, mean |scale err| %.3f%%\n", n_ok, 100 * sum / std::max(n_ok, 1));
  CHECK(n_ok >= 55 && sum / n_ok < 0.005);
}

}  // namespace

int main() {
  testGeometry();
  testSceneMatch();
  testVo();
  std::printf(g_fail ? "%d FAILED\n" : "all vision tests passed\n", g_fail);
  return g_fail ? 1 : 0;
}
