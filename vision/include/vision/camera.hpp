// 下视针孔相机模型、射线–DEM 求交与机载实时图渲染。
//
// 坐标约定（与导航核心一致）：
//   * 世界系为局部 NED：n 北、e 东、d 下；高度 h = −d（与 DEM 同一高程基准）。
//   * 地图网格（GeoMap）：像素 (row, col) ↔ n = −row·gsd，e = col·gsd。
//   * 机体系 FRD；相机光轴沿机体 +z（向下），图像上方指向机头：
//       相机 x（向右）= 机体 +y，相机 y（向下）= 机体 −x，相机 z = 机体 +z。
#pragma once

#include <random>

#include <opencv2/core.hpp>

namespace vision {

using Mat3 = cv::Matx33d;
using Vec3 = cv::Vec3d;

// ZYX 欧拉角 -> C_nb（机体到 NED），与 nav::Quat::fromEuler 一致。
Mat3 eulerToDcm(double roll, double pitch, double yaw);
// 相机 -> 机体
const Mat3& R_BC();

struct Camera {
  int width = 640, height = 640;
  double f = 1667.0;  // 焦距（像素）。1 km 离地高时地面分辨率约 0.6 m，幅宽约 384 m

  // 像素 (u, v)（左上角为 0，像素中心为 +0.5）对应的相机系射线（z = 1）
  Vec3 ray(double u, double v) const { return {(u - width / 2.0) / f, (v - height / 2.0) / f, 1.0}; }
};

struct Pose {
  double n = 0, e = 0, h = 0;  // h：高程（与 DEM 同基准）
  double roll = 0, pitch = 0, yaw = 0;

  Mat3 C_nb() const { return eulerToDcm(roll, pitch, yaw); }
  Mat3 C_nc() const { return C_nb() * R_BC(); }
};

// 地图网格上的 DEM，双线性采样（越界取边界）。NaN 用有效均值填充。
class Terrain {
 public:
  Terrain() = default;
  Terrain(const cv::Mat& dem, double gsd);

  double height(double n, double e) const;
  // 以 (n, e) 为中心、边长 2·radius 的方窗内平均高程
  double meanHeight(double n, double e, double radius) const;
  double mean() const { return mean_; }
  double gsd() const { return gsd_; }
  const cv::Mat& dem() const { return dem_; }

 private:
  cv::Mat dem_;  // CV_32F
  double gsd_ = 1, mean_ = 0;
};

// 相机系射线与地面求交，返回地面点 (n, e, h)。terrain 为 nullptr 时与高度 flat_h 的水平面求交。
Vec3 rayGround(const Pose& pose, const Mat3& C_nc, const Vec3& ray_c, const Terrain* terrain, double flat_h,
               int iters = 6);

// 世界点投影到像素 (u, v)；位于相机后方（深度 ≤ 1 m）返回 false。
bool project(const Pose& pose, const Mat3& C_nc, const Camera& cam, double n, double e, double h, double& u,
             double& v);

// 由地图网格上的正射影像渲染机载下视图像（含地形视差）。
class Renderer {
 public:
  // blur_sigma：源影像预先高斯模糊（模拟光学点扩散）
  Renderer(const cv::Mat& src, double gsd, const Terrain& terrain, double blur_sigma = 0.5);

  // 返回 uint8 图像与有效掩膜（CV_8U，0/1）。rng 不为空时加入增益/偏置/噪声，模拟相机响应差异。
  void render(const Pose& pose, const Camera& cam, cv::Mat& img, cv::Mat& valid, std::mt19937_64* rng,
              double noise = 2.0, double gain_lo = 0.85, double gain_hi = 1.15) const;

 private:
  cv::Mat src_, valid_;
  double gsd_;
  const Terrain& terrain_;
};

}  // namespace vision
