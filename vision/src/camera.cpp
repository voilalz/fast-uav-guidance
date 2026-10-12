#include "vision/camera.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace vision {

Mat3 eulerToDcm(double roll, double pitch, double yaw) {
  const double cr = std::cos(roll), sr = std::sin(roll), cp = std::cos(pitch), sp = std::sin(pitch),
               cy = std::cos(yaw), sy = std::sin(yaw);
  return Mat3(cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy,  //
              cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy,  //
              -sp, sr * cp, cr * cp);
}

const Mat3& R_BC() {
  static const Mat3 r(0, -1, 0,  //
                      1, 0, 0,   //
                      0, 0, 1);
  return r;
}

// ---- Terrain ----

Terrain::Terrain(const cv::Mat& dem, double gsd) : gsd_(gsd) {
  CV_Assert(dem.type() == CV_32F);
  double s = 0;
  long k = 0;
  for (int r = 0; r < dem.rows; ++r) {
    const float* p = dem.ptr<float>(r);
    for (int c = 0; c < dem.cols; ++c)
      if (!std::isnan(p[c])) s += p[c], ++k;
  }
  mean_ = k ? s / k : 0.0;
  dem_ = dem.clone();
  for (int r = 0; r < dem_.rows; ++r) {
    float* p = dem_.ptr<float>(r);
    for (int c = 0; c < dem_.cols; ++c)
      if (std::isnan(p[c])) p[c] = static_cast<float>(mean_);
  }
}

double Terrain::height(double n, double e) const {
  // 像素中心在整数坐标处（与 cv::remap 一致），越界复制边界
  const double x = std::clamp(e / gsd_, 0.0, dem_.cols - 1.0);
  const double y = std::clamp(-n / gsd_, 0.0, dem_.rows - 1.0);
  const int x0 = std::min(static_cast<int>(x), dem_.cols - 2), y0 = std::min(static_cast<int>(y), dem_.rows - 2);
  const double fx = x - x0, fy = y - y0;
  const float* a = dem_.ptr<float>(y0);
  const float* b = dem_.ptr<float>(y0 + 1);
  return (1 - fy) * ((1 - fx) * a[x0] + fx * a[x0 + 1]) + fy * ((1 - fx) * b[x0] + fx * b[x0 + 1]);
}

double Terrain::meanHeight(double n, double e, double radius) const {
  const int r = static_cast<int>(radius / gsd_);
  const int row = static_cast<int>(-n / gsd_), col = static_cast<int>(e / gsd_);
  const int r0 = std::max(0, row - r), r1 = std::min(dem_.rows, row + r);
  const int c0 = std::max(0, col - r), c1 = std::min(dem_.cols, col + r);
  if (r1 <= r0 || c1 <= c0) return mean_;
  return cv::mean(dem_(cv::Range(r0, r1), cv::Range(c0, c1)))[0];
}

// ---- 几何 ----

Vec3 rayGround(const Pose& pose, const Mat3& C_nc, const Vec3& ray_c, const Terrain* terrain, double flat_h,
               int iters) {
  const Vec3 d = C_nc * ray_c;  // NED 方向
  const double dd = std::max(d[2], 1e-3);
  double z = terrain ? terrain->mean() : flat_h;
  double n = pose.n, e = pose.e;
  for (int i = 0; i < (terrain ? iters : 1); ++i) {
    const double t = (pose.h - z) / dd;
    n = pose.n + t * d[0];
    e = pose.e + t * d[1];
    if (terrain) z = terrain->height(n, e);
  }
  return {n, e, z};
}

bool project(const Pose& pose, const Mat3& C_nc, const Camera& cam, double n, double e, double h, double& u,
             double& v) {
  const Vec3 p = C_nc.t() * Vec3(n - pose.n, e - pose.e, -(h - pose.h));  // 相机系
  if (p[2] <= 1.0) return false;
  u = cam.f * p[0] / p[2] + cam.width / 2.0;
  v = cam.f * p[1] / p[2] + cam.height / 2.0;
  return true;
}

// ---- Renderer ----

Renderer::Renderer(const cv::Mat& src, double gsd, const Terrain& terrain, double blur_sigma)
    : gsd_(gsd), terrain_(terrain) {
  if (blur_sigma > 0)
    cv::GaussianBlur(src, src_, cv::Size(0, 0), blur_sigma);
  else
    src_ = src;
  valid_ = src > 0;
}

void Renderer::render(const Pose& pose, const Camera& cam, cv::Mat& img, cv::Mat& valid, std::mt19937_64* rng,
                      double noise, double gain_lo, double gain_hi) const {
  cv::Mat mx(cam.height, cam.width, CV_32F), my(cam.height, cam.width, CV_32F);
  const Mat3 C = pose.C_nc();
  for (int v = 0; v < cam.height; ++v) {
    float* px = mx.ptr<float>(v);
    float* py = my.ptr<float>(v);
    for (int u = 0; u < cam.width; ++u) {
      const Vec3 g = rayGround(pose, C, cam.ray(u + 0.5, v + 0.5), &terrain_, 0);
      px[u] = static_cast<float>(g[1] / gsd_);
      py[u] = static_cast<float>(-g[0] / gsd_);
    }
  }
  // cv::remap 要求源图像行列 < 32767：按需要的区域裁剪源图像
  double lo_x, hi_x, lo_y, hi_y;
  cv::minMaxLoc(mx, &lo_x, &hi_x);
  cv::minMaxLoc(my, &lo_y, &hi_y);
  const int x0 = std::clamp(static_cast<int>(std::floor(lo_x)) - 2, 0, src_.cols);
  const int x1 = std::clamp(static_cast<int>(std::ceil(hi_x)) + 3, 0, src_.cols);
  const int y0 = std::clamp(static_cast<int>(std::floor(lo_y)) - 2, 0, src_.rows);
  const int y1 = std::clamp(static_cast<int>(std::ceil(hi_y)) + 3, 0, src_.rows);
  img = cv::Mat::zeros(cam.height, cam.width, CV_8U);
  valid = cv::Mat::zeros(cam.height, cam.width, CV_8U);
  if (x1 <= x0 || y1 <= y0) return;
  mx -= x0;
  my -= y0;
  const cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
  cv::remap(src_(roi), img, mx, my, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
  cv::remap(valid_(roi), valid, mx, my, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);
  valid = valid > 0;
  valid /= 255;
  if (rng) {
    std::uniform_real_distribution<double> ug(gain_lo, gain_hi);
    std::normal_distribution<double> nb(0, 5);
    const double g = ug(*rng), b = nb(*rng);
    cv::Mat f;
    img.convertTo(f, CV_32F, g, b);
    cv::Mat w(f.size(), CV_32F);
    cv::RNG cr((*rng)());
    cr.fill(w, cv::RNG::NORMAL, 0, noise);
    f += w;
    f.convertTo(img, CV_8U);  // 饱和截断到 [0, 255]
  }
  img.setTo(0, valid == 0);
}

}  // namespace vision
