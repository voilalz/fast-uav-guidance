#include "vision/scene_match.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace vision {
namespace {

cv::Mat clahe(const cv::Mat& img) {
  static thread_local cv::Ptr<cv::CLAHE> c = cv::createCLAHE(2.0, cv::Size(8, 8));
  cv::Mat out;
  c->apply(img, out);
  return out;
}

cv::Mat prep(const cv::Mat& img, Preprocess mode, const cv::Mat& mask = cv::Mat()) {
  cv::Mat out;
  if (mode == Preprocess::Grad) {
    cv::Mat f, gx, gy, mag;
    img.convertTo(f, CV_32F);
    cv::GaussianBlur(f, f, cv::Size(0, 0), 1.0);
    cv::Sobel(f, gx, CV_32F, 1, 0);
    cv::Sobel(f, gy, CV_32F, 0, 1);
    cv::magnitude(gx, gy, mag);
    cv::GaussianBlur(mag, out, cv::Size(0, 0), 1.0);
  } else {
    clahe(img).convertTo(out, CV_32F);
  }
  if (!mask.empty()) out.setTo(0, mask == 0);
  return out;
}

double elapsedMs(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

void orthorectify(const cv::Mat& img, const cv::Mat& valid, const Pose& pose_est, const Camera& cam,
                  const Terrain& terrain, const MatchConfig& cfg, double size_m, cv::Mat& ortho, cv::Mat& omask) {
  const int k = static_cast<int>(size_m / cfg.ortho_gsd);
  const Mat3 C = pose_est.C_nc();
  const double flat = cfg.use_dem ? 0 : terrain.meanHeight(pose_est.n, pose_est.e, 300);
  cv::Mat mu(k, k, CV_32F), mv(k, k, CV_32F);
  for (int i = 0; i < k; ++i) {
    const double n = pose_est.n - (i - k / 2.0 + 0.5) * cfg.ortho_gsd;
    float* pu = mu.ptr<float>(i);
    float* pv = mv.ptr<float>(i);
    for (int j = 0; j < k; ++j) {
      const double e = pose_est.e + (j - k / 2.0 + 0.5) * cfg.ortho_gsd;
      const double h = cfg.use_dem ? terrain.height(n, e) : flat;
      double u, v;
      if (project(pose_est, C, cam, n, e, h, u, v)) {
        pu[j] = static_cast<float>(u);
        pv[j] = static_cast<float>(v);
      } else {
        pu[j] = pv[j] = -1;
      }
    }
  }
  cv::remap(img, ortho, mu, mv, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
  cv::remap(valid, omask, mu, mv, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);
}

double grade(const MatchResult& r, double gsd) {
  if (r.stage == "fine") return std::max(2.0 * gsd * (1.0 + 30.0 / std::max(r.inliers, 1)), 2.0);
  // 粗匹配（NCC）：按跨时相评估整定，PSR 越低越不可靠；含参考库配准偏差
  return std::max(gsd * (3.0 + 20.0 / std::max(r.psr, 1.0)), 3.0);
}

SceneMatcher::SceneMatcher(const cv::Mat& ref_map, double ref_gsd, const Camera& cam, const Terrain& terrain,
                           const MatchConfig& cfg)
    : cfg_(cfg), cam_(cam), terrain_(terrain), sift_(cv::SIFT::create(1500)) {
  const double f = ref_gsd / cfg.ortho_gsd;
  if (f != 1.0)
    cv::resize(ref_map, ref_, cv::Size(), f, f, cv::INTER_AREA);
  else
    ref_ = ref_map;
}

void SceneMatcher::refCrop(double n, double e, double half_m, cv::Mat& out, int& r0, int& c0) const {
  const double g = cfg_.ortho_gsd;
  r0 = static_cast<int>(std::nearbyint((-n - half_m) / g));
  c0 = static_cast<int>(std::nearbyint((e - half_m) / g));
  const int k = static_cast<int>(2 * half_m / g);
  out = cv::Mat::zeros(k, k, CV_8U);
  const int rr0 = std::max(0, r0), cc0 = std::max(0, c0);
  const int rr1 = std::min(ref_.rows, r0 + k), cc1 = std::min(ref_.cols, c0 + k);
  if (rr1 > rr0 && cc1 > cc0)
    ref_(cv::Range(rr0, rr1), cv::Range(cc0, cc1)).copyTo(out(cv::Range(rr0 - r0, rr1 - r0), cv::Range(cc0 - c0, cc1 - c0)));
}

MatchResult SceneMatcher::match(const cv::Mat& img, const cv::Mat& valid, const Pose& pose_est,
                                double prior_sigma) {
  const auto t0 = std::chrono::steady_clock::now();
  const double g = cfg_.ortho_gsd;
  MatchResult res;
  cv::Mat ortho, om;
  orthorectify(img, valid, pose_est, cam_, terrain_, cfg_, 1.5 * cfg_.template_m, ortho, om);
  const double n0 = pose_est.n, e0 = pose_est.e;  // 正射图中心 = 估计天底点
  const int k = static_cast<int>(cfg_.template_m / g);
  const int s = (ortho.rows - k) / 2;
  const cv::Rect troi(s, s, k, k);
  const cv::Mat tpl = ortho(troi), tm = om(troi);
  cv::Scalar tmean, tstd;
  cv::meanStdDev(tpl, tmean, tstd, tm);
  const double valid_frac = cv::countNonZero(tm) / static_cast<double>(k * k);
  if (valid_frac < cfg_.min_valid || tstd[0] < 8) {  // 视场被遮挡/无数据或纹理过弱（水面、云）
    res.stage = "low_texture";
    res.ms = elapsedMs(t0);
    return res;
  }
  const double search = 3 * prior_sigma + 20;
  const double half = cfg_.template_m / 2 + search;
  cv::Mat ref;
  int r0, c0;
  refCrop(n0, e0, half, ref, r0, c0);
  if (1.0 - cv::countNonZero(ref) / static_cast<double>(ref.total()) > 0.6) {
    res.stage = "no_reference";
    res.ms = elapsedMs(t0);
    return res;
  }
  const cv::Mat A = prep(ref, cfg_.preprocess), T = prep(tpl, cfg_.preprocess, tm);
  cv::Mat corr;
  cv::matchTemplate(A, T, corr, cv::TM_CCOEFF_NORMED, tm);
  corr.forEach<float>([](float& v, const int*) {
    if (!std::isfinite(v)) v = -1;
  });
  double c1;
  cv::Point p1;
  cv::minMaxLoc(corr, nullptr, &c1, nullptr, &p1);
  const int ex = std::max(1, static_cast<int>(cfg_.exclusion_m / g));
  cv::Mat cm = corr.clone();
  cm(cv::Range(std::max(0, p1.y - ex), std::min(cm.rows, p1.y + ex + 1)),
     cv::Range(std::max(0, p1.x - ex), std::min(cm.cols, p1.x + ex + 1)))
      .setTo(-1);
  double c2;
  cv::minMaxLoc(cm, nullptr, &c2);
  cv::Scalar smean, sstd;
  cv::meanStdDev(corr, smean, sstd, cm > -1);
  const double psr = (c1 - smean[0]) / (sstd[0] + 1e-6);
  const double amb = (c1 + 1) / (c2 + 1);  // 映射到正数后之比，> 1
  // 抛物线拟合亚像素峰值
  double sx = 0, sy = 0;
  auto at = [&](int r, int c) { return static_cast<double>(corr.at<float>(r, c)); };
  if (p1.x > 0 && p1.x < corr.cols - 1) {
    const double l = at(p1.y, p1.x - 1), c = at(p1.y, p1.x), r = at(p1.y, p1.x + 1);
    sx = 0.5 * (l - r) / (l - 2 * c + r - 1e-9);
  }
  if (p1.y > 0 && p1.y < corr.rows - 1) {
    const double u = at(p1.y - 1, p1.x), c = at(p1.y, p1.x), d = at(p1.y + 1, p1.x);
    sy = 0.5 * (u - d) / (u - 2 * c + d - 1e-9);
  }
  // 粗匹配：模板左上角在参考裁剪中的位置 -> 模板中心的地面坐标（OpenCV 像素中心约定）
  const double cen_r = r0 + p1.y + std::clamp(sy, -0.5, 0.5) + k / 2.0;
  const double cen_c = c0 + p1.x + std::clamp(sx, -0.5, 0.5) + k / 2.0;
  // 模板中心在正射图中对应 (n0, e0)，因此相机水平位置修正 = 匹配中心 − (n0, e0)
  double dn = -(cen_r - 0.5) * g - n0, de = (cen_c - 0.5) * g - e0;
  res.ncc = c1;
  res.psr = psr;
  res.ambiguity = amb;
  res.stage = "coarse";
  if (cfg_.fine) {
    double f[4];
    if (fine(ortho, om, ref, r0, c0, n0, e0, dn, de, f) && f[3] >= cfg_.min_inliers &&
        std::hypot(f[0] - dn, f[1] - de) < 3 * cfg_.exclusion_m) {
      dn = f[0];
      de = f[1];
      res.dyaw = f[2];
      res.inliers = static_cast<int>(f[3]);
      res.stage = "fine";
    }
  }
  if (res.psr >= cfg_.min_psr && res.ambiguity >= cfg_.min_ambiguity) {
    res.ok = true;
    res.n = pose_est.n + dn;
    res.e = pose_est.e + de;
    res.sigma = grade(res, g);
  }
  res.ms = elapsedMs(t0);
  return res;
}

bool SceneMatcher::fine(const cv::Mat& ortho, const cv::Mat& om, const cv::Mat& ref, int r0, int c0, double n0,
                        double e0, double dn, double de, double out[4]) {
  const double g = cfg_.ortho_gsd;
  std::vector<cv::KeyPoint> k1, k2;
  cv::Mat d1, d2;
  sift_->detectAndCompute(clahe(ortho), om * 255, k1, d1);
  sift_->detectAndCompute(clahe(ref), ref > 0, k2, d2);
  if (k1.size() < 8 || k2.size() < 8) return false;
  std::vector<std::vector<cv::DMatch>> m;
  cv::BFMatcher(cv::NORM_L2).knnMatch(d1, d2, m, 2);
  // 粗匹配先验：只保留与粗位移一致的匹配（±2 × 排除半径）
  const double oc = ortho.rows / 2.0 - 0.5;  // 正射图中心（= 估计天底点）的像素坐标
  const cv::Point2f shift(static_cast<float>((e0 + de) / g - c0 - oc), static_cast<float>(-(n0 + dn) / g - r0 - oc));
  std::vector<cv::Point2f> P, Q;
  int good = 0;
  for (const auto& x : m) {
    if (x.size() < 2 || x[0].distance >= 0.8 * x[1].distance) continue;
    ++good;
    const cv::Point2f p = k1[x[0].queryIdx].pt, q = k2[x[0].trainIdx].pt;
    if (cv::norm(q - (p + shift)) < 2 * cfg_.exclusion_m / g) P.push_back(p), Q.push_back(q);
  }
  if (good < 8 || P.size() < 8) return false;
  std::vector<uchar> inl;
  const cv::Mat M = cv::estimateAffinePartial2D(P, Q, inl, cv::RANSAC, 2.0, 2000, 0.999);
  if (M.empty()) return false;
  const double qx = M.at<double>(0, 0) * oc + M.at<double>(0, 1) * oc + M.at<double>(0, 2);
  const double qy = M.at<double>(1, 0) * oc + M.at<double>(1, 1) * oc + M.at<double>(1, 2);
  out[0] = -(r0 + qy) * g - n0;
  out[1] = (c0 + qx) * g - e0;
  // 正射图北向朝上；图像中顺时针旋转 a 对应航向需增加 a
  out[2] = std::atan2(M.at<double>(1, 0), M.at<double>(0, 0));
  out[3] = cv::countNonZero(inl);
  return true;
}

}  // namespace vision
