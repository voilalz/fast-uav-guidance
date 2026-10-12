#include "vision/vo.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace vision {

const char* toString(ScaleMode m) {
  switch (m) {
    case ScaleMode::Radar: return "radar";
    case ScaleMode::DemMean: return "dem_mean";
    case ScaleMode::DemRay: return "dem_ray";
  }
  return "?";
}

namespace {

double median(std::vector<double> v) {
  const size_t k = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + k, v.end());
  if (v.size() % 2) return v[k];
  const double hi = v[k];
  return 0.5 * (hi + *std::max_element(v.begin(), v.begin() + k));
}

}  // namespace

DownwardVo::DownwardVo(const Camera& cam, const VoConfig& cfg) : cam_(cam), cfg_(cfg) {}

std::vector<cv::Point2f> DownwardVo::detect(const cv::Mat& img, const cv::Mat& valid) const {
  const int h = img.rows, w = img.cols, g = cfg_.grid;
  const int per = std::max(1, cfg_.max_pts / (g * g));
  std::vector<cv::Point2f> out;
  for (int i = 0; i < g; ++i)
    for (int j = 0; j < g; ++j) {
      const cv::Rect roi(j * w / g, i * h / g, (j + 1) * w / g - j * w / g, (i + 1) * h / g - i * h / g);
      cv::Mat m;
      if (!valid.empty()) m = valid(roi) * 255;
      std::vector<cv::Point2f> p;
      cv::goodFeaturesToTrack(img(roi), p, per, 0.01, 8, m);
      for (auto& q : p) out.emplace_back(q.x + roi.x, q.y + roi.y);
    }
  return out;
}

Vec3 DownwardVo::ground(const Pose& pose, const Mat3& C_nc, const cv::Point2f& px, ScaleMode mode, double agl,
                        const Terrain* terrain) const {
  const Vec3 ray = cam_.ray(px.x, px.y);
  if (mode == ScaleMode::DemRay) {
    const Vec3 g = rayGround(pose, C_nc, ray, terrain, 0);
    return {g[0] - pose.n, g[1] - pose.e, pose.h - g[2]};
  }
  const Vec3 d = C_nc * ray;
  return d * (agl / std::max(d[2], 1e-3));
}

VoResult DownwardVo::process(const cv::Mat& img, const Pose& pose_est, double dt, double agl,
                             const Terrain* terrain, ScaleMode mode, const cv::Mat& valid) {
  VoResult res;
  if (!has_prev_) {
    has_prev_ = true;
    prev_img_ = img.clone();
    prev_pose_ = pose_est;
    prev_agl_ = agl;
    pts_ = detect(img, valid);
    return res;
  }
  if (static_cast<int>(pts_.size()) >= cfg_.min_pts) {
    const cv::TermCriteria crit(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01);
    std::vector<cv::Point2f> p1, p0r;
    std::vector<uchar> st, st2;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_img_, img, pts_, p1, st, err, cv::Size(21, 21), 4, crit);
    cv::calcOpticalFlowPyrLK(img, prev_img_, p1, p0r, st2, err, cv::Size(21, 21), 4, crit);
    std::vector<cv::Point2f> a, b;
    for (size_t i = 0; i < pts_.size(); ++i)
      if (st[i] && st2[i] && cv::norm(p0r[i] - pts_[i]) < cfg_.fb_thresh) a.push_back(pts_[i]), b.push_back(p1[i]);
    if (static_cast<int>(a.size()) >= cfg_.min_pts) {
      const Mat3 C0 = prev_pose_.C_nc(), C1 = pose_est.C_nc();
      std::vector<Vec3> disp(a.size());
      for (size_t i = 0; i < a.size(); ++i)  // 相机位移（NED），地面点固定
        disp[i] = ground(prev_pose_, C0, a[i], mode, prev_agl_, terrain) - ground(pose_est, C1, b[i], mode, agl, terrain);
      Vec3 med, mad;
      for (int k = 0; k < 3; ++k) {
        std::vector<double> c(disp.size());
        for (size_t i = 0; i < disp.size(); ++i) c[i] = disp[i][k];
        med[k] = median(c);
        for (auto& x : c) x = std::fabs(x - med[k]);
        mad[k] = median(c) * 1.4826 + 1e-3;
      }
      std::vector<Vec3> in;
      for (const auto& d : disp) {
        bool ok = true;
        for (int k = 0; k < 3; ++k) ok = ok && std::fabs(d[k] - med[k]) < 3 * mad[k] + 0.05;
        if (ok) in.push_back(d);
      }
      if (static_cast<int>(in.size()) >= cfg_.min_pts / 2) {
        Vec3 mean(0, 0, 0);
        for (const auto& d : in) mean += d;
        mean *= 1.0 / in.size();
        double var = 0;  // 三轴方差的平均
        for (const auto& d : in) var += (d - mean).dot(d - mean);
        var /= 3.0 * in.size();
        res.ok = true;
        res.vn = mean * (1.0 / dt);
        res.vb = pose_est.C_nb().t() * res.vn;
        // 噪声：点位离散度 / sqrt(N) + 尺度来源不确定度（按 1% 计）
        const double spread = std::sqrt(var / in.size()) / dt;
        res.sigma = std::hypot(spread, 0.01 * cv::norm(res.vn));
        res.inliers = static_cast<int>(in.size());
      }
    }
    if (static_cast<int>(b.size()) >= cfg_.max_pts / 2) {
      pts_ = b;
    } else {
      pts_ = b;
      for (const auto& q : detect(img, valid)) pts_.push_back(q);
      if (static_cast<int>(pts_.size()) > cfg_.max_pts) pts_.resize(cfg_.max_pts);
    }
  } else {
    pts_ = detect(img, valid);
  }
  prev_img_ = img.clone();
  prev_pose_ = pose_est;
  prev_agl_ = agl;
  return res;
}

}  // namespace vision
