// 下视单目视觉里程计（地面近似 + 外部尺度），输出机体系速度量测，对应协议 VO_VELOCITY。
//
// 算法（与 docs/software-design.md §4.2 一致）：
//   1. 金字塔 LK 光流跟踪 + 前后向一致性检查，按网格补充特征点；
//   2. 用导航给出的两帧姿态把像点射线转到 NED（去除旋转）；
//   3. 用尺度来源给出的“相机到地面的距离”把射线落到地面，同一地面点在两帧的
//      相对位置之差即相机位移；多点取鲁棒中值（MAD 剔除外点）；
//   4. 位移 / Δt → NED 速度 → 机体系速度，协方差由内点离散度与尺度不确定度给出。
//
// 尺度来源（ScaleMode）：
//   Radar   ：雷达高度表测得的天底点离地高（林区/山地会因地形起伏引入尺度误差）
//   DemMean ：气压海拔 − 足迹范围内 DEM 平均高程（docs 推荐的林区方案）
//   DemRay  ：逐点用先验位姿与 DEM 做射线求交得到各点深度（不再假设地面平坦）
#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "vision/camera.hpp"

namespace vision {

enum class ScaleMode { Radar, DemMean, DemRay };
const char* toString(ScaleMode m);

struct VoResult {
  bool ok = false;
  Vec3 vb{0, 0, 0};  // 机体系速度 m/s
  Vec3 vn{0, 0, 0};  // NED 速度 m/s
  double sigma = 1e9;  // 每轴 1σ m/s
  int inliers = 0;
};

struct VoConfig {
  int max_pts = 400, min_pts = 40, grid = 8;
  double fb_thresh = 1.0;  // 前后向一致性阈值（像素）
};

class DownwardVo {
 public:
  explicit DownwardVo(const Camera& cam, const VoConfig& cfg = VoConfig());

  // pose_est 为导航给出的当前帧位姿（姿态用于去旋转；DemRay 模式还用其位置与高度）。
  // agl 为当前帧尺度来源给出的离地高（DemRay 模式不用）。valid 为图像有效掩膜（可为空）。
  VoResult process(const cv::Mat& img, const Pose& pose_est, double dt, double agl, const Terrain* terrain,
                   ScaleMode mode, const cv::Mat& valid = cv::Mat());

 private:
  std::vector<cv::Point2f> detect(const cv::Mat& img, const cv::Mat& valid) const;
  Vec3 ground(const Pose& pose, const Mat3& C_nc, const cv::Point2f& px, ScaleMode mode, double agl,
              const Terrain* terrain) const;

  Camera cam_;
  VoConfig cfg_;
  bool has_prev_ = false;
  cv::Mat prev_img_;
  Pose prev_pose_;
  double prev_agl_ = 0;
  std::vector<cv::Point2f> pts_;
};

}  // namespace vision
