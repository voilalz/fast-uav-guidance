// 景象匹配：实时图正射校正 → 粗匹配（掩膜 NCC 搜索）→ 精匹配（SIFT + RANSAC 相似变换）→ 质量评估。
// 输出绝对位置（及航向）量测与 1σ，对应协议 POSITION_FIX（docs/software-design.md §4.3）。
//
// 参考库为 1 m 北向朝上正射影像（docs §4.7），本模块直接在地图网格上取参考区域。
// 精匹配用 OpenCV 内置 SIFT；接口与 SuperPoint+LightGlue 等学习型匹配器一致，替换 fine() 即可。
#pragma once

#include <string>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "vision/camera.hpp"

namespace vision {

enum class Preprocess { Clahe, Grad };  // 跨时相时梯度幅值有时更稳健

struct MatchConfig {
  double ortho_gsd = 1.0;     // 正射校正与参考图分辨率 m
  double template_m = 320.0;  // 模板边长（实时图正射后中心区域）
  bool use_dem = true;        // 正射校正是否使用 DEM（false 为平地假设）
  Preprocess preprocess = Preprocess::Clahe;
  double exclusion_m = 25.0;  // 次峰排除半径
  bool fine = true;
  int min_inliers = 12;
  double min_ambiguity = 1.02;  // 主峰 / 次峰（NCC 映射到正数后之比）
  double min_psr = 4.0;
  double min_valid = 0.6;  // 模板内有效像素比例下限
};

struct MatchResult {
  bool ok = false;
  double n = NAN, e = NAN;  // 匹配得到的相机水平位置（NED，m）
  double dyaw = NAN;        // 航向修正 rad（真值 ≈ 估计 + dyaw），仅精匹配成功时有效
  double sigma = 1e9;       // 水平 1σ m
  double ncc = 0, psr = 0, ambiguity = 0;
  int inliers = 0;
  std::string stage;  // low_texture / no_reference / coarse / fine
  double ms = 0;
};

// 以估计位姿把实时图投影到以估计天底点为中心、北向朝上、ortho_gsd 分辨率的地面网格（边长 size_m）。
void orthorectify(const cv::Mat& img, const cv::Mat& valid, const Pose& pose_est, const Camera& cam,
                  const Terrain& terrain, const MatchConfig& cfg, double size_m, cv::Mat& ortho, cv::Mat& omask);

// 量测质量 -> 1σ（参数按跨时相评估结果整定）
double grade(const MatchResult& r, double gsd);

class SceneMatcher {
 public:
  // ref_map：参考正射影像（地图网格，分辨率 ref_gsd）
  SceneMatcher(const cv::Mat& ref_map, double ref_gsd, const Camera& cam, const Terrain& terrain,
               const MatchConfig& cfg = MatchConfig());

  MatchResult match(const cv::Mat& img, const cv::Mat& valid, const Pose& pose_est, double prior_sigma);

  const MatchConfig& config() const { return cfg_; }

 private:
  void refCrop(double n, double e, double half_m, cv::Mat& out, int& r0, int& c0) const;
  bool fine(const cv::Mat& ortho, const cv::Mat& om, const cv::Mat& ref, int r0, int c0, double n0, double e0,
            double dn, double de, double out[4]);

  MatchConfig cfg_;
  Camera cam_;
  const Terrain& terrain_;
  cv::Mat ref_;
  cv::Ptr<cv::SIFT> sift_;
};

}  // namespace vision
