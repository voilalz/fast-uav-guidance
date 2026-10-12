// 在真实影像 + DEM 上评估下视 VO，量化不同尺度来源在山地/林区引入的 VO 尺度误差及其时间相关性。
//
// 轨迹：沿 Ahr 河谷条带由北向南定高直线飞行，600 km/h，30 Hz，叠加小幅滚转/俯仰振荡；
// 导航给出的姿态含 0.05° 水平 / 0.3° 航向偏差与慢变误差（一阶马尔可夫 0.02°、τ 10 s），
// 位置含 30 m 误差，高度含 5 m 偏差。
//
//   vision_eval_vo [map=data/ahr_0.50m.map] [src=pre|post] [seed=1] [out=results/vision]
// 输出 vo_<mode>.csv 与 vo_summary.json；结论用于设定仿真中的 vo_scale / vo_scale_gm_sigma / vo_scale_gm_tau。
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "vision/geo_map.hpp"
#include "vision/vo.hpp"

using namespace vision;

namespace {

constexpr double kV = 600 / 3.6, kRate = 30.0, kDeg = M_PI / 180;
const ScaleMode kModes[] = {ScaleMode::Radar, ScaleMode::DemMean, ScaleMode::DemRay};

struct Row {
  double t, dist, n, terrain, agl_true, scale_err, vb_err_x, vb_err_y, vb_err_z, sigma;
  int inliers;
};

// 自相关降到 1/e 的距离（m）
double corrLength(std::vector<double> x, double step_m) {
  double m = 0;
  for (double v : x) m += v;
  m /= x.size();
  double var = 0;
  for (auto& v : x) v -= m, var += v * v;
  var /= x.size();
  if (std::sqrt(var) < 1e-9) return 0;
  for (size_t lag = 0; lag < x.size(); ++lag) {
    double s = 0;
    for (size_t i = 0; i + lag < x.size(); ++i) s += x[i] * x[i + lag];
    if (s / (var * x.size()) < std::exp(-1.0)) return lag * step_m;
  }
  return x.size() * step_m;
}

std::vector<double> movingAvg(const std::vector<double>& s, int k) {
  std::vector<double> o;
  for (size_t i = 0; i + k <= s.size(); ++i) {
    double a = 0;
    for (int j = 0; j < k; ++j) a += s[i + j];
    o.push_back(a / k);
  }
  return o;
}

double stdev(const std::vector<double>& v) {
  double m = 0, s = 0;
  for (double x : v) m += x;
  m /= v.size();
  for (double x : v) s += (x - m) * (x - m);
  return std::sqrt(s / v.size());
}

}  // namespace

int main(int argc, char** argv) {
  std::string map_path = "data/ahr_0.50m.map", out = "results/vision", src = "pre";
  unsigned seed = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto eq = a.find('=');
    const std::string k = a.substr(0, eq), v = eq == std::string::npos ? "" : a.substr(eq + 1);
    if (k == "map") map_path = v;
    else if (k == "src") src = v;
    else if (k == "seed") seed = static_cast<unsigned>(std::stoul(v));
    else if (k == "out") out = v;
    else {
      std::fprintf(stderr, "未知参数 %s\n", a.c_str());
      return 1;
    }
  }
  GeoMap map;
  if (!map.load(map_path)) {
    std::fprintf(stderr, "无法读取地图 %s\n", map_path.c_str());
    return 1;
  }
  std::filesystem::create_directories(out);
  const Terrain T(map.dem, map.gsd);
  const Camera cam;
  const Renderer rd(src == "post" ? map.post : map.pre, map.gsd, T);
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> N(0, 1);
  const double e0 = map.width() * map.gsd / 2;
  const double n_start = -260, n_end = -(map.height() * map.gsd - 260);
  const double h = T.mean() + 1000;  // 定高飞行，离地高随地形变化
  const double dt = 1 / kRate;
  const double att_bias[3] = {0.05 * kDeg, -0.05 * kDeg, 0.3 * kDeg};
  const double pos_err[2] = {30 * N(rng), 30 * N(rng)};
  double att_gm[3] = {0, 0, 0};
  const double a_gm = std::exp(-dt / 10.0);
  std::vector<DownwardVo> vos(3, DownwardVo(cam));
  std::vector<std::vector<Row>> rows(3);
  int k = 0;
  for (;; ++k) {
    const double t = k * dt, n = n_start - kV * t;
    if (n < n_end) break;
    Pose truth;
    truth.n = n;
    truth.e = e0;
    truth.h = h;
    truth.roll = 1.5 * kDeg * std::sin(2 * M_PI * 0.3 * t);
    truth.pitch = 2.0 * kDeg + 1.0 * kDeg * std::sin(2 * M_PI * 0.2 * t);
    truth.yaw = M_PI;
    cv::Mat img, valid;
    rd.render(truth, cam, img, valid, &rng);
    for (double& g : att_gm) g = a_gm * g + 0.02 * kDeg * std::sqrt(1 - a_gm * a_gm) * N(rng);
    Pose est = truth;
    est.n += pos_err[0];
    est.e += pos_err[1];
    est.h += 5.0;
    est.roll += att_bias[0] + att_gm[0];
    est.pitch += att_bias[1] + att_gm[1];
    est.yaw += att_bias[2] + att_gm[2];
    // 雷达高度表：沿机体 z 轴（= 相机光轴）到真值地形的距离 + 1 m 噪声
    const Vec3 gz = rayGround(truth, truth.C_nc(), Vec3(0, 0, 1), &T, 0);
    const double agl_radar = h - gz[2] + N(rng);
    const double agl_dem = est.h - T.meanHeight(est.n, est.e, 190);
    const Vec3 v_true = truth.C_nb().t() * Vec3(-kV, 0, 0);
    const double terr = T.meanHeight(n, e0, 190);
    for (int m = 0; m < 3; ++m) {
      const double agl = kModes[m] == ScaleMode::Radar ? agl_radar : agl_dem;
      const VoResult r = vos[m].process(img, est, dt, agl, &T, kModes[m], valid);
      if (!r.ok) continue;
      const Vec3 e = r.vb - v_true;
      rows[m].push_back({t, kV * t, n, terr, h - terr, std::hypot(r.vn[0], r.vn[1]) / kV - 1, e[0], e[1], e[2],
                         r.sigma, r.inliers});
    }
  }

  std::ostringstream js;
  js << "{\n";
  double relief_lo = 1e9, relief_hi = -1e9;
  for (int m = 0; m < 3; ++m) {
    const std::string name = toString(kModes[m]);
    FILE* f = std::fopen((out + "/vo_" + name + ".csv").c_str(), "w");
    std::fprintf(f, "t,dist,n,terrain,agl_true,scale_err,vb_err_x,vb_err_y,vb_err_z,sigma,inliers\n");
    std::vector<double> s, ex, ey, ez, sig;
    for (const auto& r : rows[m]) {
      std::fprintf(f, "%.4f,%.2f,%.2f,%.3f,%.3f,%.6f,%.4f,%.4f,%.4f,%.4f,%d\n", r.t, r.dist, r.n, r.terrain,
                   r.agl_true, r.scale_err, r.vb_err_x, r.vb_err_y, r.vb_err_z, r.sigma, r.inliers);
      s.push_back(r.scale_err), ex.push_back(r.vb_err_x), ey.push_back(r.vb_err_y), ez.push_back(r.vb_err_z);
      sig.push_back(r.sigma);
      if (m == 0) relief_lo = std::min(relief_lo, r.terrain), relief_hi = std::max(relief_hi, r.terrain);
    }
    std::fclose(f);
    const int kk = static_cast<int>(kRate);
    std::vector<double> s1;  // 1 s 分块平均（滤除帧间噪声，保留时间相关分量）
    for (size_t i = 0; i + kk <= s.size(); i += kk) {
      double a = 0;
      for (int j = 0; j < kk; ++j) a += s[i + j];
      s1.push_back(a / kk);
    }
    double sm = 0;
    for (double x : s) sm += x;
    sm /= s.size();
    auto rms = [](const std::vector<double>& v) {
      double a = 0;
      for (double x : v) a += x * x;
      return std::sqrt(a / v.size());
    };
    std::nth_element(sig.begin(), sig.begin() + sig.size() / 2, sig.end());
    const double L = corrLength(movingAvg(s, kk), kV / kRate);
    char buf[1024];
    std::snprintf(buf, sizeof buf,
                  "{\n  \"frames_ok_pct\": %.4f,\n  \"scale_err_mean_pct\": %.4f,\n  \"scale_err_std_pct\": %.4f,\n"
                  "  \"scale_err_1s_std_pct\": %.4f,\n  \"scale_corr_length_m\": %.2f,\n  \"scale_corr_time_s\": "
                  "%.4f,\n  \"vb_err_rms\": [%.4f, %.4f, %.4f],\n  \"reported_sigma_median\": %.4f\n }",
                  100.0 * s.size() / k, 100 * sm, 100 * stdev(s), 100 * stdev(s1), L, L / kV, rms(ex), rms(ey),
                  rms(ez), sig[sig.size() / 2]);
    std::printf("%s %s\n", name.c_str(), buf);
    js << " \"" << name << "\": " << buf << ",\n";
  }
  js << " \"terrain_relief_m\": " << relief_hi - relief_lo << "\n}\n";
  FILE* f = std::fopen((out + "/vo_summary.json").c_str(), "w");
  std::fputs(js.str().c_str(), f);
  std::fclose(f);
  return 0;
}
