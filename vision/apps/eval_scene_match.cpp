// 在真实多时相影像上评估景象匹配：精度、成功率、错误匹配率与形态，用于整定仿真中的量测模型。
//
// 实时图由 POST-event 影像（洪水后、不同卫星与时相、含薄云）按随机位姿渲染，参考图为 PRE-event 影像；
// 另跑一组“同源”（实时图也由 PRE 渲染）作为上界。
//
//   vision_eval_scene_match [map=data/ahr_0.50m.map] [trials=300] [seed=1] [out=results/vision] [cases=a,b]
// 输出 <out>/scene_match_<case>.csv 与 scene_match_summary.json；图由 tools/plot_vision.py 绘制。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "vision/geo_map.hpp"
#include "vision/scene_match.hpp"

using namespace vision;

namespace {

struct Case {
  std::string name;
  bool post;  // 实时图来源：POST（跨时相）或 PRE（同源）
  MatchConfig cfg;
  double prior_sigma;
};

std::vector<Case> allCases() {
  MatchConfig grad;
  grad.preprocess = Preprocess::Grad;
  MatchConfig flat;
  flat.use_dem = false;
  MatchConfig nopyr;  // 关闭由粗到精，单级全分辨率 NCC（对照）
  nopyr.pyramid_factor = 1;
  return {{"same_source", false, {}, 50},
          {"cross_time", true, {}, 50},
          {"cross_time_grad", true, grad, 50},
          {"cross_time_flat", true, flat, 50},
          {"cross_time_prior150", true, {}, 150},
          {"cross_time_prior150_nopyr", true, nopyr, 150},
          {"cross_time_prior400", true, {}, 400},
          {"cross_time_prior400_nopyr", true, nopyr, 400}};
}

struct Row {
  int i, ok;
  std::string stage;
  double err_n, err_e, err, sigma, ncc, psr, amb;
  int inliers;
  double prior_err, yaw_prior_err, yaw_err, ms;
};

double wrapPi(double a) { return std::remainder(a, 2 * M_PI); }

// 随机真值位姿：足迹在有效数据内，航向沿条带（南北向）±20°
Pose samplePose(std::mt19937_64& rng, const Terrain& T, const cv::Mat& mask, double gsd, double margin = 230) {
  std::uniform_real_distribution<double> U(0, 1);
  std::normal_distribution<double> N(0, 1);
  const int k = static_cast<int>(150 / gsd);
  for (int it = 0; it < 1000; ++it) {
    const double n = -(margin + U(rng) * (mask.rows * gsd - 2 * margin));
    const double e = mask.cols * gsd / 2 - 60 + 120 * U(rng);
    const int r = static_cast<int>(-n / gsd), c = static_cast<int>(e / gsd);
    if (r - k < 0 || c - k < 0 || r + k > mask.rows || c + k > mask.cols) continue;
    if (cv::countNonZero(mask(cv::Range(r - k, r + k), cv::Range(c - k, c + k))) < 0.97 * 4 * k * k) continue;
    Pose p;
    p.n = n;
    p.e = e;
    p.h = T.meanHeight(n, e, 300) + 1000 + 30 * N(rng);
    p.yaw = (U(rng) < 0.5 ? 0 : M_PI) + 10 * M_PI / 180 * N(rng);
    p.roll = 2 * M_PI / 180 * N(rng);
    p.pitch = 2 * M_PI / 180 * N(rng);
    return p;
  }
  throw std::runtime_error("找不到有效位姿");
}

// 导航先验：位置/高度/姿态误差（GNSS 拒止若干分钟后的典型量级）
Pose perturb(std::mt19937_64& rng, const Pose& p, double sig) {
  std::normal_distribution<double> N(0, 1);
  const double d = M_PI / 180;
  Pose q = p;
  q.n += sig * N(rng);
  q.e += sig * N(rng);
  q.h += 10 * N(rng);
  q.roll += 0.1 * d * N(rng);
  q.pitch += 0.1 * d * N(rng);
  q.yaw += 0.5 * d * N(rng);
  return q;
}

std::vector<Row> runCase(const Case& cs, const GeoMap& map, const Terrain& T, int trials, unsigned seed) {
  const Camera cam;
  SceneMatcher m(map.pre, map.gsd, cam, T, cs.cfg);
  const Renderer rd(cs.post ? map.post : map.pre, map.gsd, T);
  std::mt19937_64 rng(seed);
  std::vector<Row> rows;
  for (int i = 0; i < trials; ++i) {
    const Pose truth = samplePose(rng, T, map.mask, map.gsd);
    const Pose est = perturb(rng, truth, cs.prior_sigma);
    cv::Mat img, valid;
    rd.render(truth, cam, img, valid, &rng);
    const MatchResult r = m.match(img, valid, est, cs.prior_sigma);
    Row w{};
    w.i = i;
    w.ok = r.ok;
    w.stage = r.stage;
    w.err_n = r.ok ? r.n - truth.n : NAN;
    w.err_e = r.ok ? r.e - truth.e : NAN;
    w.err = std::hypot(w.err_n, w.err_e);
    w.sigma = r.sigma;
    w.ncc = r.ncc;
    w.psr = r.psr;
    w.amb = r.ambiguity;
    w.inliers = r.inliers;
    w.prior_err = std::hypot(est.n - truth.n, est.e - truth.e);
    w.yaw_prior_err = (est.yaw - truth.yaw) * 180 / M_PI;
    w.yaw_err = r.ok && !std::isnan(r.dyaw) ? wrapPi(est.yaw + r.dyaw - truth.yaw) * 180 / M_PI : NAN;
    w.ms = r.ms;
    rows.push_back(w);
  }
  return rows;
}

void writeCsv(const std::string& path, const std::vector<Row>& rows) {
  FILE* f = std::fopen(path.c_str(), "w");
  std::fprintf(f, "i,ok,stage,err_n,err_e,err,sigma,ncc,psr,amb,inliers,prior_err,yaw_prior_err,yaw_err,ms\n");
  for (const auto& r : rows)
    std::fprintf(f, "%d,%d,%s,%.4f,%.4f,%.4f,%.4f,%.5f,%.4f,%.5f,%d,%.4f,%.5f,%.5f,%.2f\n", r.i, r.ok,
                 r.stage.c_str(), r.err_n, r.err_e, r.err, r.sigma, r.ncc, r.psr, r.amb, r.inliers, r.prior_err,
                 r.yaw_prior_err, r.yaw_err, r.ms);
  std::fclose(f);
}

// ---- 统计 ----
double mean(const std::vector<double>& v) {
  return v.empty() ? NAN : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
double stdev(const std::vector<double>& v) {
  if (v.empty()) return NAN;
  const double m = mean(v);
  double s = 0;
  for (double x : v) s += (x - m) * (x - m);
  return std::sqrt(s / v.size());
}
double pct(std::vector<double> v, double p) {  // 线性插值百分位（与 numpy 默认一致）
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  const double k = p / 100 * (v.size() - 1);
  const size_t a = static_cast<size_t>(k);
  return a + 1 < v.size() ? v[a] + (k - a) * (v[a + 1] - v[a]) : v[a];
}
std::string num(double x) {
  if (!std::isfinite(x)) return "NaN";
  std::ostringstream s;
  s.precision(10);
  s << x;
  return s.str();
}

std::string stats(const std::vector<Row>& rows, double outlier_m = 30) {
  std::vector<double> en, ee, err, nis, en_o, err_o, yaw, yaw_prior, ms;
  int ok = 0, fine = 0;
  std::map<std::string, int> fail;
  for (const auto& r : rows) {
    yaw_prior.push_back(r.yaw_prior_err);
    ms.push_back(r.ms);
    if (!r.ok) {
      ++fail[r.stage];
      continue;
    }
    ++ok;
    fine += r.stage == "fine";
    if (!std::isnan(r.yaw_err)) yaw.push_back(r.yaw_err);
    if (r.err < outlier_m) {
      en.push_back(r.err_n), ee.push_back(r.err_e), err.push_back(r.err);
      nis.push_back((r.err_n * r.err_n + r.err_e * r.err_e) / (r.sigma * r.sigma));
    } else {
      err_o.push_back(r.err);
    }
  }
  const double n = rows.size();
  std::ostringstream o;
  o << "{\n  \"trials\": " << rows.size() << ",\n  \"reported_pct\": " << num(100 * ok / n)
    << ",\n  \"fine_pct\": " << num(ok ? 100.0 * fine / ok : 0)
    << ",\n  \"outlier_pct_of_reported\": " << num(ok ? 100.0 * err_o.size() / ok : 0)
    << ",\n  \"bias_n\": " << num(mean(en)) << ",\n  \"bias_e\": " << num(mean(ee))
    << ",\n  \"std_n\": " << num(stdev(en)) << ",\n  \"std_e\": " << num(stdev(ee))
    << ",\n  \"err_median\": " << num(pct(err, 50)) << ",\n  \"err_p95_inliers\": " << num(pct(err, 95))
    << ",\n  \"outlier_err_min\": " << num(err_o.empty() ? NAN : *std::min_element(err_o.begin(), err_o.end()))
    << ",\n  \"outlier_err_median\": " << num(pct(err_o, 50))
    << ",\n  \"outlier_err_max\": " << num(err_o.empty() ? NAN : *std::max_element(err_o.begin(), err_o.end()))
    << ",\n  \"nis_mean_inliers\": " << num(mean(nis)) << ",\n  \"yaw_err_std_deg\": " << num(stdev(yaw))
    << ",\n  \"yaw_prior_std_deg\": " << num(stdev(yaw_prior)) << ",\n  \"ms_median\": " << num(pct(ms, 50))
    << ",\n  \"fail_reasons\": {";
  bool first = true;
  for (const auto& [k, v] : fail) o << (first ? "" : ", ") << "\"" << k << "\": " << v, first = false;
  o << "}\n }";
  return o.str();
}

}  // namespace

int main(int argc, char** argv) {
  std::string map_path = "data/ahr_0.50m.map", out = "results/vision", cases_arg;
  int trials = 300;
  unsigned seed = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto eq = a.find('=');
    const std::string k = a.substr(0, eq), v = eq == std::string::npos ? "" : a.substr(eq + 1);
    if (k == "map") map_path = v;
    else if (k == "trials") trials = std::stoi(v);
    else if (k == "seed") seed = static_cast<unsigned>(std::stoul(v));
    else if (k == "out") out = v;
    else if (k == "cases") cases_arg = v;
    else {
      std::fprintf(stderr, "未知参数 %s\n", a.c_str());
      return 1;
    }
  }
  GeoMap map;
  if (!map.load(map_path)) {
    std::fprintf(stderr, "无法读取地图 %s（先运行 vision_build_map）\n", map_path.c_str());
    return 1;
  }
  std::filesystem::create_directories(out);
  const Terrain T(map.dem, map.gsd);
  std::vector<Case> cases;
  for (const auto& c : allCases())
    if (cases_arg.empty() || ("," + cases_arg + ",").find("," + c.name + ",") != std::string::npos)
      cases.push_back(c);
  std::vector<std::vector<Row>> res(cases.size());
  std::vector<std::thread> th;
  for (size_t i = 0; i < cases.size(); ++i)
    th.emplace_back([&, i] { res[i] = runCase(cases[i], map, T, trials, seed); });
  for (auto& t : th) t.join();
  std::ostringstream js;
  js << "{\n";
  for (size_t i = 0; i < cases.size(); ++i) {
    writeCsv(out + "/scene_match_" + cases[i].name + ".csv", res[i]);
    const std::string s = stats(res[i]);
    std::printf("%s %s\n", cases[i].name.c_str(), s.c_str());
    js << " \"" << cases[i].name << "\": " << s << (i + 1 < cases.size() ? ",\n" : "\n");
  }
  js << "}\n";
  FILE* f = std::fopen((out + "/scene_match_summary.json").c_str(), "w");
  std::fputs(js.str().c_str(), f);
  std::fclose(f);
  return 0;
}
