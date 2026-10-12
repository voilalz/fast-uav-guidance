// 论文 demo：图像在环的闭环导航仿真。
//
// 飞机以 600 km/h、离地约 1 km 沿 Ahr 河谷影像条带（0.84 km × 5.43 km）往返飞行，条带两端各延伸 12 km
// 后按 45° 坡度限制掉头，由航线跟踪制导律重新切入航线。每次飞越条带时，按 1 Hz 关键帧：
//   1. 用 POST-event 影像（洪水后、不同卫星与时相）在真值位姿渲染机载下视图像；
//   2. 以导航当前估计位姿与估计 1σ 作先验，用 PRE-event 参考库运行景象匹配（vision::SceneMatcher）；
//   3. 匹配成功则以 250 ms 时延把位置（及航向）量测送入 NavCore。
// 条带之外没有绝对修正，只有 IMU + 统计 VO + 空速 + 气压 + 磁航向。IMU/VO 误差使用 profile=realistic。
//
// 对比的滤波器配置（同一组随机种子、同一组图像）：
//   original          原有模型（未建模加计漂移与时变 VO 尺度，无锁死恢复）
//   original_recovery_sep  原有模型 + 锁死恢复（最小间隔 3 s）
//   matched           过程噪声与真实误差匹配（VO 尺度一阶马尔可夫、加计零偏随机游走 4e-4）
//   matched_recovery  matched + 门限锁死检测与协方差恢复（连续 2 次一致的被拒修正即恢复）
//   matched_recovery_sep  同上，但被拒修正需间隔 ≥ 3 s（视场不重叠）才计为独立证据
//
//   nav_demo [map=data/ahr_0.50m.map] [runs=8] [seed0=1] [threads=4] [out=results/demo/demo]
//            [variants=original,matched,...] [key=value ...]   # 其余 key=value 传给 SimConfig
// 输出 <out>_runs.csv、<out>_series.csv、<out>_fixes.csv；图由 tools/plot_demo.py 绘制。
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "nav/earth.hpp"
#include "sim_core.hpp"
#include "vision/geo_map.hpp"
#include "vision/scene_match.hpp"

using nav::kDeg;
using nav::real;

namespace {

constexpr real kV = 600.0 / 3.6;
constexpr real kExtend = 12000.0;  // 条带两端延伸距离 m

struct Variant {
  std::string name;
  std::vector<std::pair<std::string, std::string>> params;
};

const std::vector<Variant>& allVariants() {
  static const std::vector<Variant> v = {
      {"original", {}},
      {"original_recovery_sep", {{"recovery", "1"}, {"recovery_min_sep", "3"}}},
      {"matched", {{"filter_vo_scale_sigma", "0.01"}, {"filter_vo_scale_tau", "120"}, {"filter_accel_bias_rw", "4e-4"}}},
      {"matched_recovery",
       {{"filter_vo_scale_sigma", "0.01"}, {"filter_vo_scale_tau", "120"}, {"filter_accel_bias_rw", "4e-4"},
        {"recovery", "1"}}},
      {"matched_recovery_sep",
       {{"filter_vo_scale_sigma", "0.01"}, {"filter_vo_scale_tau", "120"}, {"filter_accel_bias_rw", "4e-4"},
        {"recovery", "1"}, {"recovery_min_sep", "3"}}},
  };
  return v;
}

// 局部切平面：仿真起点 (lat0, lon0) 对应地图坐标 (n_off, e_off)
struct Frame {
  real lat0, lon0, rn, re_cos, n_off, e_off, h_off;
  void toMap(real lat, real lon, real& n, real& e) const {
    n = (lat - lat0) * rn + n_off;
    e = (lon - lon0) * re_cos + e_off;
  }
  void toLatLon(real n, real e, real& lat, real& lon) const {
    lat = lat0 + (n - n_off) / rn;
    lon = lon0 + (e - e_off) / re_cos;
  }
  vision::Pose pose(const nav::NavState& s) const {
    vision::Pose p;
    toMap(s.lat, s.lon, p.n, p.e);
    p.h = s.h + h_off;
    const nav::Vec3 eul = s.q.toEuler();
    p.roll = eul[0];
    p.pitch = eul[1];
    p.yaw = eul[2];
    return p;
  }
};

// 航线跟踪制导：在 A、B 两点之间往返，航向指令 = 航线方向 − atan(偏航距 / L)，坡度 ≤ 45°，坡度变化率受限
class Guidance {
 public:
  Guidance(const Frame& f, real n_a, real n_b, real e_c) : f_(f), na_(n_a), nb_(n_b), ec_(e_c) {}

  real yawRate(real t, const nav::NavState& s) {
    real n, e;
    f_.toMap(s.lat, s.lon, n, e);
    const real to = to_b_ ? nb_ : na_;
    const real dir = to_b_ ? -1.0 : 1.0;  // A 在北、B 在南
    if ((to - n) * dir < 0) to_b_ = !to_b_;  // 越过端点后掉头
    const real line_course = to_b_ ? nav::kPi : 0.0;
    // 偏航距：以航线前进方向为基准，右侧为正
    const real xte = to_b_ ? -(e - ec_) : (e - ec_);
    const real course_cmd = line_course - std::atan(xte / 1000.0);
    const real course = std::atan2(s.vn[1], s.vn[0]);
    const real max_rate = 9.8 * std::tan(45.0 * kDeg) / kV;
    real cmd = std::clamp(0.1 * nav::wrapPi(course_cmd - course), -max_rate, max_rate);
    // 航向角速度变化率限制（约 5 s 从零到最大坡度）
    const real dt = last_t_ < 0 ? 0 : t - last_t_;
    last_t_ = t;
    const real step = max_rate / 5.0 * dt;
    rate_ += std::clamp(cmd - rate_, -step, step);
    return rate_;
  }

 private:
  Frame f_;
  real na_, nb_, ec_;
  bool to_b_ = true;
  real rate_ = 0, last_t_ = -1;
};

struct ProviderLog {
  real t;
  std::string stage;
  real psr, sigma, err;
};

struct RunOut {
  std::string variant;
  unsigned seed;
  sim::Result res;
  std::vector<ProviderLog> log;
  int attempts = 0;
};

}  // namespace

int main(int argc, char** argv) {
  std::string map_path = "data/ahr_0.50m.map", out = "results/demo/demo", variants_arg = "original,original_recovery_sep,matched,matched_recovery,matched_recovery_sep";
  int runs = 8, threads = static_cast<int>(std::thread::hardware_concurrency());
  unsigned seed0 = 1;
  std::vector<char*> rest = {argv[0]};
  for (int i = 1; i < argc; ++i) {
    if (!std::strncmp(argv[i], "map=", 4)) map_path = argv[i] + 4;
    else if (!std::strncmp(argv[i], "runs=", 5)) runs = std::atoi(argv[i] + 5);
    else if (!std::strncmp(argv[i], "threads=", 8)) threads = std::atoi(argv[i] + 8);
    else if (!std::strncmp(argv[i], "seed0=", 6)) seed0 = static_cast<unsigned>(std::atoi(argv[i] + 6));
    else if (!std::strncmp(argv[i], "out=", 4)) out = argv[i] + 4;
    else if (!std::strncmp(argv[i], "variants=", 9)) variants_arg = argv[i] + 9;
    else rest.push_back(argv[i]);
  }
  vision::GeoMap map;
  if (!map.load(map_path)) {
    std::fprintf(stderr, "无法读取地图 %s（先运行 tools/download_data.sh 与 vision_build_map）\n", map_path.c_str());
    return 1;
  }
  const vision::Terrain T(map.dem, map.gsd);
  const vision::Camera cam;
  const vision::Renderer renderer(map.post, map.gsd, T);
  const real strip_len = map.height() * map.gsd, strip_w = map.width() * map.gsd, e_c = strip_w / 2;

  // 基础配置：真实误差剖面，1 Hz 关键帧景象匹配，关闭三维地形匹配（只看景象匹配）
  sim::SimConfig base;
  sim::setParam(base, "profile", "realistic");
  base.t3d = false;
  base.fix_interval = 1.0;
  base.fix_first = 1.0;
  base.fix_delay = 0.25;
  std::string err;
  if (!sim::parseArgs(base, static_cast<int>(rest.size()), rest.data(), 1, &err)) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 2;
  }

  std::vector<Variant> variants;
  for (const auto& v : allVariants())
    if (("," + variants_arg + ",").find("," + v.name + ",") != std::string::npos) variants.push_back(v);

  // 起点：条带北端以北 12 km，沿条带中线向南
  sim::Scenario sc0;
  sc0.lat0 = 50.53 * kDeg;
  sc0.lon0 = 7.05 * kDeg;
  sc0.h0 = 1500.0;
  sc0.yaw0 = nav::kPi;
  const nav::earth::Radii r0 = nav::earth::radii(sc0.lat0);
  const Frame frame{sc0.lat0, sc0.lon0, r0.rn, r0.re * std::cos(sc0.lat0), kExtend, e_c,
                    T.mean() + 1000.0 - sc0.h0};

  std::vector<RunOut> outs;
  for (const auto& v : variants)
    for (int i = 0; i < runs; ++i) outs.push_back({v.name, seed0 + i, {}, {}, 0});
  std::atomic<int> next{0};
  std::vector<std::thread> pool;
  for (int w = 0; w < std::max(1, threads); ++w)
    pool.emplace_back([&] {
      for (int k; (k = next++) < static_cast<int>(outs.size());) {
        RunOut& o = outs[k];
        sim::SimConfig c = base;
        c.seed = o.seed;
        for (const auto& v : variants)
          if (v.name == o.variant)
            for (const auto& [key, val] : v.params) sim::setParam(c, key, val);
        sim::Scenario sc = sc0;
        auto guid = std::make_shared<Guidance>(frame, kExtend, -(strip_len + kExtend), e_c);
        sc.yaw_rate = [guid](real t, const nav::NavState& s) { return guid->yawRate(t, s); };
        vision::SceneMatcher matcher(map.pre, map.gsd, cam, T);
        std::mt19937_64 rng(o.seed * 1000003ull + 7);  // 图像噪声：与滤波器配置无关，各配置看到同一组图像
        sc.fix_provider = [&](real t, const nav::NavState& truth, const nav::NavState& est, real sigma_ne,
                              nav::PositionFix& z, bool& outlier) {
          const vision::Pose tp = frame.pose(truth);
          if (tp.n > 150 || tp.n < -strip_len - 150 || std::fabs(tp.e - e_c) > strip_w / 2 + 150) return false;
          ++o.attempts;
          cv::Mat img, valid;
          renderer.render(tp, cam, img, valid, &rng);
          const vision::Pose ep = frame.pose(est);
          // 搜索窗口取 3σ：滤波器 σ 过于乐观时真实位置落在窗口外，匹配器直接失败（匹配器层面的“锁死”）
          const vision::MatchResult r = matcher.match(img, valid, ep, std::clamp(sigma_ne, 10.0, 400.0));
          const real e = r.ok ? std::hypot(r.n - tp.n, r.e - tp.e) : NAN;
          o.log.push_back({t, r.stage, r.psr, r.sigma, e});
          if (!r.ok) return false;
          frame.toLatLon(r.n, r.e, z.lat, z.lon);
          z.sigma_ne = r.sigma;
          if (!std::isnan(r.dyaw)) {
            z.yaw = nav::wrapPi(ep.yaw + r.dyaw);
            z.sigma_yaw = 0.3 * kDeg;
          }
          outlier = e > 30.0;
          return true;
        };
        o.res = sim::run(c, sc, true, false);
        std::fprintf(stderr, "  %-21s seed %3u  RMS %8.1f m  终值 %8.1f m  匹配 %d/%d  锁死 %d  恢复 %u\n",
                     o.variant.c_str(), o.seed, o.res.rms_h, o.res.final_h,
                     o.res.true_accepted + o.res.true_rejected + o.res.outlier_accepted + o.res.outlier_rejected,
                     o.attempts, o.res.lockout_episodes, o.res.recoveries);
      }
    });
  for (auto& th : pool) th.join();

  std::filesystem::create_directories(std::filesystem::path(out).parent_path());
  FILE* fr = std::fopen((out + "_runs.csv").c_str(), "w");
  FILE* fs = std::fopen((out + "_series.csv").c_str(), "w");
  FILE* ff = std::fopen((out + "_fixes.csv").c_str(), "w");
  std::fprintf(fr, "variant,seed,rms_h,max_h,final_h,attempts,true_accepted,true_rejected,outlier_accepted,"
                   "outlier_rejected,lockout_episodes,lockout_longest_s,recoveries,nees_pos_mean,nees_kin_mean\n");
  std::fprintf(fs, "variant,seed,t,n_true,e_true,n_est,e_est,err_h,sigma_ne,nees_pos,mode\n");
  std::fprintf(ff, "variant,seed,t,n_fix,e_fix,n_true,e_true,err,outlier,accepted,recovered,nis,sigma,stage,psr\n");
  for (const auto& o : outs) {
    const sim::Result& r = o.res;
    std::fprintf(fr, "%s,%u,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d,%d,%.0f,%u,%.3f,%.3f\n", o.variant.c_str(), o.seed, r.rms_h,
                 r.max_h, r.final_h, o.attempts, r.true_accepted, r.true_rejected, r.outlier_accepted,
                 r.outlier_rejected, r.lockout_episodes, r.lockout_longest_s, r.recoveries, r.nees_pos_mean,
                 r.nees_kin_mean);
    for (const auto& s : r.samples) {
      real nt, et, ne, ee;
      frame.toMap(s.lat, s.lon, nt, et);
      frame.toMap(s.est_lat, s.est_lon, ne, ee);
      std::fprintf(fs, "%s,%u,%.0f,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f,%.3f,%d\n", o.variant.c_str(), o.seed, s.t, nt, et, ne,
                   ee, s.err_h, s.sigma_ne, s.nees_pos, s.mode);
    }
    // 成功匹配的提供者日志与仿真中的量测记录一一对应（按时间顺序）
    std::vector<const ProviderLog*> ok;
    for (const auto& l : o.log)
      if (std::isfinite(l.err)) ok.push_back(&l);
    for (size_t i = 0; i < r.fixes.size(); ++i) {
      const sim::FixRecord& x = r.fixes[i];
      real nf, ef, nt, et;
      frame.toMap(x.lat, x.lon, nf, ef);
      frame.toMap(x.truth_lat, x.truth_lon, nt, et);
      const ProviderLog* l = i < ok.size() ? ok[i] : nullptr;
      std::fprintf(ff, "%s,%u,%.2f,%.1f,%.1f,%.1f,%.1f,%.2f,%d,%d,%d,%.3f,%.2f,%s,%.2f\n", o.variant.c_str(), o.seed, x.t,
                   nf, ef, nt, et, x.innov, x.outlier, x.accepted, x.recovered, x.nis, l ? l->sigma : NAN,
                   l ? l->stage.c_str() : "", l ? l->psr : NAN);
    }
  }
  std::fclose(fr);
  std::fclose(fs);
  std::fclose(ff);

  std::printf("%-21s %9s %9s %9s %8s %8s %8s %6s\n", "variant", "RMS中位", "RMS最大", "终值中位", "真接受",
              "真拒绝", "错接受", "锁死");
  for (const auto& v : variants) {
    std::vector<real> rms, fin;
    int ta = 0, tr = 0, oa = 0, lk = 0;
    for (const auto& o : outs)
      if (o.variant == v.name) {
        rms.push_back(o.res.rms_h), fin.push_back(o.res.final_h);
        ta += o.res.true_accepted, tr += o.res.true_rejected, oa += o.res.outlier_accepted;
        lk += o.res.lockout_episodes > 0;
      }
    std::sort(rms.begin(), rms.end());
    std::sort(fin.begin(), fin.end());
    std::printf("%-21s %9.1f %9.1f %9.1f %8d %8d %8d %3d/%zu\n", v.name.c_str(), rms[rms.size() / 2], rms.back(),
                fin[fin.size() / 2], ta, tr, oa, lk, rms.size());
  }
  return 0;
}
