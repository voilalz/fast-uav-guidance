// 论文 demo：图像在环的闭环导航仿真。
//
// 飞机以 600 km/h、离地约 1 km 沿地图长边的中线往返飞行（默认 Ahr 河谷条带 0.84 km × 5.43 km；
// Louisiana 地图 5.5 km × 17 km 用 map=… lat=29.95 lon=-89.95），两端各延伸 12 km 后按 45° 坡度限制掉头，
// 由航线跟踪制导律重新切入航线。每次飞越地图时，按 1 Hz 关键帧：
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
//   *_window          搜索窗口下限：匹配器先验 σ ≥ 2 m/s × 距上次被接受修正的时间（滤波器协方差不变）
//   *_floor           协方差下限：NavCore 把水平位置 σ 抬高到 2 m/s × 距上次被接受修正的时间（窗口与门限一起放大）
//   *_afloor          自适应协方差下限：连续 10 次有覆盖的匹配尝试都没有被接受的修正才启用下限，
//                     且下限不超过参考覆盖允许的 σ（3σ + 20 m ≤ 地图短边的一半）
//   *_sh / *_fading / *_huber / *_studentt   对比基线：景象匹配更新改用 Sage–Husa 自适应 R、自适应渐消、
//                     Huber M 估计、Student-t 变分贝叶斯（见 nav::FixUpdateMode），不用锁死恢复与下限
//
//   nav_demo [map=data/ahr_0.50m.map] [lat=50.53 lon=7.05] [max_prior=1000] [runs=8] [seed0=1] [threads=4]
//            [out=results/demo/demo]
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
  // 匹配搜索窗口下限：先验 σ ≥ window_rate × 距上次被接受修正的时间（m/s）；0 表示只用滤波器 σ
  real window_rate = 0;
};

const std::vector<Variant>& allVariants() {
  static const std::vector<Variant> v = {
      {"original", {}},
      {"original_recovery_sep", {{"recovery", "1"}, {"recovery_min_sep", "3"}}},
      {"original_window", {}, 2.0},
      {"original_recovery_sep_window", {{"recovery", "1"}, {"recovery_min_sep", "3"}}, 2.0},
      {"matched", {{"filter_vo_scale_sigma", "0.01"}, {"filter_vo_scale_tau", "120"}, {"filter_accel_bias_rw", "4e-4"}}},
      {"matched_recovery",
       {{"filter_vo_scale_sigma", "0.01"}, {"filter_vo_scale_tau", "120"}, {"filter_accel_bias_rw", "4e-4"},
        {"recovery", "1"}}},
      {"matched_recovery_sep",
       {{"filter_vo_scale_sigma", "0.01"}, {"filter_vo_scale_tau", "120"}, {"filter_accel_bias_rw", "4e-4"},
        {"recovery", "1"}, {"recovery_min_sep", "3"}}},
      {"matched_recovery_sep_window",
       {{"filter_vo_scale_sigma", "0.01"}, {"filter_vo_scale_tau", "120"}, {"filter_accel_bias_rw", "4e-4"},
        {"recovery", "1"}, {"recovery_min_sep", "3"}},
       2.0},
      {"original_floor", {{"sigma_floor_rate", "2"}}},
      {"original_recovery_sep_floor", {{"recovery", "1"}, {"recovery_min_sep", "3"}, {"sigma_floor_rate", "2"}}},
      {"matched_recovery_sep_floor",
       {{"filter_vo_scale_sigma", "0.01"}, {"filter_vo_scale_tau", "120"}, {"filter_accel_bias_rw", "4e-4"},
        {"recovery", "1"}, {"recovery_min_sep", "3"}, {"sigma_floor_rate", "2"}}},
      // 自适应协方差下限（sigma_floor_max 在 main 中按地图覆盖补上）
      {"original_afloor", {{"sigma_floor_rate", "2"}, {"sigma_floor_trigger", "10"}}},
      {"original_recovery_sep_afloor",
       {{"recovery", "1"}, {"recovery_min_sep", "3"}, {"sigma_floor_rate", "2"}, {"sigma_floor_trigger", "10"}}},
      {"matched_recovery_sep_afloor",
       {{"filter_vo_scale_sigma", "0.01"}, {"filter_vo_scale_tau", "120"}, {"filter_accel_bias_rw", "4e-4"},
        {"recovery", "1"}, {"recovery_min_sep", "3"}, {"sigma_floor_rate", "2"}, {"sigma_floor_trigger", "10"}}},
      // 基线：自适应 / 鲁棒滤波
      {"original_sh", {{"fix_update", "1"}}},
      {"original_fading", {{"fix_update", "2"}}},
      {"original_huber", {{"fix_update", "3"}}},
      {"original_studentt", {{"fix_update", "4"}}},
      {"matched_sh", {{"filter_vo_scale_sigma", "0.01"}, {"filter_vo_scale_tau", "120"}, {"filter_accel_bias_rw", "4e-4"}, {"fix_update", "1"}}},
      {"matched_fading", {{"filter_vo_scale_sigma", "0.01"}, {"filter_vo_scale_tau", "120"}, {"filter_accel_bias_rw", "4e-4"}, {"fix_update", "2"}}},
      {"matched_huber", {{"filter_vo_scale_sigma", "0.01"}, {"filter_vo_scale_tau", "120"}, {"filter_accel_bias_rw", "4e-4"}, {"fix_update", "3"}}},
      {"matched_studentt", {{"filter_vo_scale_sigma", "0.01"}, {"filter_vo_scale_tau", "120"}, {"filter_accel_bias_rw", "4e-4"}, {"fix_update", "4"}}},
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

// 航线跟踪制导：在 A、B 两点（地图坐标）之间往返，航向指令 = 航线方向 − atan(偏航距 / L)，
// 坡度 ≤ 45°，坡度变化率受限
class Guidance {
 public:
  Guidance(const Frame& f, real na, real ea, real nb, real eb) : f_(f), na_(na), ea_(ea) {
    len_ = std::hypot(nb - na, eb - ea);
    un_ = (nb - na) / len_;
    ue_ = (eb - ea) / len_;
  }

  real yawRate(real t, const nav::NavState& s) {
    real n, e;
    f_.toMap(s.lat, s.lon, n, e);
    const real along = (n - na_) * un_ + (e - ea_) * ue_;
    const real cross = -(n - na_) * ue_ + (e - ea_) * un_;  // A→B 方向右侧为正
    if (to_b_ && along > len_) to_b_ = false;                // 越过端点后掉头
    if (!to_b_ && along < 0) to_b_ = true;
    const real line_course = std::atan2(to_b_ ? ue_ : -ue_, to_b_ ? un_ : -un_);
    const real xte = to_b_ ? cross : -cross;
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
  real na_, ea_, un_, ue_, len_;
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
  real lat_deg = 50.53, lon_deg = 7.05, max_prior = 1000;
  std::vector<char*> rest = {argv[0]};
  for (int i = 1; i < argc; ++i) {
    if (!std::strncmp(argv[i], "map=", 4)) map_path = argv[i] + 4;
    else if (!std::strncmp(argv[i], "runs=", 5)) runs = std::atoi(argv[i] + 5);
    else if (!std::strncmp(argv[i], "threads=", 8)) threads = std::atoi(argv[i] + 8);
    else if (!std::strncmp(argv[i], "seed0=", 6)) seed0 = static_cast<unsigned>(std::atoi(argv[i] + 6));
    else if (!std::strncmp(argv[i], "out=", 4)) out = argv[i] + 4;
    else if (!std::strncmp(argv[i], "variants=", 9)) variants_arg = argv[i] + 9;
    else if (!std::strncmp(argv[i], "lat=", 4)) lat_deg = std::atof(argv[i] + 4);
    else if (!std::strncmp(argv[i], "lon=", 4)) lon_deg = std::atof(argv[i] + 4);
    else if (!std::strncmp(argv[i], "max_prior=", 10)) max_prior = std::atof(argv[i] + 10);
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
  // 航线沿影像区域的长轴穿过中心，两端各延伸 kExtend
  const real H = map.height() * map.gsd, W = map.width() * map.gsd;
  const bool ns = H >= W;
  const real na = ns ? kExtend : -H / 2, ea = ns ? W / 2 : -kExtend;
  const real nb = ns ? -H - kExtend : -H / 2, eb = ns ? W / 2 : W + kExtend;

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

  // 起点：航线 A 端（区域外 12 km），沿航线飞向 B
  sim::Scenario sc0;
  sc0.lat0 = lat_deg * kDeg;
  sc0.lon0 = lon_deg * kDeg;
  sc0.h0 = 1500.0;
  sc0.yaw0 = std::atan2(eb - ea, nb - na);
  const nav::earth::Radii r0 = nav::earth::radii(sc0.lat0);
  const Frame frame{sc0.lat0, sc0.lon0, r0.rn, r0.re * std::cos(sc0.lat0), na, ea, T.mean() + 1000.0 - sc0.h0};

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
        real window_rate = 0;
        for (const auto& v : variants)
          if (v.name == o.variant) {
            for (const auto& [key, val] : v.params) sim::setParam(c, key, val);
            window_rate = v.window_rate;
            // 自适应下限的上界：3σ + 20 m 不超过地图短边的一半（参考覆盖），且不超过匹配器先验上限
            if (c.sigma_floor_trigger > 0 && c.sigma_floor_max <= 0)
              c.sigma_floor_max = std::min((std::min(H, W) / 2 - 20.0) / 3.0, max_prior);
          }
        sim::Scenario sc = sc0;
        auto guid = std::make_shared<Guidance>(frame, na, ea, nb, eb);
        sc.yaw_rate = [guid](real t, const nav::NavState& s) { return guid->yawRate(t, s); };
        vision::SceneMatcher matcher(map.pre, map.gsd, cam, T);
        std::mt19937_64 rng(o.seed * 1000003ull + 7);  // 图像噪声：与滤波器配置无关，各配置看到同一组图像
        sc.fix_provider = [&, window_rate](real t, const nav::NavState& truth, const nav::NavState& est,
                                           real sigma_ne, real t_since_fix, nav::PositionFix& z, bool& outlier) {
          const vision::Pose tp = frame.pose(truth);
          if (tp.n > 150 || tp.n < -H - 150 || tp.e < -150 || tp.e > W + 150) return -1;
          ++o.attempts;
          cv::Mat img, valid;
          renderer.render(tp, cam, img, valid, &rng);
          const vision::Pose ep = frame.pose(est);
          // 搜索窗口取 3σ：滤波器 σ 过于乐观时真实位置落在窗口外，匹配器直接失败（匹配器层面的“锁死”）。
          // window 配置按“距上次被接受修正的时间 × 漂移率”给 σ 设下限，与滤波器自身的 σ 无关
          const real prior = std::clamp(std::max(sigma_ne, window_rate * t_since_fix), 10.0, max_prior);
          const vision::MatchResult r = matcher.match(img, valid, ep, prior);
          const real e = r.ok ? std::hypot(r.n - tp.n, r.e - tp.e) : NAN;
          o.log.push_back({t, r.stage, r.psr, r.sigma, e});
          if (!r.ok) return 0;
          frame.toLatLon(r.n, r.e, z.lat, z.lon);
          z.sigma_ne = r.sigma;
          if (!std::isnan(r.dyaw)) {
            z.yaw = nav::wrapPi(ep.yaw + r.dyaw);
            z.sigma_yaw = 0.3 * kDeg;
          }
          outlier = e > 30.0;
          return 1;
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
