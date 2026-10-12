// 闭环数字仿真：600 km/h、1 km 高度、商用级 MEMS + 视觉里程计 + 景象/地形匹配 + 空速 + 气压 + 磁航向。
// 用于验证 NavCore/ESKF 的实现与方案的精度预算。
//   ./nav_sim [seconds] [seed] [key=value ...]
//   例：./nav_sim 1200 1 est_mag_bias=0          （磁航向偏差按白噪声处理）
//       ./nav_sim 1200 1 profile=realistic recovery=1
//       ./nav_sim 1200 1 csv=run.csv              （输出逐秒误差/σ/NEES 时间序列）
// 参数全集见 sim/sim_core.hpp 中的 SimConfig。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "sim_core.hpp"

using namespace sim;

int main(int argc, char** argv) {
  SimConfig cfg;
  int first_kv = 1;
  if (argc > 1 && !std::strchr(argv[1], '=')) { cfg.duration = std::atof(argv[1]); first_kv = 2; }
  if (argc > 2 && !std::strchr(argv[2], '=')) { cfg.seed = static_cast<unsigned>(std::atoi(argv[2])); first_kv = 3; }
  std::string csv;
  for (int i = first_kv; i < argc; ++i)
    if (std::strncmp(argv[i], "csv=", 4) == 0) { csv = argv[i] + 4; argv[i] = const_cast<char*>("profile=default"); }
  std::string err;
  if (!parseArgs(cfg, argc, argv, first_kv, &err)) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 2;
  }
  const real duration = cfg.duration;
  const real km = duration * 600.0 / 3.6 / 1000;
  std::printf("配置：%s\n\n", describe(cfg).c_str());

  std::printf("场景 A：组合导航（VO + 景象/三维地形匹配 + 空速 + 气压 + 磁航向），航程 %.0f km\n", km);
  const Result a = run(cfg, Scenario{}, !csv.empty(), true);
  std::printf("  => 水平 RMS %.1f m，最大 %.1f m，输出帧数 %d（%.1f Hz），绝对修正 接受 %u / 拒绝 %u / 超时 %u\n",
              a.rms_h, a.max_h, a.outputs, a.outputs / duration, a.fix.accepted, a.fix.rejected, a.fix.too_old);
  std::printf("     正确修正被拒 %d，错误匹配被接受 %d，门限锁死 %d 次（最长 %.0f s），锁死恢复 %u 次，"
              "位置 NEES 均值 %.1f（理想 3），95%% 区间内比例 %.0f%%\n\n",
              a.true_rejected, a.outlier_accepted, a.lockout_episodes, a.lockout_longest_s, a.recoveries,
              a.nees_pos_mean, a.nees_pos_in95 * 100);
  if (!csv.empty()) {
    if (FILE* f = std::fopen(csv.c_str(), "w")) {
      std::fprintf(f, "t,err_n,err_e,err_d,err_h,sigma_ne,nees_pos,nees_kin,mode\n");
      for (const Sample& s : a.samples)
        std::fprintf(f, "%.0f,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%d\n", s.t, s.err_n, s.err_e, s.err_d, s.err_h,
                     s.sigma_ne, s.nees_pos, s.nees_kin, s.mode);
      std::fclose(f);
    }
  }

  std::printf("场景 B：仅视觉里程计 + 空速 + 气压 + 磁航向（无绝对修正）\n");
  Scenario b;
  b.fixes = false;
  const Result rb = run(cfg, b, false, false);
  std::printf("  => 1/5/10 min 水平误差 %.0f / %.0f / %.0f m，终点 %.0f m\n\n", rb.err_at[0], rb.err_at[1], rb.err_at[2],
              rb.final_h);

  std::printf("场景 C：纯惯导（仅气压定高）\n");
  Scenario c;
  c.vo = c.fixes = c.airspeed = c.heading = false;
  const Result rc = run(cfg, c, false, false);
  std::printf("  => 1/5/10 min 水平误差 %.0f / %.0f / %.0f m\n", rc.err_at[0], rc.err_at[1], rc.err_at[2]);

  // 作为回归测试：组合导航 RMS 应在设计指标量级内，输出频率不低于 30 Hz
  const bool ok = a.rms_h < 50.0 && a.outputs / duration >= 29.5;
  std::printf("\n%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
