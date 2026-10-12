// Monte-Carlo 批量仿真：同一配置下运行 N 个随机种子，输出逐次统计与逐秒聚合时间序列。
//   ./nav_mc runs=100 out=results/baseline [threads=4] [seed0=1] [key=value ...]
// 输出：
//   <out>_runs.csv     每次运行一行：RMS、最大误差、修正接受/拒绝分类、锁死次数、NEES 等
//   <out>_series.csv   逐秒：水平误差均值/中位数/95 分位、平均估计 σ、平均位置/运动学 NEES（ANEES）
//   <out>_fixes.csv    每次绝对修正：是否错误匹配、是否被接受、NIS、新息大小
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "sim_core.hpp"

using namespace sim;

int main(int argc, char** argv) {
  SimConfig cfg;
  int runs = 100, threads = static_cast<int>(std::thread::hardware_concurrency());
  unsigned seed0 = 1;
  std::string out = "mc";
  std::vector<char*> rest = {argv[0]};
  for (int i = 1; i < argc; ++i) {
    if (!std::strncmp(argv[i], "runs=", 5)) runs = std::atoi(argv[i] + 5);
    else if (!std::strncmp(argv[i], "threads=", 8)) threads = std::atoi(argv[i] + 8);
    else if (!std::strncmp(argv[i], "seed0=", 6)) seed0 = static_cast<unsigned>(std::atoi(argv[i] + 6));
    else if (!std::strncmp(argv[i], "out=", 4)) out = argv[i] + 4;
    else rest.push_back(argv[i]);
  }
  std::string err;
  if (!parseArgs(cfg, static_cast<int>(rest.size()), rest.data(), 1, &err)) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 2;
  }
  if (threads < 1) threads = 1;
  std::fprintf(stderr, "Monte-Carlo %d 次，%d 线程：%s\n", runs, threads, describe(cfg).c_str());

  std::vector<Result> res(runs);
  std::atomic<int> next{0};
  std::vector<std::thread> pool;
  for (int w = 0; w < threads; ++w)
    pool.emplace_back([&] {
      for (int i; (i = next++) < runs;) {
        SimConfig c = cfg;
        c.seed = seed0 + i;
        res[i] = run(c, Scenario{}, true, false);
      }
    });
  for (auto& th : pool) th.join();

  FILE* f = std::fopen((out + "_runs.csv").c_str(), "w");
  if (!f) { std::perror("open"); return 1; }
  std::fprintf(f, "seed,rms_h,max_h,final_h,true_accepted,true_rejected,outlier_accepted,outlier_rejected,"
                  "lockout_episodes,lockout_longest_s,recoveries,nees_pos_mean,nees_kin_mean,nees_pos_in95\n");
  for (int i = 0; i < runs; ++i) {
    const Result& r = res[i];
    std::fprintf(f, "%u,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d,%.0f,%u,%.3f,%.3f,%.3f\n", seed0 + i, r.rms_h, r.max_h, r.final_h,
                 r.true_accepted, r.true_rejected, r.outlier_accepted, r.outlier_rejected, r.lockout_episodes,
                 r.lockout_longest_s, r.recoveries, r.nees_pos_mean, r.nees_kin_mean, r.nees_pos_in95);
  }
  std::fclose(f);

  f = std::fopen((out + "_series.csv").c_str(), "w");
  std::fprintf(f, "t,err_mean,err_median,err_p95,sigma_mean,anees_pos,anees_kin\n");
  const std::size_t T = res[0].samples.size();
  std::vector<real> col(runs);
  for (std::size_t k = 0; k < T; ++k) {
    real em = 0, sm = 0, np = 0, nk = 0;
    for (int i = 0; i < runs; ++i) {
      const Sample& s = res[i].samples[k];
      col[i] = s.err_h;
      em += s.err_h; sm += s.sigma_ne; np += s.nees_pos; nk += s.nees_kin;
    }
    std::sort(col.begin(), col.end());
    std::fprintf(f, "%.0f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f\n", res[0].samples[k].t, em / runs, col[runs / 2],
                 col[std::min(runs - 1, static_cast<int>(0.95 * runs))], sm / runs, np / runs, nk / runs);
  }
  std::fclose(f);

  f = std::fopen((out + "_fixes.csv").c_str(), "w");
  std::fprintf(f, "seed,t,source,outlier,accepted,recovered,nis,innov_m\n");
  for (int i = 0; i < runs; ++i)
    for (const FixRecord& x : res[i].fixes)
      std::fprintf(f, "%u,%.2f,%d,%d,%d,%d,%.3f,%.2f\n", seed0 + i, x.t, x.source, x.outlier, x.accepted, x.recovered,
                   x.nis, x.innov);
  std::fclose(f);

  // 控制台汇总
  std::vector<real> rms(runs);
  int lock_runs = 0, tr = 0, ta = 0, oa = 0, orj = 0;
  real anees = 0;
  for (int i = 0; i < runs; ++i) {
    rms[i] = res[i].rms_h;
    lock_runs += res[i].lockout_episodes > 0;
    tr += res[i].true_rejected; ta += res[i].true_accepted;
    oa += res[i].outlier_accepted; orj += res[i].outlier_rejected;
    anees += res[i].nees_pos_mean;
  }
  std::sort(rms.begin(), rms.end());
  std::printf("RMS 中位数 %.1f m，95 分位 %.1f m，最大 %.1f m；发生锁死的运行 %d/%d；"
              "正确修正错拒率 %.1f%%，错误匹配误接率 %.1f%%；位置 ANEES %.1f（理想 3）\n",
              rms[runs / 2], rms[std::min(runs - 1, static_cast<int>(0.95 * runs))], rms[runs - 1], lock_runs, runs,
              100.0 * tr / std::max(1, tr + ta), 100.0 * oa / std::max(1, oa + orj), anees / runs);
  return 0;
}
