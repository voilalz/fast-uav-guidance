// 机载 DEM 访问与 TERCOM 批处理地形匹配。
// DEM 数据由地图瓦片管理器加载到外部缓冲区，本类只持有指针，不分配内存。
#pragma once

#include <cstddef>

#include "nav/math.hpp"

namespace nav {

// 局部 NED 规则网格 DEM：(row, col) = (北向, 东向)，原点为网格西南角
class Dem {
 public:
  Dem() = default;
  Dem(const float* data, std::size_t rows, std::size_t cols, real spacing_m)
      : d_(data), rows_(rows), cols_(cols), sp_(spacing_m) {}

  bool contains(real n, real e) const {
    return d_ && n >= 0 && e >= 0 && n < (rows_ - 1) * sp_ && e < (cols_ - 1) * sp_;
  }
  // 双线性插值高度；越界返回 false
  bool height(real n, real e, real& h) const;
  // 中心差分梯度 dh/dn, dh/de
  bool gradient(real n, real e, real& gn, real& ge) const;

 private:
  float at(std::size_t r, std::size_t c) const { return d_[r * cols_ + c]; }
  const float* d_ = nullptr;
  std::size_t rows_ = 0, cols_ = 0;
  real sp_ = 30.0;
};

struct TercomResult {
  bool valid = false;
  real dn = 0, de = 0;     // 惯导航迹应施加的修正（m）
  real cost = 0;           // 最优 MSD（去均值）
  real ambiguity = 0;      // 次优/最优 代价比，越大越可靠
  real terrain_sigma = 0;  // 剖面地形标准差（地形粗糙度）
};

// 批处理 TERCOM：剖面点 i 的惯导位置 (n_i, e_i)，测得地形高 t_i = 海拔 - 雷达高。
// 在 ±search_m 内按 step_m 网格搜索平移，代价为去均值均方差（对气压高度偏差不敏感）。
template <std::size_t MaxPts>
class Tercom {
 public:
  bool add(real n, real e, real terrain_h) {
    if (n_ >= MaxPts) return false;
    pn_[n_] = n; pe_[n_] = e; ph_[n_] = terrain_h;
    ++n_;
    return true;
  }
  void clear() { n_ = 0; }
  std::size_t size() const { return n_; }

  TercomResult match(const Dem& dem, real search_m, real step_m, real exclusion_m) const {
    TercomResult res;
    if (n_ < 10) return res;
    // 地形粗糙度
    real m = 0, v = 0;
    for (std::size_t i = 0; i < n_; ++i) m += ph_[i];
    m /= n_;
    for (std::size_t i = 0; i < n_; ++i) v += (ph_[i] - m) * (ph_[i] - m);
    res.terrain_sigma = std::sqrt(v / n_);

    real best = 1e30, best_dn = 0, best_de = 0;
    const int k = static_cast<int>(search_m / step_m);
    for (int a = -k; a <= k; ++a)
      for (int b = -k; b <= k; ++b) {
        const real c = cost(dem, a * step_m, b * step_m);
        if (c < best) { best = c; best_dn = a * step_m; best_de = b * step_m; }
      }
    if (best >= 1e29) return res;
    // 次优：排除最优点邻域后的最小代价
    real second = 1e30;
    for (int a = -k; a <= k; ++a)
      for (int b = -k; b <= k; ++b) {
        const real dn = a * step_m, de = b * step_m;
        if (std::fabs(dn - best_dn) < exclusion_m && std::fabs(de - best_de) < exclusion_m) continue;
        const real c = cost(dem, dn, de);
        if (c < second) second = c;
      }
    res.valid = true;
    res.dn = best_dn;
    res.de = best_de;
    res.cost = best;
    res.ambiguity = second / (best + 1e-6);
    return res;
  }

 private:
  real cost(const Dem& dem, real dn, real de) const {
    // 单遍累加：var = E[d^2] - E[d]^2，避免栈上临时数组
    real sum = 0, sum2 = 0;
    for (std::size_t i = 0; i < n_; ++i) {
      real h;
      if (!dem.height(pn_[i] + dn, pe_[i] + de, h)) return 1e30;
      const real d = ph_[i] - h;
      sum += d;
      sum2 += d * d;
    }
    const real mean = sum / n_;
    return sum2 / n_ - mean * mean;
  }

  real pn_[MaxPts], pe_[MaxPts], ph_[MaxPts];
  std::size_t n_ = 0;
};

}  // namespace nav
