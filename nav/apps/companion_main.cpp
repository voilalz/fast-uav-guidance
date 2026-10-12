// 视觉计算机（Jetson Orin NX，Linux + PREEMPT_RT）软件框架。
//
// 线程划分（均为预分配缓冲，运行期不在热路径上分配内存）：
//   CaptureThread   : 可见光/红外相机采集（硬件触发），用 MCU 的 CAM_TRIGGER 时间戳给帧打时间
//   VoThread        : 每帧：特征跟踪 + 单应分解 -> 机体系速度 -> VO_VELOCITY（30 Hz）
//   MatchThread     : 关键帧（约 1 Hz）：正射校正 -> 粗匹配 -> 精匹配 -> 质量评估 -> POSITION_FIX
//   Terrain3DThread : 滑窗运动恢复结构 -> 局部 DSM -> 与参考 DSM 配准 -> POSITION_FIX（含高度）
//   LinkThread      : 与 MCU 的 UDP 收发
// 真实后端：VO 用 OpenCV/VPI（GPU 光流），匹配用 TensorRT（SuperPoint + LightGlue INT8/FP16），
// 本文件只定义接口并用桩实现跑通数据流，宿主机可编译。
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "nav/math.hpp"
#include "nav/protocol.h"

namespace companion {

using nav::real;

struct Frame {
  std::uint64_t t_mid_exposure_us = 0;
  std::uint32_t frame_id = 0;
  std::uint8_t camera = 0;  // 0 可见光 1 红外
  int width = 0, height = 0;
  const std::uint8_t* pixels = nullptr;  // 指向预分配环形缓冲
};

struct Prior {  // 来自 MCU 的 NAV_PRIOR / NAV_OUTPUT
  double lat = 0, lon = 0;
  real alt = 0, agl = 0;
  real roll = 0, pitch = 0, yaw = 0;
  real sigma_ne = 0;
};

class ICamera {
 public:
  virtual ~ICamera() = default;
  virtual bool grab(Frame& f) = 0;
};

class IVisualOdometry {
 public:
  virtual ~IVisualOdometry() = default;
  // 输出机体系速度及其 1σ；agl 为尺度来源（气压-DEM 或 SfM 地表高）
  virtual bool process(const Frame& f, const Prior& p, real vb[3], real& sigma, int& inliers) = 0;
};

struct MatchResult {
  double lat = 0, lon = 0;
  real yaw = NAN;
  real sigma_ne = 0;
  int inliers = 0;
  real psr = 0;         // 峰值旁瓣比
  real ambiguity = 0;   // 次优/最优
};

class ISceneMatcher {
 public:
  virtual ~ISceneMatcher() = default;
  virtual bool match(const Frame& f, const Prior& p, MatchResult& r) = 0;
};

// ---- 参考库瓦片管理 ----
// 瓦片按 0.01°（约 1 km）经纬网格划分，文件名 tiles/<lat_idx>_<lon_idx>.{dom,dsm,feat}
struct TileId { int lat_idx, lon_idx; };

class TileStore {
 public:
  static constexpr double kTileDeg = 0.01;

  // 计算覆盖先验 3σ 搜索窗口的瓦片列表，返回数量（最多 cap 个）
  static int tilesForWindow(double lat_rad, double lon_rad, real radius_m, TileId* out, int cap) {
    const double deg = 180.0 / nav::kPi;
    const double lat = lat_rad * deg, lon = lon_rad * deg;
    const double dlat = radius_m / 111320.0;
    const double dlon = radius_m / (111320.0 * std::cos(lat_rad));
    const int i0 = static_cast<int>(std::floor((lat - dlat) / kTileDeg));
    const int i1 = static_cast<int>(std::floor((lat + dlat) / kTileDeg));
    const int j0 = static_cast<int>(std::floor((lon - dlon) / kTileDeg));
    const int j1 = static_cast<int>(std::floor((lon + dlon) / kTileDeg));
    int n = 0;
    for (int i = i0; i <= i1; ++i)
      for (int j = j0; j <= j1; ++j)
        if (n < cap) out[n++] = {i, j};
    return n;
  }
};

// 量测质量 -> 协方差：内点少、PSR 低、歧义大时放大 σ，过差直接不报
inline bool gradeMatch(const MatchResult& r, real gsd_m, real& sigma) {
  if (r.inliers < 30 || r.ambiguity < 1.3) return false;
  const real k = 1.0 + 50.0 / r.inliers + 2.0 / (r.ambiguity - 1.0);
  sigma = std::fmax(3.0 * gsd_m * k, 5.0);
  return true;
}

// ---- 桩实现 ----
struct StubCamera : ICamera {
  std::uint32_t id = 0;
  bool grab(Frame& f) override {
    f.frame_id = ++id;
    f.t_mid_exposure_us = 33333ull * id;
    f.width = 2448;
    f.height = 2048;
    return id <= 90;
  }
};
struct StubVo : IVisualOdometry {
  bool process(const Frame&, const Prior&, real vb[3], real& sigma, int& inliers) override {
    vb[0] = 166.7; vb[1] = 0; vb[2] = 0; sigma = 0.6; inliers = 200;
    return true;
  }
};
struct StubMatcher : ISceneMatcher {
  bool match(const Frame&, const Prior& p, MatchResult& r) override {
    r.lat = p.lat; r.lon = p.lon; r.inliers = 120; r.psr = 12; r.ambiguity = 2.5;
    return true;
  }
};

}  // namespace companion

int main() {
  using namespace companion;
  StubCamera cam;
  StubVo vo;
  StubMatcher matcher;
  Prior prior;
  prior.lat = 30 * nav::kDeg;
  prior.lon = 110 * nav::kDeg;
  prior.sigma_ne = 50;

  std::vector<std::uint8_t> tx;  // 实机为 UDP socket
  tx.reserve(64 * 1024);
  std::uint16_t seq = 0;
  int n_vo = 0, n_fix = 0;
  Frame f;
  while (cam.grab(f)) {
    real vb[3], sig;
    int inl;
    if (vo.process(f, prior, vb, sig, inl)) {
      nav_vo_velocity_t m{};
      m.t_us = f.t_mid_exposure_us;
      m.frame_id = f.frame_id;
      for (int i = 0; i < 3; ++i) m.vb[i] = static_cast<float>(vb[i]);
      m.sigma = static_cast<float>(sig);
      m.inliers = static_cast<std::uint16_t>(inl);
      std::uint8_t buf[NAV_MAX_FRAME];
      const size_t n = nav_encode(NAV_MSG_VO_VELOCITY, seq++, &m, sizeof m, buf, sizeof buf);
      tx.insert(tx.end(), buf, buf + n);
      ++n_vo;
    }
    if (f.frame_id % 30 == 0) {  // 关键帧约 1 Hz
      TileId tiles[64];
      const int nt = TileStore::tilesForWindow(prior.lat, prior.lon, 3 * prior.sigma_ne, tiles, 64);
      MatchResult r;
      real sigma;
      if (nt > 0 && matcher.match(f, prior, r) && gradeMatch(r, 0.5, sigma)) {
        nav_position_fix_t m{};
        m.t_us = f.t_mid_exposure_us;
        m.frame_id = f.frame_id;
        m.lat_rad = r.lat;
        m.lon_rad = r.lon;
        m.sigma_ne_m = static_cast<float>(sigma);
        m.sigma_d_m = -1;
        m.yaw_rad = static_cast<float>(r.yaw);
        m.source = 1;
        m.inliers = static_cast<std::uint16_t>(r.inliers);
        std::uint8_t buf[NAV_MAX_FRAME];
        const size_t n = nav_encode(NAV_MSG_POSITION_FIX, seq++, &m, sizeof m, buf, sizeof buf);
        tx.insert(tx.end(), buf, buf + n);
        ++n_fix;
      }
    }
  }
  std::printf("3 s 数据：VO 消息 %d 条，位置修正 %d 条，发送 %zu 字节\n", n_vo, n_fix, tx.size());
  return (n_vo == 90 && n_fix == 3) ? 0 : 1;
}
