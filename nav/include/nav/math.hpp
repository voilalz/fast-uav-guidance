// 定长矩阵/向量/四元数工具。
// 嵌入式约束：无动态内存、无异常，全部在栈上，尺寸编译期确定。
#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace nav {

using real = double;  // STM32H7 具备硬件双精度 FPU

constexpr real kPi = 3.14159265358979323846;
constexpr real kDeg = kPi / 180.0;

template <std::size_t R, std::size_t C>
struct Mat {
  std::array<real, R * C> a{};

  static constexpr std::size_t rows = R;
  static constexpr std::size_t cols = C;

  real& operator()(std::size_t r, std::size_t c) { return a[r * C + c]; }
  real operator()(std::size_t r, std::size_t c) const { return a[r * C + c]; }
  real& operator[](std::size_t i) { return a[i]; }
  real operator[](std::size_t i) const { return a[i]; }

  static Mat zeros() { return Mat{}; }
  static Mat identity() {
    static_assert(R == C, "identity requires square matrix");
    Mat m{};
    for (std::size_t i = 0; i < R; ++i) m(i, i) = 1.0;
    return m;
  }

  Mat& operator+=(const Mat& o) {
    for (std::size_t i = 0; i < R * C; ++i) a[i] += o.a[i];
    return *this;
  }
  Mat& operator-=(const Mat& o) {
    for (std::size_t i = 0; i < R * C; ++i) a[i] -= o.a[i];
    return *this;
  }
  Mat& operator*=(real s) {
    for (auto& v : a) v *= s;
    return *this;
  }

  Mat<C, R> t() const {
    Mat<C, R> m;
    for (std::size_t r = 0; r < R; ++r)
      for (std::size_t c = 0; c < C; ++c) m(c, r) = (*this)(r, c);
    return m;
  }

  template <std::size_t BR, std::size_t BC>
  Mat<BR, BC> block(std::size_t r0, std::size_t c0) const {
    Mat<BR, BC> m;
    for (std::size_t r = 0; r < BR; ++r)
      for (std::size_t c = 0; c < BC; ++c) m(r, c) = (*this)(r0 + r, c0 + c);
    return m;
  }

  template <std::size_t BR, std::size_t BC>
  void setBlock(std::size_t r0, std::size_t c0, const Mat<BR, BC>& b) {
    for (std::size_t r = 0; r < BR; ++r)
      for (std::size_t c = 0; c < BC; ++c) (*this)(r0 + r, c0 + c) = b(r, c);
  }
};

template <std::size_t R, std::size_t C>
Mat<R, C> operator+(Mat<R, C> x, const Mat<R, C>& y) { return x += y; }
template <std::size_t R, std::size_t C>
Mat<R, C> operator-(Mat<R, C> x, const Mat<R, C>& y) { return x -= y; }
template <std::size_t R, std::size_t C>
Mat<R, C> operator*(Mat<R, C> x, real s) { return x *= s; }
template <std::size_t R, std::size_t C>
Mat<R, C> operator*(real s, Mat<R, C> x) { return x *= s; }
template <std::size_t R, std::size_t C>
Mat<R, C> operator-(Mat<R, C> x) { return x *= -1.0; }

template <std::size_t R, std::size_t K, std::size_t C>
Mat<R, C> operator*(const Mat<R, K>& x, const Mat<K, C>& y) {
  Mat<R, C> m;
  for (std::size_t r = 0; r < R; ++r)
    for (std::size_t k = 0; k < K; ++k) {
      const real v = x(r, k);
      if (v == 0.0) continue;  // 误差状态转移矩阵稀疏，跳过零元素
      for (std::size_t c = 0; c < C; ++c) m(r, c) += v * y(k, c);
    }
  return m;
}

// 高斯-约旦消元求逆（部分主元）。量测维数很小（≤3），足够。失败返回 false。
template <std::size_t N>
bool inverse(const Mat<N, N>& in, Mat<N, N>& out) {
  Mat<N, N> a = in;
  out = Mat<N, N>::identity();
  for (std::size_t c = 0; c < N; ++c) {
    std::size_t piv = c;
    for (std::size_t r = c + 1; r < N; ++r)
      if (std::fabs(a(r, c)) > std::fabs(a(piv, c))) piv = r;
    if (std::fabs(a(piv, c)) < 1e-15) return false;
    if (piv != c)
      for (std::size_t k = 0; k < N; ++k) {
        real t = a(c, k); a(c, k) = a(piv, k); a(piv, k) = t;
        t = out(c, k); out(c, k) = out(piv, k); out(piv, k) = t;
      }
    const real d = 1.0 / a(c, c);
    for (std::size_t k = 0; k < N; ++k) { a(c, k) *= d; out(c, k) *= d; }
    for (std::size_t r = 0; r < N; ++r) {
      if (r == c) continue;
      const real f = a(r, c);
      if (f == 0.0) continue;
      for (std::size_t k = 0; k < N; ++k) { a(r, k) -= f * a(c, k); out(r, k) -= f * out(c, k); }
    }
  }
  return true;
}

using Vec2 = Mat<2, 1>;
using Vec3 = Mat<3, 1>;
using Mat3 = Mat<3, 3>;

inline Vec3 vec3(real x, real y, real z) { Vec3 v; v[0] = x; v[1] = y; v[2] = z; return v; }
inline real dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline real norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return vec3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}
inline Mat3 skew(const Vec3& v) {
  Mat3 m;
  m(0, 1) = -v[2]; m(0, 2) = v[1];
  m(1, 0) = v[2];  m(1, 2) = -v[0];
  m(2, 0) = -v[1]; m(2, 1) = v[0];
  return m;
}

// Hamilton 四元数 (w, x, y, z)。本工程中 q 表示 机体系 -> 导航系(NED) 的旋转。
struct Quat {
  real w = 1, x = 0, y = 0, z = 0;

  Quat operator*(const Quat& o) const {
    return {w * o.w - x * o.x - y * o.y - z * o.z,
            w * o.x + x * o.w + y * o.z - z * o.y,
            w * o.y - x * o.z + y * o.w + z * o.x,
            w * o.z + x * o.y - y * o.x + z * o.w};
  }
  Quat conj() const { return {w, -x, -y, -z}; }
  void normalize() {
    const real n = std::sqrt(w * w + x * x + y * y + z * z);
    w /= n; x /= n; y /= n; z /= n;
    if (w < 0) { w = -w; x = -x; y = -y; z = -z; }
  }

  // 旋转向量 -> 四元数
  static Quat exp(const Vec3& r) {
    const real th = norm(r);
    if (th < 1e-12) return {1.0, 0.5 * r[0], 0.5 * r[1], 0.5 * r[2]};
    const real s = std::sin(0.5 * th) / th;
    return {std::cos(0.5 * th), s * r[0], s * r[1], s * r[2]};
  }
  // 四元数 -> 旋转向量
  Vec3 log() const {
    Quat q = *this;
    if (q.w < 0) { q.w = -q.w; q.x = -q.x; q.y = -q.y; q.z = -q.z; }
    const real sn = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
    const real th = 2.0 * std::atan2(sn, q.w);
    const real k = sn < 1e-12 ? 2.0 : th / sn;
    return vec3(k * q.x, k * q.y, k * q.z);
  }

  Mat3 toDcm() const {
    Mat3 m;
    m(0, 0) = 1 - 2 * (y * y + z * z); m(0, 1) = 2 * (x * y - w * z);     m(0, 2) = 2 * (x * z + w * y);
    m(1, 0) = 2 * (x * y + w * z);     m(1, 1) = 1 - 2 * (x * x + z * z); m(1, 2) = 2 * (y * z - w * x);
    m(2, 0) = 2 * (x * z - w * y);     m(2, 1) = 2 * (y * z + w * x);     m(2, 2) = 1 - 2 * (x * x + y * y);
    return m;
  }

  // ZYX 欧拉角（滚转、俯仰、偏航）
  static Quat fromEuler(real roll, real pitch, real yaw) {
    const real cr = std::cos(roll / 2), sr = std::sin(roll / 2);
    const real cp = std::cos(pitch / 2), sp = std::sin(pitch / 2);
    const real cy = std::cos(yaw / 2), sy = std::sin(yaw / 2);
    return {cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy};
  }
  Vec3 toEuler() const {
    const Mat3 c = toDcm();
    return vec3(std::atan2(c(2, 1), c(2, 2)), std::asin(-clamp1(c(2, 0))), std::atan2(c(1, 0), c(0, 0)));
  }

 private:
  static real clamp1(real v) { return v > 1 ? 1 : (v < -1 ? -1 : v); }
};

inline real wrapPi(real a) {
  while (a > kPi) a -= 2 * kPi;
  while (a < -kPi) a += 2 * kPi;
  return a;
}

}  // namespace nav
