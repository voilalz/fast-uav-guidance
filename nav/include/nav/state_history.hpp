// 名义状态历史环形缓冲：用于时延量测（图像匹配结果滞后 100–300 ms）回溯到曝光时刻。
#pragma once

#include <cstddef>

#include "nav/types.hpp"

namespace nav {

template <std::size_t Capacity>
class StateHistory {
 public:
  void push(const NavState& s) {
    buf_[head_] = s;
    head_ = (head_ + 1) % Capacity;
    if (size_ < Capacity) ++size_;
  }

  // 取时间上最接近 t_us 的历史状态；超出缓冲范围返回 false
  bool lookup(TimeUs t_us, NavState& out) const {
    if (size_ == 0) return false;
    const NavState& newest = at(size_ - 1);
    const NavState& oldest = at(0);
    if (t_us > newest.t_us + kTolUs || t_us + kTolUs < oldest.t_us) return false;
    std::size_t best = 0;
    TimeUs best_d = ~TimeUs{0};
    for (std::size_t i = 0; i < size_; ++i) {
      const TimeUs ti = at(i).t_us;
      const TimeUs d = ti > t_us ? ti - t_us : t_us - ti;
      if (d < best_d) { best_d = d; best = i; }
    }
    out = at(best);
    return true;
  }

  std::size_t size() const { return size_; }

 private:
  static constexpr TimeUs kTolUs = 20000;
  // i = 0 为最旧
  const NavState& at(std::size_t i) const { return buf_[(head_ + Capacity - size_ + i) % Capacity]; }

  NavState buf_[Capacity];
  std::size_t head_ = 0, size_ = 0;
};

}  // namespace nav
