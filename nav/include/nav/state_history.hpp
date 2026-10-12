// 名义状态历史环形缓冲：用于时延量测（图像匹配结果滞后 100–300 ms）回溯到曝光时刻。
#pragma once

#include <cstddef>

#include "nav/types.hpp"

namespace nav {

// Entry 需含 t_us 成员；默认直接存名义状态。
template <std::size_t Capacity, class Entry = NavState>
class StateHistory {
 public:
  void push(const Entry& s) {
    buf_[head_] = s;
    head_ = (head_ + 1) % Capacity;
    if (size_ < Capacity) ++size_;
  }

  // 取时间上最接近 t_us 的历史状态；超出缓冲范围返回 false
  bool lookup(TimeUs t_us, Entry& out) const {
    if (size_ == 0) return false;
    const Entry& newest = at(size_ - 1);
    const Entry& oldest = at(0);
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
  const Entry& at(std::size_t i) const { return buf_[(head_ + Capacity - size_ + i) % Capacity]; }

  Entry buf_[Capacity];
  std::size_t head_ = 0, size_ = 0;
};

}  // namespace nav
