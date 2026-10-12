// 单生产者单消费者无锁队列：中断（生产者）-> 主循环/任务（消费者）。
#pragma once

#include <atomic>
#include <cstddef>

namespace nav {

template <class T, std::size_t N>
class SpscQueue {
  static_assert((N & (N - 1)) == 0, "N must be a power of two");

 public:
  bool push(const T& v) {
    const std::size_t h = head_.load(std::memory_order_relaxed);
    if (h - tail_.load(std::memory_order_acquire) == N) return false;  // 满
    buf_[h & (N - 1)] = v;
    head_.store(h + 1, std::memory_order_release);
    return true;
  }
  bool pop(T& v) {
    const std::size_t t = tail_.load(std::memory_order_relaxed);
    if (t == head_.load(std::memory_order_acquire)) return false;  // 空
    v = buf_[t & (N - 1)];
    tail_.store(t + 1, std::memory_order_release);
    return true;
  }
  std::size_t size() const { return head_.load() - tail_.load(); }

 private:
  T buf_[N];
  std::atomic<std::size_t> head_{0}, tail_{0};
};

}  // namespace nav
