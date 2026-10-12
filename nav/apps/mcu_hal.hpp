// 导航 MCU（STM32H743）硬件抽象层接口。
// 目标板实现基于 STM32Cube HAL/LL + FreeRTOS；宿主机实现见 mcu_main.cpp 中的 Host* 桩。
#pragma once

#include <cstddef>
#include <cstdint>

#include "nav/imu_array.hpp"
#include "nav/types.hpp"

namespace hal {

using nav::TimeUs;

// 64 位微秒时基：TIM2(32 bit, 1 MHz) + 溢出计数；PPS 由 TIM2 输入捕获锁存
class ITimebase {
 public:
  virtual ~ITimebase() = default;
  virtual TimeUs nowUs() const = 0;
  virtual bool lastPps(TimeUs& t) const = 0;
};

// IMU 阵列：4 × ADIS16470 共用 SYNC（TIM 输出 2 kHz），各自 SPI + DMA 读取
constexpr std::size_t kImuCount = 4;
class IImuArrayDriver {
 public:
  virtual ~IImuArrayDriver() = default;
  // 在 DR 中断/DMA 完成回调中调用：读取一组同步采样，返回 false 表示本周期无新数据
  virtual bool read(nav::RawImu (&out)[kImuCount], TimeUs& t_sample) = 0;
};

class IAirData {  // Aeroprobe μADC 或 MS4525DO-005D + BMP390
 public:
  virtual ~IAirData() = default;
  virtual bool readAirspeed(nav::Airspeed& z) = 0;
  virtual bool readBaro(nav::BaroAlt& z) = 0;
};

class IMagnetometer {  // PNI RM3100，含硬/软铁补偿后输出航向
 public:
  virtual ~IMagnetometer() = default;
  virtual bool readHeading(nav::MagHeading& z) = 0;
};

class IRadarAltimeter {
 public:
  virtual ~IRadarAltimeter() = default;
  virtual bool read(nav::RadarAlt& z) = 0;
};

// 相机触发：TIM 输出比较产生 30 Hz 脉冲，曝光中点 = 触发沿 + 固定延迟 + 曝光/2
class ICameraTrigger {
 public:
  virtual ~ICameraTrigger() = default;
  virtual bool poll(TimeUs& t_mid_exposure, std::uint32_t& frame_id, std::uint8_t& cam) = 0;
};

// 字节流链路：以太网 UDP（与 Jetson）、UART（与飞控）
class ILink {
 public:
  virtual ~ILink() = default;
  virtual std::size_t write(const std::uint8_t* d, std::size_t n) = 0;
  virtual std::size_t read(std::uint8_t* d, std::size_t cap) = 0;
};

}  // namespace hal
