# 宿主机（x86 / Jetson aarch64）总构建：导航核心库、传感器驱动、视觉服务、仿真、工具与全部测试。
# 导航 MCU 固件是单独的交叉编译工程，见 firmware/。
cmake_minimum_required(VERSION 3.16)
project(fast_uav_guidance C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()

foreach(dir nav drivers companion sim tools/maplib vision demo)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
  endif()
endforeach()
