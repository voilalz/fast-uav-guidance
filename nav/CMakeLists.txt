cmake_minimum_required(VERSION 3.16)
project(fast_uav_nav C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 导航核心库：按嵌入式约束编译（无异常、无 RTTI），可直接移植到 STM32H7（arm-none-eabi-g++）
add_library(navcore STATIC
  src/strapdown.cpp
  src/eskf.cpp
  src/nav_core.cpp
  src/terrain.cpp
  src/protocol.c)
target_include_directories(navcore PUBLIC include)
target_compile_options(navcore PRIVATE -Wall -Wextra
  $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions -fno-rtti>)

add_executable(nav_sim sim/sim_main.cpp)
target_link_libraries(nav_sim navcore)

add_executable(nav_mcu_host apps/mcu_main.cpp)
target_link_libraries(nav_mcu_host navcore)

add_executable(nav_companion_host apps/companion_main.cpp)
target_link_libraries(nav_companion_host navcore)

enable_testing()
add_executable(nav_tests tests/test_main.cpp)
target_link_libraries(nav_tests navcore)
add_test(NAME unit COMMAND nav_tests)
add_test(NAME closed_loop_sim COMMAND nav_sim 1200 1)
add_test(NAME mcu_app_host COMMAND nav_mcu_host)
add_test(NAME companion_host COMMAND nav_companion_host)
