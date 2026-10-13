# 可单独构建（cd nav && cmake -B build），也可由仓库根目录 CMakeLists.txt 统一构建
cmake_minimum_required(VERSION 3.16)
if(NOT DEFINED PROJECT_NAME)
  project(fast_uav_nav C CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_C_STANDARD 99)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
  enable_testing()
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

add_library(navsim STATIC sim/sim_core.cpp)
target_link_libraries(navsim PUBLIC navcore)
target_compile_options(navsim PRIVATE -Wall -Wextra)

add_executable(nav_sim sim/sim_main.cpp)
target_link_libraries(nav_sim navsim)

find_package(Threads REQUIRED)
add_executable(nav_mc sim/mc_main.cpp)
target_link_libraries(nav_mc navsim Threads::Threads)

add_executable(nav_mcu_host apps/mcu_main.cpp)
target_link_libraries(nav_mcu_host navcore)

add_executable(nav_companion_host apps/companion_main.cpp)
target_link_libraries(nav_companion_host navcore)

add_executable(nav_tests tests/test_main.cpp)
target_link_libraries(nav_tests navcore)
add_test(NAME unit COMMAND nav_tests)
add_test(NAME closed_loop_sim COMMAND nav_sim 1200 1)
add_test(NAME mcu_app_host COMMAND nav_mcu_host)
add_test(NAME companion_host COMMAND nav_companion_host)

# 计算耗时与内存基准（不进 ctest）
add_executable(nav_bench bench/nav_bench.cpp)
target_link_libraries(nav_bench navcore)
