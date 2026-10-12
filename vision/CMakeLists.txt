# 视觉导航离线管线（C++）：地图构建、下视相机渲染、VO、景象匹配与评估程序。
# 依赖 OpenCV（core/imgproc/video/features2d/calib3d）与 GDAL（仅 build_map 用于重投影）。
# 缺少依赖时整体跳过，不影响导航核心的构建。
find_package(OpenCV QUIET COMPONENTS core imgproc video features2d calib3d)
if(NOT OpenCV_FOUND)
  message(STATUS "vision: 未找到 OpenCV，跳过视觉管线")
  return()
endif()
find_package(Threads REQUIRED)

add_library(vision STATIC
  src/geo_map.cpp
  src/camera.cpp
  src/vo.cpp
  src/scene_match.cpp)
target_include_directories(vision PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vision PUBLIC ${OpenCV_LIBS})
target_compile_options(vision PRIVATE -Wall -Wextra)

add_executable(vision_eval_scene_match apps/eval_scene_match.cpp)
target_link_libraries(vision_eval_scene_match vision Threads::Threads)

add_executable(vision_eval_vo apps/eval_vo.cpp)
target_link_libraries(vision_eval_vo vision)

add_executable(vision_tests tests/test_vision.cpp)
target_link_libraries(vision_tests vision)
add_test(NAME vision_unit COMMAND vision_tests)

find_package(GDAL QUIET)
if(GDAL_FOUND)
  add_executable(vision_build_map apps/build_map.cpp)
  target_link_libraries(vision_build_map vision GDAL::GDAL)
else()
  message(STATUS "vision: 未找到 GDAL，跳过 vision_build_map")
endif()
