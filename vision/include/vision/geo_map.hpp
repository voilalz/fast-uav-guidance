// 局部地图网格：两期正射灰度影像 + 有效掩膜 + DEM，均在同一北向朝上的 UTM 网格上。
//
// 网格约定：像素 (row, col) 中心对应局部坐标 n = −row·gsd，e = col·gsd（原点为网格左上角像素中心）。
// 数据由 vision_build_map 从公开数据（SpaceNet 8 Ahr 河谷两期 Maxar 影像 + Copernicus GLO-30 DEM）
// 重投影生成，保存为简单二进制文件（见 save/load）。
#pragma once

#include <string>

#include <opencv2/core.hpp>

namespace vision {

struct GeoMap {
  int epsg = 32632;
  double x0 = 0, y0 = 0;  // 网格左上角（UTM，m）
  double gsd = 0.5;       // m/像素
  cv::Mat pre;            // CV_8U：PRE-event 灰度（参考库）
  cv::Mat post;           // CV_8U：POST-event 灰度（用于渲染机载实时图）
  cv::Mat mask;           // CV_8U：两期影像与 DEM 均有效处为 1
  cv::Mat dem;            // CV_32F：高程 m，无数据为 NaN

  int width() const { return pre.cols; }
  int height() const { return pre.rows; }

  bool save(const std::string& path) const;
  bool load(const std::string& path);
};

}  // namespace vision
