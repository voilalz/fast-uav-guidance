#include "vision/geo_map.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>

namespace vision {
namespace {

// 文件布局：Header，随后 pre / post / mask（uint8，行优先）与 dem（float32，行优先）
struct Header {
  char magic[8];  // "FUGMAP1"
  std::int32_t width, height, epsg, reserved;
  double x0, y0, gsd;
};
constexpr char kMagic[8] = "FUGMAP1";

bool writeMat(std::ofstream& f, const cv::Mat& m) {
  for (int r = 0; r < m.rows; ++r) f.write(m.ptr<char>(r), static_cast<std::streamsize>(m.cols * m.elemSize()));
  return static_cast<bool>(f);
}

bool readMat(std::ifstream& f, cv::Mat& m, int rows, int cols, int type) {
  m.create(rows, cols, type);
  f.read(m.ptr<char>(0), static_cast<std::streamsize>(m.total() * m.elemSize()));
  return static_cast<bool>(f);
}

}  // namespace

bool GeoMap::save(const std::string& path) const {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  Header h{};
  std::memcpy(h.magic, kMagic, sizeof kMagic);
  h.width = width();
  h.height = height();
  h.epsg = epsg;
  h.x0 = x0;
  h.y0 = y0;
  h.gsd = gsd;
  f.write(reinterpret_cast<const char*>(&h), sizeof h);
  return writeMat(f, pre) && writeMat(f, post) && writeMat(f, mask) && writeMat(f, dem);
}

bool GeoMap::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  Header h{};
  f.read(reinterpret_cast<char*>(&h), sizeof h);
  if (!f || std::memcmp(h.magic, kMagic, sizeof kMagic) != 0) return false;
  epsg = h.epsg;
  x0 = h.x0;
  y0 = h.y0;
  gsd = h.gsd;
  return readMat(f, pre, h.height, h.width, CV_8U) && readMat(f, post, h.height, h.width, CV_8U) &&
         readMat(f, mask, h.height, h.width, CV_8U) && readMat(f, dem, h.height, h.width, CV_32F);
}

}  // namespace vision
