// 把公开数据重投影到同一局部 UTM 网格，生成 GeoMap 二进制文件。
//
//   tools/download_data.sh data                       # 下载 SpaceNet 8 瓦片与 Copernicus DEM
//   vision_build_map data [gsd=0.5] [out=data/ahr_0.50m.map]
//
// 数据（不随仓库分发）：
//   * SpaceNet 8（CC BY-SA 4.0）：德国 Ahr 河谷（森林山地 + 城镇）同一地区两期 Maxar 影像，
//     PRE-event（2021 年洪水前，0.42 m）作参考影像库，POST-event（洪水后、不同卫星与时相，0.70 m）
//     用来渲染机载实时图，构成“跨时相、跨传感器”的景象匹配条件。
//   * Copernicus GLO-30 DEM（30 m）：渲染实时图（含地形视差）、正射校正与 VO 尺度。
// 网格覆盖全部 PRE 瓦片的外包框，北向朝上，左上角为原点。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <gdal_priv.h>
#include <gdal_utils.h>
#include <ogr_spatialref.h>

#include "vision/geo_map.hpp"

namespace fs = std::filesystem;

namespace {

struct Grid {
  double x0, y0, gsd;
  int w, h;
};

std::vector<std::string> listTif(const fs::path& dir) {
  std::vector<std::string> out;
  if (fs::exists(dir))
    for (const auto& e : fs::directory_iterator(dir))
      if (e.path().extension() == ".tif") out.push_back(e.path().string());
  std::sort(out.begin(), out.end());
  return out;
}

OGRSpatialReference utm(int epsg) {
  OGRSpatialReference s;
  s.importFromEPSG(epsg);
  s.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  return s;
}

Grid gridFor(const std::vector<std::string>& files, double gsd, int epsg) {
  double left = 1e300, bottom = 1e300, right = -1e300, top = -1e300;
  const OGRSpatialReference dst = utm(epsg);
  for (const auto& f : files) {
    GDALDataset* d = static_cast<GDALDataset*>(GDALOpen(f.c_str(), GA_ReadOnly));
    double gt[6];
    d->GetGeoTransform(gt);
    OGRSpatialReference src(*d->GetSpatialRef());
    src.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    OGRCoordinateTransformation* ct = OGRCreateCoordinateTransformation(&src, &dst);
    const double x0 = gt[0], y1 = gt[3], x1 = gt[0] + gt[1] * d->GetRasterXSize(),
                 y0 = gt[3] + gt[5] * d->GetRasterYSize();
    double l, b, r, t;
    ct->TransformBounds(x0, y0, x1, y1, &l, &b, &r, &t, 21);
    left = std::min(left, l), bottom = std::min(bottom, b), right = std::max(right, r), top = std::max(top, t);
    OGRCoordinateTransformation::DestroyCT(ct);
    GDALClose(d);
  }
  return {left, top, gsd, static_cast<int>((right - left) / gsd), static_cast<int>((top - bottom) / gsd)};
}

GDALDataset* memDataset(const Grid& g, int bands, GDALDataType type, int epsg, double init) {
  GDALDriver* mem = GetGDALDriverManager()->GetDriverByName("MEM");
  GDALDataset* d = mem->Create("", g.w, g.h, bands, type, nullptr);
  double gt[6] = {g.x0, g.gsd, 0, g.y0, 0, -g.gsd};
  d->SetGeoTransform(gt);
  const OGRSpatialReference s = utm(epsg);
  d->SetSpatialRef(&s);
  for (int b = 1; b <= bands; ++b) d->GetRasterBand(b)->Fill(init);
  return d;
}

// 与源数据同地理参照的单波段内存数据集
GDALDataset* likeSource(GDALDataset* src, const std::vector<float>& data) {
  GDALDriver* mem = GetGDALDriverManager()->GetDriverByName("MEM");
  const int w = src->GetRasterXSize(), h = src->GetRasterYSize();
  GDALDataset* d = mem->Create("", w, h, 1, GDT_Float32, nullptr);
  double gt[6];
  src->GetGeoTransform(gt);
  d->SetGeoTransform(gt);
  d->SetSpatialRef(src->GetSpatialRef());
  CPLErr e = d->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, w, h, const_cast<float*>(data.data()), w, h,
                                           GDT_Float32, 0, 0);
  (void)e;
  return d;
}

void warp(GDALDataset* src, GDALDataset* dst, std::vector<std::string> args) {
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);
  GDALWarpAppOptions* o = GDALWarpAppOptionsNew(argv.data(), nullptr);
  GDALDatasetH s = src;
  int usage = 0;
  GDALWarp(nullptr, dst, 1, &s, o, &usage);
  GDALWarpAppOptionsFree(o);
}

cv::Mat readBand(GDALDataset* d, const Grid& g) {
  cv::Mat m(g.h, g.w, CV_32F);
  CPLErr e = d->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, g.w, g.h, m.ptr<float>(0), g.w, g.h, GDT_Float32, 0, 0);
  (void)e;
  return m;
}

// 多张 RGB 瓦片重投影拼接（重叠处平均），返回灰度图与有效掩膜
void mosaicGray(const std::vector<std::string>& files, const Grid& g, int epsg, cv::Mat& gray, cv::Mat& valid) {
  cv::Mat acc = cv::Mat::zeros(g.h, g.w, CV_32F), cnt = cv::Mat::zeros(g.h, g.w, CV_32F);
  for (const auto& f : files) {
    GDALDataset* d = static_cast<GDALDataset*>(GDALOpen(f.c_str(), GA_ReadOnly));
    const int w = d->GetRasterXSize(), h = d->GetRasterYSize();
    std::vector<float> rgb(3 * static_cast<size_t>(w) * h), gr(static_cast<size_t>(w) * h), va(gr.size());
    int bands[3] = {1, 2, 3};
    CPLErr e = d->RasterIO(GF_Read, 0, 0, w, h, rgb.data(), w, h, GDT_Float32, 3, bands, 0, 0, 0, nullptr);
    (void)e;
    const size_t n = gr.size();
    for (size_t i = 0; i < n; ++i) {
      gr[i] = 0.299f * rgb[i] + 0.587f * rgb[n + i] + 0.114f * rgb[2 * n + i];
      va[i] = (rgb[i] + rgb[n + i] + rgb[2 * n + i]) > 0 ? 1.f : 0.f;
    }
    GDALDataset* sg = likeSource(d, gr);
    GDALDataset* sv = likeSource(d, va);
    GDALDataset* dg = memDataset(g, 1, GDT_Float32, epsg, 0);
    GDALDataset* dv = memDataset(g, 1, GDT_Float32, epsg, 0);
    warp(sg, dg, {"-r", "bilinear", "-srcnodata", "None", "-dstnodata", "None"});
    warp(sv, dv, {"-r", "near", "-srcnodata", "None", "-dstnodata", "None"});
    const cv::Mat mg = readBand(dg, g), mv = readBand(dv, g);
    acc += mg.mul(mv);
    cnt += mv;
    for (GDALDataset* x : {sg, sv, dg, dv, d}) GDALClose(x);
  }
  cv::Mat img = acc / cv::max(cnt, 1.0);
  img.convertTo(gray, CV_8U);
  valid = cnt > 0;
  valid /= 255;
}

cv::Mat demOnGrid(const std::vector<std::string>& files, const Grid& g, int epsg) {
  cv::Mat out(g.h, g.w, CV_32F, cv::Scalar(NAN));
  for (const auto& f : files) {
    GDALDataset* d = static_cast<GDALDataset*>(GDALOpen(f.c_str(), GA_ReadOnly));
    GDALDataset* t = memDataset(g, 1, GDT_Float32, epsg, NAN);
    warp(d, t, {"-r", "bilinear", "-dstnodata", "nan"});
    const cv::Mat m = readBand(t, g);
    for (int r = 0; r < g.h; ++r) {
      float* o = out.ptr<float>(r);
      const float* s = m.ptr<float>(r);
      for (int c = 0; c < g.w; ++c)
        if (std::isnan(o[c])) o[c] = s[c];
    }
    GDALClose(t);
    GDALClose(d);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string data = argc > 1 ? argv[1] : "data";
  double gsd = 0.5;
  std::string out;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a.rfind("gsd=", 0) == 0) gsd = std::stod(a.substr(4));
    if (a.rfind("out=", 0) == 0) out = a.substr(4);
  }
  if (out.empty()) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "ahr_%.2fm.map", gsd);
    out = (fs::path(data) / buf).string();
  }
  GDALAllRegister();
  const auto pre_f = listTif(fs::path(data) / "sn8" / "pre"), post_f = listTif(fs::path(data) / "sn8" / "post"),
             dem_f = listTif(fs::path(data) / "dem");
  if (pre_f.empty() || post_f.empty() || dem_f.empty()) {
    std::fprintf(stderr, "缺少数据：先运行 tools/download_data.sh %s\n", data.c_str());
    return 1;
  }
  vision::GeoMap map;
  const Grid g = gridFor(pre_f, gsd, map.epsg);
  map.x0 = g.x0;
  map.y0 = g.y0;
  map.gsd = gsd;
  cv::Mat mp, mq;
  mosaicGray(pre_f, g, map.epsg, map.pre, mp);
  mosaicGray(post_f, g, map.epsg, map.post, mq);
  map.dem = demOnGrid(dem_f, g, map.epsg);
  map.mask = mp & mq & (map.dem == map.dem);  // NaN != NaN
  if (!map.save(out)) {
    std::fprintf(stderr, "写入失败：%s\n", out.c_str());
    return 1;
  }
  double lo, hi;
  cv::minMaxLoc(map.dem, &lo, &hi, nullptr, nullptr, map.mask);
  std::printf("网格 %d×%d 像素，%.2f km × %.2f km，有效 %.0f%%，DEM %.0f–%.0f m -> %s\n", g.w, g.h, g.w * gsd / 1000,
              g.h * gsd / 1000, 100.0 * cv::countNonZero(map.mask) / map.mask.total(), lo, hi, out.c_str());
  return 0;
}
