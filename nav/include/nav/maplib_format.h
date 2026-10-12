/*
 * 参考库瓦片格式（地面建库工具写入，视觉计算机读取）。C99 头文件，小端。
 *
 * 瓦片划分：经纬度 0.01° × 0.01° 网格。瓦片索引
 *   lat_idx = floor(lat_deg / 0.01)，lon_idx = floor(lon_deg / 0.01)
 * 瓦片覆盖 [lat_idx*0.01, (lat_idx+1)*0.01] × [lon_idx*0.01, (lon_idx+1)*0.01]。
 *
 * 目录：<library_root>/<lat_idx>_<lon_idx>/
 *   meta.bin  ：maplib_tile_meta_t
 *   dom.png   ：8 位灰度正射影像，dom_w × dom_h，北向朝上，第 0 行为北边界，第 0 列为西边界，
 *               像素按经纬度等间隔（每像素 0.01/dom_h 度纬度、0.01/dom_w 度经度），像素中心采样
 *   dsm.bin   ：float32 高程（m，椭球高）dsm_w × dsm_h，行列约定同 dom
 *   feat.bin  ：可选，预提取特征：maplib_feat_header_t + count × maplib_feat_t
 *
 * 库根目录下 index.bin：maplib_index_header_t + count × (int32 lat_idx, int32 lon_idx)
 */
#ifndef NAV_MAPLIB_FORMAT_H
#define NAV_MAPLIB_FORMAT_H

#include <stdint.h>

#define MAPLIB_TILE_DEG 0.01
#define MAPLIB_MAGIC_TILE 0x544C4D46u  /* "FMLT" */
#define MAPLIB_MAGIC_FEAT 0x54414546u  /* "FEAT" */
#define MAPLIB_MAGIC_INDEX 0x58444E49u /* "INDX" */
#define MAPLIB_VERSION 1u

#pragma pack(push, 1)

typedef struct {
  uint32_t magic;       /* MAPLIB_MAGIC_TILE */
  uint32_t version;     /* MAPLIB_VERSION */
  int32_t lat_idx, lon_idx;
  uint32_t dom_w, dom_h;
  uint32_t dsm_w, dsm_h;
  float matchability;   /* 0..1，可匹配性评分（纹理丰富度/稳定性） */
  float terrain_sigma;  /* 瓦片内地形高程标准差 m（地形匹配可用性） */
  uint64_t source_time; /* 影像获取时间（Unix 秒） */
  uint32_t crc32;       /* dom.png 与 dsm.bin 文件内容的 CRC32（按此顺序连续计算） */
} maplib_tile_meta_t;

typedef struct {
  uint32_t magic;   /* MAPLIB_MAGIC_FEAT */
  uint32_t version;
  uint32_t count;
  uint32_t desc_bytes; /* 每个描述子字节数（ORB = 32） */
} maplib_feat_header_t;

typedef struct {
  float x, y;          /* dom 像素坐标 */
  float response, angle, size;
  uint8_t desc[32];
} maplib_feat_t;

typedef struct {
  uint32_t magic;   /* MAPLIB_MAGIC_INDEX */
  uint32_t version;
  uint32_t count;
} maplib_index_header_t;

#pragma pack(pop)

#endif /* NAV_MAPLIB_FORMAT_H */
