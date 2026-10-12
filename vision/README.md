# 视觉导航离线管线（C++）

论文实验用的图像级实现：在真实多时相卫星影像与 DEM 上渲染机载下视图像，运行视觉里程计与景象匹配，
统计其误差分布，再把统计结果回灌到 `nav/sim` 的量测模型中；`demo/nav_demo` 把景象匹配器直接接入闭环导航仿真。
机载实时实现（Jetson，TensorRT）的接口见 `nav/apps/companion_main.cpp`，两者算法流程一致。

依赖 OpenCV ≥ 4.4（core / imgproc / video / features2d / calib3d，含 SIFT）；`vision_build_map` 另需 GDAL ≥ 3.4。
缺少 OpenCV 时 CMake 自动跳过本目录，不影响导航核心构建。

| 文件 | 内容 |
|---|---|
| `include/vision/geo_map.hpp` | 局部地图网格（两期正射灰度影像 + 掩膜 + DEM）及其二进制文件格式 |
| `include/vision/camera.hpp` | 下视针孔相机、射线–DEM 求交、投影、带地形视差的实时图渲染 |
| `include/vision/vo.hpp` | 下视单目 VO：LK 光流 + 姿态去旋转 + 地面求交，三种尺度来源（雷达高 / 气压−DEM / 逐点 DEM 射线） |
| `include/vision/scene_match.hpp` | 景象匹配：DEM 正射校正 → 掩膜 NCC 粗匹配（PSR、歧义比）→ SIFT + RANSAC 相似变换精化 → 量测 σ |
| `apps/build_map.cpp` | `vision_build_map`：用 GDAL 把 SpaceNet 8 瓦片与 Copernicus DEM 重投影到同一 UTM 网格 |
| `apps/eval_scene_match.cpp` | `vision_eval_scene_match`：300 帧随机位姿评估（同源 / 跨时相 / 梯度预处理 / 平地正射 / 大先验误差） |
| `apps/eval_vo.cpp` | `vision_eval_vo`：沿河谷定高直线飞行 600 km/h、30 Hz，评估 VO 尺度误差及其相关长度 |
| `tests/test_vision.cpp` | 合成纹理 + 合成地形上的几何一致性、景象匹配与 VO 精度测试（不依赖外部数据） |

## 运行

```bash
sudo apt install libopencv-dev libgdal-dev          # Ubuntu
cmake -B build && cmake --build build -j
tools/download_data.sh data                          # 下载约 150 MB 公开数据到 data/
build/vision/vision_build_map data                   # 生成 data/ahr_0.50m.map（约 130 MB）
build/vision/vision_eval_scene_match trials=300 out=results/vision
build/vision/vision_eval_vo out=results/vision
python3 tools/plot_vision.py results/vision          # 绘图（matplotlib）
```

## 数据说明

- 参考库：PRE-event（0.42 m，洪水前）重采样到 1 m，对应 `docs/navigation-design.md` §4.7 的 1 m 参考库。
- 实时图：POST-event（0.70 m，洪水后、不同卫星、含薄云与雾霾）按相机模型渲染，地面分辨率约 0.6 m，
  幅宽约 384 m（1 km 离地高）。渲染使用 DEM 射线求交，含真实地形视差（区域地形高差约 200 m）。
- 网格约定：像素 (row, col) ↔ 局部坐标 n = −row·gsd，e = col·gsd（北向朝上）。
- 局限：渲染无法再现真实的离天底视角建筑倾斜和 BRDF 差异；两期影像之间本身存在数米的配准偏差，
  评估中以“偏差（bias）”单独列出。

## 与早期 Python 原型的一致性

本目录由 Python 原型（numpy + OpenCV + rasterio）移植而来。同一输入下：GDAL 重建的地图与 rasterio 版
DEM 和掩膜逐像素一致、灰度差异 < 1 级（舍入）；300 帧评估的统计量在随机数差异范围内一致
（同源 2.2 m / 99.7%，跨时相报告率 28% → 32%、误差中位数 5.3 → 4.5 m；VO 雷达高尺度误差 2.3% → 2.2%）。
