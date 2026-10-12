# 视觉导航离线管线（Python）

论文实验用的图像级实现：在真实多时相卫星影像与 DEM 上渲染机载下视图像，运行视觉里程计与景象匹配，
统计其误差分布，再把统计结果回灌到 `nav/sim` 的量测模型中。机载实时实现（Jetson，C++/TensorRT）
的接口见 `nav/apps/companion_main.cpp`，两者算法流程一致。

| 文件 | 内容 |
|---|---|
| `mapdata.py` | 下载 SpaceNet 8（Ahr 河谷洪水前/后两期 Maxar 影像，CC BY-SA 4.0）与 Copernicus GLO-30 DEM，重投影到同一 UTM 网格 |
| `camera.py` | 下视针孔相机、射线–DEM 求交、带地形视差的实时图渲染 |
| `vo.py` | 下视单目 VO：LK 光流 + 姿态去旋转 + 地面求交，三种尺度来源（雷达高 / 气压−DEM / 逐点 DEM 射线） |
| `scene_match.py` | 景象匹配：DEM 正射校正 → 掩膜 NCC 粗匹配（PSR、歧义比）→ SIFT + RANSAC 相似变换精化 → 量测 σ |
| `eval_scene_match.py` | 300 帧随机位姿评估：同源 / 跨时相 / 梯度预处理 / 平地正射 / 大先验误差 |
| `eval_vo.py` | 沿河谷定高直线飞行 600 km/h、30 Hz，评估 VO 尺度误差及其相关长度 |

## 运行

```bash
python3 -m pip install numpy opencv-python-headless rasterio matplotlib
python3 -m vision.mapdata                      # 下载约 150 MB 数据到 data/，构建 0.5 m 网格
python3 -m vision.eval_scene_match --trials 300 --out results/vision
python3 -m vision.eval_vo --out results/vision
```

## 数据说明

- 参考库：PRE-event（0.42 m，洪水前）重采样到 1 m，对应 `docs/navigation-design.md` §4.7 的 1 m 参考库。
- 实时图：POST-event（0.70 m，洪水后、不同卫星、含薄云与雾霾）按相机模型渲染，地面分辨率约 0.6 m，
  幅宽约 384 m（1 km 离地高）。渲染使用 DEM 射线求交，含真实地形视差（区域地形高差约 200 m）。
- 局限：渲染无法再现真实的离天底视角建筑倾斜和 BRDF 差异；两期影像之间本身存在数米的配准偏差，
  评估中以“偏差（bias）”单独列出。
