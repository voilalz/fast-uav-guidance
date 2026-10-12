"""公开数据下载与本地地图网格构建。

数据（均为公开数据，不随仓库分发，由本脚本下载到 data/）：
  * SpaceNet 8（CC BY-SA 4.0）：德国 Ahr 河谷（森林山地 + 城镇）同一地区两期 Maxar 影像，
    PRE-event（2021 年洪水前，0.42 m）作参考影像库，POST-event（洪水后、不同卫星与时相，0.70 m）
    用来渲染机载实时图，构成“跨时相、跨传感器”的景象匹配条件。
  * Copernicus GLO-30 DEM（30 m）：渲染实时图（含地形视差）、正射校正与 VO 尺度。

所有数据重投影到同一个局部 UTM 32N 网格（北向朝上，默认 0.5 m），以网格左上角为原点：
  x = 东向（列方向，m），y = 北向（m），像素 (row, col) 对应 x = col·gsd，y = −row·gsd。
"""
import csv
import os
import urllib.request

import numpy as np

SN8 = "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/Germany_Training_Public/"
DEM_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N50_00_E00{e}_00_DEM/"
           "Copernicus_DSM_COG_10_N50_00_E00{e}_00_DEM.tif")
# 选取的连续瓦片块：SN8 Germany 网格第 28–29 行、第 59–68 列（约 3.5 km × 1.1 km，Ahr 河谷）
DEFAULT_ROWS = (28, 29)
DEFAULT_COLS = range(59, 69)


def _get(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def download(data_dir, rows=DEFAULT_ROWS, cols=DEFAULT_COLS):
    """下载所选瓦片的 PRE/POST 影像和覆盖它们的 DEM，返回 (pre 文件列表, post 文件列表, dem 文件列表)。"""
    mapping = _get(SN8 + "Germany_Training_Public_label_image_mapping.csv",
                   os.path.join(data_dir, "sn8", "mapping.csv"))
    pre, post = [], []
    for r in csv.DictReader(open(mapping)):
        _, a, b = r["label"].replace(".geojson", "").split("_")
        if int(a) in rows and int(b) in cols:
            pre.append(_get(SN8 + "PRE-event/" + r["pre-event image"],
                            os.path.join(data_dir, "sn8", "pre", r["pre-event image"])))
            post.append(_get(SN8 + "POST-event/" + r["post-event image 1"],
                             os.path.join(data_dir, "sn8", "post", r["post-event image 1"])))
    dem = [_get(DEM_URL.format(e=e), os.path.join(data_dir, "dem", f"cop30_N50_E00{e}.tif")) for e in (6, 7)]
    return pre, post, dem


class LocalGrid:
    """局部 UTM 网格（北向朝上）。"""

    def __init__(self, crs, x0, y0, width, height, gsd):
        self.crs, self.x0, self.y0, self.w, self.h, self.gsd = crs, x0, y0, width, height, gsd

    @property
    def transform(self):
        from rasterio.transform import from_origin
        return from_origin(self.x0, self.y0, self.gsd, self.gsd)

    def resampled(self, gsd):
        return LocalGrid(self.crs, self.x0, self.y0, int(self.w * self.gsd / gsd), int(self.h * self.gsd / gsd), gsd)


def grid_for(files, gsd=0.5, crs="EPSG:32632"):
    """覆盖全部影像瓦片交集外包框的网格。"""
    import rasterio
    from rasterio.warp import transform_bounds
    bs = []
    for f in files:
        with rasterio.open(f) as d:
            bs.append(transform_bounds(d.crs, crs, *d.bounds))
    bs = np.array(bs)
    left, bottom, right, top = bs[:, 0].min(), bs[:, 1].min(), bs[:, 2].max(), bs[:, 3].max()
    return LocalGrid(crs, left, top, int((right - left) / gsd), int((top - bottom) / gsd), gsd)


def mosaic_gray(files, grid, resampling="bilinear"):
    """把多张 RGB 瓦片重投影拼接到网格，返回 uint8 灰度图和有效掩膜。"""
    import rasterio
    from rasterio.warp import Resampling, reproject
    rs = getattr(Resampling, resampling)
    acc = np.zeros((grid.h, grid.w), np.float32)
    cnt = np.zeros((grid.h, grid.w), np.float32)
    for f in files:
        with rasterio.open(f) as d:
            src = d.read().astype(np.float32)
            gray = 0.299 * src[0] + 0.587 * src[1] + 0.114 * src[2]
            valid = (src.sum(axis=0) > 0).astype(np.float32)
            dst = np.zeros((grid.h, grid.w), np.float32)
            dv = np.zeros((grid.h, grid.w), np.float32)
            reproject(gray, dst, src_transform=d.transform, src_crs=d.crs, dst_transform=grid.transform,
                      dst_crs=grid.crs, resampling=rs)
            reproject(valid, dv, src_transform=d.transform, src_crs=d.crs, dst_transform=grid.transform,
                      dst_crs=grid.crs, resampling=Resampling.nearest)
            acc += dst * dv
            cnt += dv
    img = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0)
    return np.clip(img, 0, 255).astype(np.uint8), cnt > 0


def dem_on_grid(dem_files, grid):
    """DEM 双线性重采样到网格（米，EGM2008 高程）。"""
    import rasterio
    from rasterio.warp import Resampling, reproject
    out = np.full((grid.h, grid.w), np.nan, np.float32)
    for f in dem_files:
        with rasterio.open(f) as d:
            tmp = np.full((grid.h, grid.w), np.nan, np.float32)
            reproject(d.read(1).astype(np.float32), tmp, src_transform=d.transform, src_crs=d.crs,
                      dst_transform=grid.transform, dst_crs=grid.crs, resampling=Resampling.bilinear,
                      dst_nodata=np.nan)
            out = np.where(np.isnan(out), tmp, out)
    return out


def build(data_dir, gsd=0.5, cache=True):
    """下载并构建 (pre 灰度, post 灰度, 有效掩膜, DEM, grid)，结果缓存为 npz。"""
    path = os.path.join(data_dir, f"ahr_{gsd:.2f}m.npz")
    if cache and os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        g = z["grid"].item()
        return z["pre"], z["post"], z["mask"], z["dem"], LocalGrid(**g)
    pre_f, post_f, dem_f = download(data_dir)
    grid = grid_for(pre_f, gsd)
    pre, mp = mosaic_gray(pre_f, grid)
    post, mq = mosaic_gray(post_f, grid)
    dem = dem_on_grid(dem_f, grid)
    mask = mp & mq & ~np.isnan(dem)
    g = dict(crs=grid.crs, x0=grid.x0, y0=grid.y0, width=grid.w, height=grid.h, gsd=grid.gsd)
    np.savez_compressed(path, pre=pre, post=post, mask=mask, dem=dem, grid=np.array(g, dtype=object))
    return pre, post, mask, dem, grid


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--gsd", type=float, default=0.5)
    a = ap.parse_args()
    pre, post, mask, dem, g = build(a.data, a.gsd)
    print(f"网格 {g.w}×{g.h} 像素，{g.w * g.gsd / 1000:.2f} km × {g.h * g.gsd / 1000:.2f} km，"
          f"有效 {mask.mean() * 100:.0f}%，DEM {np.nanmin(dem):.0f}–{np.nanmax(dem):.0f} m")
