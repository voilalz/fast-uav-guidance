"""下视针孔相机模型与机载实时图渲染。

坐标约定（与导航核心一致）：
  * 世界系为局部 NED：n 北、e 东、d 下；高度 h = −d（与 DEM 同一高程基准）。
  * 地图网格（mapdata.LocalGrid）：像素 (row, col) ↔ n = −row·gsd，e = col·gsd。
  * 机体系 FRD；相机光轴沿机体 +z（向下），图像上方指向机头：
      相机 x（向右）= 机体 +y，相机 y（向下）= 机体 −x，相机 z = 机体 +z。
"""
from dataclasses import dataclass

import cv2
import numpy as np

R_BC = np.array([[0.0, -1.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0]])  # 相机 -> 机体


def euler_to_dcm(roll, pitch, yaw):
    """ZYX 欧拉角 -> C_nb（机体到 NED），与 nav::Quat::fromEuler 一致。"""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
                     [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
                     [-sp, sr * cp, cr * cp]])


@dataclass
class Camera:
    width: int = 640
    height: int = 640
    f: float = 1667.0  # 焦距（像素）。1 km 高度下地面分辨率约 0.6 m，幅宽约 384 m

    @property
    def K(self):
        return np.array([[self.f, 0, self.width / 2], [0, self.f, self.height / 2], [0, 0, 1.0]])

    def pixel_rays(self, step=1):
        """全部像素（或按 step 抽稀）的相机系单位前向射线，形状 (H, W, 3)。"""
        u, v = np.meshgrid(np.arange(0, self.width, step) + 0.5, np.arange(0, self.height, step) + 0.5)
        return np.stack([(u - self.width / 2) / self.f, (v - self.height / 2) / self.f, np.ones_like(u)], -1)


@dataclass
class Pose:
    n: float
    e: float
    h: float           # 高程（与 DEM 同基准）
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def C_nc(self):
        return euler_to_dcm(self.roll, self.pitch, self.yaw) @ R_BC


class Terrain:
    """地图网格上的 DEM，双线性采样（越界取边界）。"""

    def __init__(self, dem, gsd):
        d = dem.copy()
        d[np.isnan(d)] = np.nanmean(d)
        self.dem, self.gsd = d.astype(np.float32), gsd

    def height(self, n, e):
        shape = np.shape(n)
        col = (np.asarray(e, np.float32) / self.gsd).ravel()
        row = (-np.asarray(n, np.float32) / self.gsd).ravel()
        k = col.size
        w = 1024  # cv2.remap 要求映射表行列 < 32767，按 1024 列折叠
        pad = (-k) % w
        col = np.pad(col, (0, pad)).reshape(-1, w)
        row = np.pad(row, (0, pad)).reshape(-1, w)
        out = cv2.remap(self.dem, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return out.ravel()[:k].reshape(shape)

    def mean_height(self, n, e, radius):
        r = int(radius / self.gsd)
        row, col = int(-n / self.gsd), int(e / self.gsd)
        sub = self.dem[max(0, row - r):row + r, max(0, col - r):col + r]
        return float(sub.mean()) if sub.size else float(self.dem.mean())


def ray_ground(pose, rays_c, terrain=None, flat_h=None, iters=6):
    """相机射线与地面求交，返回地面点 (n, e, h)。terrain 为 None 时与高度 flat_h 的水平面求交。"""
    d = rays_c @ pose.C_nc().T  # NED 方向
    dd = np.maximum(d[..., 2], 1e-3)
    z = np.full(d.shape[:-1], flat_h if flat_h is not None else terrain.dem.mean(), np.float32)
    for _ in range(iters if terrain is not None else 1):
        t = (pose.h - z) / dd
        n = pose.n + t * d[..., 0]
        e = pose.e + t * d[..., 1]
        if terrain is not None:
            z = terrain.height(n, e)
    return n, e, z


def project(pose, cam, n, e, h):
    """世界点投影到像素坐标 (u, v)；位于相机后方的点返回 NaN。"""
    p = np.stack([n - pose.n, e - pose.e, -(h - pose.h)], -1) @ pose.C_nc()  # 相机系
    z = p[..., 2]
    ok = z > 1.0
    u = np.where(ok, cam.f * p[..., 0] / np.where(ok, z, 1) + cam.width / 2, np.nan)
    v = np.where(ok, cam.f * p[..., 1] / np.where(ok, z, 1) + cam.height / 2, np.nan)
    return u, v


def render(src, src_gsd, terrain, pose, cam, rng=None, noise=2.0, gain=(0.85, 1.15), blur_sigma=0.5):
    """以 pose 从正射影像 src（地图网格）渲染下视相机图像（含地形视差）。

    返回 (uint8 图像, 有效掩膜)。rng 不为 None 时加入增益/偏置/噪声，模拟相机响应差异。
    """
    rays = cam.pixel_rays()
    n, e, _ = ray_ground(pose, rays, terrain)
    col = (e / src_gsd).astype(np.float32)
    row = (-n / src_gsd).astype(np.float32)
    s = cv2.GaussianBlur(src, (0, 0), blur_sigma) if blur_sigma > 0 else src
    img = cv2.remap(s, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.remap((src > 0).astype(np.uint8), col, row, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
    if rng is not None:
        g = rng.uniform(*gain)
        img = img.astype(np.float32) * g + rng.normal(0, 5) + rng.normal(0, noise, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
    img[~valid] = 0
    return img, valid
