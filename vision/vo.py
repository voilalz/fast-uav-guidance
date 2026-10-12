"""下视单目视觉里程计（地面近似平面 + 外部尺度），输出机体系速度量测，对应协议 VO_VELOCITY。

算法（与 docs/software-design.md §4.2 一致）：
  1. 金字塔 LK 光流跟踪 + 前后向一致性检查，按网格补充特征点；
  2. 用导航给出的两帧姿态把像点射线转到 NED（去除旋转）；
  3. 用尺度来源给出的“相机到地面的距离”把射线落到地面，同一地面点在两帧的
     相对位置之差即相机位移；多点取鲁棒中值（MAD 剔除外点）；
  4. 位移 / Δt → NED 速度 → 机体系速度，协方差由内点离散度与尺度不确定度给出。

尺度来源（scale_mode）：
  "radar"     ：雷达高度表测得的天底点离地高（林区/山地会因地形起伏引入尺度误差）
  "dem_mean"  ：气压海拔 − 足迹范围内 DEM 平均高程（docs 推荐的林区方案）
  "dem_ray"   ：逐点用先验位姿与 DEM 做射线求交得到各点深度（不再假设地面平坦）
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import euler_to_dcm, ray_ground


@dataclass
class VoResult:
    ok: bool
    vb: np.ndarray        # 机体系速度 m/s
    vn: np.ndarray        # NED 速度 m/s
    sigma: float          # 每轴 1σ m/s
    inliers: int


class DownwardVo:
    def __init__(self, cam, max_pts=400, min_pts=40, grid=8, fb_thresh=1.0):
        self.cam, self.max_pts, self.min_pts, self.grid, self.fb = cam, max_pts, min_pts, grid, fb_thresh
        self.prev = None
        self.pts = None
        self.lk = dict(winSize=(21, 21), maxLevel=4,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def _detect(self, img, mask=None):
        h, w = img.shape
        per = max(1, self.max_pts // (self.grid * self.grid))
        out = []
        for i in range(self.grid):
            for j in range(self.grid):
                r0, r1, c0, c1 = i * h // self.grid, (i + 1) * h // self.grid, j * w // self.grid, (j + 1) * w // self.grid
                m = None if mask is None else (mask[r0:r1, c0:c1].astype(np.uint8) * 255)
                p = cv2.goodFeaturesToTrack(img[r0:r1, c0:c1], per, 0.01, 8, mask=m)
                if p is not None:
                    out.append(p.reshape(-1, 2) + [c0, r0])
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 2), np.float32)

    def _rays(self, px):
        return np.stack([(px[:, 0] - self.cam.width / 2) / self.cam.f,
                         (px[:, 1] - self.cam.height / 2) / self.cam.f, np.ones(len(px))], -1)

    def _ground(self, pose, rays, scale_mode, agl, terrain):
        """射线 -> 相对相机的地面点 NED 偏移。"""
        d = rays @ pose.C_nc().T
        if scale_mode == "dem_ray":
            n, e, z = ray_ground(pose, rays, terrain)
            return np.stack([n - pose.n, e - pose.e, pose.h - z], -1)
        t = agl / np.maximum(d[:, 2], 1e-3)
        return d * t[:, None]

    def process(self, img, pose_est, dt, agl=None, terrain=None, scale_mode="radar", valid=None):
        """pose_est 为导航给出的当前帧位姿（姿态用于去旋转；dem_ray 模式还用其位置与高度）。"""
        if self.prev is None:
            self.prev = (img, pose_est, agl)
            self.pts = self._detect(img, valid)
            return VoResult(False, np.zeros(3), np.zeros(3), np.inf, 0)
        img0, pose0, agl0 = self.prev
        res = VoResult(False, np.zeros(3), np.zeros(3), np.inf, 0)
        if len(self.pts) >= self.min_pts:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(img0, img, self.pts, None, **self.lk)
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(img, img0, p1, None, **self.lk)
            good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(p0r - self.pts, axis=1) < self.fb)
            a, b = self.pts[good], p1[good]
            if len(a) >= self.min_pts:
                g0 = self._ground(pose0, self._rays(a), scale_mode, agl0, terrain)
                # 当前帧：dem_ray 用 pose0 + 预测位移近似当前位置做射线求交
                g1 = self._ground(pose_est, self._rays(b), scale_mode, agl, terrain)
                disp = g0 - g1  # 相机位移（NED），地面点固定
                med = np.median(disp, axis=0)
                mad = np.median(np.abs(disp - med), axis=0) * 1.4826 + 1e-3
                inl = np.all(np.abs(disp - med) < 3 * mad + 0.05, axis=1)
                if inl.sum() >= self.min_pts // 2:
                    d = disp[inl].mean(axis=0)
                    vn = d / dt
                    C_nb = euler_to_dcm(pose_est.roll, pose_est.pitch, pose_est.yaw)
                    vb = C_nb.T @ vn
                    # 噪声：点位离散度 / sqrt(N) + 尺度来源不确定度（按 1% 计）
                    spread = np.sqrt(np.mean(np.var(disp[inl], axis=0)) / inl.sum()) / dt
                    sigma = float(np.hypot(spread, 0.01 * np.linalg.norm(vn)))
                    res = VoResult(True, vb, vn, sigma, int(inl.sum()))
            self.pts = b if len(b) >= self.max_pts // 2 else np.concatenate([b, self._detect(img, valid)])[:self.max_pts]
        else:
            self.pts = self._detect(img, valid)
        self.prev = (img, pose_est, agl)
        return res
