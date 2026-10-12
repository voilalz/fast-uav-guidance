"""景象匹配：实时图正射校正 → 粗匹配（NCC 搜索）→ 精匹配（局部特征 + RANSAC 相似变换）→ 质量评估。
输出绝对位置（及航向）量测与 1σ，对应协议 POSITION_FIX（docs/software-design.md §4.3）。

参考库为 1 m 北向朝上正射影像（docs §4.7），本模块直接在地图网格上取参考区域。
精匹配默认用 SIFT（OpenCV 内置、无需网络权重）；接口与 SuperPoint+LightGlue 等学习型匹配器一致，
替换 FineMatcher 即可。
"""
from dataclasses import dataclass, field
import time

import cv2
import numpy as np

from .camera import project


@dataclass
class MatchConfig:
    ortho_gsd: float = 1.0       # 正射校正与参考图分辨率 m
    template_m: float = 320.0    # 模板边长（实时图正射后中心区域）
    use_dem: bool = True         # 正射校正是否使用 DEM（False 为平地假设）
    preprocess: str = "clahe"    # "clahe" | "grad"：跨时相用梯度幅值更稳健
    exclusion_m: float = 25.0    # 次峰排除半径
    fine: bool = True
    min_inliers: int = 12
    min_ambiguity: float = 1.02  # 主峰 / 次峰（NCC 映射到正数后之比）
    min_psr: float = 4.0
    min_valid: float = 0.6       # 模板内有效像素比例下限（足迹外/无数据区域由掩膜排除）


@dataclass
class MatchResult:
    ok: bool
    n: float = np.nan            # 匹配得到的相机水平位置（NED，m）
    e: float = np.nan
    dyaw: float = np.nan         # 航向修正 rad（真值 ≈ 估计 + dyaw）
    sigma: float = np.inf        # 水平 1σ m
    ncc: float = 0.0
    psr: float = 0.0
    ambiguity: float = 0.0
    inliers: int = 0
    stage: str = ""
    ms: float = 0.0
    extra: dict = field(default_factory=dict)


def _prep(img, mode, mask=None):
    if mode == "grad":
        f = cv2.GaussianBlur(img, (0, 0), 1.0).astype(np.float32)
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1)
        out = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.0)
    else:
        out = cv2.createCLAHE(2.0, (8, 8)).apply(img).astype(np.float32)
    if mask is not None:
        out[~mask] = 0
    return out


def orthorectify(img, valid, pose_est, cam, terrain, cfg, size_m):
    """以估计位姿把实时图投影到以估计天底点为中心、北向朝上、ortho_gsd 分辨率的地面网格。

    返回 (正射图, 有效掩膜, 网格中心 (n, e))。中心取相机正下方，因此匹配得到的偏移直接是水平位置修正。
    """
    k = int(size_m / cfg.ortho_gsd)
    off = (np.arange(k) - k / 2 + 0.5) * cfg.ortho_gsd
    ee, nn = np.meshgrid(pose_est.e + off, pose_est.n - off)
    hh = terrain.height(nn, ee) if cfg.use_dem else np.full_like(nn, terrain.mean_height(pose_est.n, pose_est.e, 300))
    u, v = project(pose_est, cam, nn, ee, hh)
    u = np.nan_to_num(u, nan=-1).astype(np.float32)
    v = np.nan_to_num(v, nan=-1).astype(np.float32)
    o = cv2.remap(img, u, v, cv2.INTER_AREA if cfg.ortho_gsd > 0 else cv2.INTER_LINEAR, borderValue=0)
    m = cv2.remap(valid.astype(np.uint8), u, v, cv2.INTER_NEAREST, borderValue=0) > 0
    return o, m, (pose_est.n, pose_est.e)


class SceneMatcher:
    def __init__(self, ref_map, ref_gsd, cam, terrain, cfg=MatchConfig()):
        self.cfg, self.cam, self.terrain = cfg, cam, terrain
        f = ref_gsd / cfg.ortho_gsd
        self.ref = cv2.resize(ref_map, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f != 1 else ref_map
        self.sift = cv2.SIFT_create(nfeatures=1500)

    def _ref_crop(self, n, e, half_m):
        g = self.cfg.ortho_gsd
        r0, c0 = int(round((-n - half_m) / g)), int(round((e - half_m) / g))
        k = int(2 * half_m / g)
        H, W = self.ref.shape
        out = np.zeros((k, k), np.uint8)
        rr0, cc0, rr1, cc1 = max(0, r0), max(0, c0), min(H, r0 + k), min(W, c0 + k)
        if rr1 > rr0 and cc1 > cc0:
            out[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = self.ref[rr0:rr1, cc0:cc1]
        return out, r0, c0

    def match(self, img, valid, pose_est, prior_sigma):
        t0 = time.perf_counter()
        cfg, g = self.cfg, self.cfg.ortho_gsd
        foot = 1.5 * cfg.template_m
        ortho, om, (n0, e0) = orthorectify(img, valid, pose_est, self.cam, self.terrain, cfg, foot)
        k = int(cfg.template_m / g)
        s = (ortho.shape[0] - k) // 2
        tpl, tm = ortho[s:s + k, s:s + k], om[s:s + k, s:s + k]
        res = MatchResult(False)
        if tm.mean() < cfg.min_valid or tpl[tm].std() < 8:  # 视场被遮挡/无数据或纹理过弱（水面、云）
            res.stage, res.ms = "low_texture", (time.perf_counter() - t0) * 1e3
            return res
        search = 3 * prior_sigma + 20
        half = cfg.template_m / 2 + search
        ref, r0, c0 = self._ref_crop(n0, e0, half)
        if (ref == 0).mean() > 0.6:
            res.stage, res.ms = "no_reference", (time.perf_counter() - t0) * 1e3
            return res
        A, T = _prep(ref, cfg.preprocess), _prep(tpl, cfg.preprocess, tm)
        corr = cv2.matchTemplate(A, T, cv2.TM_CCOEFF_NORMED, mask=tm.astype(np.uint8))
        corr = np.nan_to_num(corr, nan=-1, posinf=-1, neginf=-1)
        _, c1, _, p1 = cv2.minMaxLoc(corr)
        ex = max(1, int(cfg.exclusion_m / g))
        cm = corr.copy()
        cm[max(0, p1[1] - ex):p1[1] + ex + 1, max(0, p1[0] - ex):p1[0] + ex + 1] = -1
        c2 = cm.max()
        side = corr[cm > -1]
        psr = (c1 - side.mean()) / (side.std() + 1e-6)
        amb = (c1 + 1) / (c2 + 1)  # 映射到正数后之比，> 1
        # 抛物线拟合亚像素峰值
        sx = sy = 0.0
        if 0 < p1[0] < corr.shape[1] - 1:
            l, c, r_ = corr[p1[1], p1[0] - 1], corr[p1[1], p1[0]], corr[p1[1], p1[0] + 1]
            sx = 0.5 * (l - r_) / (l - 2 * c + r_ - 1e-9)
        if 0 < p1[1] < corr.shape[0] - 1:
            u_, c, d_ = corr[p1[1] - 1, p1[0]], corr[p1[1], p1[0]], corr[p1[1] + 1, p1[0]]
            sy = 0.5 * (u_ - d_) / (u_ - 2 * c + d_ - 1e-9)
        # 粗匹配：模板左上角在参考裁剪中的位置 -> 模板中心的地面坐标
        cen_r = r0 + p1[1] + float(np.clip(sy, -0.5, 0.5)) + k / 2
        cen_c = c0 + p1[0] + float(np.clip(sx, -0.5, 0.5)) + k / 2
        n_c, e_c = -(cen_r - 0.5) * g, (cen_c - 0.5) * g  # OpenCV 像素中心约定
        # 模板中心在正射图中对应 (n0, e0)，因此相机水平位置修正 = 匹配中心 − (n0, e0)
        dn, de = n_c - n0, e_c - e0
        res.ncc, res.psr, res.ambiguity, res.stage = float(c1), float(psr), float(amb), "coarse"
        dyaw = np.nan
        inl = 0
        if cfg.fine:
            fine = self._fine(ortho, om, ref, r0, c0, n0, e0, dn, de)
            if fine is not None and fine[3] >= cfg.min_inliers and np.hypot(fine[0] - dn, fine[1] - de) < 3 * cfg.exclusion_m:
                dn, de, dyaw, inl = fine
                res.stage = "fine"
        res.inliers = int(inl)
        ok = res.psr >= cfg.min_psr and res.ambiguity >= cfg.min_ambiguity
        if ok:
            res.ok = True
            res.n, res.e, res.dyaw = pose_est.n + dn, pose_est.e + de, dyaw
            res.sigma = grade(res, g)
        res.ms = (time.perf_counter() - t0) * 1e3
        return res

    def _fine(self, ortho, om, ref, r0, c0, n0, e0, dn, de):
        """在粗匹配位置附近用 SIFT + RANSAC 相似变换精化，返回 (dn, de, dyaw, inliers)。"""
        g = self.cfg.ortho_gsd
        k1, d1 = self.sift.detectAndCompute(cv2.createCLAHE(2.0, (8, 8)).apply(ortho), om.astype(np.uint8) * 255)
        k2, d2 = self.sift.detectAndCompute(cv2.createCLAHE(2.0, (8, 8)).apply(ref), (ref > 0).astype(np.uint8) * 255)
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            return None
        m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
        good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.8 * b.distance]
        if len(good) < 8:
            return None
        P = np.float32([k1[x.queryIdx].pt for x in good])
        Q = np.float32([k2[x.trainIdx].pt for x in good])
        # 粗匹配先验：只保留与粗位移一致的匹配（±2 × 排除半径）
        oc = ortho.shape[0] / 2 - 0.5  # 正射图中心（= 估计天底点）的像素坐标
        exp_q = P - oc + np.float32([(e0 + de) / g - c0, -(n0 + dn) / g - r0])
        near = np.linalg.norm(Q - exp_q, axis=1) < 2 * self.cfg.exclusion_m / g
        if near.sum() < 8:
            return None
        M, inl = cv2.estimateAffinePartial2D(P[near], Q[near], method=cv2.RANSAC, ransacReprojThreshold=2.0,
                                             maxIters=2000, confidence=0.999)
        if M is None:
            return None
        ninl = int(inl.sum())
        q = M @ np.array([oc, oc, 1.0])
        n_c = -(r0 + q[1]) * g
        e_c = (c0 + q[0]) * g
        # 正射图北向朝上；图像中顺时针旋转 a 对应航向需增加 a
        a = np.arctan2(M[1, 0], M[0, 0])
        return n_c - n0, e_c - e0, float(a), ninl


def grade(r, gsd):
    """量测质量 -> 1σ，与 nav/apps/companion_main.cpp 中 gradeMatch 的形式一致（参数按跨时相评估结果整定）。"""
    if r.stage == "fine":
        return float(max(2.0 * gsd * (1.0 + 30.0 / max(r.inliers, 1)), 2.0))
    # 粗匹配（NCC）：按跨时相评估整定，PSR 越低越不可靠；含参考库配准偏差
    return float(max(gsd * (3.0 + 20.0 / max(r.psr, 1.0)), 3.0))
