"""数据管道。

两条铁律在工程上落实：
1. **目标域 DataLoader 不返回标签字段**（而非返回后"自律不用"，修 A3）；
2. 训练与评估走**同一条几何管线**（letterbox），修 C1。

域编号约定：0=BUSI, 1=BUSBRA, 2=BUS-UCLM（无标签目标域）。
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..data.preprocess import (PreprocConfig, letterbox, preprocess_image,
                               preprocess_mask, tone_normalize)
from .augment import (rendering_randomize, scale_randomize,
                      wavelet_amplitude_jitter)

DOMAIN_ID = {"busi": 0, "busbra": 1, "busuclm": 2}
EXTERNAL_DOMAIN_ID = -1          # 外部队列（BrEaST / OASBUD）不参与域判别


class BreastDataset(Dataset):
    """image, label, domain, index。target 模式下 label 返回 -1。"""

    def __init__(self, entries: list[dict], cfg: PreprocConfig = PreprocConfig(),
                 train: bool = False, with_mask: bool = False, with_label: bool = True,
                 render_aug: bool = False, wavelet_jitter: bool = False,
                 scale_aug: bool = False):
        self.entries = entries
        self.cfg = cfg
        self.train = train
        self.with_mask = with_mask
        self.with_label = with_label
        self.render_aug = render_aug and train
        self.wavelet_jitter = wavelet_jitter and train
        self.scale_aug = scale_aug and train
        self.aug = _build_aug() if train else None
        self._seed_base = 0

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        """**固定位置**五元组，避免掩膜/标签开关移动字段位置：

            (x, y, domain, index, mask)

        with_label=False 时 y = -1（训练循环只对源域用 y，目标域永不读取）；
        with_mask=False 时 mask 为全零，仅占位。
        """
        e = self.entries[i]
        # 几何走 preprocess 的统一入口（P4；旧 letterbox 仅作附录消融臂）
        # ratio 由条目携带（仅 OASBUD 非 1，见 data/registry 与 oasbud_rf）
        ratio = float(e.get("pixel_ratio", 1.0))
        x_np = preprocess_image(e["image"], self.cfg, as_rgb=False, ratio=ratio)
        im = Image.fromarray(x_np)
        # 渲染/尺度随机化（几何之后施加；参数范围 a priori 固定）
        if self.render_aug or self.wavelet_jitter or self.scale_aug:
            rng = np.random.default_rng((self._seed_base + i) & 0x7FFFFFFF)
            g = np.asarray(im)
            if self.render_aug:
                g = rendering_randomize(g, rng)
            if self.scale_aug:
                g = scale_randomize(g, rng)
            if self.wavelet_jitter:
                g = wavelet_amplitude_jitter(g, rng)
            im = Image.fromarray(g)
        im = im.convert("RGB")
        if self.aug is not None:
            im = self.aug(im)
        # 只传 uint8（通道优先），归一化搬到最后在 GPU 上做：
        # float32 传输的共享内存带宽是 uint8 的 4 倍，Windows 上这项开销显著。
        arr = np.asarray(im)
        x = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()

        y = torch.tensor(int(e["label"]) if self.with_label else -1, dtype=torch.long)
        # 外部队列（BrEaST/OASBUD）不在训练域内，给 -1；它们只用于评估，
        # 域判别头不会见到（训练 loader 只装 busi/busbra/busuclm）
        dom = torch.tensor(DOMAIN_ID.get(e["dataset"], EXTERNAL_DOMAIN_ID),
                           dtype=torch.long)
        idx = torch.tensor(i, dtype=torch.long)

        m = np.zeros((self.cfg.size, self.cfg.size), dtype=np.uint8)
        if self.with_mask:
            acc = np.zeros((self.cfg.size, self.cfg.size), dtype=bool)
            for mp in e.get("masks") or []:
                acc |= preprocess_mask(mp, self.cfg.size, self.cfg.geometry, ratio,
                                       image_path=e["image"])
            m = acc.astype(np.uint8)
        return x, y, dom, idx, torch.from_numpy(m)


def _build_aug():
    """轻量增广。渲染随机化在 batch 层做（见 augment.py），此处只做几何与噪声。"""
    import torchvision.transforms as T
    return T.Compose([
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.RandomHorizontalFlip(),
        # 垂直翻转违反超声深度轴物理，禁止
    ])


def make_loader(entries: list[dict], *, cfg=PreprocConfig(), batch_size=32,
                train=False, with_mask=False, with_label=True, num_workers=8,
                shuffle=None, seed=0, drop_last=False,
                render_aug=False, wavelet_jitter=False, scale_aug=False) -> DataLoader:
    ds = BreastDataset(entries, cfg, train=train, with_mask=with_mask,
                       with_label=with_label, render_aug=render_aug,
                       wavelet_jitter=wavelet_jitter, scale_aug=scale_aug)
    if shuffle is None:
        shuffle = train
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True, drop_last=drop_last, persistent_workers=num_workers > 0,
                      generator=g)
