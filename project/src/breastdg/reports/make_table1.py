"""make_table1.py —— 从原始数据重算全部账目，论文数字的唯一来源。

用法：
    python -m breastdg.reports.make_table1            # 控制台输出
    python -m breastdg.reports.make_table1 --json     # 同时写 reports/table1.json

纪律：论文中出现的任何数据集样本量必须来自本脚本输出，不得手工填写。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from .. import config
from ..data import registry


def _summarize(entries: list[dict]) -> dict:
    labels = collections.Counter(e["label"] for e in entries)
    patients = {e["patient_id"] for e in entries}
    lesions = {e["lesion_id"] for e in entries}
    with_mask = sum(1 for e in entries if e["masks"])
    return dict(
        n_images=len(entries),
        n_benign=labels.get(0, 0),
        n_malignant=labels.get(1, 0),
        n_other=int(labels.get(None, 0)),
        n_patients=len(patients),
        n_lesions=len(lesions),
        n_with_mask=with_mask,
        images_per_patient=round(len(entries) / max(len(patients), 1), 3),
    )


def find_duplicates(entries: list[dict]) -> list[list[str]]:
    """近重复检测（感知哈希）。无 imagehash 时返回空表。"""
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        return []
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for e in entries:
        try:
            with Image.open(e["image"]) as im:
                h = imagehash.phash(im.convert("L"))
        except Exception:
            continue
        groups[str(h)].append(e["image"])
    return [g for g in groups.values() if len(g) > 1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="写入 reports/table1.json")
    ap.add_argument("--dupes", action="store_true", help="运行感知哈希近重复检测")
    args = ap.parse_args(argv)

    print("=" * 74)
    print("Table 1 —— 数据集账目（由原始数据重算）")
    print("=" * 74)

    report: dict = {}

    # ---- 训练用公开数据集 ----
    for name, fn in registry.ALL_PUBLIC.items():
        entries = fn()
        if not entries:
            print(f"[skip] {name}: 未找到数据")
            continue
        s = _summarize(entries)
        report[name] = s
        print(f"\n{name:10s} 图像 {s['n_images']:5d} "
              f"(良 {s['n_benign']:4d} : 恶 {s['n_malignant']:4d})  "
              f"病例 {s['n_patients']:4d}  病灶 {s['n_lesions']:4d}  "
              f"含掩膜 {s['n_with_mask']:4d}  图/病例 {s['images_per_patient']}")
        if args.dupes:
            d = find_duplicates(entries)
            print(f"{'':10s} 近重复组: {len(d)}")
            for g in d[:10]:
                print("           ", [Path(x).name for x in g])
            report[name]["duplicate_groups"] = [[Path(x).name for x in g] for g in d]

    # ---- OASBUD：病灶级，200 张等于 100 病灶 × 2 切面 ----
    les = registry.oasbud_lesions()
    if les:
        lab = collections.Counter(x["label"] for x in les)
        n_views = sum(x["n_views"] for x in les)
        report["oasbud"] = dict(n_lesions=len(les), n_views=n_views,
                                n_benign=lab.get(0, 0), n_malignant=lab.get(1, 0))
        print(f"\n{'oasbud':10s} 病灶 {len(les):5d} (良 {lab.get(0,0)} : 恶 {lab.get(1,0)})  "
              f"→ 重建图像 {n_views} 张 (每病灶 2 个 RF 切面)")

    # ---- 正常类（被任务排除，须在论文中说明） ----
    normal = registry.busi_entries(include_normal=True)
    n_normal = sum(1 for e in normal if e["label"] is None)
    all_u = registry.busuclm_entries()
    print(f"\n[说明] 已排除的无病灶图像：BUSI normal {n_normal} 张；"
          f"BUS-UCLM normal（未进入本次登记，见 config.BUSUCLM_USE_LESIONS_ONLY）")
    report["excluded"] = {"busi_normal": n_normal}

    print("\n" + "=" * 74)
    if args.json:
        out = config.REPORTS_DIR / "table1.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
