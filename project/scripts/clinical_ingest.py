#!/usr/bin/env python3
"""临床队列**接入**：把外部临床数据规范成与 BrEaST 同构的 manifest。

⚠ 本脚本属于「临床数据附加层」，为论文**完整性服务**，不是核心结论的一部分。
   完全独立：**不修改任何既有文件**。删除清单见
   `env/clinical_layer_files.txt`，删掉即可还原成"只有公开队列"的状态。

数据放哪
--------
**仓库外**（关键）：

    C:\\Users\\15315\\Desktop\\paper1st\\data\\_clinical\\<name>\\

为什么放仓库外：`config.DATA_ROOT` = `paper1st/data`，是 `meliorate/` 的**同级**
目录，**不在**将要上传 git 的范围内（git 根是 `meliorate/project`）。
即：数据**物理上**不可能混进提交，不依赖 .gitignore 的正确性。
（同 `config.py` 里"把 runs 放数据盘以保证镜像不含结果"的同一思路。）

本脚本只**读**原始数据、只**写**到上面那个目录，不碰 `runs/` 也不碰代码。

两种输入形态（任选）
--------------------
A. **已有清单**（推荐，临床数据通常自带表）：

     python scripts/clinical_ingest.py --name hospA --manifest 原始表.csv \
            --src 图像目录 --col-image 文件名 --col-label 病理

   CSV 需能给出：文件名、标签。`case_id` 可选（缺则用文件名主干）。
   `label` 支持 0/1，也支持文本（自动识别 benign/malignant/良/恶）。

B. **目录即标签**（数据已按类分文件夹）：

     python scripts/clinical_ingest.py --name hospA --src 数据根 --layout dirs

   约定 `<src>/benign*/` 与 `<src>/malignant*/`（大小写不敏感，支持中文
   `良性/恶性`）。每个文件一张图。

输出（与 BrEaST 同 schema，故下游零改动可读）
--------------------------------------------
    data/_clinical/<name>/manifest.csv     case_id,image,label,lesion_id,birads,view
    data/_clinical/<name>/images/          （**软链/复制**原图）

⚠ **cluster bootstrap 的分组单位是 lesion_id**（F23 同源）。
   临床数据若"一例多图"，必须给出 `case_id` 让同病例的多张图归为一簇；
   否则同一病灶的多视图会被当成独立样本，CI 会被**低估**（假精确）。
   本脚本据此设 `lesion_id = case_id`（缺 case_id 时才退化为按图）。

用法示例（干跑：只报告，不写盘）
-------------------------------
    python scripts/clinical_ingest.py --name hospA --src /path --layout dirs --dry-run
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")     # 入口自护（preflight 坑1）

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from breastdg import config                   # noqa: E402

# 临床数据一律放仓库**外**的 _clinical/ 下（见 docstring）
CLINICAL_ROOT = config.DATA_ROOT / "_clinical"

_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# 标签文本 → 0/1。中文与常见英文写法都收。
_POS = re.compile(r"malign|恶性|癌|cancer|carcinoma|^1$", re.I)
_NEG = re.compile(r"benign|良性|^0$", re.I)


def parse_label(v: str) -> int | None:
    """把标签字段解析成 0/1；无法判定返回 None（**不猜**）。"""
    s = str(v).strip()
    if _POS.search(s) and not _NEG.search(s):
        return 1
    if _NEG.search(s):
        return 0
    return None


def scan_dirs(src: Path) -> list[dict]:
    """形态 B：子目录名即标签。`<src>/benign*/` 与 `<src>/malignant*/`。"""
    out = []
    subs = [d for d in sorted(src.iterdir()) if d.is_dir()]
    if not subs:
        raise SystemExit(f"  ✗ {src} 下没有子目录，无法按目录判标签")
    hit = False
    for d in subs:
        lab = parse_label(d.name)
        if lab is None:
            print(f"  [skip] 目录名无法判标签: {d.name}")
            continue
        hit = True
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.suffix.lower() in _IMG_EXT:
                out.append({"case_id": f.stem, "image_src": f, "label": lab,
                            "birads": "", "view": ""})
    if not hit:
        raise SystemExit(
            f"  ✗ {src} 下没有可识别为良/恶的子目录。\n"
            f"     看到: {[d.name for d in subs]}\n"
            f"     命名须含 benign/malignant（或 良性/恶性）")
    return out


def scan_manifest(src: Path, man: Path, col_image: str, col_label: str,
                  col_case: str | None, col_birads: str | None) -> list[dict]:
    """形态 A：读既有 CSV。只认**用户显式指定的列名**，不猜列。"""
    rows = list(csv.DictReader(open(man, encoding="utf-8-sig")))
    if not rows:
        raise SystemExit(f"  ✗ {man} 无数据行")
    for c in (col_image, col_label):
        if c not in rows[0]:
            raise SystemExit(
                f"  ✗ 列名 {c!r} 不在表里。实际列: {list(rows[0])}\n"
                f"     用 --col-image/--col-label 指定正确列名")
    out, bad = [], 0
    for r in rows:
        lab = parse_label(r[col_label])
        if lab is None:
            bad += 1
            continue
        img = src / str(r[col_image]).strip()
        if not img.exists():                      # 允许表里已是相对子路径
            cand = list(src.rglob(Path(str(r[col_image])).name))
            if len(cand) == 1:
                img = cand[0]
            else:
                print(f"  [skip] 找不到图: {r[col_image]}"
                      f"{'（同名多份，歧义）' if cand else ''}")
                continue
        case = (str(r[col_case]).strip() if col_case and r.get(col_case)
                else img.stem)
        out.append({"case_id": case, "image_src": img, "label": lab,
                    "birads": (r.get(col_birads, "") if col_birads else ""),
                    "view": ""})
    if bad:
        print(f"  ⚠ {bad} 行的标签无法解析，已跳过（未猜）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="队列名（如 hospA）")
    ap.add_argument("--src", required=True, help="原始图像根目录")
    ap.add_argument("--layout", default="dirs", choices=["dirs", "manifest"])
    ap.add_argument("--manifest", default=None, help="layout=manifest 时的 CSV")
    ap.add_argument("--col-image", default="image")
    ap.add_argument("--col-label", default="label")
    ap.add_argument("--col-case", default=None)
    ap.add_argument("--col-birads", default=None)
    ap.add_argument("--link", default="copy", choices=["copy", "hardlink"],
                    help="默认 copy（最稳）；hardlink 省空间但跨盘不行")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = Path(a.src).resolve()
    if not src.exists():
        raise SystemExit(f"  ✗ 源目录不存在: {src}")
    dst = CLINICAL_ROOT / a.name
    print(f"  源  {src}")
    print(f"  目标 {dst}")

    if a.layout == "dirs":
        recs = scan_dirs(src)
    else:
        if not a.manifest:
            raise SystemExit("  ✗ layout=manifest 须给 --manifest")
        recs = scan_manifest(src, Path(a.manifest), a.col_image, a.col_label,
                             a.col_case, a.col_birads)

    n_pos = sum(r["label"] for r in recs)
    n_case = len({r["case_id"] for r in recs})
    print(f"\n  图 {len(recs)}   恶性 {n_pos}   良性 {len(recs)-n_pos}   "
          f"case_id 数 {n_case}")
    if n_case < len(recs):
        print(f"  → 一例多图：{len(recs)}/{n_case} = "
              f"{len(recs)/max(n_case,1):.2f} 图/例（簇键=case_id，故 CI 正确）")
    else:
        print("  → 一图一例（或未给 case_id）：簇键退化为按图")
    if not recs:
        raise SystemExit("  ✗ 没有可用记录")
    if a.dry_run:
        print("\n  --dry-run：未写盘。")
        return 0

    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"  ✗ {dst} 已存在且非空。要覆盖请先手动删除"
                         f"（**不自动覆盖**，避免误删已整理的数据）")
    (dst / "images").mkdir(parents=True, exist_ok=True)
    man_rows = []
    for i, r in enumerate(recs):
        ext = r["image_src"].suffix.lower()
        fn = f"{r['case_id']}_{i:05d}{ext}"       # 唯一化，防同名打架
        target = dst / "images" / fn
        if a.link == "hardlink":
            try:
                target.hardlink_to(r["image_src"])
            except OSError:
                shutil.copy2(r["image_src"], target)
        else:
            shutil.copy2(r["image_src"], target)
        man_rows.append({"case_id": r["case_id"], "image": fn,
                         "label": r["label"], "lesion_id": r["case_id"],
                         "birads": r["birads"], "view": r["view"]})

    with open(dst / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "image", "label",
                                          "lesion_id", "birads", "view"])
        w.writeheader()
        w.writerows(man_rows)
    with open(dst / "SOURCE.md", "w", encoding="utf-8") as f:
        f.write(f"# 临床队列 `{a.name}`（附加层，非核心结论）\n\n"
                f"- 来源目录：`{src}`（**本机路径，不入 git**）\n"
                f"- 接入方式：layout={a.layout}"
                f"{'  manifest=' + str(a.manifest) if a.manifest else ''}\n"
                f"- 图 {len(recs)}（恶性 {n_pos} / 良性 {len(recs)-n_pos}），"
                f"case 数 {n_case}\n"
                f"- 簇键 `lesion_id = case_id`（F23 同源；一例多图必须如此）\n"
                f"- ⚠ 本队列为论文**完整性**服务，**不进入**锁箱主张\n")
    print(f"\n  已写入 {dst/'manifest.csv'}")
    print(f"  已写入 {dst/'SOURCE.md'}")
    print(f"  图像 {len(man_rows)} 张 → {dst/'images'}/")
    print(f"\n  下一步（评估）：\n"
          f"    python scripts/clinical_eval.py --name {a.name}")
    return 0


if __name__ == "__main__":      # Windows spawn 守卫（preflight 坑3）
    sys.exit(main())
