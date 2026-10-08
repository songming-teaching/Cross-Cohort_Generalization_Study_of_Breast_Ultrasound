"""核验 check1 第③条：`ALL_2nd` 里含 scale（采集协议线索），
故它不是"纯二阶纹理"。本脚本算出**纯纹理子集**（speckle/snr/entropy/grad）
的域可分性，与含 scale 的 ALL_2nd 对照。
复用 diagnose_trivial_cues_tier2.cues2，口径完全一致。
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
from breastdg.data import registry
from breastdg.data.dedup import dedupe_entries
from breastdg.data.preprocess import PreprocConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

spec = importlib.util.spec_from_file_location("t2", "scripts/diagnose_trivial_cues_tier2.py")
t2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t2)

def cv(X, y, g, seed=0):
    sk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    a = []
    for tr, te in sk.split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        if len(set(y[te].tolist())) > 1:
            a.append(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te]))))
    return float(np.mean(a))

cfg = PreprocConfig(size=256, geometry="p4")
busi, _ = dedupe_entries(registry.busi_entries())
busbra = registry.busbra_entries()
busuclm = registry.busuclm_entries()
C = {}
for nm, ents in (("BUSI", busi), ("BUSBRA", busbra), ("BUS-UCLM", busuclm)):
    C[nm] = np.array([t2.cues2(e, cfg) for e in ents], float)
    print(f"  {nm} n={len(ents)}", flush=True)

N = t2.NAMES
tex = [N.index(k) for k in ("speckle", "snr", "entropy", "grad")]
print()
print("=" * 70)
for pair, other in (("BUSI_vs_BUS-UCLM", "BUS-UCLM"), ("BUSI_vs_BUSBRA", "BUSBRA")):
    y = np.r_[np.zeros(C["BUSI"].shape[0], int), np.ones(C[other].shape[0], int)]
    g = np.array([e["patient_id"] for e in busi] + [e["patient_id"] for e in
                 (busbra if other == "BUSBRA" else busuclm)])
    Xall = np.concatenate([C["BUSI"], C[other]], 0)
    Xt = np.concatenate([C["BUSI"][:, tex], C[other][:, tex]], 0)
    Xs = np.concatenate([C["BUSI"][:, N.index("scale")][:, None],
                         C[other][:, N.index("scale")][:, None]], 0)
    print(f"[{pair}]")
    print(f"  纯 scale 单 cue           = {cv(Xs, y, g):.4f}")
    print(f"  ALL_2nd (scale+4纹理)     = {cv(Xall, y, g):.4f}")
    print(f"  纯二阶纹理(去 scale)      = {cv(Xt, y, g):.4f}")
    print()
