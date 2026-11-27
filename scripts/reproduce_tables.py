"""Rebuild the manuscript's result tables from the saved per-run test predictions.

Usage:  python scripts/reproduce_tables.py            (from the repository root; CPU only, ~3-4 min)

Reads   results/preds/<config>__s<seed>__<benchmark>.npz   (one file per run and test set)
Writes  results/reproduced/*.csv  and prints a check against results/tables/*.csv
All metrics use a fixed 0.5 threshold, exactly as in the training notebook.
"""
import glob, os, ast
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, average_precision_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED, TAB = f'{ROOT}/results/preds', f'{ROOT}/results/tables'
OUT = f'{ROOT}/results/reproduced'; os.makedirs(OUT, exist_ok=True)
N_BOOT = 1000

def load(cfg, seed, bench):
    p = f'{PRED}/{cfg}__s{seed}__{bench}.npz'
    return np.load(p) if os.path.exists(p) else None

def has_modality(d):
    return (d['n_present'] > 0).mean() > 0

def metrics(t, p):
    yh = (p > 0.5).astype(int)
    return {'Acc': accuracy_score(t, yh), 'F1': f1_score(t, yh, zero_division=0),
            'F1_macro': f1_score(t, yh, average='macro', zero_division=0),
            'AUC': roc_auc_score(t, p), 'MCC': matthews_corrcoef(t, yh), 'AP': average_precision_score(t, p)}

def paired_boot(t, pa, pb, seed):
    rng = np.random.default_rng(seed); n = len(t); dA, dM = [], []
    for _ in range(N_BOOT):
        ix = rng.integers(0, n, n); tt = t[ix]
        if len(np.unique(tt)) < 2: continue
        dA.append(roc_auc_score(tt, pa[ix]) - roc_auc_score(tt, pb[ix]))
        dM.append(matthews_corrcoef(tt, (pa[ix] > .5).astype(int)) - matthews_corrcoef(tt, (pb[ix] > .5).astype(int)))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return q(dA), q(dM)

# ---- per-seed metrics (basis of Tables 6 and 9) -------------------------------------------
rows = []
for f in sorted(glob.glob(f'{PRED}/*.npz')):
    cfg, s, bench = os.path.basename(f)[:-4].split('__'); d = np.load(f)
    if not has_modality(d) or len(np.unique(d['true'])) < 2: continue
    rows.append({'config': cfg, 'seed': int(s[1:]), 'bench': bench, 'N': len(d['true']), **metrics(d['true'], d['prob'])})
per_seed = pd.DataFrame(rows).sort_values(['config', 'bench', 'seed'])
per_seed.to_csv(f'{OUT}/per_seed.csv', index=False)
summary = per_seed.groupby(['config', 'bench'])[['Acc', 'F1', 'F1_macro', 'AUC', 'MCC', 'AP']].agg(['mean', 'std', 'count'])
summary.round(4).to_csv(f'{OUT}/table6_table9_mean_std.csv')

# ---- paired comparisons on seed-averaged predictions (Table 7, Figs. 11 and 13) ------------
COMPARATORS = ['mean_masked', 'gating_masked', 'late_masked', 'mlca_masked', 'cmfusion_masked', 'tcaf_original',
               'tcaf_masked_no_audio', 'tcaf_masked_no_crema', 'image_only', 'text_only', 'audio_only']
prow = []
for b in COMPARATORS:
    for bench in ('m3h', 'hmm', 'ucb'):
        A = [load('tcaf_masked', s, bench) for s in (42, 43, 44)]
        B = [load(b, s, bench) for s in (42, 43, 44)]
        pairs = [(a, c) for a, c in zip(A, B) if a is not None and c is not None and has_modality(a) and has_modality(c)]
        if len(pairs) < 2: continue
        t = pairs[0][0]['true']; pa = np.mean([a['prob'] for a, _ in pairs], 0); pb = np.mean([c['prob'] for _, c in pairs], 0)
        ciA, ciM = paired_boot(t, pa, pb, seed=99)
        prow.append({'comparator': b, 'bench': bench,
                     'dAUC': roc_auc_score(t, pa) - roc_auc_score(t, pb), 'dAUC_lo': ciA[0], 'dAUC_hi': ciA[1],
                     'dMCC': matthews_corrcoef(t, (pa > .5).astype(int)) - matthews_corrcoef(t, (pb > .5).astype(int)),
                     'dMCC_lo': ciM[0], 'dMCC_hi': ciM[1]})
paired = pd.DataFrame(prow); paired.round(4).to_csv(f'{OUT}/table7_paired_differences.csv', index=False)

# ---- auxiliary heads (Table 8) -------------------------------------------------------------
arow = []
for s in (42, 43, 44):
    for bench in ('m3h', 'hmm', 'ucb'):
        d = load('tcaf_masked', s, bench)
        for h in ('h2', 'h3'):
            m = d[f'{h}_true'] >= 0
            if m.sum():
                arow.append({'seed': s, 'bench': bench, 'head': h.upper(), 'N': int(m.sum()),
                             'Acc': accuracy_score(d[f'{h}_true'][m], d[f'{h}_pred'][m]),
                             'F1_macro': f1_score(d[f'{h}_true'][m], d[f'{h}_pred'][m], average='macro', zero_division=0)})
aux = pd.DataFrame(arow).groupby(['bench', 'head'])[['N', 'Acc', 'F1_macro']].agg(['mean', 'std'])
aux.round(4).to_csv(f'{OUT}/table8_aux_heads.csv')

# ---- check against the tables written by the training notebook -----------------------------
st = pd.read_csv(f'{TAB}/v3_per_seed.csv')
m = per_seed.merge(st, on=['config', 'seed', 'bench'], suffixes=('', '_nb'))
d1 = max(float((m[k] - m[f'{k}_nb']).abs().max()) for k in ['Acc', 'F1', 'F1_macro', 'AUC', 'MCC', 'AP'])
sp = pd.read_csv(f'{TAB}/v3_paired_comparisons.csv'); sp = sp[(sp.seed == 'ensemble') & (sp.A == 'tcaf_masked')]
mm = paired.merge(sp, left_on=['comparator', 'bench'], right_on=['B', 'bench'])
d2 = max(max(abs(r.dAUC_lo - ast.literal_eval(r.dAUC_CI)[0]), abs(r.dAUC_hi - ast.literal_eval(r.dAUC_CI)[1]),
             abs(r.dMCC_lo - ast.literal_eval(r.dMCC_CI)[0]), abs(r.dMCC_hi - ast.literal_eval(r.dMCC_CI)[1]),
             abs(r.dAUC_x - r.dAUC_y), abs(r.dMCC_x - r.dMCC_y)) for r in mm.itertuples())
print(f'Runs: {len(per_seed)} (config x seed x test set) | max |diff| vs notebook per-seed table: {d1:.1e}')
print(f'Paired comparisons: {len(mm)} | max |diff| vs notebook paired table (incl. CIs): {d2:.1e}')
print(f'Reproduced tables written to {OUT}')
