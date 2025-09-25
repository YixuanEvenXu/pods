"""
To make plots for the multi-GPU experiment results of the paper
"Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning"
by Yixuan Even Xu*, Yash Savani*, Fei Fang, and Zico Kolter
https://arxiv.org/abs/2504.13818

Developed by: Yixuan Even Xu in 2025
"""


import csv, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# ╭─────────────────────────────╮
# │  USER SETTINGS              │
# ╰─────────────────────────────╯
CSV_ACC = Path('results/a100s-acc.csv')
CSV_LEN = Path('results/a100s-length.csv')
OUT_DIR = Path('figures/a100s')

SEEDS        = None                  # e.g. [11,12]; None = all seeds
TARGET_MIN   = np.arange(0, 30+1, 2)   # 0,5,…,30 minutes
TIME_PLOT_WID = 5                    # right-panel width (in)

ALG_CONF = {
    'GRPO-GA'  : dict(pat=r'GR7B-s\d+-p8-n32-m32-ga16',  ga=16, col='#BB0011'),
    'GRPO-PODS': dict(pat=r'GR7B-s\d+-p8-n128-m32-ga4',  ga=4,  col='#1166CC'),
}

def soften(hexcode, alpha=0.7):
    r,g,b=(int(hexcode[i:i+2],16) for i in (1,3,5))
    return '#{:02X}{:02X}{:02X}'.format(int(r*alpha+255*(1-alpha)),
                                        int(g*alpha+255*(1-alpha)),
                                        int(b*alpha+255*(1-alpha)))
for cfg in ALG_CONF.values():
    cfg['col']=soften(cfg['col'],0.7)

plt.rcParams.update({'lines.linewidth':2,'lines.markersize':8,
                     'font.size':22,'axes.labelsize':22,
                     'axes.titlesize':26,'legend.fontsize':22})

OUT_DIR.mkdir(parents=True,exist_ok=True)
seed_re=re.compile(r'-s(\d+)-')

# ╭──────── helper: read csv → dict ───────╮
def read_csv(path):
    with path.open(newline='') as f:
        rdr = csv.reader(f)
        hdr = [h.strip('"') for h in next(rdr)]
        rows = list(rdr)
    cols={h:[] for h in hdr}
    for row in rows:
        for h,v in zip(hdr,row):
            cols[h].append(v if v else None)
    times=np.array([float(x) for x in cols['eval/time']])/60.0
    return hdr,cols,times

hdr_acc,acc_cols,t_acc = read_csv(CSV_ACC)
hdr_len,len_cols,t_len = read_csv(CSV_LEN)

# ╭──────── collect per-run series ─────────╮
acc_runs=defaultdict(lambda:defaultdict(list))
step_runs=defaultdict(lambda:defaultdict(list))
len_runs=defaultdict(lambda:defaultdict(list))

for col_name in hdr_acc:
    for alg,cfg in ALG_CONF.items():
        pat=cfg['pat']
        if not re.fullmatch(fr'data/{pat} - (?:_step|eval/rewards/accuracy_reward/mean)',col_name):
            continue
        run=col_name.split(' ')[0].replace('data/','')
        m=seed_re.search(run);   seed=int(m.group(1)) if m else None
        if seed is None or (SEEDS and seed not in SEEDS): continue
        series=[(t,float(v)) for t,v in zip(t_acc,acc_cols[col_name]) if v]
        if '_step' in col_name:
            step_runs[alg][seed]=[(t,int(v)) for t,v in series]
        else:
            acc_runs[alg][seed]=series

for col_name in hdr_len:
    for alg,cfg in ALG_CONF.items():
        pat=cfg['pat']
        if not re.fullmatch(fr'data/{pat} - eval/completions/mean_length',col_name):
            continue
        run=col_name.split(' ')[0].replace('data/','')
        m=seed_re.search(run);   seed=int(m.group(1)) if m else None
        if seed is None or (SEEDS and seed not in SEEDS): continue
        series=[(t,float(v)) for t,v in zip(t_len,len_cols[col_name]) if v]
        len_runs[alg][seed]=series

# ╭──────── align helper ────────╮
def nearest(series,tgt):
    arr=np.array([t for t,_ in series])
    return series[np.abs(arr-tgt).argmin()][1]

# ╭──────── aggregate per algorithm ────────╮
acc_mat,len_mat,sec_gl={}, {}, {}
for alg,cfg in ALG_CONF.items():
    seeds=set(acc_runs[alg]) & set(step_runs[alg]) & set(len_runs[alg])
    acc_mat[alg]=np.array([[nearest(acc_runs[alg][s],tm) for tm in TARGET_MIN] for s in seeds])
    len_mat[alg]=np.array([[nearest(len_runs[alg][s],tm) for tm in TARGET_MIN] for s in seeds])
    steps_30=[nearest(step_runs[alg][s],TARGET_MIN[-1]) for s in seeds]  # step @ ~30 min
    sec_gl[alg]=(30*60)/(np.mean(steps_30)/cfg['ga'])                    # sec / global step

mean_acc={alg:m.mean(0) for alg,m in acc_mat.items()}
sem_acc ={alg:sem(m,0)  for alg,m in acc_mat.items()}
mean_len={alg:m.mean(0) for alg,m in len_mat.items()}
sem_len ={alg:sem(m,0)  for alg,m in len_mat.items()}

# ╭──────── FIG 1 : composite ──────────────╮
fig,(ax_l,ax_r)=plt.subplots(
    1,2,figsize=(18,4),dpi=100,
    gridspec_kw={'width_ratios':[18-TIME_PLOT_WID,TIME_PLOT_WID],'wspace':0.1},
    constrained_layout=True)

for alg,cfg in ALG_CONF.items():
    c=cfg['col']; m,s=mean_acc[alg],sem_acc[alg]
    ax_l.plot(TARGET_MIN,m*100,marker='o',color=c,label=alg)
    lb = (m-1.96*s)*100
    lb = [max(x,0) for x in lb]
    ub = (m+1.96*s)*100
    ub = [min(x,100) for x in ub]
    ax_l.fill_between(TARGET_MIN,lb,ub,color=c,alpha=0.25)
ax_l.set_xlabel('Training Time on 8 A100s (minutes)')
ax_l.set_ylabel('Test Accuracy (%)')
ax_l.set_xlim(0,30); ax_l.grid(alpha=0.3); ax_l.legend()

x=np.arange(len(ALG_CONF))
ax_r.bar(x,[sec_gl[a] for a in ALG_CONF],color=[cfg['col'] for cfg in ALG_CONF.values()])
ax_r.set_xticks(x); ax_r.set_xticklabels(list(ALG_CONF))
ax_r.set_ylabel('Seconds per \nGlobal Training Step')
ax_r.set_xlabel('Algorithm')

fig.savefig(OUT_DIR/'a100s_main.png',dpi=200)
fig.savefig(OUT_DIR/'a100s_main.pdf')
plt.close(fig)

# ╭──────── FIG 2 : length curve ───────────╮
plt.figure(figsize=(18,4))
for alg,cfg in ALG_CONF.items():
    c=cfg['col']; m,s=mean_len[alg],sem_len[alg]
    plt.plot(TARGET_MIN,m,marker='o',color=c,label=alg)
    plt.fill_between(TARGET_MIN,m-1.96*s,m+1.96*s,color=c,alpha=0.25)
plt.xlabel('Training Time on 8 A100s (minutes)')
plt.ylabel('Average\nCompletion Length')
plt.xlim(0,30); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR/'a100s_length_curve.png',dpi=200)
plt.savefig(OUT_DIR/'a100s_length_curve.pdf',dpi=200)
plt.close()

print('✅  30-minute figures saved to', OUT_DIR)



def compute_pods_speedup_99(TARGET_MIN, mean_acc, alg_grpo='GRPO-GA', alg_pods='GRPO-PODS'):
    """
    Multi-GPU version:
    - Take the max accuracy on GRPO-GA's mean curve (exclude t=0).
    - Let T = 0.99 * (that max).
    - Find the earliest time on GRPO-GA with acc ≥ T, and earliest time on GRPO-PODS with acc ≥ T.
    - Print and return the ratio t_GRPO / t_PODS (≥1 ⇒ PODS is faster).
    """
    if alg_grpo not in mean_acc or alg_pods not in mean_acc:
        print("⚠️  Need both GRPO-GA and GRPO-PODS in mean_acc; skipping.")
        return None

    t = np.asarray(TARGET_MIN, dtype=float)
    mask = t > 0  # ignore potential t=0 anchor
    t = t[mask]
    y_g = np.asarray(mean_acc[alg_grpo])[mask]   # GRPO-GA
    y_p = np.asarray(mean_acc[alg_pods])[mask]   # GRPO-PODS

    if t.size == 0 or y_g.size != t.size or y_p.size != t.size:
        print("⚠️  Inconsistent time/accuracy arrays; skipping.")
        return None

    T = 0.99 * float(y_g.max())

    def first_idx_at_least(y, thr):
        idxs = np.where(y >= thr)[0]
        return int(idxs[0]) if idxs.size else None

    i_g = first_idx_at_least(y_g, T)
    i_p = first_idx_at_least(y_p, T)

    if i_g is None or i_p is None:
        print("ℹ️  Threshold not reached on both curves; cannot compute speedup.")
        return None

    ratio = t[i_g] / t[i_p]
    print(f"✅  PODS speedup to 99% of GRPO-GA best: {ratio:.2f}×  "
          f"(GRPO-GA {t[i_g]:.3g} min vs PODS {t[i_p]:.3g} min; "
          f"threshold={T*100:.2f}%)")
    return ratio

# ╭──────────────────────────────────────────╮
# │  Compute PODS speedup at 99% GRPO best  │
# ╰──────────────────────────────────────────╯
compute_pods_speedup_99(TARGET_MIN, mean_acc)
