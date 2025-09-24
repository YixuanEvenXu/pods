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
CSV_ACC = Path('results/h100s-acc.csv')
CSV_LEN = Path('results/h100s-length.csv')
OUT_DIR = Path('figures/h100s')

SEEDS        = None                  # e.g. [11,12]; None = all seeds
TARGET_MIN   = np.arange(0, 31, 2)   # 0,5,…,30 minutes
TIME_PLOT_WID = 5                    # right-panel width (in)

ALG_CONF = {
    'GRPO-GA'  : dict(pat=r'GRPO-s\d+-p8-n32-m32-ga16',  ga=16, col='#BB0011'),
    'GRPO-PODS': dict(pat=r'GRPO-s\d+-p8-n128-m32-ga4',  ga=4,  col='#1166CC'),
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
    ax_l.fill_between(TARGET_MIN,(m-1.96*s)*100,(m+1.96*s)*100,color=c,alpha=0.25)
ax_l.set_xlabel('Training Time on 8 H100s (minutes)')
ax_l.set_ylabel('Test Accuracy (%)')
ax_l.set_xlim(0,30); ax_l.grid(alpha=0.3); ax_l.legend()

x=np.arange(len(ALG_CONF))
ax_r.bar(x,[sec_gl[a] for a in ALG_CONF],color=[cfg['col'] for cfg in ALG_CONF.values()])
ax_r.set_xticks(x); ax_r.set_xticklabels(list(ALG_CONF))
ax_r.set_ylabel('Seconds per \nGlobal Training Step')
ax_r.set_xlabel('Algorithm')

fig.savefig(OUT_DIR/'h100s_main.png',dpi=200)
fig.savefig(OUT_DIR/'h100s_main.pdf')
plt.close(fig)

# ╭──────── FIG 2 : length curve ───────────╮
plt.figure(figsize=(18,4))
for alg,cfg in ALG_CONF.items():
    c=cfg['col']; m,s=mean_len[alg],sem_len[alg]
    plt.plot(TARGET_MIN,m,marker='o',color=c,label=alg)
    plt.fill_between(TARGET_MIN,m-1.96*s,m+1.96*s,color=c,alpha=0.25)
plt.xlabel('Training Time on 8 H100s (minutes)')
plt.ylabel('Average\nCompletion Length')
plt.xlim(0,30); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR/'h100s_length_curve.png',dpi=200)
plt.savefig(OUT_DIR/'h100s_length_curve.pdf',dpi=200)
plt.close()

print('✅  30-minute figures saved to', OUT_DIR)
