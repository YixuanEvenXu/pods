"""
To make plots for the single-GPU experiment results of the paper
"Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning"
by Yixuan Even Xu*, Yash Savani*, Fei Fang, and Zico Kolter
https://arxiv.org/abs/2504.13818

Developed by: Yixuan Even Xu in 2025
"""

import os, json, re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# ╭─────────────────────────────╮
# │  USER-EDITABLE PARAMETERS   │
# ╰─────────────────────────────╯
setting_num = 6

# ───── Setting definitions ─────────────────────────────────
if setting_num == 1:
    exp_name = 'gsm8k_n_scaling'
    settings = [
        'Instruct-GSM8K-512-16-16',
        'Instruct-GSM8K-512-32-16',
        'Instruct-GSM8K-512-64-16',
        'Instruct-GSM8K-512-128-16',
        'Instruct-GSM8K-512-256-16',
    ]
    tol_rainbow = ["#BB0011", "#EE6611", "#1166CC", "#00AA55", "#AA4499"]
    display_names = [f'$N = {n}$' for n in [16, 32, 64, 128, 256]]
    run_names = [f'Run{i}' for i in range(1, 11)]
    dataset_key = 'gsm8k'
    time_plot_wid = 9
    max_hours   = 6
    ckpt_int    = 0.5
    metric_key  = 'ans_acc'

elif setting_num == 2:
    exp_name = 'gsm8k_m_scaling'
    settings = [
        'Instruct-GSM8K-512-64-16',
        'Instruct-GSM8K-512-64-8',
        'Instruct-GSM8K-512-64-4',
        'Instruct-GSM8K-512-64-2',
    ]
    tol_rainbow = ["#1166CC", "#EE6611", "#00AA55", "#AA4499"]
    display_names = [f'$M = {m}$' for m in [16, 8, 4, 2]]
    run_names = [f'Run{i}' for i in range(1, 11)]
    dataset_key = 'gsm8k'
    time_plot_wid = 9
    max_hours   = 6
    ckpt_int    = 0.5
    metric_key  = 'ans_acc'

elif setting_num == 3:
    exp_name = 'gsm8k_main'
    settings = [
        'Instruct-GSM8K-512-16-16',
        'Instruct-GSM8K-512-64-16',
    ]
    tol_rainbow = ["#BB0011", "#1166CC"]
    display_names = ['GRPO', 'GRPO-PODS']
    run_names = [f'Run{i}' for i in range(1, 11)]
    dataset_key = 'gsm8k'
    time_plot_wid = 5
    max_hours   = 6
    ckpt_int    = 0.5
    metric_key  = 'ans_acc'

elif setting_num == 4:
    exp_name = 'math_main'
    settings = [
        'Instruct-MATH-NF-512-8-8',
        'Instruct-MATH-NF-512-32-8',
    ]
    tol_rainbow = ["#BB0011", "#1166CC"]
    display_names = ['GRPO', 'GRPO-PODS']
    run_names = [f'Run{i}' for i in range(1, 11)]
    dataset_key = 'math500'
    time_plot_wid = 5
    max_hours   = 6
    ckpt_int    = 0.5
    metric_key  = 'ans_acc'

elif setting_num == 5:
    exp_name = 'down_sampling_rule'
    settings = [
        'Instruct-GSM8K-512-64-16',
        'Instruct-GSM8K-MaxR-512-64-16',
        'Instruct-GSM8K-Rand-512-64-16',
    ]
    tol_rainbow = ["#1166CC", "#00AA55", "#EE6611"]
    display_names = ['Max-Variance', 'Max-Reward', 'Random']
    run_names = [f'Run{i}' for i in range(1, 11)]
    dataset_key = 'gsm8k'
    time_plot_wid = 8
    max_hours   = 6
    ckpt_int    = 0.5
    metric_key  = 'ans_acc'

elif setting_num == 6:
    exp_name = 'llama_main'
    settings = [
        'Instruct-Llama-512-16-16-0.04',
        'Instruct-Llama-512-64-16-0.04',
    ]
    tol_rainbow = ["#BB0011", "#1166CC"]
    display_names = ['GRPO', 'GRPO-PODS']
    run_names = [f'Run{i}' for i in range(1, 11)]
    dataset_key = 'gsm8k'
    time_plot_wid = 5
    max_hours   = 1.5
    ckpt_int    = 1/6
    metric_key  = 'both_acc'

# ╭──────────────────────────────╮
# │  File / colour prep          │
# ╰──────────────────────────────╯
fig_root  = Path('figures') / exp_name; fig_root.mkdir(parents=True, exist_ok=True)
cache_dir = Path('results') / 'step_time'; cache_dir.mkdir(parents=True, exist_ok=True)

disp_map   = dict(zip(settings, display_names))
def soften(hexcode, alpha=0.7):
    r, g, b = (int(hexcode[i:i+2], 16) for i in (1, 3, 5))
    r = int(r*alpha + 255*(1-alpha)); g = int(g*alpha + 255*(1-alpha)); b = int(b*alpha + 255*(1-alpha))
    return f'#{r:02X}{g:02X}{b:02X}'
color_map  = {lab: soften(tol_rainbow[i % len(tol_rainbow)], 0.7)
              for i, lab in enumerate(display_names)}

plt.rcParams.update({'lines.linewidth': 2, 'lines.markersize': 8,
                     'font.size': 22, 'axes.labelsize': 22,
                     'axes.titlesize': 26, 'legend.fontsize': 22})

# ╭─────────────────────────────╮
# │  containers                 │
# ╰─────────────────────────────╯
acc_curve   = defaultdict(lambda: defaultdict(list))
len_curve   = defaultdict(lambda: defaultdict(list))
step_counts = defaultdict(list)
step_time   = defaultdict(list)

# ╭─────────────────────────────╮
# │  parse logs                 │
# ╰─────────────────────────────╯
for setting in settings:
    for run in run_names:
        p = Path('results') / f'{setting}-{run}.json'
        if not p.exists(): continue
        log=json.loads(p.read_text())
        cps=sorted(log,key=lambda k:int(k.split('-')[-1]))
        if not cps: continue
        for idx,cp in enumerate(cps):
            t_hr=(idx+1)*ckpt_int
            if t_hr>max_hours: break
            block=log[cp].get(dataset_key,{})
            accs=block.get(metric_key,[])
            lens=block.get('lengths',[])
            if accs: acc_curve[setting][t_hr].append(np.mean(accs))
            if lens: len_curve[setting][t_hr].append(np.mean(lens))
        idx6=min(len(cps)-1,int(max_hours/ckpt_int)-1)
        steps=int(cps[idx6].split('-')[-1])
        step_counts[setting].append(steps)
        tot=max_hours*3600/steps
        step_time[setting].append(tot)

# ╭──────────────────────────────────────────────╮
# │  Add base-model anchor, keep original y-axis │
# ╰──────────────────────────────────────────────╯
if setting_num == 6:
    base_path = Path('results') / 'base-llama.json'
else:
    base_path = Path('results') / 'base-qwen.json'
if base_path.exists():
    base_json = json.loads(base_path.read_text()).get(dataset_key, {})
    base_vals_acc = base_json.get(metric_key, [])
    base_vals_len = base_json.get('lengths', [])
    if base_vals_acc:
        base_acc = np.mean(base_vals_acc)
        for setting in settings:
            acc_curve[setting][0.0].append(base_acc)
    if base_vals_len:
        base_len = np.mean(base_vals_len)
        for setting in settings:
            len_curve[setting][0.0].append(base_len)
else:
    print('⚠️  Base model log not found; skipping base anchor')

# axis range (accuracy)
y_vals=[v*100 for s in settings for t,vlist in acc_curve[s].items() if t>0 for v in vlist]
ymin,ymax=min(y_vals),max(y_vals); margin=0.01*(ymax-ymin)

# ╭──────── completion-length curve ────────╮
plt.figure(figsize=(18,4))
for s in settings:
    lab=disp_map[s]; c=color_map[lab]
    t=sorted(len_curve[s]); y=[np.mean(len_curve[s][ti]) for ti in t]
    e=[sem(len_curve[s][ti]) if len(len_curve[s][ti])>1 else 0 for ti in t]
    plt.plot(t,y,marker='o',color=c,label=lab)
    plt.fill_between(t,np.array(y)-1.96*np.array(e),np.array(y)+1.96*np.array(e),
                     color=c,alpha=0.25)
plt.xlabel('Training Time on One L40S (hours)')
plt.ylabel('Average\nCompletion Length')
plt.xlim(0,max_hours)
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(fig_root/'length_over_time.png',dpi=200)
plt.savefig(fig_root/'length_over_time.pdf',dpi=200)
plt.close()

# ╭──────── composite panel ────────╮
ratio_l,ratio_r=18-time_plot_wid,time_plot_wid
fig,(ax_l,ax_r)=plt.subplots(1,2,figsize=(18,4),dpi=100,
    gridspec_kw={'width_ratios':[ratio_l,ratio_r],'wspace':0.1},
    constrained_layout=True)

# accuracy curve (left)
for s in settings:
    lab=disp_map[s]; c=color_map[lab]
    t=sorted(acc_curve[s]); y=[np.mean(acc_curve[s][ti])*100 for ti in t]
    e=[sem(acc_curve[s][ti])*100 if len(acc_curve[s][ti])>1 else 0 for ti in t]
    ax_l.plot(t,y,marker='o',color=c,label=lab)
    ax_l.fill_between(t,np.array(y)-1.96*np.array(e),np.array(y)+1.96*np.array(e),
                      color=c,alpha=0.25)
ax_l.set_xlabel('Training Time on One L40S (hours)')
ax_l.set_ylabel('Test Accuracy (%)')
ax_l.set_xlim(0,max_hours); ax_l.set_ylim(ymin-margin,ymax+margin)
ax_l.grid(alpha=0.3); ax_l.legend(loc='best')

# stacked time bar (right)
x=np.arange(len(settings))
times=[np.mean(step_time[s]) for s in settings]
labels=[disp_map[s] for s in settings]; x=np.arange(len(labels))
colors=[color_map[l] for l in labels]
ax_r.bar(x,times,color=colors)
ax_r.set_xticks(x); ax_r.set_xticklabels(labels)
ax_r.set_xlabel('Algorithm'); ax_r.set_ylabel('Seconds per\nTraining Step')

fig.savefig(fig_root/'main.png',dpi=200)
fig.savefig(fig_root/'main.pdf'); plt.close(fig)

print('✅  All plots saved under', fig_root)

def compute_pods_speedup_99(settings, disp_map, acc_curve):
    """
    Pick the highest point on GRPO's curve (excluding t=0 anchors).
    Let T = 0.99 * (that max accuracy).
    Find the earliest time on GRPO with acc ≥ T, and the earliest time on PODS with acc ≥ T.
    Return/print the ratio t_GRPO / t_PODS (≥1 ⇒ PODS is faster).
    """
    # Identify GRPO and GRPO-PODS settings by display labels
    grpo_setting = next((s for s in settings if disp_map.get(s) == 'GRPO'), None)
    pods_setting = next((s for s in settings if disp_map.get(s) == 'GRPO-PODS'), None)
    if grpo_setting is None or pods_setting is None:
        print("⚠️  Need both 'GRPO' and 'GRPO-PODS' to compute speedup; skipping.")
        return None

    # Build (time, mean_acc) series in raw units (0..1); ignore t=0 anchors
    def series(setting):
        ts = sorted(t for t in acc_curve[setting].keys() if t > 0)
        ys = [float(np.mean(acc_curve[setting][t])) for t in ts]
        return ts, ys

    ts_g, ys_g = series(grpo_setting)
    ts_p, ys_p = series(pods_setting)
    if not ts_g or not ts_p:
        print("⚠️  Missing accuracy points for GRPO or GRPO-PODS; skipping.")
        return None

    # Threshold = 0.99 * max GRPO accuracy
    max_grpo = max(ys_g)
    T = 0.99 * max_grpo

    # Earliest times where acc ≥ T
    def first_time_at_least(ts, ys, thr):
        for t, y in zip(ts, ys):
            if y >= thr:
                return t
        return None

    t_grpo = first_time_at_least(ts_g, ys_g, T)
    t_pods = first_time_at_least(ts_p, ys_p, T)

    if t_grpo is None or t_pods is None:
        print("ℹ️  Could not find threshold crossings on both curves (GRPO and PODS).")
        return None

    ratio = t_grpo / t_pods
    print(f"✅  PODS speedup to 99% of GRPO's best: {ratio:.2f}×  "
          f"(GRPO {t_grpo:.3g}h vs PODS {t_pods:.3g}h; "
          f"threshold={T*100:.2f}%)")
    return ratio

# ╭──────────────────────────────╮
# │  Compute PODS speedup vs GRPO │
# ╰──────────────────────────────╯
compute_pods_speedup_99(settings, disp_map, acc_curve)
