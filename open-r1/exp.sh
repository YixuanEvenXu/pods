#!/usr/bin/env bash
set -e

# Activate env
source openr1/bin/activate

# Common accelerate settings
ACC_CFG=recipes/accelerate_configs/zero2.yaml
PY=src/open_r1/grpo.py
BASE_CFG=recipes/Qwen2.5-3B-Instruct/config_pods.yaml
NP=8                 # --num_processes
NG=32                # --num_generations (m32 in dir names)

# ───── experiments copied from screen-grab ────────────────────────────────
accelerate launch --config_file $ACC_CFG --num_processes $NP $PY \
  --config $BASE_CFG --pods_full_size=128 --num_generations=$NG --gradient_accumulation_steps=4 \
  --seed=25 --output_dir=data/GRPO-s25-p8-n128-m32-ga4

accelerate launch --config_file $ACC_CFG --num_processes $NP $PY \
  --config $BASE_CFG --pods_full_size=32  --num_generations=$NG --gradient_accumulation_steps=16 \
  --seed=25 --output_dir=data/GRPO-s25-p8-n32-m32-ga16

accelerate launch --config_file $ACC_CFG --num_processes $NP $PY \
  --config $BASE_CFG --pods_full_size=32  --num_generations=$NG --gradient_accumulation_steps=16 \
  --seed=24 --output_dir=data/GRPO-s24-p8-n32-m32-ga16

accelerate launch --config_file $ACC_CFG --num_processes $NP $PY \
  --config $BASE_CFG --pods_full_size=128 --num_generations=$NG --gradient_accumulation_steps=4  \
  --seed=23 --output_dir=data/GRPO-s23-p8-n128-m32-ga4

accelerate launch --config_file $ACC_CFG --num_processes $NP $PY \
  --config $BASE_CFG --pods_full_size=32  --num_generations=$NG --gradient_accumulation_steps=16 \
  --seed=23 --output_dir=data/GRPO-s23-p8-n32-m32-ga16

accelerate launch --config_file $ACC_CFG --num_processes $NP $PY \
  --config $BASE_CFG --pods_full_size=128 --num_generations=$NG --gradient_accumulation_steps=4  \
  --seed=22 --output_dir=data/GRPO-s22-p8-n128-m32-ga4

accelerate launch --config_file $ACC_CFG --num_processes $NP $PY \
  --config $BASE_CFG --pods_full_size=32  --num_generations=$NG --gradient_accumulation_steps=16 \
  --seed=22 --output_dir=data/GRPO-s22-p8-n32-m32-ga16

accelerate launch --config_file $ACC_CFG --num_processes $NP $PY \
  --config $BASE_CFG --pods_full_size=128 --num_generations=$NG --gradient_accumulation_steps=4  \
  --seed=21 --output_dir=data/GRPO-s21-p8-n128-m32-ga4

accelerate launch --config_file $ACC_CFG --num_processes $NP $PY \
  --config $BASE_CFG --pods_full_size=32  --num_generations=$NG --gradient_accumulation_steps=16 \
  --seed=21 --output_dir=data/GRPO-s21-p8-n32-m32-ga16

accelerate launch --config_file $ACC_CFG --num_processes $NP $PY \
  --config $BASE_CFG --pods_full_size=128 --num_generations=$NG --gradient_accumulation_steps=4  \
  --seed=12 --output_dir=data/GRPO-s12-p8-n128-m32-ga4

accelerate launch --config_file $ACC_CFG --num_processes $NP $PY \
  --config $BASE_CFG --pods_full_size=128 --num_generations=$NG --gradient_accumulation_steps=4  \
  --seed=15 --output_dir=data/GRPO-s15-p8-n128-m32-ga4

$ACCELERATE_LOG_LEVEL accelerate launch --config_file $ACC_CFG --num_processes $NP $PY \
  --config $BASE_CFG --pods_full_size=32  --num_generations=$NG --gradient_accumulation_steps=16 \
  --seed=15 --output_dir=data/GRPO-s15-p8-n32-m32-ga16

$ACCELERATE_LOG_LEVEL accelerate launch --config_file $ACC_CFG --num_processes $NP $PY \
  --config $BASE_CFG --pods_full_size=128 --num_generations=$NG --gradient_accumulation_steps=4  \
  --seed=14 --output_dir=data/GRPO-s14-p8-n128-m32-ga4

$ACCELERATE_LOG_LEVEL accelerate launch --config_file $ACC_CFG --num_processes $NP $PY \
  --config $BASE_CFG --pods_full_size=32  --num_generations=$NG --gradient_accumulation_steps=16 \
  --seed=14 --output_dir=data/GRPO-s14-p8-n32-m32-ga16

$ACCELERATE_LOG_LEVEL accelerate launch --config_file $ACC_CFG --num_processes $NP $PY \
  --config $BASE_CFG --pods_full_size=128 --num_generations=$NG --gradient_accumulation_steps=4  \
  --seed=13 --output_dir=data/GRPO-s13-p8-n128-m32-ga4

$ACCELERATE_LOG_LEVEL accelerate launch --config_file $ACC_CFG --num_processes $NP $PY \
  --config $BASE_CFG --pods_full_size=32  --num_generations=$NG --gradient_accumulation_steps=16 \
  --seed=13 --output_dir=data/GRPO-s13-p8-n32-m32-ga16

$ACCELERATE_LOG_LEVEL accelerate launch --config_file $ACC_CFG --num_processes $NP $PY \
  --config $BASE_CFG --pods_full_size=32  --num_generations=$NG --gradient_accumulation_steps=16 \
  --seed=11 --output_dir=data/GRPO-s11-p8-n32-m32-ga16
