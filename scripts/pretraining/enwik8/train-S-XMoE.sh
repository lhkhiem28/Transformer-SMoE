#!/bin/bash
echo 'Run training...'
cd source/
python -u train.py \
    --cuda \
    --data ../datasets/pretraining/enwik8/ \
    --dataset enwik8 \
    --n_layer 1 \
    --d_model 256 \
    --n_head 8 \
    --d_head 64 \
    --d_inner 512 \
    --dropout 0.1 \
    --dropatt 0.0 \
    --optim adam \
    --lr 0.00025 \
    --warmup_step 0 \
    --max_step 100000 \
    --tgt_len 512 \
    --mem_len 512 \
    --eval_tgt_len 128 \
    --batch_size 64 \
    --multi_gpu \
    --moe --moe-num-expert 16 --moe-top-k 2 \
    --gate_name CustomNaiveGate_XMoE --load_balance 0.01 \
    --moe_index 0 \
    --work_dir ../ckps/pretraining/XMoE-S \
    # --dynamic_moe \
    # --dynamic_moe_mode linear_increase \
    # --dynamic_overall_steps 100000 \
    # --moe-top-k-min 16 \
    # --moe-top-k-max 16 \