#!/bin/bash

song_name="nocturne_op9_no_2-1"
model_arch="MLP"
discount_factor=0.84
wandb_mode="online" #   mode: ['online', 'offline', 'disabled']
total_timestep=4_000_000

export PYTHONPATH="$PWD/robopianist"

WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python train.py \
    --root-dir /tmp/robopianist/rl/ \
    --batch-size 256\
    --warmup-steps 5000 \
    --total-steps $total_timestep \
    --discount $discount_factor \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.rnn-hidden-size 256 \
    --agent-config.fc-hidden-sizes 256 256 256 \
    --trim-silence \
    --gravity-compensation \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --environment-name "RoboPianist-Model-{$model_arch}-Song-{$song_name}-Discount-{$discount_factor}" \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --midi-path "$PWD/robopianist/robopianist/music/data/pig_single_finger/$song_name.proto" \
    --mode $wandb_mode
    # --reduced-action-space \
    # --action-reward-observation \
    # --name ""
