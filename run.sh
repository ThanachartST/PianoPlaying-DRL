#!/bin/bash

export PYTHONPATH="$PWD/robopianist"

WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python train.py \
    --root-dir /tmp/robopianist/rl/ \
    --batch-size 256\
    --warmup-steps 5000 \
    --total-steps 3_000_000 \
    --discount 0.99 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.rnn-hidden-size 256 \
    --agent-config.fc-hidden-sizes 256 256 256 \
    --trim-silence \
    --gravity-compensation \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --environment-name "RoboPianist-{music_name}" \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --midi-path "$PWD/robopianist/robopianist/music/data/pig_single_finger/clair_de_lune-1.proto"
    # --reduced-action-space \
    # --action-reward-observation \
    # --name ""
