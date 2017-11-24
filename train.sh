#!/usr/bin/env sh
set -x

export CUDA_VISIBLE_DEVICES=0

batchSize=64
microBatch=64

E=1
dataset="fold"
# dataset="svhn"

arch='densenet'
k=12
L=100
archConfig="depth=${L},growthRate=${k}"

lr=0.1
sgdMomentum=0.9
bnMomentum=0.1

nGPU=1
seed=0

SAVE="./checkpoints/scale_32_f5"
# RESUME="${SAVE}/latest.pth"
python train_model.py \
  --dataset $dataset \
  --E $E \
  --save $SAVE \
  --batchSize $batchSize \
  --microBatch $microBatch \
  --nGPU $nGPU \
  --manualSeed $seed \
  --sgdMomentum $sgdMomentum \
  --bnMomentum $bnMomentum \
  --arch $arch \
  --archConfig $archConfig \
  --workers 4 \
  --lr $lr \
  --imageSize 32 \
  # --probs \
  # --resume $RESUME \
