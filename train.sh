#!/usr/bin/env sh
set -x

export CUDA_VISIBLE_DEVICES=0

batchSize=64
microBatch=32

E=1
probs='True'
dataset="cifar100"
# dataset="svhn"

arch='densenet'
archConfig='depth=190'

sgdMomentum=0.9
bnMomentum=0.1

nGPU=2
seed=0

SAVE="densenet_k${k}_L${L}_${dataset}_sm_${numNet}_0"
# RESUME="densenet_k${k}_L${L}_${dataset}_sm_${numNet}_0/latest.pth"
python train_model.py \
  --dataset $dataset \
  --E $E \
  --save $SAVE \
  --batchSize $batchSize \
  --microBatch $microBatch \
  --nGPU $nGPU \
  --manualSeed $seed \
  --probs \
  --sgdMomentum $sgdMomentum \
  --bnMomentum $bnMomentum \
  --arch $arch \
  --archConfig $archConfig \
  # --resume $RESUME \
