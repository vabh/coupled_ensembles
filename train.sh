#!/usr/bin/env sh
set -x

export CUDA_VISIBLE_DEVICES=0

k=12
L=148
numNet=1
batchSize=64
microBatch=64

probs='True'
dataset="cifar100"
# dataset="svhn"

nGPU=2
seed=0

SAVE="densenet_k${k}_L${L}_${dataset}_sm_${numNet}_0"
# RESUME="densenet_k${k}_L${L}_${dataset}_sm_${numNet}_5/net_epoch_100.pth"
# RESUME="densenet_k${k}_L${L}_${dataset}_sm_${numNet}_5/latest.pth"
python train_densenet.py \
  --dataset $dataset \
  --k $k \
  --L $L \
  --num $numNet \
  --save $SAVE \
  --batchSize $batchSize \
  --microBatch $microBatch \
  --nGPU $nGPU \
  --manualSeed $seed \
  --probs \
  # --resume $RESUME \
