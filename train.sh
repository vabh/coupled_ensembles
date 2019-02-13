#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

set -x

seed=1000

gpu=2
mB=64
uB=64
wd=0.0001
lr=0.1
niter=300

dataroot="../data"

dataset='cifar100'
# dataset='imagenet'
# dataset='stl10'
k=12
L=56
E=2

arch="resnet"
archConfig="depth=${L}"
# arch="densenet169"
# archConfig="growthRate=${k},depth=${L},num_classes=10"
# archConfig="num_classes=1000"

save="densenet_k${k}_L${L}_${dataset}_sm_${E}_0"
resume="${save}/latest.pth"
python train_model.py \
  --dataroot $dataroot \
  --dataset $dataset \
  --batchSize $mB \
  --microBatch $uB \
  --E $E \
  --arch $arch \
  --save $save \
  --nGPU $gpu\
  --weightDecay $wd \
  --lr $lr \
  --manualSeed $seed \
  --niter $niter \
  --archConfig $archConfig \
  # --probs \
  # --resume $resume \
  # --testOnly \
