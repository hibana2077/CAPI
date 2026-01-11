#!/bin/bash
#PBS -P yp87
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=8GB
#PBS -l walltime=08:00:00
#PBS -l wd
#PBS -l storage=scratch/yp87

RUN_TAG="capi"

module load cuda/12.6.2

source /scratch/yp87/sl5952/CAPI/.venv/bin/activate
export HF_HOME="/scratch/yp87/sl5952/CARROT/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -u src/main.py \
  --dataset cub_200_2011 --download --model resnet50 --pretrained \
  --epochs 300 \
  --lambda_lie 0.1 \
  --batch_size 64 \
  --capi_dim 64 \
  --seed 42 >> "T006.log" 2>&1