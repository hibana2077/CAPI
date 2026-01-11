#!/bin/bash
#PBS -P yp87
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=12GB
#PBS -l walltime=08:00:00
#PBS -l wd
#PBS -l storage=scratch/yp87

RUN_TAG="capi_sweep"

module load cuda/12.6.2

source /scratch/yp87/sl5952/CAPI/.venv/bin/activate
export HF_HOME="/scratch/yp87/sl5952/CARROT/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -u src/main.py \
  --dataset stanford_cars --download --model resnet50 --pretrained \
  --epochs 300 \
  --capi_dim 32 \
  --lambda_lie 0.01 \
  --gamma 0.5 \
  --seed 42 >> "SW011.log" 2>&1
