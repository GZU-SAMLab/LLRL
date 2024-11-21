#!/bin/sh
#SBATCH -J Name
#SBATCH -o ./out/train.out.%j
#SBATCH -e ./err/train.err.%j
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
source activate env
python ./main/mainBackBone.py \
        --file_root ./dataset/backBone/sum1 \
        --max_steps 40000 --model_type small \
        --batch_size 16 --lr 2e-4 --gpu_id 0
