#!/bin/sh
#SBATCH -J Name
#SBATCH -o ./out/train.out.%j
#SBATCH -e ./err/train.err.%j
#SBATCH --partition=gpu-l20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
source activate env
python ./main/mainClassifierVal.py --num_class 6 \
      --data-path ./datset \
      --model-path ./weight/ResNet18-model-new.pth

