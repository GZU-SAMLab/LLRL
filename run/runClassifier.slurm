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
python ./main/mainClassifier.py --num_class 6 --epochs 300 \
      --data-path ./dataset \
      --model-path ./weight/best_model103Epoch.pth
