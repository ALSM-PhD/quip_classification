#!/bin/bash
#SBATCH -p GPU-AI
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:volta16:1
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shahira 
#SBATCH --output=tcga_Res50_2class_adam_b128_CEloss_lr10-4_out_new.txt


#echo commands to stdout
set -x

source /etc/profile.d/modules.sh
module load singularity/2.5.1
cd /home/shahira/NNFramework

singularity exec --writable  --bind /pylon5/ac3uump --nv $SCRATCH/containers/tensorflow/tf-18.11-py3-w  python sa_runners/tf-classifier_runner.py $HOME/NNFramework/config/config_tcga_resnet-50.ini >> res50_out_new.txt

exit 0