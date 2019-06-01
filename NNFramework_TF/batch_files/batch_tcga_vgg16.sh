#!/bin/bash
#SBATCH -p GPU-AI
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:volta16:1
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alsmeirelles@gmail.com 


#echo commands to stdout
set -x

source /etc/profile.d/modules.sh
module load singularity/2.5.1
module load cuda/10.0
cd /pylon5/ac3uump/alsm/active-learning/quip_classification/NNFramework_TF

export PYTHONPATH=/home/alsm/.local/lib/python3.5/site-packages/:/usr/local/lib/python3.5/dist-packages/:/opt/packages/TensorFlow/gnu/tf1.8_py3_gpu/lib/python3.6/site-packages:/opt/packages/python/gnu_openmpi/3.6.4_np1.14.5/lib/python3.6/site-packages

singularity exec --writable  --bind /pylon5/ac3uump --nv ../../containers/tf-18.11-py3-w python3 sa_runners/tf_classifier_runner.py config/config_vgg-all.ini 0


exit 0
