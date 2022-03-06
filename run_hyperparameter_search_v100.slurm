#!/bin/bash
#SBATCH --job-name=roberta_hyperparameter_search
#SBATCH --output=/scratch/xl3139/MLLU/MLLU-HW3/log/Mar06_%j_%x.out
#SBATCH --error=/scratch/xl3139/MLLU/MLLU-HW3/log/Mar06_%j_%x.err
#SBATCH --export=ALL
#SBATCH --mail-user=xl3139@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 4


## TODO: Run your hyperparameter search in a singularity container. Remember to 
## activate your Conda environment in the container before running your Python script.
singularity exec --nv --bind /scratch/xl3139 --overlay /scratch/xl3139/back-sing-nlp.ext3:ro /scratch/xl3139/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate nlp
cd /scratch/xl3139/MLLU/MLLU-HW3
python run_hyperparameter_search.py 'data/BoolQ'
"
