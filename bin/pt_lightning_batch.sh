#!/bin/sh
#SBATCH -J pytorch_lightning # job name
#SBATCH --time=24:00:00 # walltime 
#SBATCH --comment=pytorch # application name
#SBATCH -p amd_a100nv_8 # partition name (queue or class)
#SBATCH --nodes=2 # the number of nodes
#SBATCH --ntasks-per-node=2 # number of tasks per node
#SBATCH --gres=gpu:2 # number of GPUs per node
#SBATCH --cpus-per-task=4 # number of cpus per task
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load gcc/10.2.0 cuda/11.7
source ~/.bashrc
conda activate lightning

# The num_nodes argument should be specified to be the same number as in the #SBATCH --nodes=xxx  
srun python /scratch/qualis/git-projects/distributed-training-on-supercomputer-with-pytorch-lightning/src/pt_bert_nsmc_lightning.py --num_nodes 2 
