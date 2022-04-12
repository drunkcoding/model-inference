#!/bin/bash
# Author(s): James Owers (james.f.owers@gmail.com)
#
# example usage:
# ```
# EXPT_FILE=experiments.txt  # <- this has a command to run on each line
# NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
# MAX_PARALLEL_JOBS=12
# sbatch --array=1-${NR_EXPTS}%${MAX_PARALLEL_JOBS} slurm_arrayjob.sh $EXPT_FILE
# ```
#
# or, equivalently and as intended, with provided `run_experiement`:
# ```
# run_experiment -b slurm_arrayjob.sh -e experiments.txt -m 12
# ```

#SBATCH --mail-user=leyang.xue@ed.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --partition=small

# ====================
# Options for sbatch
# ====================
# Maximum number of nodes to use for the job
# #SBATCH --nodes=1

# Megabytes of RAM required. Check `cluster-status` for node configurations
# #SBATCH --mem=32000

# Number of CPUs to use. Check `cluster-status` for node configurations
# #SBATCH --cpus-per-task=8

# Maximum time for the job to run, format: days-hours:minutes:seconds
# #SBATCH --time=1-00:00:00

# #SBATCH --gres=gpu:2
# =====================
# Logging information
# =====================
# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Environment setup
# ===================
echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
# source ~/.bashrc
module load python/anaconda3
module load gcc
source activate torch

# Make script bail out after first error
set -e

# ==============================
# Finally, run the experiment!
# ==============================
# Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# inclusive.

# experiment_text_file=$1
# COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${experiment_text_file}`"

#source /etc/profile.d/modules.sh
#module load cuda


cd /jmain02/home/J2AD002/jxm12/lxx22-jxm12/model-inference

# DEPLOY=( "t5_sst2_S" "t5_sst2_M" "t5_sst2_L" "t5_sst2_XL" "t5_sst2_S_r2" "t5_sst2_M_r2" "t5_sst2_L_r2" "t5_sst2_XL_r2" )
# TAGS=( "ray-S" "ray-M" "ray-L" "ray-XL" "ray-S-R2" "ray-M-R2" "ray-L-R2" "ray-XL-R2" )

DEPLOY=( "t5_sst2_S_r2" "t5_sst2_M_r2" "t5_sst2_L_r2" "t5_sst2_XL_r2" )
TAGS=( "ray-S-R2" "ray-M-R2" "ray-L-R2" "ray-XL-R2" )

for i in ${!TAGS[@]}; do
    echo "$i, ${TAGS[$i]}, ${DEPLOY[$i]}"
    bash ray/run_serve.sh ${DEPLOY[$i]}
    sleep 15m
    ${HOME}/.conda/envs/torch/bin/python tests/test_cost_model.py --dataset_name glue --task_name sst2 --model_name_or_path ~/HuggingFace/google/t5-small-lm-adapt/ --tag ${TAGS[$i]}
    bash ray/kill_serve.sh
    sleep 1m
done

# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
