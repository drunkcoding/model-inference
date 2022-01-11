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
#SBATCH --gres=gpu:8
#SBATCH --mincpus=30
#SBATCH --partition=big

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
source ~/.bashrc

# Make script bail out after first error
set -e

task_name=$1
base_dir=$2
batch_size=2
learning_rate=$3

deepspeed t5train/run_glue_deepspeed_seq2seq.py \
    --deepspeed deepspeed_cfg_auto.json \
    --model_name_or_path  ${base_dir}/${model_name} \
    --task_name ${task_name} \
    --dataset_name glue \
    --max_seq_length 128 \
    --do_eval \
    --per_device_train_batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --num_train_epochs 10 \
    --save_strategy steps \
    --logging_strategy steps \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --save_steps 40 \
    --gradient_accumulation_steps 8 \
    --logging_steps 40 \
    --save_total_limit 5 \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --overwrite_output_dir \
    --output_dir ./outputs/${model_name}/${task_name}/ \
    &> ./log/${model_name}/${task_name}/glue_bsz${batch_size}_lr${learning_rate}.log


# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"

