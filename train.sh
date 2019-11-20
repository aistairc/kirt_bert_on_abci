#!/bin/bash

#$ -l rt_G.large=1
#$ -l h_rt=72:00:00
#$ -t 1-2
#$ -j y
#$ -cwd
#$ -m ea
#$ -M abci@virtualemail.info

source /etc/profile.d/modules.sh

module load openmpi/3.1.3
module load python/3.6/3.6.5
module load cuda/10.0/10.0.130
module load cudnn/7.5/7.5.0
module load nccl/2.4/2.4.2-1
module load singularity/2.6.1

export PYTHONUNBUFFERED=1

CACHE_DIR=".cache"

echo "Task ID is $SGE_TASK_ID"

NUM_NODES=2
NUM_GPUS_PER_NODE=4

ARGS="train_on_pregenerated.py --pregenerated_data=data/generated/epochs/ --bert_model=data/generated/abci_bert_base/ --output_dir=data/generated/abci_bert_base/model/ --train_batch_size=8 --epochs=5 --learning_rate=1e-4 --optimizer=RADAM --large_train_data --fp16 --save_checkpoint_steps=5"

echo "Training..."

singularity run --nv ~/pytorch-19.06-py3.simg python prepare_args.py $CACHE_DIR $NUM_NODES $NUM_GPUS_PER_NODE $SGE_TASK_ID $ARGS
