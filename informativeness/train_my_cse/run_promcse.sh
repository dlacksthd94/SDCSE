#!/bin/bash

cd ../PromCSE

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=4

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \

for BATCH_SIZE in 64 128; do
    for LR in 1e-3 1e-4 1e-5; do
        for PROMPT_LEN in 10 12 14 16; do
            taskset -c 120-127 \
            python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
                --model_name_or_path bert-base-uncased \
                --train_file data/wiki1m_for_simcse.txt \
                --output_dir result/my-unsup-promcse-bert-base-uncased_${BATCH_SIZE}_${LR}_${PROMPT_LEN} \
                --num_train_epochs 1 \
                --per_device_train_batch_size ${BATCH_SIZE} \
                --learning_rate ${LR} \
                --max_seq_length 32 \
                --evaluation_strategy steps \
                --metric_for_best_model stsb_spearman \
                --load_best_model_at_end \
                --eval_steps `expr 128 \* 64 / ${BATCH_SIZE}` \
                --pooler_type cls_before_pooler \
                --pre_seq_len ${PROMPT_LEN} \
                --overwrite_output_dir \
                --temp 0.05 \
                --do_train \
                --do_eval \
                --do_mlm
                --fp16 \
                --seed 0 \
                "$@"
        done
    done
done