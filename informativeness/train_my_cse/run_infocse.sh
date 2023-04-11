#!/bin/bash

cd ../InfoCSE

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \

for BATCH_SIZE in 64 128 256; do
    for LR in 1e-4 1e-5 1e-6; do
        for EPOCH in 1 2; do
            taskset -c 120-127 \
            python train.py \
                --model_name_or_path bert-base-uncased \
                --train_file data/wiki1m_for_simcse.txt \
                --output_dir result/my-unsup-infocse-bert-base-uncased_${BATCH_SIZE}_${LR} \
                --num_train_epochs ${EPOCH} \
                --per_device_train_batch_size ${BATCH_SIZE} \
                --learning_rate ${LR} \
                --max_seq_length 32 \
                --evaluation_strategy steps \
                --metric_for_best_model stsb_spearman \
                --load_best_model_at_end \
                --eval_steps `expr 128 \* 64 / ${BATCH_SIZE}` \
                --pooler_type cls_before_pooler \
                --overwrite_output_dir \
                --temp 0.05 \
                --do_train \
                --do_eval \
                --fp16 \
                --seed 0 \
                "$@"
        done
    done
done