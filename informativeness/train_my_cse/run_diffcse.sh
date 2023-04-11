#!/bin/bash

cd ../DiffCSE
source ~/anaconda3/etc/profile.d/conda.sh
conda activate diffcse
conda env list

for BATCH_SIZE in 64; do
    for LR in 1e-4; do
        for EPOCH in 1; do
            for SEED in 0; do
                taskset -c 120-127 \
                python train.py \
                    --model_name_or_path bert-base-uncased \
                    --generator_name distilbert-base-uncased \
                    --train_file data/wiki1m_for_simcse.txt \
                    --output_dir result/my-unsup-diffcse-bert-base-uncased_${BATCH_SIZE}_${LR}_${EPOCH}_${SEED} \
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
                    --lambda_weight 0.005 \
                    --masking_ratio 0.30 \
                    --fp16 \
                    --seed ${SEED} \
                    "$@"
            done
        done
    done
done