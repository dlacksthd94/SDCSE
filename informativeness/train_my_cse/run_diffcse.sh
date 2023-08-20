#!/bin/bash

ENCODER_NAME='DiffCSE'
ENCODER_NAME_LOWER=$(echo ${ENCODER_NAME} | tr '[:upper:]' '[:lower:]')
cd ../${ENCODER_NAME}
source ~/anaconda3/etc/profile.d/conda.sh
conda activate diffcse
conda env list

declare -A dict_plm=(
    ["bert_base"]="bert-base-uncased"
    ["bert_large"]="bert-large-uncased"
    ["roberta_base"]="roberta-base"
    ["roberta_large"]="roberta-large"
)

declare -A dict_data=(
    ["constituency_parsing"]='../data/backup_1000000/wiki1m_tree_cst_lg_large_subsentence.json'
    ["none"]='data/wiki1m_for_simcse.txt'
    ["dropout"]='data/wiki1m_for_simcse.txt'
    ["mask_token"]='data/wiki1m_for_simcse.txt'
    ["unk_token"]='data/wiki1m_for_simcse.txt'
    ["pad_token"]='data/wiki1m_for_simcse.txt'
)

declare -A dict_lr=(
    ["bert_base"]=7e-6
    # ["bert_large"]=1e-5
    # ["roberta_base"]=1e-5
    # ["roberta_large"]=3e-5
)

declare -A dict_metric=(
    ["stsb"]="stsb_spearman"
    ["sickr"]="sickr_spearman"
    ["sts"]="avg_sts"
    ["transfer"]="avg_transfer --eval_transfer"
)

declare -A dict_pooler=(
    ["ap"]="--pooler_type cls"
    ["wp"]="--pooler_type cls --mlp_only_train"
    ["wop"]="--pooler_type cls_before_pooler"
)

declare -A dict_lambda_w2=(
    ["bert_base"]=5e-3
    # ["bert_large"]=1e-5
    ["roberta_base"]=5e-3
    # ["roberta_large"]=3e-5
)

declare -A dict_mask_ratio=(
    ["bert_base"]=3e-1
    # ["bert_large"]=1e-5
    ["roberta_base"]=3e-1
    # ["roberta_large"]=3e-5
)

RESULT_ROOT_FOLDER='.'
# RESULT_ROOT_FOLDER='/data1/csl'

for PLM in bert_base; do
    for BATCH_SIZE in 128; do
        for LR in ${dict_lr[${PLM}]}; do
            for EPOCH in 2; do
                for SEED in 0; do
                    for MAX_LEN in 32; do
                        for LAMBDA in 0e-0; do
                            for PERTURB_TYPE in none; do
                                for PERTURB_NUM in 0; do
                                    for PERTURB_STEP in 0; do
                                        for LOSS in mse; do
                                            for POOLER in wp; do
                                                for METRIC in stsb; do
                                                    for SIM in 0; do
                                                        for MARGIN in 0e-0; do
                                                            for LAMBDA2 in ${dict_lambda_w2[${PLM}]}; do
                                                                for MASK_RATIO in ${dict_mask_ratio[${PLM}]}; do
                                                                    CUDA_VISIBLE_DEVICES=0,1,2,3 \
                                                                    taskset -c 112-127 \
                                                                    python train.py \
                                                                        --model_name_or_path ${dict_plm[${PLM}]} \
                                                                        --generator_name distilbert-base-uncased \
                                                                        --train_file ${dict_data[${PERTURB_TYPE}]} \
                                                                        --output_dir ${RESULT_ROOT_FOLDER}/result/my-unsup-${ENCODER_NAME_LOWER}-${dict_plm[${PLM}]}_${BATCH_SIZE}_${LR}_${EPOCH}_${SEED}_${MAX_LEN}_${LAMBDA}_${PERTURB_TYPE}_${PERTURB_NUM}_${PERTURB_STEP}_${LOSS}_${POOLER}_${METRIC}_${MARGIN}_${LAMBDA2}_${MASK_RATIO} \
                                                                        --num_train_epochs ${EPOCH} \
                                                                        --per_device_train_batch_size ${BATCH_SIZE} \
                                                                        --learning_rate ${LR} \
                                                                        --max_seq_length ${MAX_LEN} \
                                                                        --evaluation_strategy steps \
                                                                        --metric_for_best_model ${dict_metric[${METRIC}]} \
                                                                        --load_best_model_at_end \
                                                                        --eval_steps 125 \
                                                                        ${dict_pooler[${POOLER}]} \
                                                                        --overwrite_output_dir \
                                                                        --temp 0.05 \
                                                                        --do_train \
                                                                        --do_eval \
                                                                        --batchnorm \
                                                                        --lambda_weight ${LAMBDA2} \
                                                                        --masking_ratio ${MASK_RATIO} \
                                                                        --fp16 \
                                                                        --seed ${SEED} \
                                                                        "$@"
                                                                done
                                                            done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done