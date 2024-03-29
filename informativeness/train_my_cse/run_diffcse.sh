#!/bin/bash

start=`date`

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
    # ["dropout"]='data/wiki1m_for_simcse_test.txt'
    ["mask_token"]='data/wiki1m_for_simcse.txt'
    ["unk_token"]='data/wiki1m_for_simcse.txt'
    ["pad_token"]='data/wiki1m_for_simcse.txt'
)

declare -A dict_lr=(
    ["bert_base"]=7e-6 # sts
    # ["bert_base"]=2e-6 # transfer
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

declare -A dict_lambda_diffcse=(
    ["bert_base"]=5e-3 # sts
    # ["bert_base"]=5e-2 # transfer
)

declare -A dict_mask_ratio=(
    ["bert_base"]=3e-1 # sts
    # ["bert_base"]=15e-2 # transfer
)

RESULT_ROOT_FOLDER='result'
# RESULT_ROOT_FOLDER='/data1/csl/DiffCSE'

for PLM in bert_base; do
    for BATCH_SIZE in 64; do
        for LR in ${dict_lr[${PLM}]}; do
            for EPOCH in 2; do
                for SEED in 0; do
                    for MAX_LEN in 32; do
                        for LAMBDA_SDCSE in 1e-1; do
                            for PERTURB_TYPE in dropout; do
                                for PERTURB_NUM in 1; do
                                    for PERTURB_STEP in 2; do
                                        for LOSS in margin; do
                                            for POOLER in wp; do
                                                for METRIC in stsb; do
                                                    for SIM in 0; do
                                                        for MARGIN in 1e-1; do
                                                            for LAMBDA_DIFFCSE in ${dict_lambda_diffcse[${PLM}]}; do
                                                                for MASK_RATIO in ${dict_mask_ratio[${PLM}]}; do
                                                                    for PROMPT_LEN in 0; do
                                                                        CUDA_VISIBLE_DEVICES=1 \
                                                                        python train.py \
                                                                            --model_name_or_path ${dict_plm[${PLM}]} \
                                                                            --generator_name distilbert-base-uncased \
                                                                            --train_file ${dict_data[${PERTURB_TYPE}]} \
                                                                            --output_dir ${RESULT_ROOT_FOLDER}/my-unsup-${ENCODER_NAME_LOWER}-${dict_plm[${PLM}]}_${BATCH_SIZE}_${LR}_${EPOCH}_${SEED}_${MAX_LEN}_${LAMBDA_SDCSE}_${PERTURB_TYPE}_${PERTURB_NUM}_${PERTURB_STEP}_${LOSS}_${POOLER}_${METRIC}_${MARGIN}_${LAMBDA_DIFFCSE}_${MASK_RATIO}_${PROMPT_LEN} \
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
                                                                            --lambda_weight_diffcse ${LAMBDA_DIFFCSE} \
                                                                            --masking_ratio ${MASK_RATIO} \
                                                                            --lambda_weight_sdcse ${LAMBDA_SDCSE} \
                                                                            --perturbation_type ${PERTURB_TYPE} \
                                                                            --perturbation_num ${PERTURB_NUM} \
                                                                            --perturbation_step ${PERTURB_STEP} \
                                                                            --loss_type ${LOSS} \
                                                                            --num_informative_pair ${SIM} \
                                                                            --margin ${MARGIN} \
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
done

# taskset -c 112-127 \

end=`date`
echo $start
echo $end