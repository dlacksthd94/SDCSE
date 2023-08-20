# !/bin/bash

declare -A dict_encoder=(
    ["SimCSE"]="simcse"
    ["DiffCSE"]="diffcse"
    ["PromCSE"]="promcse"
    ["MCSE"]="mcse"
    ["SNCSE"]="sncse"
    ["SDCSE"]="sdcse"
)

declare -A dict_encoder_modelpath=(
    ["SimCSE"]="princeton-nlp/unsup-simcse-bert-base-uncased"
    ["DiffCSE"]="voidism/diffcse-bert-base-uncased-sts"
    ["PromCSE"]="PromCSE/models/my-unsup-promcse-bert-base-uncased"
    ["MCSE"]="MCSE/models/flickr_mcse_bert"
    ["SNCSE"]="SNCSE/SNCSE/models/my-unsup-sncse-bert-base-uncased"
)

declare -A dict_epoch=(
    ["SimCSE"]=1
    ["DiffCSE"]=2
    ["SDCSE"]=1
)

declare -A dict_pooler_method=(
    ["ap"]="cls"
    ["wp"]="cls_before_pooler"
    ["wop"]="cls_before_pooler"
)

declare -A dict_plm=(
    ["bert_base"]="bert-base-uncased"
    ["bert_large"]="bert-large-uncased"
    ["roberta_base"]="roberta-base"
    ["roberta_large"]="roberta-large"
)

# declare -A dict_lr=(
#     ["bert_base"]=3e-5
#     ["bert_large"]=1e-5
#     ["roberta_base"]=1e-5
#     ["roberta_large"]=3e-5
# )

declare -A dict_lr=(
    ["bert_base"]=7e-6
)

list_encoder="SimCSE DiffCSE PromCSE MCSE SNCSE SDCSE"

TASK_SET='full' # sts / transfer / full
MODE='test' # test / dev / fasttest (applied only on transfer tasks)
# ENCODER='SDCSE'
ENCODER='DiffCSE'
RESULT_ROOT_FOLDER=${ENCODER}/result
# RESULT_ROOT_FOLDER='/data1/csl'
# RESULT_FOLDER='backup_eval_token_sim1'
RESULT_FOLDER='backup_eval_dropout_sim0_nocls'
# RESULT_FOLDER='backup_eval_dropout_sim0_all'
# RESULT_FOLDER='1gpu_125step'
# RESULT_FOLDER='4gpu_32step'
GPU_ID=3

for training_method in unsup; do
    for plm in bert_base; do
        for batch_size in 128; do
            for lr in ${dict_lr[${plm}]}; do
                for epoch in ${dict_epoch[${ENCODER}]}; do
                    for seed in 3 4; do
                        for max_len in 32; do
                            for LAMBDA in 0e-0; do
                                for PERTURB_TYPE in none; do
                                    for PERTURB_NUM in 0; do
                                        for PERTURB_STEP in 0; do
                                            for LOSS in mse; do
                                                for POOLER in wp; do
                                                    for METRIC in stsb; do
                                                        for MARGIN in 0e-0; do
                                                            for LAMBDA2 in 5e-3; do
                                                                for MASK_RATIO in 3e-1; do
                                                                    save_folder="result/evaluation/${dict_encoder[${ENCODER}]}/${RESULT_FOLDER}"
                                                                    file_name=${MODE}_${TASK_SET}_${training_method}_${dict_encoder[${ENCODER}]}_${plm}_${batch_size}_${lr}_${epoch}_${seed}_${max_len}_${LAMBDA}_${PERTURB_TYPE}_${PERTURB_NUM}_${PERTURB_STEP}_${LOSS}_${POOLER}_${METRIC}_${MARGIN}_${LAMBDA2}_${MASK_RATIO}.txt
                                                                    if [ ! -d ${save_folder} ]; then
                                                                        mkdir ${save_folder}
                                                                    fi
                                                                    echo ${training_method} ${ENCODER} ${plm} ${POOLER} ${batch_size} ${lr} ${epoch} ${seed} ${max_len} ${LAMBDA} ${PERTURB_TYPE} ${PERTURB_NUM} ${PERTURB_STEP} ${LOSS} ${POOLER} ${METRIC} ${MARGIN} ${LAMBDA2} ${MASK_RATIO}
                                                                    if [ ${ENCODER} = PromCSE ]; then
                                                                        # taskset -c 120-127 \
                                                                        python evaluation.py \
                                                                            --model_name_or_path ${RESULT_ROOT_FOLDER}/${RESULT_FOLDER}/my-${training_method}-${dict_encoder[${ENCODER}]}-${dict_plm[${plm}]}_${batch_size}_${lr}_${epoch}_${seed}_${max_len}_${LAMBDA}_${PERTURB_TYPE}_${PERTURB_NUM}_${PERTURB_STEP}_${LOSS}_${POOLER}_${METRIC}_${MARGIN}_${LAMBDA2}_${MASK_RATIO} \
                                                                            --pooler ${dict_pooler_method[${POOLER}]} \
                                                                            --task_set ${TASK_SET} \
                                                                            --tasks STS12 \
                                                                            --pre_seq_len 16 \
                                                                            --mode ${MODE} \
                                                                            --gpu_id ${GPU_ID} > ${save_folder}/${file_name}
                                                                    else
                                                                        # taskset -c 120-127 \
                                                                        python evaluation.py \
                                                                            --model_name_or_path ${RESULT_ROOT_FOLDER}/${RESULT_FOLDER}/my-${training_method}-${dict_encoder[${ENCODER}]}-${dict_plm[${plm}]}_${batch_size}_${lr}_${epoch}_${seed}_${max_len}_${LAMBDA}_${PERTURB_TYPE}_${PERTURB_NUM}_${PERTURB_STEP}_${LOSS}_${POOLER}_${METRIC}_${MARGIN}_${LAMBDA2}_${MASK_RATIO} \
                                                                            --pooler ${dict_pooler_method[${POOLER}]} \
                                                                            --task_set ${TASK_SET} \
                                                                            --tasks STS12 \
                                                                            --mode ${MODE} \
                                                                            --gpu_id ${GPU_ID} > ${save_folder}/${file_name}
                                                                    fi
                                                                    echo
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
