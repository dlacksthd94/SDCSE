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

declare -A dict_pooler_method=(
    ["wp"]="cls_before_pooler"
    ["wop"]="cls_before_pooler"
)

list_encoder="SimCSE DiffCSE PromCSE MCSE SNCSE SDCSE"

# for encoder in ${list_encoder}; do
#     cd ${ENCODER}
#     for training_method in unsup; do
#         for plm in bert; do
#             for pooler_method in wp wop; do
#                 file_name=result_${training_method}_${dict_encoder[${ENCODER}]}_${plm}_${pooler_method}.txt
#                 echo ${training_method} ${ENCODER} ${plm} ${pooler_method}
#                 if [ ${ENCODER} = PromCSE ]; then
#                     python evaluation.py \
#                         --model_name_or_path ${dict_encoder_modelpath[${ENCODER}]} \
#                         --pooler ${dict_pooler_method[${pooler_method}]} \
#                         --task_set ${TASK_SET} \
#                         --pre_seq_len 16 \
#                         --mode ${MODE} > "../result/${file_name}"
#                 else
#                     python evaluation.py \
#                         --model_name_or_path ${dict_encoder_modelpath[${ENCODER}]} \
#                         --pooler ${dict_pooler_method[${pooler_method}]} \
#                         --task_set ${TASK_SET} \
#                         --mode ${MODE} > "../result/${file_name}"
#                 fi
#                 echo
#             done
#         done
#     done
#     cd ..
# done

# encoder=MCSE
# cd ${ENCODER}
#     for training_method in unsup; do
#         for plm in bert; do
#             for pooler_method in wp wop; do
#                 file_name=result_${training_method}_${dict_encoder[${ENCODER}]}_${plm}_${pooler_method}.txt
#                 echo ${training_method} ${ENCODER} ${plm} ${pooler_method}
#                 if [ ${ENCODER} = PromCSE ]; then
#                     python evaluation.py \
#                         --model_name_or_path ${dict_encoder_modelpath[${ENCODER}]} \
#                         --pooler ${dict_pooler_method[${pooler_method}]} \
#                         --task_set full \
#                         --pre_seq_len 16 \
#                         --mode ${MODE} > "../result/${file_name}"
#                 else
#                     python evaluation.py \
#                         --model_name_or_path ${dict_encoder_modelpath[${ENCODER}]} \
#                         --pooler ${dict_pooler_method[${pooler_method}]} \
#                         --task_set full \
#                         --mode ${MODE} > "../result/${file_name}"
#                 fi
#                 echo
#             done
#         done
#     done
#     cd ..

TASK_SET='full' # sts / transfer / full
MODE='test' # test / dev / fasttest (applied only on transfer tasks)
RESULT_FOLDER='backup_eval_token_sim2'
ENCODER='SDCSE'
GPU_ID=2

for training_method in unsup; do
    for plm in bert; do
        for batch_size in 64; do
            for lr in 3e-5; do
                for epoch in 1; do
                    for seed in 1; do
                        for max_len in 32; do
                            for lambda_weight in 1e-0; do
                                for PERTURB_TYPE in mask_token; do
                                    for PERTURB_NUM in 1; do
                                        for PERTURB_STEP in 2; do
                                            for LOSS in mse; do
                                                for POOLER in wp; do
                                                    for METRIC in stsb; do
                                                        file_name=${MODE}_${training_method}_${dict_encoder[${ENCODER}]}_${plm}_${batch_size}_${lr}_${epoch}_${seed}_${max_len}_${lambda_weight}_${PERTURB_TYPE}_${PERTURB_NUM}_${PERTURB_STEP}_${LOSS}_${POOLER}_${METRIC}.txt
                                                        save_folder="result/evaluation/${dict_encoder[${ENCODER}]}/${RESULT_FOLDER}"
                                                        if [ ! -d ${save_folder} ]; then
                                                            mkdir ${save_folder}
                                                        fi
                                                        echo ${training_method} ${ENCODER} ${plm} ${POOLER} ${batch_size} ${lr} ${epoch} ${seed} ${max_len} ${lambda_weight} ${PERTURB_TYPE} ${PERTURB_NUM} ${PERTURB_STEP} ${LOSS} ${POOLER} ${METRIC}
                                                        if [ ${ENCODER} = PromCSE ]; then
                                                            # taskset -c 120-127 \
                                                            python evaluation.py \
                                                                --model_name_or_path ${ENCODER}/result/${RESULT_FOLDER}/my-${training_method}-${dict_encoder[${ENCODER}]}-${plm}-base-uncased_${batch_size}_${lr}_${epoch}_${seed}_${max_len}_${lambda_weight}_${PERTURB_TYPE}_${PERTURB_NUM}_${PERTURB_STEP}_${LOSS}_${POOLER}_${METRIC} \
                                                                --pooler ${dict_pooler_method[${POOLER}]} \
                                                                --task_set ${TASK_SET} \
                                                                --tasks STS12 \
                                                                --pre_seq_len 16 \
                                                                --mode ${MODE} \
                                                                --gpu_id ${GPU_ID} > ${save_folder}/${file_name}
                                                        else
                                                            # taskset -c 120-127 \
                                                            python evaluation.py \
                                                                --model_name_or_path ${ENCODER}/result/${RESULT_FOLDER}/my-${training_method}-${dict_encoder[${ENCODER}]}-${plm}-base-uncased_${batch_size}_${lr}_${epoch}_${seed}_${max_len}_${lambda_weight}_${PERTURB_TYPE}_${PERTURB_NUM}_${PERTURB_STEP}_${LOSS}_${POOLER}_${METRIC} \
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

# TASK_SET='full' # sts / transfer / full
# MODE='test' # test / dev / fasttest (applied only on transfer tasks)
# RESULT_FOLDER='backup'
# ENCODER='SimCSE'
# GPU_ID=3

# for training_method in unsup; do
#     for plm in bert; do
#         for batch_size in 64; do
#             for lr in 3e-5; do
#                 for epoch in 1; do
#                     for seed in 0 1 2 3 4; do
#                         for max_len in 32; do
#                             for POOLER in wp wop; do
#                                 for METRIC in stsb; do
#                                     file_name=result_${training_method}_${dict_encoder[${ENCODER}]}_${plm}_${batch_size}_${lr}_${epoch}_${seed}_${max_len}_${POOLER}_${METRIC}.txt
#                                     save_folder="result/evaluation/${dict_encoder[${ENCODER}]}/${RESULT_FOLDER}"
#                                     if [ ! -d ${save_folder} ]; then
#                                         mkdir ${save_folder}
#                                     fi
#                                     echo ${training_method} ${ENCODER} ${plm} ${POOLER} ${batch_size} ${lr} ${epoch} ${seed} ${max_len} ${POOLER} ${METRIC}
#                                     if [ ${ENCODER} = PromCSE ]; then
#                                         taskset -c 120-127 \
#                                         python evaluation.py \
#                                             --model_name_or_path ${ENCODER}/result/${RESULT_FOLDER}/my-${training_method}-${dict_encoder[${ENCODER}]}-${plm}-base-uncased_${batch_size}_${lr}_${epoch}_${seed}_${max_len}_${POOLER}_${METRIC} \
#                                             --pooler ${dict_pooler_method[${POOLER}]} \
#                                             --task_set ${TASK_SET} \
#                                             --tasks STS12 \
#                                             --pre_seq_len 16 \
#                                             --mode ${MODE} \
#                                             --gpu_id ${GPU_ID} > ${save_folder}/${file_name}
#                                     else
#                                         taskset -c 120-127 \
#                                         python evaluation.py \
#                                             --model_name_or_path ${ENCODER}/result/${RESULT_FOLDER}/my-${training_method}-${dict_encoder[${ENCODER}]}-${plm}-base-uncased_${batch_size}_${lr}_${epoch}_${seed}_${max_len}_${POOLER}_${METRIC} \
#                                             --pooler ${dict_pooler_method[${POOLER}]} \
#                                             --task_set ${TASK_SET} \
#                                             --tasks STS12 \
#                                             --mode ${MODE} \
#                                             --gpu_id ${GPU_ID} > ${save_folder}/${file_name}
#                                     fi
#                                     echo
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done
