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
    ["wp"]="cls"
    ["wop"]="cls_before_pooler"
)

list_encoder="SimCSE DiffCSE PromCSE MCSE SNCSE SDCSE"

# for encoder in ${list_encoder}; do
#     cd ${encoder}
#     for training_method in unsup; do
#         for plm in bert; do
#             for pooler_method in wp wop; do
#                 file_name=result_${training_method}_${dict_encoder[${encoder}]}_${plm}_${pooler_method}.txt
#                 echo ${training_method} ${encoder} ${plm} ${pooler_method}
#                 if [ ${encoder} = PromCSE ]; then
#                     python evaluation.py \
#                         --model_name_or_path ${dict_encoder_modelpath[${encoder}]} \
#                         --pooler ${dict_pooler_method[${pooler_method}]} \
#                         --task_set ${TASK_SET} \
#                         --pre_seq_len 16 \
#                         --mode ${MODE} > "../result/${file_name}"
#                 else
#                     python evaluation.py \
#                         --model_name_or_path ${dict_encoder_modelpath[${encoder}]} \
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
# cd ${encoder}
#     for training_method in unsup; do
#         for plm in bert; do
#             for pooler_method in wp wop; do
#                 file_name=result_${training_method}_${dict_encoder[${encoder}]}_${plm}_${pooler_method}.txt
#                 echo ${training_method} ${encoder} ${plm} ${pooler_method}
#                 if [ ${encoder} = PromCSE ]; then
#                     python evaluation.py \
#                         --model_name_or_path ${dict_encoder_modelpath[${encoder}]} \
#                         --pooler ${dict_pooler_method[${pooler_method}]} \
#                         --task_set full \
#                         --pre_seq_len 16 \
#                         --mode ${MODE} > "../result/${file_name}"
#                 else
#                     python evaluation.py \
#                         --model_name_or_path ${dict_encoder_modelpath[${encoder}]} \
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
MODE='fasttest' # test / dev / fasttest (applied only on transfer tasks)
RESULT_FOLDER='backup_diff_seed_eval_only'

encoder=SDCSE

for training_method in unsup; do
    for plm in bert; do
        for pooler_method in wop; do
            for batch_size in 256; do
                for lr in 1e-4; do
                    for epoch in 1; do
                        for seed in 1 2; do
                            for max_len in 32; do
                                for lambda_weight in 0e-0 1e-0 5e-1 1e-1 5e-2 1e-2; do
                                    file_name=result_${training_method}_${dict_encoder[${encoder}]}_${plm}_${pooler_method}_${batch_size}_${lr}_${epoch}_${seed}_${max_len}_${lambda_weight}.txt
                                    save_folder="result/evaluation/${dict_encoder[${encoder}]}/${RESULT_FOLDER}"
                                    if [ ! -d ${save_folder} ]; then
                                        mkdir ${save_folder}
                                    fi
                                    echo ${training_method} ${encoder} ${plm} ${pooler_method} ${batch_size} ${lr} ${epoch} ${seed} ${max_len} ${lambda_weight}
                                    if [ ${encoder} = PromCSE ]; then
                                        taskset -c 120-127 \
                                        python evaluation.py \
                                            --model_name_or_path ${encoder}/result/${RESULT_FOLDER}/my-${training_method}-${dict_encoder[${encoder}]}-${plm}-base-uncased_${batch_size}_${lr}_${epoch}_${seed}_${max_len}_${lambda_weight} \
                                            --pooler ${dict_pooler_method[${pooler_method}]} \
                                            --task_set ${TASK_SET} \
                                            --tasks STS12 \
                                            --pre_seq_len 16 \
                                            --mode ${MODE} > ${save_folder}/${file_name}
                                    else
                                        taskset -c 120-127 \
                                        python evaluation.py \
                                            --model_name_or_path ${encoder}/result/${RESULT_FOLDER}/my-${training_method}-${dict_encoder[${encoder}]}-${plm}-base-uncased_${batch_size}_${lr}_${epoch}_${seed}_${max_len}_${lambda_weight} \
                                            --pooler ${dict_pooler_method[${pooler_method}]} \
                                            --task_set ${TASK_SET} \
                                            --tasks STS12 \
                                            --mode ${MODE} > ${save_folder}/${file_name}
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
