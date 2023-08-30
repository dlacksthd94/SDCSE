#!/bin/bash

(
start=`date`

ENCODER_NAME='MixCSE'
ENCODER_NAME_LOWER=$(echo ${ENCODER_NAME} | tr '[:upper:]' '[:lower:]')
cd ../${ENCODER_NAME}

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \

declare -A dict_pooler=(
    ["ap"]="--pooler_type cls"
    ["wp"]="--pooler_type cls --mlp_only_train"
    ["wop"]="--pooler_type cls_before_pooler"
)

declare -A dict_metric=(
    ["stsb"]="stsb_spearman"
    ["sickr"]="sickr_spearman"
    ["sts"]="avg_sts"
    ["transfer"]="avg_transfer --eval_transfer"
)

declare -A dict_plm=(
    ["bert_base"]="bert-base-uncased"
    ["bert_large"]="bert-large-uncased"
    ["roberta_base"]="roberta-base"
    ["roberta_large"]="roberta-large"
)

declare -A dict_lr=(
    ["bert_base"]=3e-5
    ["bert_large"]=1e-5
)

declare -A dict_data=(
    ["none"]='data/wiki1m_for_simcse.txt'
    ["dropout"]='data/wiki1m_for_simcse.txt'
    # ["dropout"]='data/wiki1m_for_simcse_test.txt'
)

RESULT_ROOT_FOLDER='result/backup_eval_dropout_sim0_nocls_1gpu'
# RESULT_ROOT_FOLDER='/data1/csl/SDCSE'

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=4

for PLM in bert_base; do
    for BATCH_SIZE in 64; do
        for LR in ${dict_lr[${PLM}]}; do
            for EPOCH in 1; do
                for SEED in 0; do
                    for MAX_LEN in 32; do
                        for LAMBDA_SDCSE in 1e-2; do
                            for PERTURB_TYPE in dropout; do
                                for PERTURB_NUM in 1; do
                                    for PERTURB_STEP in 2; do
                                        for LOSS in margin; do
                                            for POOLER in wp; do
                                                for METRIC in stsb; do
                                                    for SIM in 0; do
                                                        for MARGIN in 1e-2; do
                                                            for LAMBDA_DIFFCSE in 0e-0; do
                                                                for MASK_RATIO in 0e-0; do
                                                                    for PROMPT_LEN in 0; do
                                                                        CUDA_VISIBLE_DEVICES=0 \
                                                                        python train.py \
                                                                            --model_name_or_path ${dict_plm[${PLM}]} \
                                                                            --train_file ${dict_data[${PERTURB_TYPE}]} \
                                                                            --eval_path SentEval/data/downstream/STS/STSBenchmark/sts-dev.csv \
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
                                                                            --fp16 \
                                                                            --seed ${SEED} \
                                                                            --no_remove_unused_columns \
                                                                            --lambda_weight ${LAMBDA_SDCSE} \
                                                                            --perturbation_type ${PERTURB_TYPE} \
                                                                            --perturbation_num ${PERTURB_NUM} \
                                                                            --perturbation_step ${PERTURB_STEP} \
                                                                            --loss_type ${LOSS} \
                                                                            --num_informative_pair ${SIM} \
                                                                            --margin ${MARGIN} \
                                                                            --lambdas 0.2 \
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

# python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \

#############################

cd ..

declare -A dict_encoder=(
    ["SimCSE"]="simcse"
    ["SDCSE"]="sdcse"
    ["DiffCSE"]="diffcse"
    ["PromCSE"]="promcse"
    ["MixCSE"]="mixcse"
)

declare -A dict_epoch=(
    ["SimCSE"]=1
    ["SDCSE"]=1
    ["DiffCSE"]=2
    ["PromCSE"]=1
    ["MixCSE"]=1
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

list_encoder="SimCSE DiffCSE PromCSE MixCSE"

TASK_SET='full' # sts / transfer / full
MODE='test' # test / dev / fasttest (applied only on transfer tasks)
# ENCODER='SDCSE'
# ENCODER='DiffCSE'
# ENCODER='PromCSE'
ENCODER='MixCSE'
RESULT_ROOT_FOLDER=${ENCODER}/result
# RESULT_ROOT_FOLDER="/data1/csl/${ENCODER}"
# RESULT_FOLDER='backup_eval_token_sim1'
# RESULT_FOLDER='backup_eval_dropout_sim0_nocls'
# RESULT_FOLDER='backup_eval_dropout_sim0_all'
RESULT_FOLDER='backup_eval_dropout_sim0_nocls_1gpu'
# RESULT_FOLDER='backup_eval_dropout_sim0_nocls_sts_1gpu'
# RESULT_FOLDER='backup_eval_dropout_sim0_nocls_sts_nobn_1gpu'
# RESULT_FOLDER='1gpu_125step'
# RESULT_FOLDER='4gpu_32step'

if [ ${ENCODER} = SDCSE ]; then
    declare -A dict_bs=(
        ["bert_base"]=64
        ["bert_large"]=64
        ["roberta_base"]=512
        ["roberta_large"]=512
    )
    declare -A dict_lr=(
        ["bert_base"]=3e-5
        ["bert_large"]=1e-5
        ["roberta_base"]=1e-5
        ["roberta_large"]=3e-5
    )
    declare -A dict_lambda_diffcse=(
        ["bert_base"]=0e-0
    )
    declare -A dict_mask_ratio=(
        ["bert_base"]=0e-0
    )
    declare -A dict_prom_len=(
        ["bert_base"]=0
    )

elif [ ${ENCODER} = DiffCSE ]; then
    declare -A dict_bs=(
        ["bert_base"]=64 # sts
        ["roberta_base"]=64 # sts
        # ["bert_base"]=64 # transfer
        # ["roberta_base"]=128 # transfer
    )
    declare -A dict_lr=(
        ["bert_base"]=7e-6
    )        
    declare -A dict_lambda_diffcse=(
        ["bert_base"]=5e-3 # sts
        # ["bert_base"]=5e-2 # transfer
    )
    declare -A dict_mask_ratio=(
        ["bert_base"]=3e-1 # sts
        # ["bert_base"]=15e-2 # transfer
    )
    declare -A dict_prom_len=(
        ["bert_base"]=0
    )

elif [ ${ENCODER} = PromCSE ]; then
    declare -A dict_bs=(
        ["bert_base"]=256
    )
    declare -A dict_lr=(
        ["bert_base"]=3e-2
    )        
    declare -A dict_lambda_diffcse=(
        ["bert_base"]=0e-0
    )
    declare -A dict_mask_ratio=(
        ["bert_base"]=0e-0
    )
    declare -A dict_prom_len=(
        ["bert_base"]=16
    )

elif [ ${ENCODER} = MixCSE ]; then
    declare -A dict_bs=(
        ["bert_base"]=64
    )
    declare -A dict_lr=(
        ["bert_base"]=3e-5
    )
    declare -A dict_lambda_diffcse=(
        ["bert_base"]=0e-0
    )
    declare -A dict_mask_ratio=(
        ["bert_base"]=0e-0
    )
    declare -A dict_prom_len=(
        ["bert_base"]=0
    )
fi

for training_method in unsup; do
    for plm in bert_base; do
        for batch_size in ${dict_bs[${plm}]}; do
            for lr in ${dict_lr[${plm}]}; do
                for epoch in ${dict_epoch[${ENCODER}]}; do
                    for seed in 0; do
                        for max_len in 32; do
                            for LAMBDA_SDCSE in 1e-2; do
                                for PERTURB_TYPE in dropout; do
                                    for PERTURB_NUM in 1; do
                                        for PERTURB_STEP in 2; do
                                            for LOSS in margin; do
                                                for POOLER in wp; do
                                                    for METRIC in stsb; do
                                                        for MARGIN in 1e-2; do
                                                            for LAMBDA_DIFFCSE in ${dict_lambda_diffcse[${plm}]}; do
                                                                for MASK_RATIO in ${dict_mask_ratio[${plm}]}; do
                                                                    for PROMPT_LEN in ${dict_prom_len[${plm}]}; do
                                                                        GPU_ID=0
                                                                        save_folder="result/evaluation/${dict_encoder[${ENCODER}]}/${RESULT_FOLDER}"
                                                                        file_name=${MODE}_${TASK_SET}_${training_method}_${dict_encoder[${ENCODER}]}_${plm}_${batch_size}_${lr}_${epoch}_${seed}_${max_len}_${LAMBDA_SDCSE}_${PERTURB_TYPE}_${PERTURB_NUM}_${PERTURB_STEP}_${LOSS}_${POOLER}_${METRIC}_${MARGIN}_${LAMBDA_DIFFCSE}_${MASK_RATIO}_${PROMPT_LEN}.txt
                                                                        if [ ! -d ${save_folder} ]; then
                                                                            mkdir -p ${save_folder}
                                                                        fi
                                                                        echo ${training_method} ${ENCODER} ${plm} ${POOLER} ${batch_size} ${lr} ${epoch} ${seed} ${max_len} ${LAMBDA_SDCSE} ${PERTURB_TYPE} ${PERTURB_NUM} ${PERTURB_STEP} ${LOSS} ${POOLER} ${METRIC} ${MARGIN} ${LAMBDA_DIFFCSE} ${MASK_RATIO} ${PROMPT_LEN}
                                                                        if [ ${ENCODER} = PromCSE ]; then
                                                                            # taskset -c 120-127 \
                                                                            python evaluation.py \
                                                                                --model_name_or_path ${RESULT_ROOT_FOLDER}/${RESULT_FOLDER}/my-${training_method}-${dict_encoder[${ENCODER}]}-${dict_plm[${plm}]}_${batch_size}_${lr}_${epoch}_${seed}_${max_len}_${LAMBDA_SDCSE}_${PERTURB_TYPE}_${PERTURB_NUM}_${PERTURB_STEP}_${LOSS}_${POOLER}_${METRIC}_${MARGIN}_${LAMBDA_DIFFCSE}_${MASK_RATIO}_${PROMPT_LEN} \
                                                                                --pooler ${dict_pooler_method[${POOLER}]} \
                                                                                --task_set ${TASK_SET} \
                                                                                --tasks STS12 \
                                                                                --pre_seq_len ${PROMPT_LEN} \
                                                                                --mode ${MODE} \
                                                                                --gpu_id ${GPU_ID} > ${save_folder}/${file_name}
                                                                        elif [ ${ENCODER} = MixCSE ]; then
                                                                            cd ${ENCODER}
                                                                            # taskset -c 120-127 \
                                                                            CUDA_VISIBLE_DEVICES=${GPU_ID} \
                                                                            python evaluation.py \
                                                                                --model_name_or_path result/${RESULT_FOLDER}/my-${training_method}-${dict_encoder[${ENCODER}]}-${dict_plm[${plm}]}_${batch_size}_${lr}_${epoch}_${seed}_${max_len}_${LAMBDA_SDCSE}_${PERTURB_TYPE}_${PERTURB_NUM}_${PERTURB_STEP}_${LOSS}_${POOLER}_${METRIC}_${MARGIN}_${LAMBDA_DIFFCSE}_${MASK_RATIO}_${PROMPT_LEN} \
                                                                                --eval_path SentEval/data/downstream/STS/STSBenchmark/sts-dev.csv \
                                                                                --pooler ${dict_pooler_method[${POOLER}]} \
                                                                                --task_set ${TASK_SET} \
                                                                                --tasks STS12 \
                                                                                --mode ${MODE} > ../${save_folder}/${file_name}
                                                                            cd ..
                                                                        else
                                                                            # taskset -c 120-127 \
                                                                            python evaluation.py \
                                                                                --model_name_or_path ${RESULT_ROOT_FOLDER}/${RESULT_FOLDER}/my-${training_method}-${dict_encoder[${ENCODER}]}-${dict_plm[${plm}]}_${batch_size}_${lr}_${epoch}_${seed}_${max_len}_${LAMBDA_SDCSE}_${PERTURB_TYPE}_${PERTURB_NUM}_${PERTURB_STEP}_${LOSS}_${POOLER}_${METRIC}_${MARGIN}_${LAMBDA_DIFFCSE}_${MASK_RATIO}_${PROMPT_LEN} \
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
done

end=`date`
echo $start
echo $end
)