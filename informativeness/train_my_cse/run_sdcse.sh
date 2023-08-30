#!/bin/bash

start=`date`

ENCODER_NAME='SDCSE'
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
    ["roberta_base"]=1e-5
    ["roberta_large"]=3e-5
)

declare -A dict_data=(
    ["constituency_parsing"]='../data/backup_1000000/wiki1m_tree_cst_lg_large_subsentence.json'
    ["none"]='data/wiki1m_for_simcse.txt'
    ["dropout"]='data/wiki1m_for_simcse.txt'
    ["mask_token"]='data/wiki1m_for_simcse.txt'
    ["unk_token"]='data/wiki1m_for_simcse.txt'
    ["pad_token"]='data/wiki1m_for_simcse.txt'
)

RESULT_ROOT_FOLDER='result'
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
                                                        for MARGIN in 1e-1; do
                                                            for LAMBDA_DIFFCSE in 0e-0; do
                                                                for MASK_RATIO in 0e-0; do
                                                                    for PROMPT_LEN in 0; do
                                                                        CUDA_VISIBLE_DEVICES=0 \
                                                                        python train.py \
                                                                            --model_name_or_path ${dict_plm[${PLM}]} \
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

end=`date`
echo $start
echo $end