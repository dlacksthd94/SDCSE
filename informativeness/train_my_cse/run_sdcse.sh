#!/bin/bash

cd ../SDCSE

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

for PLM in bert_base; do
    for BATCH_SIZE in 64; do
        for LR in ${dict_lr[${PLM}]}; do
            for EPOCH in 1; do
                for SEED in 0 1 2 3 4; do
                    for MAX_LEN in 32; do
                        for LAMBDA in 0e-0; do
                            for PERTURB_TYPE in none; do
                                for PERTURB_NUM in 0; do
                                    for PERTURB_STEP in 0; do
                                        for LOSS in mse; do
                                            for POOLER in ap; do
                                                for METRIC in stsb; do
                                                    for SIM in 1; do
                                                        for MARGIN in 0e-0; do
                                                            taskset -c 120-127 \
                                                            python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
                                                                --model_name_or_path ${dict_plm[${PLM}]} \
                                                                --train_file data/wiki1m_for_simcse.txt \
                                                                --output_dir result/my-unsup-sdcse-${dict_plm[${PLM}]}_${BATCH_SIZE}_${LR}_${EPOCH}_${SEED}_${MAX_LEN}_${LAMBDA}_${PERTURB_TYPE}_${PERTURB_NUM}_${PERTURB_STEP}_${LOSS}_${POOLER}_${METRIC}_${MARGIN} \
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
                                                                --lambda_weight ${LAMBDA} \
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