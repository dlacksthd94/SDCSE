#!/bin/bash

# In this example, we show how to train DCPCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

NUM_GPU=4

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=25901

# Allow multiple threads
export OMP_NUM_THREADS=8

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path roberta-base \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir models/my-unsup-promcse-roberta-base \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-2 \
    --pre_seq_len 14 \
    --do_mlm \
    --num_train_epochs 1 \
    --eval_steps 125 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
