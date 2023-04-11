# python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
#!/bin/bash

# In this example, we show how to train PromCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=4

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=25901

# Allow multiple threads
export OMP_NUM_THREADS=8

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/nli_for_simcse.csv \
    --output_dir models/my-sup-promcse-bert-base-uncased \
    --per_device_train_batch_size 256 \
    --learning_rate 1e-2 \
    --pre_seq_len 12 \
    --num_train_epochs 10 \
    --eval_steps 125 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_eh_loss \
    --eh_loss_margin 0.2 \
    --eh_loss_weight 10 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
