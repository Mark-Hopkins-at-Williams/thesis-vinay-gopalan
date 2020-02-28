export GLUE_DIR=data
export TASK_NAME=SST-3


python3 run_glue.py \
    --model_type bert \
    --model_name_or_path bert-large-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --overwrite_output_dir  \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --weight_decay 0 \
    --adam_epsilon 1e-8 \
    --num_train_epochs 3.0 \
    --output_dir $GLUE_DIR/$TASK_NAME/checks/ \