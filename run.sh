export PYTHONPATH=FastChat
export WANDB_PROJECT="token-language-adaptation"

export BASE_MODEL=meta-llama/Llama-2-7b-hf

#TOKENIZER=$BASE_MODEL
TOKENIZER="tokenizer_models/Llama-2-7b-hf_hi_50000_merged_300000.model"
RESIZE_EMBED_STRATEGY=mean

TRAIN_DATA_PATH="data/mc4_hi_train[:200k]_text.json"
EVAL_DATA_PATH="data/mc4_hi_validation[:10k]_text.json"

NUM_EPOCHS=1
EVAL_STEPS=625
LR=6e-5
BSZ=8

export OUTPUT_DIR=saved_models/Llama2_7b_200k_hi_50kTok

CUDA_LAUNCH_BLOCKING=1 deepspeed --exclude localhost:4 FastChat/fastchat/train/train_mem.py \
  --model_name_or_path $BASE_MODEL \
  --tokenizer_name_or_path $TOKENIZER \
  --resize_embed_strategy $RESIZE_EMBED_STRATEGY \
  --data_path $TRAIN_DATA_PATH \
  --eval_data_path $EVAL_DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs $NUM_EPOCHS \
  --evaluation_strategy "steps" \
  --eval_steps $EVAL_STEPS \
  --save_steps 100000 \
  --save_total_limit 1 \
  --learning_rate $LR \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --bf16 \
  --per_device_train_batch_size $BSZ \
  --deepspeed deepspeed.json \
  --model_max_length 1024 \
  --gradient_checkpointing True \
  --lazy_preprocess True \
  --optim "adamw_bnb_8bit"

