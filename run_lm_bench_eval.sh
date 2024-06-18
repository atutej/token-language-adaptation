export CUDA_VISIBLE_DEVICES=0
LANG=$1
MODEL=$2
LOG_FILE="./results/lm_eval-$LANG-$MODEL.txt"
echo "Running lm eval on $LANG, $MODEL" |& tee -a $LOG_FILE
lm_eval --model hf --model_args pretrained=$MODEL --tasks mlama_$LANG --device cuda:0 --num_fewshot 0 --batch_size auto:4 |& tee -a $LOG_FILE

#lm_eval --model hf --model_args pretrained=$MODEL --tasks indicsentiment_$LANG,xnli_$LANG,xcopa_$LANG --device cuda:0 --num_fewshot 5 --batch_size auto:4 |& tee -a $LOG_FILE
lm_eval --model hf --model_args pretrained=$MODEL --tasks indicsentiment_$LANG,xnli_$LANG,xstorycloze_$LANG --device cuda:0 --num_fewshot 5 --batch_size auto:4 |& tee -a $LOG_FILE
#lm_eval --model hf --model_args pretrained=$MODEL --tasks indicsentiment_$LANG,xcopa_$LANG --device cuda:0 --num_fewshot 5 --batch_size auto:4 |& tee -a $LOG_FILE
echo "Done." |& tee -a $LOG_FILE 

