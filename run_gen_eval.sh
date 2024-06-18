export CUDA_VISIBLE_DEVICES=0

python scripts/eval_flores_mt.py --lang hin_Deva --langname Hindi --max_tokens 200 --model saved_models/results_Llama2_7b_200k_hi_50kTok --batch_size 4 --shot 5
python scripts/eval_xlsum.py --lang hindi --langname Hindi --max_tokens 50 --model saved_models/results_Llama2_7b_200k_hi_50kTok --batch_size 4 --shot 1



