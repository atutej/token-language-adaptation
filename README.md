Language-Specific LLM Adaptation
==================================================

## 1. Overview

This repository contains the codebase for the paper ["Exploring Design Choices for Building Language-Specific LLMs"](https://arxiv.org/abs/2406.14670).

## 2. Setup
### 2.2 Installation
```bash
virtualenv -p [PATH to python3.10 binary] tla
source tla/bin/activate
pip install -r requirements.txt
cd lm-evaluation-harness
pip install -e .
cd ../
```

### 2.3 Preprocessing

The ```scripts/``` folder contains preprocessing scripts to generate language-speecific vocabularies and CPT data. 

##### 2.3.1 Generating CPT Data

The following example command generates 200000 training and 10000 validation examples from the [mC4](https://huggingface.co/datasets/allenai/c4) corpus for Hindi and saves into the ```data/``` folder. Language ISO codes for other languages: (ar, tr, ta). Codes for all other languages from the mC4 dataset can be found [here](https://www.semanticscholar.org/paper/mT5%3A-A-Massively-Multilingual-Pre-trained-Xue-Constant/74276a37bfa50f90dfae37f767b2b67784bd402a/figure/5).
```
python create_mc4_data.py --lang hi --data_size 200000 --split train
python create_mc4_data.py --lang hi --data_size 10000 --split validation
```

##### 2.3.1 Generating Augmented Vocabularies

For models that use [SentencePiece](https://github.com/google/sentencepiece) (Llama-2, Mistral, Gemma, XGLM, etc.), the following script generates the new vocabularies and merges them with that of the original model:
```
python augment_tokenizers.py --vocab_size 50000 --train_from_scratch True --lang "hi" --model_base meta-llama/Llama-2-7b-hf --dataset_size 300000
```
For models that use the [Tokenizers](https://github.com/huggingface/tokenizers) library (Bloom, Phi-2), the following script generates the new vocabularies and merges them with that of the original model:
```
python augment_df_tokenizers.py --vocab_size 50000 --lang "tr" --model_base bloom --dataset_size 300000
```

## 3. Running Experiments

### 3.1 Training

```run.sh``` contains an example scropts that trains LLaMA-2-7B on Hindi (hi), with augmented vocabulary of 50K tokens on 200K examples. The model will be saved at ```saved_models/```.

### 3.2 Evaluation
See ```run_gen_eval.sh``` for an example to evaluation on generation tasks (Flores, XL-Sum). ```run_lm_bench_eval.sh``` runs evluations on NLU benchmarks. Example on Hindi (hi):
```
run_lm_bench_eval.sh hi saved_models/results_Llama2_7b_200k_hi_50kTok
```

##### Some of the code was forked from the following repositories
* [FastChat](https://github.com/lm-sys/FastChat)
* [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
* [focus](https://github.com/konstantinjdobler/focus)

## Cite

If our work was helpful in your research, please kindly cite us as follows:
```
@misc{tejaswi2024exploring,
      title={Exploring Design Choices for Building Language-Specific LLMs}, 
      author={Atula Tejaswi and Nilesh Gupta and Eunsol Choi},
      year={2024},
      eprint={2406.14670},
      archivePrefix={arXiv}
}
```
 
## References
[1] Xue, Linting, et al. "mT5: A massively multilingual pre-trained text-to-text transformer." arXiv preprint arXiv:2010.11934 (2020).

[2] Kudo, Taku, and John Richardson. "Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing." arXiv preprint arXiv:1808.06226 (2018).

[3] Dobler, Konstantin, and Gerard De Melo. "FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models." The 2023 Conference on Empirical Methods in Natural Language Processing. 2023.