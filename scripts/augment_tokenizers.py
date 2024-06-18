from transformers import AutoTokenizer, LlamaTokenizer, GemmaTokenizer, XGLMTokenizer
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
import sentencepiece.sentencepiece_model_pb2 as model
import numpy as np

import sentencepiece as spm
import tempfile
from datasets import load_dataset
import itertools
import argparse
import glob
from tqdm import tqdm
import random
import spacy
from collections import OrderedDict
from ftlangdetect import detect

parser = argparse.ArgumentParser(description='Merge Tokenizers')
parser.add_argument('--vocab_size', default=10000, type=int)
parser.add_argument('--train_from_scratch', default=True, type=bool)
parser.add_argument('--lang', default="hi", type=str)
parser.add_argument('--model_base', default="meta-llama/Llama-2-7b-hf", type=str)
parser.add_argument('--cache_dir', default="./cache", type=str)
parser.add_argument('--dataset_size', default=300000, type=int)
args = parser.parse_args()

TRAIN_FROM_SCRATCH = args.train_from_scratch
VOCAB_SIZE = args.vocab_size
CACHE_DIR = args.cache_dir
languages_base = [args.lang]
model_base = args.model_base

if not os.path.exists("tokenizer_models"):
    os.makedirs("tokenizer_models")


m = model.ModelProto()
t = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

with tempfile.TemporaryDirectory() as tmpdirname:
    t.save_pretrained(tmpdirname)
    
    model_files = glob.glob(os.path.join(tmpdirname, '*.model'))
    if not model_files:
        raise FileNotFoundError("No .model file found in the tokenizer directory.")
    
    tokenizer_model_path = model_files[0]
    with open(tokenizer_model_path, 'rb') as f:
        tokenizer_bytes = f.read()
    
    m.ParseFromString(tokenizer_bytes)


if TRAIN_FROM_SCRATCH:
    for lang in languages_base:
        dataset = load_dataset("mc4", lang, split="train", streaming=True)
        dataset = dataset.take(args.dataset_size)

        nlp = spacy.blank(lang)
        nlp.add_pipe('sentencizer')

        text_list = [i['text'] for i in dataset]
        random.shuffle(text_list)

        with tempfile.NamedTemporaryFile(delete=True, mode='w', encoding='utf-8') as temp_file:
            for text in tqdm(text_list):
                doc = nlp(text)
                for sent in doc.sents:
                    try:
                        if detect(text=sent.text.replace("\n"," "), low_memory=True)["lang"] == lang:
                            temp_file.write(sent.text + '\n')
                            num_bytes = len(sent.text.encode('utf-8'))
                    except KeyboardInterrupt:
                        exit()
                    except Exception as e:
                        print(e)
                        continue

            if not os.path.exists("./tokenizer_models/sp_models"):
                os.makedirs("./tokenizer_models/sp_models")
            
            spm.SentencePieceTrainer.train(input=temp_file.name, model_prefix='./tokenizer_models/sp_models/tokenizer_' + lang + "_" + str(VOCAB_SIZE), vocab_size=VOCAB_SIZE, byte_fallback=True, model_type="bpe", normalization_rule_name="identity", character_coverage=0.98, split_by_whitespace=True, accept_language=lang)
        temp_file.close()

VOCAB_SIZE = str(VOCAB_SIZE)

res = {}
for languages in list(itertools.permutations(languages_base)):
    print(languages)
    for idx, lang in enumerate(languages):
        res[str(languages)] = {}
        if idx==0:
            base_tokenizer = AutoTokenizer.from_pretrained(model_base)

        else:
            if model_base == "meta-llama/Llama-2-7b-hf":
                base_tokenizer = LlamaTokenizer("models/Llama-2-7b-hf_merged.model", legacy=True)
            elif model_base == "mistralai/Mistral-7B-v0.1":
                base_tokenizer = LlamaTokenizer("models/mistral-7b_merged.model", legacy=True)
            elif model_base == "facebook/xglm-7.5B": 
                base_tokenizer = XGLMTokenizer("models/xglm-7.5B_merged.model")
            else:
                base_tokenizer = GemmaTokenizer("models/gemma-7b_merged.model", legacy=True)


        m_delta = model.ModelProto()
        m_delta.ParseFromString(open("./tokenizer_models/sp_models/tokenizer_" + lang + "_" + VOCAB_SIZE + ".model", "rb").read())
        new_vocab = [p.piece for p in m_delta.pieces]
        delta_scores = [p.score for p in m_delta.pieces]
     
        custom_vocab = [p.piece for p in m.pieces]
        dict1 = {p.piece: p.score for p in m.pieces}
        dict2 = {delta_vocab: delta_scores for delta_vocab, delta_scores in zip(new_vocab, delta_scores)} 
        for i, piece in enumerate(m.pieces):
            piece.score = piece.score - len(custom_vocab)
        
        SCORE = m.pieces[-1].score - 1
        for i, v in tqdm(enumerate(new_vocab)):
            if v not in custom_vocab:
                new_token = model.ModelProto().SentencePiece()
                new_token.piece = v
                new_token.score = delta_scores[i] + SCORE
                m.pieces.append(new_token)
        
        
        if model_base == "meta-llama/Llama-2-7b-hf":
            prefix = "Llama-2-7b-hf_"
        elif model_base == "facebook/xglm-7.5B":
            prefix = "xglm-7.5B_"
        elif model_base == "mistralai/Mistral-7B-v0.1":
            prefix = "mistral-7b_"
        else:
            prefix = "gemma-7b_"

        model_path = "tokenizer_models/" + prefix + lang + "_" + VOCAB_SIZE + "_merged_" + str(args.dataset_size) + ".model"

        with open(os.path.join(model_path), 'wb') as f:
            f.write(m.SerializeToString())

        print(len(m.pieces))