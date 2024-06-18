import os
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import random
import spacy
from ftlangdetect import detect
from datasets import load_dataset

parser = argparse.ArgumentParser(description='Merge HF tokenizers')
parser.add_argument('--model_base', default='bloom', type=str)
parser.add_argument('--lang', default='hi', type=str)
parser.add_argument('--vocab_size', default=10000, type=int)
parser.add_argument('--dataset_size', default=300000, type=int)
args = parser.parse_args()

BASE_TOKENIZER = args.model_base
LANG = args.lang
NUM_NEW_TOKENS = args.vocab_size

if not os.path.exists("tokenizer_models"):
    os.makedirs("tokenizer_models")

llama3_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
llama3_tokenizer_json = json.loads(llama3_tokenizer._tokenizer.to_str())
llama3_tokenizer_json['pre_tokenizer']['pretokenizers'][0]['pattern']['Regex'] = """'s|'t|'re|'ve|'m|'ll|'d| ?\p{N}+|(?=\s)|\s+(?!\S)|\s+"""
modified_pre_tokenizer = llama3_tokenizer.backend_tokenizer.from_str(json.dumps(llama3_tokenizer_json)).pre_tokenizer

if BASE_TOKENIZER == 'phi2':
    tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')
    tokenizer.backend_tokenizer.pre_tokenizer = modified_pre_tokenizer
elif BASE_TOKENIZER == 'bloom':
    tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-7b1')
    tokenizer.backend_tokenizer.pre_tokenizer = modified_pre_tokenizer
    
tokenizer.save_pretrained(f'./tokenizer_models/{BASE_TOKENIZER}_tokenizer')
tokenizer = AutoTokenizer.from_pretrained(f'./tokenizer_models/{BASE_TOKENIZER}_tokenizer')


dataset = load_dataset("mc4", LANG, split="train", streaming=True)
dataset = dataset.take(args.dataset_size*2)

nlp = spacy.blank(LANG)
nlp.add_pipe('sentencizer')

text_list = [i['text'] for i in dataset]
random.shuffle(text_list)

new_corpus = []
for text in tqdm(text_list):
    doc = nlp(text)
    for sent in doc.sents:
        try:
            if detect(text=sent.text.replace("\n"," "), low_memory=True)["lang"] == LANG:
                num_bytes = len(sent.text.encode('utf-8'))
                new_corpus.append(sent.text)
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)   
            continue

new_corpus = new_corpus[:args.dataset_size]

def get_training_corpus():
    dataset = new_corpus
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples

new_tokenizer = tokenizer.train_new_from_iterator(get_training_corpus(), NUM_NEW_TOKENS)
base_json = json.loads(tokenizer.backend_tokenizer.to_str())
new_json = json.loads(new_tokenizer.backend_tokenizer.to_str())

new_vocab = base_json['model']['vocab']
new_merges = base_json['model']['merges']
for k, v in tqdm(new_json['model']['vocab'].items()):
    if k not in new_vocab:
        new_vocab[k] = len(new_vocab)

for m in tqdm(new_json['model']['merges']):
    if m not in new_merges:
        new_merges.append(m)

print(len(base_json['model']['merges']), len(new_merges))

new_json['model']['vocab'] = new_vocab
new_json['model']['merges'] = new_merges
new_tokenizer._tokenizer = new_tokenizer.backend_tokenizer.from_str(json.dumps(new_json))
new_tokenizer.save_pretrained(f'./tokenizer_models/{BASE_TOKENIZER}_tokenizer_{LANG}_extended_{NUM_NEW_TOKENS}')