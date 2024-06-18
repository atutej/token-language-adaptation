from datasets import load_dataset
from tqdm import tqdm
from ftlangdetect import detect
import argparse
import os

parser = argparse.ArgumentParser(description='MC4 data creation')
parser.add_argument('--lang', default="hi", type=str)
parser.add_argument('--data_size', default=100000, type=int)
parser.add_argument('--split', default="train", type=str)
args = parser.parse_args()

DATA_SIZE = args.data_size
LANG = args.lang
all_texts = []

if not os.path.exists("../data"):
    os.makedirs("../data")

mc4_dataset = load_dataset(
    "mc4", LANG,
    split=args.split,
    streaming=True,
)

for text in tqdm(mc4_dataset):
    if len(all_texts)%1000 == 0:
        print(len(all_texts))
    if len(text['text'].split()) > 500:
        if detect(text=text['text'].replace("\n"," "), low_memory=True)["lang"] == LANG:
            all_texts.append(text['text'])
            if len(all_texts) > DATA_SIZE:
                break

res = []
for i, row in enumerate(all_texts):
    res.append({"id": i, "conversations": row})
import json
json.dump(res, open(f'../data/mc4_{LANG}_{args.split}[:{DATA_SIZE//1000}k]_text.json', 'w'))