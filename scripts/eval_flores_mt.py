import os
import json
import argparse
from tqdm import tqdm

import numpy as np
import torch
import transformers
import evaluate
from datasets import load_dataset

parser = argparse.ArgumentParser(description='FLORES evaluation')
parser.add_argument('--model', default="meta-llama/Llama-2-7b-hf", type=str)
parser.add_argument('--lang', default="hin_Deva", type=str)
parser.add_argument('--langname', default="Hindi", type=str)
parser.add_argument('--shot', type=int, default=5)
parser.add_argument('--noshuffle', action="store_true")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--device', default="cuda:0")
parser.add_argument('--load_in_8bit', action="store_true")
parser.add_argument('--max_tokens', type=int, default=200)
args = parser.parse_args()

def write_pretty_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as write_file:
        json.dump(data, write_file, indent=4, ensure_ascii=False)

second_lang = args.lang
model_path = args.model
num_shots=args.shot

data_set_test={
        "path": "facebook/flores",
        "name": "eng_Latn-" + second_lang,
        "split": "devtest",
}
data_set_shot={
        "path": "facebook/flores",
        "name": "eng_Latn-" + second_lang,
        "split": "dev",
}

model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map=args.device, load_in_8bit=args.load_in_8bit, max_position_embeddings=4096)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
tokenizer.model_max_length = 4096
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
pipeline.tokenizer.padding_side = 'left'

ds_predict = load_dataset(**data_set_test)
ds_predict = ds_predict.rename_column("sentence_" + second_lang, "sentence_second_lang")
ds_shots = load_dataset(**data_set_shot)
ds_shots = ds_shots.rename_column("sentence_" + second_lang, "sentence_second_lang")
prompt_template="English: {sentence_eng_Latn}\n" + args.langname + ": {sentence_second_lang}"

def construct_contexts(ds, i):
    if num_shots == 0:
        prompt_examples=""
    else:
        if args.noshuffle:
            ds_examples=ds.select(range(0, num_shots))
        else:
            ds_examples = ds.shuffle(seed=(42+i)).select(range(num_shots))
        prompt_examples = "\n\n".join([prompt_template.format(**row) for row in ds_examples])
    return prompt_examples

prompts=[(construct_contexts(ds_shots, i)+"\n\n"+prompt_template).format(sentence_eng_Latn=d["sentence_eng_Latn"],sentence_second_lang="")[:-1] for i, d in enumerate(ds_predict)]
prompts_generator=(p for p in prompts)

gen_config = {
        "max_new_tokens": args.max_tokens,
        "do_sample": False
}

results={
        "model": model_path,
        "dataset": data_set_test["path"] + "_" + data_set_test["name"],
        "num_shots": num_shots,
        "num_translations": 0,
        "bleu_score": None,
        "chrf++": None,
        "rouge": None,
        "perc": None,
        "avg_len": None,
        "translations": [],
}

chrf = evaluate.load("chrf", word_order=2)
sacrebleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")
percs = []
lens = []

pbar = tqdm(pipeline(prompts_generator, batch_size=args.batch_size, add_special_tokens=True, **gen_config),total=len(prompts),desc="")
for i, out in enumerate(pbar):
    prediction=out[0]["generated_text"][len(prompts[i])+1:]
    prediction_raw=prediction
    leng = len(pipeline.tokenizer.encode(prediction_raw))
    lens.append(leng)
    prediction=prediction_raw.split("\n")[0].strip() if "\n" in prediction_raw else prediction_raw.strip()
    prediction = prediction.split("English:")[0].strip() if "English:" in prediction else prediction.strip()
    
    reference=ds_predict[i]["sentence_second_lang"]
    original=ds_predict[i]["sentence_eng_Latn"]

    results["translations"].append({"input": original, "reference":reference, "prediction": prediction, "prediction_raw": prediction_raw})
    results["num_translations"]+=1

    x = pipeline.tokenizer.encode(prediction)
    x1 = [l for l in x if l>=32000]
    perc = len(x1)/len(x)
    percs.append(perc)
    sacrebleu_results=sacrebleu.compute(predictions=[t["prediction"] for t in results["translations"]], references=[t["reference"] for t in results["translations"]], tokenize="flores101")
    pbar.set_description(f'spBleu: {sacrebleu_results["score"]:0.2f}')

sacrebleu_results=sacrebleu.compute(predictions=[t["prediction"] for t in results["translations"]], references=[t["reference"] for t in results["translations"]], tokenize="flores101")
chrf_results=chrf.compute(predictions=[t["prediction"] for t in results["translations"]], references=[t["reference"] for t in results["translations"]])
rouge_results=rouge.compute(predictions=[t["prediction"] for t in results["translations"]], references=[t["reference"] for t in results["translations"]])

results["bleu_score"]=sacrebleu_results["score"]
results["chrf++"] = chrf_results["score"]
results["rouge"] = rouge_results
results["avg_len"] = np.mean(lens)
results["perc"] = np.mean(percs)

if not os.path.exists("results"):
    os.makedirs("results")

write_pretty_json("results/" + second_lang.replace("_", "-") + "_spbleu-" + ''.join(model_path.split("/")[-2:]) + f"_{num_shots}-shot_" + str(args.max_tokens) + "maxlen" + ".json",results)
