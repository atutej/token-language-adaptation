import transformers
import evaluate
import torch
import json
import os
from tqdm import tqdm
import time
from datasets import load_dataset
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='XLSUM evaluation')
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
data_set_test={
        "path": "csebuetnlp/xlsum",
        "name": args.langname.lower(),
        "split": "validation",
}
data_set_shot={
        "path": "csebuetnlp/xlsum",
        "name": args.langname.lower(),
        "split": "validation",
}
num_shots=args.shot

pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            model_kwargs={"load_in_8bit": args.load_in_8bit},
            
        )
pipeline.tokenizer.padding_side = 'left'
pipeline.tokenizer.add_special_tokens({"pad_token":"<pad>"})
pipeline.model.resize_token_embeddings(len(pipeline.tokenizer))
pipeline.model.config.pad_token_id = pipeline.tokenizer.pad_token_id


ds_predict = load_dataset(**data_set_test)
ds_shots = load_dataset(**data_set_shot)
prompt_template="Summary: {text}\n" + "Title: {summary}"

def construct_contexts(ds, i):
    if num_shots == 0:
        prompt_examples=""
    else:
        if args.noshuffle:
            ds_examples=ds.select(range(0, num_shots))
        else:
            ds_examples = ds.shuffle(seed=(42+i)).select(range(num_shots))
        prompt_examples = "\n\n".join([prompt_template.format(text=row["summary"].replace("{", '').replace("}", ''), summary=row["title"].replace("{", '').replace("}", '')) for row in ds_examples])
    return prompt_examples

prompts=[(construct_contexts(ds_shots, i)+"\n\n"+prompt_template).format(text=d["summary"].replace("{", '').replace("}", ''),summary="")[:-1] for i, d in enumerate(ds_predict)]
prompts_generator=(p for p in prompts)

gen_config = {
        "max_new_tokens": args.max_tokens,
        "do_sample": False
}

results={
        "model": model_path,
        "dataset": data_set_test["path"] + "_" + data_set_test["name"],
        "gen_config": gen_config,
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

    reference=ds_predict[i]["title"]
    original=ds_predict[i]["summary"]

    results["translations"].append({"input": original, "reference":reference, "prediction": prediction, "prediction_raw": prediction_raw})
    results["num_translations"]+=1

    x = pipeline.tokenizer.encode(prediction)
    x1 = [l for l in x if l>=32000]
    perc = len(x1)/(len(x) + 1e-6)
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

if not os.path.exists("results_xlsum"):
    os.makedirs("results_xlsum")

write_pretty_json("results_xlsum/" + second_lang.replace("_", "-") + "_spbleu-" + ''.join(model_path.split("/")[-2:]) + f"_{num_shots}-shot_" + str(args.max_tokens) + "maxlen" + ".json",results)