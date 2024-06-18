# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import torch.nn.functional as F

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from transformers import LlamaTokenizer

from my_models import get_model_and_tokenizer

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    tokenizer_name_or_path: Optional[str] = field(default=None)
    orig_tokenizer_name_or_path: Optional[str] = field(default=None)
    resize_embed_strategy: Optional[str] = field(default=None)
    freeze_transformer: bool = field(default=False)
    freeze_embeddings: bool = field(default=False)
    freeze_lmhead: bool = field(default=False)
    freeze_old_tok_params: bool = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    preprocess_fn: str = field(
        default=None, metadata={"help": ""}
    ) # options translate_orig_tok_to_new, consume_orig_spit_new


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    pad_to_multiple_of: Optional[int] = field(default=16)
    metric_for_best_model: str = field(default="Sum NLL")


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def mask_inst(A, tokenizer, mask_token_id=-100):
    # Define the sequences to search for
    sequence1 = tokenizer("[INST]", add_special_tokens=False, return_tensors="pt" if isinstance(A, torch.Tensor) else None).input_ids[0]
    sequence2 = tokenizer("[/INST]", add_special_tokens=False, return_tensors="pt" if isinstance(A, torch.Tensor) else None).input_ids[0]

    search = True
    while search:
        # Find the starting indices of the sequences in A
        start_index1 = -1
        start_index2 = -1

        for i in range(len(A) - len(sequence1) + 1):
            if all(A[i:i+len(sequence1)] == sequence1):
                start_index1 = i
                break

        for i in range(len(A) - len(sequence2) + 1):
            if all(A[i:i+len(sequence2)] == sequence2):
                start_index2 = i
                break

        # If both sequences are found and sequence2 occurs after sequence1
        if start_index1 != -1 and start_index2 != -1 and start_index2 > start_index1:
            search = True
            # Set the values in B to 0 from start_index1 to the end of sequence2
            end_index2 = start_index2 + len(sequence2)
            for i in range(start_index1, end_index2):
                A[i] = mask_token_id
        else:
            search = False

    return A


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    **kwargs
) -> Dict:

    # Tokenize conversations
    input_ids = tokenizer(
        sources,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    targets[targets == tokenizer.pad_token_id] = IGNORE_TOKEN_ID

    for i in range(input_ids.shape[0]):
        targets[i] = mask_inst(targets[i], tokenizer, IGNORE_TOKEN_ID)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, preprocess_fn=preprocess):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess_fn(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, preprocess_fn=preprocess):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.preprocess_fn = preprocess_fn
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        ret = self.preprocess_fn([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = {k : v[0] for k, v in ret.items()}
        return ret

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    preprocess_fn = preprocess

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, preprocess_fn=preprocess_fn)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, preprocess_fn=preprocess_fn)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def compute_metrics(eval_preds):
    metrics, _ = eval_preds
    return {"Sum NLL": metrics[0].mean(), "Mean NLL": metrics[1].mean(), "Perplexity": np.exp(metrics[1]).mean(), "Mean text length": metrics[2].mean()}

@torch.no_grad()
def preprocess_logits_for_metrics(logits, labels):
    shifted_logits = logits[:, :-1].transpose(1, 2)
    shifted_targets = labels[:, 1:]
    shifted_attn_mask = (labels[:, 1:].ne(IGNORE_TOKEN_ID)).float()
    sum_nll = (F.cross_entropy(shifted_logits, shifted_targets, reduction='none')*shifted_attn_mask).sum(1)
    mean_nll = sum_nll / shifted_attn_mask.sum(1)
    return sum_nll, mean_nll, shifted_attn_mask.sum(1)

class MyCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=1):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        b = {k: [x[k] for x in batch] for k in batch[0].keys()}
        for k, v_list in b.items():
            if isinstance(v_list[0], torch.Tensor):
                max_len = max([len(v) for v in v_list])
                max_len = (max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of
                b[k] = torch.stack([F.pad(v, (0, max_len - len(v)), value=self.tokenizer.pad_token_id) for v in v_list])
        if "label" in b:
            b["labels"] = b["label"]
            del b["label"]
        if "label_ids" in b:
            b["labels"] = b["label_ids"]
            del b["label_ids"]
        b["labels"][b["labels"] == self.tokenizer.pad_token_id] = IGNORE_TOKEN_ID
        return b

def make_trainer(model, tokenizer, training_args, **data_module):
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        data_collator=MyCollator(tokenizer, pad_to_multiple_of=training_args.pad_to_multiple_of),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        **data_module
    )
    return trainer

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    training_args.label_names = ["labels"]

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    model, tokenizer = get_model_and_tokenizer(model_args, training_args, config=config)

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    print("Done")


    # Start trainner
    trainer = make_trainer(model, tokenizer, training_args, **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
