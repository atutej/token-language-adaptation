import transformers
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import torch.distributed as dist
from tqdm import tqdm

def get_model_and_tokenizer(model_args, training_args, **model_kwargs):
        # Load model and tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        ).to("cuda")

        tokenizer, orig_tokenizer = get_tokenizer(model_args, training_args)
        if tokenizer.vocab_size > orig_tokenizer.vocab_size:
            print("Resizing token embeddings")
            model = resize_token_embeddings(model, tokenizer, orig_tokenizer, model_args, training_args)

        if model_args.freeze_transformer:
            print("Freezing transformer")
            for n, p in model.named_parameters():
                if model_args.freeze_embeddings:
                    if 'lm_head' not in n:
                        p.requires_grad = False
                    else:
                        print('Trainable: ', n)
                elif model_args.freeze_lmhead:
                    if 'embed' not in n:
                        p.requires_grad = False
                    else:
                        print('Trainable: ', n)
                else:
                    if 'embed' not in n and 'lm_head' not in n:
                        p.requires_grad = False
                    else:
                        print('Trainable: ', n)

        return model, tokenizer

def get_tokenizer(model_args, training_args):
    if model_args.tokenizer_name_or_path is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    elif model_args.tokenizer_name_or_path.endswith(".model"):
        if "xglm" in model_args.tokenizer_name_or_path:
            from transformers import XGLMTokenizer
            tokenizer = XGLMTokenizer(
                model_args.tokenizer_name_or_path,
                legacy=True,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False
            )
        elif "gemma" in model_args.tokenizer_name_or_path:
            from transformers import GemmaTokenizer
            tokenizer = GemmaTokenizer(
                model_args.tokenizer_name_or_path,
                legacy=True,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False
            )
        elif "bloom" in model_args.tokenizer_name_or_path:
            from transformers import BloomTokenizer
            tokenizer = BloomTokenizer(
                model_args.tokenizer_name_or_path,
                legacy=True,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False
            )
        else:
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer(
            model_args.tokenizer_name_or_path, 
            legacy=True,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False
            )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
    tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token

    if model_args.orig_tokenizer_name_or_path is None:
        orig_tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=True,
            )
    elif model_args.orig_tokenizer_name_or_path.endswith(".model"):
        from transformers import LlamaTokenizer
        orig_tokenizer = LlamaTokenizer(
        model_args.orig_tokenizer_name_or_path, 
        legacy=True,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False
        )
    else:
        orig_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.orig_tokenizer_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
    orig_tokenizer.pad_token = orig_tokenizer.unk_token if orig_tokenizer.unk_token else orig_tokenizer.eos_token
    
    return tokenizer, orig_tokenizer

def resize_token_embeddings(model, tokenizer, orig_tokenizer, model_args, training_args):
    strategy = model_args.resize_embed_strategy
    model.resize_token_embeddings(len(tokenizer))

    if strategy == 'mean':
        print('Using mean strategy')
        new_vocab = tokenizer.get_vocab()
        vocab = orig_tokenizer.get_vocab()
        new_tokens = list(set(list(new_vocab.keys())) - set(list(vocab.keys())))
        new_token_ii = orig_tokenizer(new_tokens, add_special_tokens=False).input_ids
        out_embeds = model.get_output_embeddings()
        inp_embeds = model.get_input_embeddings()
        for token, token_ii in tqdm(zip(new_tokens, new_token_ii)):
            inp_embeds.weight.data[new_vocab[token]] = inp_embeds.weight.data[token_ii].mean(dim=0)
            out_embeds.weight.data[new_vocab[token]] = out_embeds.weight.data[token_ii].mean(dim=0)

    elif "focus" in strategy:
        from deepfocus import FOCUS
        focus_lang = strategy.split('_')[1]
        print('Using focus strategy for language: ', focus_lang)
        target_embeddings = FOCUS(
        source_embeddings=model.get_input_embeddings().weight,
        source_tokenizer=orig_tokenizer,
        target_tokenizer=tokenizer,
        auxiliary_embedding_mode="fasttext-wordlevel",
        language_identifier=focus_lang,
        )
        model.resize_token_embeddings(len(tokenizer))
        model.get_input_embeddings().weight.data = target_embeddings

        target_output_embeddings = FOCUS(
        source_embeddings=model.get_output_embeddings().weight,
        source_tokenizer=orig_tokenizer,
        target_tokenizer=tokenizer,
        auxiliary_embedding_mode="fasttext-wordlevel",
        language_identifier=focus_lang,
        )
        model.get_output_embeddings().weight.data = target_output_embeddings
    
    return model