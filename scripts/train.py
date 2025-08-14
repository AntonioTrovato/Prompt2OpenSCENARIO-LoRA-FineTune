import os, json, math, argparse, random
from dataclasses import dataclass
from typing import Dict, List, Any
from datasets import load_dataset, DatasetDict
#from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from transformers import (AutoModelForCausalLM, AutoTokenizer)
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import torch
from huggingface_hub import HfApi

def format_example(ex, sys_tmpl, stop_seq):
    system = ex["system"].strip()
    user = ex["user"].strip()
    assistant = ex["assistant"].strip()
    prompt = sys_tmpl.format(system=system, user=user)
    # Importante: assicura una newline dopo [/INST] così il trainer può mascherare correttamente
    text = prompt + assistant + ("" if assistant.endswith(stop_seq) else stop_seq)
    return text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="config/lora-codellama13b.yaml")
    args = p.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.cfg))

    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    # Tokenizer & modello (LoRA)
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=False)
    # Aggiungiamo esplicitamente la stop sequence come token normale (non speciale)
    if "</OpenSCENARIO>" not in tok.get_vocab():
        tok.add_tokens(["</OpenSCENARIO>"])
    #bnb = BitsAndBytesConfig(
    #    load_in_4bit=cfg["load_in_4bit"],
    #    bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
    #    bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    #    bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"])
    #)
    #model = AutoModelForCausalLM.from_pretrained(
    #    cfg["base_model"],
    #    quantization_config=bnb,
    #    torch_dtype=torch.bfloat16,
    #    device_map="auto"
    #)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype = torch.bfloat16,  # o torch.float16 se preferisci
        device_map = "auto"
    )

    # allunghiamo le embedding se abbiamo aggiunto token
    model.resize_token_embeddings(len(tok))

    # Dataset da HF Datasets
    ds = load_dataset(cfg["hf_dataset_repo"], split="train")
    # split train/val
    ds = ds.train_test_split(test_size=cfg.get("val_split_ratio", 0.1), seed=cfg["seed"])
    ds = DatasetDict({"train": ds["train"], "validation": ds["test"]})

    # Template
    sys_tmpl = open("prompt_templates/codellama_inst.txt", "r", encoding="utf-8").read()
    stop_seq = cfg["stop_sequence"]

    def formatting_func(batch):
        return [format_example(ex, sys_tmpl, stop_seq) for ex in batch]

    # LoRA
    peft_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias=cfg["bias"],
        task_type="CAUSAL_LM"
    )

    # Trainer
    sft_cfg = SFTConfig(
        max_seq_length=cfg["max_seq_len"],
        packing=cfg["packing"],
        dataset_text_field=None,  # usiamo formatting_func
        formatting_func=formatting_func,
        train_on_source=False,    # maschera il prompt, addestra solo sulla risposta
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["train_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        lr_scheduler_type="cosine",
        warmup_ratio=cfg["warmup_ratio"],
        output_dir=cfg["output_dir"],
        report_to=["tensorboard"],
        bf16=True,
        gradient_checkpointing=cfg["gradient_checkpointing"]
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        peft_config=peft_cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        args=sft_cfg
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])

    # Push su HF Models
    repo_id = cfg["hf_model_repo"]
    trainer.push_to_hub(repo_id=repo_id, commit_message="Upload LoRA adapters")
    print(f"Adapters pushed to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
