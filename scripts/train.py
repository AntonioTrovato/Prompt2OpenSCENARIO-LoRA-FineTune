import glob
import os, json, argparse, random
from typing import Dict, List, Any
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import torch
import xml.etree.ElementTree as ET
from lxml import etree
import yaml

import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any
import re

from lxml import etree

# -----
# Main
# -----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="config/lora-codellama13b.yaml")
    p.add_argument("--jsonl_path", default="")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Template single-turn compatibile con assistant_only_loss
    tok.chat_template = (
        "{{ bos_token }}"
        "{% for m in messages %}"
        "{% if m['role'] == 'user' %}"
        "[INST] {{ m['content'] }} [/INST]"
        "{% elif m['role'] == 'assistant' %}"
        "{% generation %}{{ m['content'] }}{% endgeneration %}{{ eos_token }}"
        "{% endif %}"
        "{% endfor %}"
    )

    from transformers import BitsAndBytesConfig

    use_4bit = bool(cfg.get("load_in_4bit", False))
    dtype = torch.bfloat16 if str(cfg.get("torch_dtype", "bfloat16")) == "bfloat16" else torch.float16

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=torch.bfloat16 if str(
                cfg.get("bnb_4bit_compute_dtype", "bfloat16")) == "bfloat16" else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=dtype if not use_4bit else None,
        quantization_config=bnb_config if use_4bit else None,
        device_map="auto" if use_4bit else None,  # utile con 4-bit
    )
    model.config.pad_token_id = tok.pad_token_id
    if cfg.get("gradient_checkpointing", False):
        model.config.use_cache = False

    # Dataset
    if args.jsonl_path:
        rows = [json.loads(line) for line in open(args.jsonl_path, "r", encoding="utf-8")]
        ds = Dataset.from_list(rows)
    else:
        ds = load_dataset(cfg["hf_dataset_repo"], split="train")

    ds = ds.train_test_split(test_size=cfg.get("val_split_ratio", 0.1), seed=cfg["seed"])
    ds = DatasetDict({"train": ds["train"], "validation": ds["test"]})

    stop_seq = cfg["stop_sequence"]

    peft_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias=cfg["bias"],
        task_type="CAUSAL_LM"
    )

    def to_messages(ex):
        system = ex["system"].strip()
        user = ex["user"].strip()
        assistant = ex["assistant"].strip()
        if not assistant.endswith(stop_seq):
            assistant += stop_seq

        # fondi il system nel primo user (stile CodeLlama)
        user_with_sys = f"<<SYS>>\n{system}\n<</SYS>>\n\n{user}"

        return {
            "messages": [
                {"role": "user", "content": user_with_sys},
                {"role": "assistant", "content": assistant},
            ]
        }

    ds = ds.map(to_messages)

    def _ok_len(batch):
        return [
            len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False))
            <= cfg["max_length"]
            for msgs in batch["messages"]
        ]

    ds["train"] = ds["train"].filter(_ok_len, batched=True)

    sft_cfg = SFTConfig(
        max_length=cfg["max_length"],
        packing=cfg["packing"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["train_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg["warmup_ratio"],
        output_dir=cfg["output_dir"],
        report_to=["tensorboard"],
        bf16=(str(cfg.get("torch_dtype", "bfloat16")) == "bfloat16"),
        gradient_checkpointing=cfg["gradient_checkpointing"],
        assistant_only_loss=True,
        save_total_limit=cfg.get("save_total_limit", 3),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=peft_cfg,
        processing_class=tok,
    )

    last_ckpt = None
    if os.path.isdir(cfg["output_dir"]):
        candidates = sorted(glob.glob(os.path.join(cfg["output_dir"], "checkpoint-*")),
                            key=os.path.getmtime)
        if candidates:
            last_ckpt = candidates[-1]

    if last_ckpt:
        print(f"Resuming from checkpoint: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        trainer.train()

    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])

if __name__ == "__main__":
    main()
