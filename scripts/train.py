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
import xml.etree.ElementTree as ET

def extract_features_from_xosc(xosc_text: str) -> dict:
    """
    Estrae feature salienti dall'OpenSCENARIO 1.0 (best-effort, robusta a mancanze).
    Se alcuni campi non ci sono, li lascia vuoti.
    """
    feat = {
        "map": None,
        "time_of_day": None,
        "weather": {"cloud_state": None, "precipitation": None, "fog": None, "wind": None},
        "speed_limits": [],
        "entities": [],   # [{"name":..., "type": "ego|vehicle|pedestrian|misc"}]
        "initial_positions": [],  # [{"entity": name, "x":.., "y":.., "z":.., "h":..}]
        "events": [],     # descrizioni brevi di azioni/trigger ("after 5s lead brakes", ecc.)
        "notes": []
    }
    try:
        root = ET.fromstring(xosc_text)
    except Exception:
        feat["notes"].append("xml_parse_failed")
        return feat

    # RoadNetwork / LogicFile
    rn = root.find(".//RoadNetwork/LogicFile")
    if rn is not None:
        feat["map"] = rn.attrib.get("filepath") or rn.attrib.get("file") or None

    # Environment / TimeOfDay / Weather
    tod = root.find(".//Environment/TimeOfDay")
    if tod is not None:
        feat["time_of_day"] = tod.attrib.get("dateTime") or tod.attrib.get("animation") or None

    w = root.find(".//Environment/Weather")
    if w is not None:
        feat["weather"]["cloud_state"] = w.attrib.get("cloudState")
        feat["weather"]["precipitation"] = w.attrib.get("precipitationType") or w.attrib.get("precipitation")
        fog = w.find(".//Fog")
        if fog is not None:
            feat["weather"]["fog"] = fog.attrib.get("visualRange")
        wind = w.find(".//Wind")
        if wind is not None:
            feat["weather"]["wind"] = wind.attrib.get("direction") or wind.attrib.get("speed")

    # Speed limits (molti scenari li esprimono come ParameterDeclaration o TrafficRules)
    for sl in root.findall(".//SpeedLimitAction") + root.findall(".//SpeedAction"):
        maxkph = sl.attrib.get("max") or sl.attrib.get("target")
        if maxkph:
            feat["speed_limits"].append(maxkph)

    # Entities
    for ent in root.findall(".//Entities/*"):
        tag = ent.tag.lower()
        name = ent.attrib.get("name") or ent.attrib.get("nameRef")
        etype = "vehicle"
        if "pedestrian" in tag:
            etype = "pedestrian"
        elif "misc" in tag:
            etype = "misc"
        if name:
            feat["entities"].append({"name": name, "type": etype})

    # Initial positions (Init -> Private -> TeleportAction / WorldPosition)
    for priv in root.findall(".//Init/Actions/Private"):
        name = priv.attrib.get("entityRef")
        wp = priv.find(".//WorldPosition")
        if wp is not None:
            feat["initial_positions"].append({
                "entity": name,
                "x": wp.attrib.get("x"), "y": wp.attrib.get("y"),
                "z": wp.attrib.get("z"), "h": wp.attrib.get("h")
            })

    # Events & Triggers (best-effort: estraiamo label, delay, tipo azione)
    for ev in root.findall(".//Storyboard//Event"):
        ev_name = ev.attrib.get("name")
        act = ev.find(".//Action")
        trig = ev.find(".//StartTrigger") or ev.find(".//ConditionGroup")
        desc = []
        if ev_name: desc.append(ev_name)
        if trig is not None:
            # cerca after delay
            for cond in trig.findall(".//ByValueCondition/SimulationTimeCondition"):
                delay = cond.attrib.get("value")
                if delay:
                    desc.append(f"after {delay}s")
        if act is not None:
            # tipo azione comune (SpeedAction, LaneChangeAction, etc.)
            atag = next((c.tag for c in list(act) if isinstance(c.tag, str)), None)
            if atag:
                desc.append(atag)
        if desc:
            feat["events"].append(" ".join(desc))

    return feat

def format_example(ex, sys_tmpl, stop_seq, use_feature_hints):
    system = ex["system"].strip()
    user = ex["user"].strip()
    assistant = ex["assistant"].strip()
    # HINTS opzionali dal target
    hints_block = ""
    if use_feature_hints:
        try:
            feats = extract_features_from_xosc(assistant)
            hints_block = "\n\n<HINTS>\n" + json.dumps(feats, ensure_ascii=False) + "\n</HINTS>\n"
        except Exception:
            hints_block = ""
    prompt = sys_tmpl.format(system=system, user=(user + hints_block))
    # Salvaguardia: garantisci newline dopo [/INST] se manca
    if prompt.endswith("[/INST]"):
        prompt = prompt + "\n"
    # Concatena output gold; NON aggiungere eos come stop token personalizzato,
    # ma chiudi sempre con lo stop_sequence nel target (utile anche a inferenza).
    text = prompt + assistant
    if not assistant.strip().endswith(stop_seq):
        text += stop_seq
    return text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="config/lora-codellama13b.yaml")
    p.add_argument("--jsonl_path", default="")
    p.add_argument("--use_feature_hints", action="store_true")
    args = p.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.cfg))

    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    # Tokenizer & modello (LoRA)
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=False)
    # LLaMA: pad_token = eos_token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
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

    dtype = torch.bfloat16 if str(cfg.get("torch_dtype", "bfloat16")) == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype = dtype,
        device_map = "auto",
        max_memory = {0: "24GiB", "cpu" : "48GiB"}
    )
    model.config.pad_token_id = tok.pad_token_id

    # Se gradient checkpointing: disattiva cache
    if cfg.get("gradient_checkpointing", False):
        model.config.use_cache = False

    # (Niente token extra: non modifichiamo il vocab)

    # Dataset da HF Datasets
    if args.jsonl_path:
        # carica JSONL locale pre-processato
        import json, datasets
        rows = [json.loads(line) for line in open(args.jsonl_path, "r", encoding="utf-8")]
        ds = datasets.Dataset.from_list(rows)
    else:
        ds = load_dataset(cfg["hf_dataset_repo"], split="train")
    # split train/val
    ds = ds.train_test_split(test_size=cfg.get("val_split_ratio", 0.1), seed=cfg["seed"])
    ds = DatasetDict({"train": ds["train"], "validation": ds["test"]})

    # Template
    sys_tmpl = open("prompt_templates/codellama_inst.txt", "r", encoding="utf-8").read()
    stop_seq = cfg["stop_sequence"]

    def formatting_func(batch):
        return [format_example(ex, sys_tmpl, stop_seq, use_feature_hints=args.use_feature_hints) for ex in batch]

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
        lr_scheduler_type=cfg.get("lr_scheduler_type","cosine"),
        warmup_ratio=cfg["warmup_ratio"],
        output_dir=cfg["output_dir"],
        report_to=["tensorboard"],
        bf16 = (str(cfg.get("torch_dtype", "bfloat16")) == "bfloat16"),
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
    if cfg.get("push_to_hub", False):
        repo_id = cfg.get("hub_model_id") or cfg["hf_model_repo"]
        trainer.push_to_hub(repo_id=repo_id, commit_message="Upload LoRA adapters")
        print(f"Adapters pushed to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
