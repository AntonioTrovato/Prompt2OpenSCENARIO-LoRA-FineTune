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

EGO_NAME = "ego_vehicle"

def reduce_xosc(xosc_content: str) -> str:
    """
    Riduce un file OpenSCENARIO 1.0 secondo le regole:
    - Niente fusioni tra nodi gemelli (stesso tag).
    - In ogni gruppo di gemelli per tag:
        * Se almeno uno 'porta' a entityRef=ego o entityRef=secondo_veicolo, tieni TUTTI quelli che portano a (eccezione).
        * Altrimenti tieni solo il primo.
    - Non eliminare <ScenarioObject name="ego_vehicle"> né i suoi figli.
    - Tra i non-ego referenziati, scegli il primo incontrato come 'secondo_veicolo' e tieni solo quello.
    - Vietato eliminare rami che portano a entityRef=ego o entityRef=secondo_veicolo.
    """
    tree = ET.ElementTree(ET.fromstring(xosc_content))
    root = tree.getroot()

    # -------------------------------------------------------------
    # 1) Raccogli i riferimenti entityRef in ORDINE di documento
    #    (non un set, così il "primo referenziato" è deterministico)
    # -------------------------------------------------------------
    ordered_entity_refs: List[str] = []
    for elem in root.iter():
        val = elem.attrib.get("entityRef")
        if val is not None:
            ordered_entity_refs.append(val)

    # Scegli il "secondo veicolo": primo entityRef != ego
    second_vehicle = next((v for v in ordered_entity_refs if v != EGO_NAME), None)

    # -------------------------------------------------------------
    # 2) Mantieni solo ego + (eventuale) secondo_veicolo in <Entities>
    # -------------------------------------------------------------
    entities = root.find(".//Entities")
    if entities is not None:
        to_remove = []
        for so in entities.findall("ScenarioObject"):
            name = so.attrib.get("name")
            if name == EGO_NAME:
                continue
            if name == second_vehicle:
                continue
            # rimuovi tutti gli altri non-ego
            to_remove.append(so)
        for so in to_remove:
            entities.remove(so)

    protected_targets = {EGO_NAME}
    if second_vehicle:
        protected_targets.add(second_vehicle)

    # -------------------------------------------------------------
    # 3) Helper: il nodo 'porta' a un protected? (ego o secondo)
    #    True se nel suo sottoalbero esiste entityRef in protected_targets
    # -------------------------------------------------------------
    def leads_to_protected(node: ET.Element) -> bool:
        for n in node.iter():
            v = n.attrib.get("entityRef")
            if v in protected_targets:
                return True
        return False

    # -------------------------------------------------------------
    # 4) Deduplicazione senza fusioni, con eccezione per multipli protetti
    #    Ragioniamo per gruppi di figli con lo stesso tag
    # -------------------------------------------------------------
    def reduce_node(node: ET.Element):
        # Gruppo i figli per tag
        children = list(node)
        by_tag: Dict[str, List[ET.Element]] = {}
        for ch in children:
            by_tag.setdefault(ch.tag, []).append(ch)

        # Per ogni gruppo di gemelli per tag, applico la regola
        for tag, group in by_tag.items():
            if len(group) <= 1:
                continue  # niente gemelli

            # Partiziona: protetti vs non protetti
            protected_group = [g for g in group if leads_to_protected(g)]
            non_protected_group = [g for g in group if g not in protected_group]

            # Caso A: esistono uno o più protetti -> tieni TUTTI i protetti, elimina tutti i non-protetti
            if protected_group:
                to_keep = set(protected_group)
                to_drop = [g for g in group if g not in to_keep]
            else:
                # Caso B: nessun protetto -> tieni SOLO il primo (ordine documento), elimina gli altri
                to_keep = {group[0]}
                to_drop = group[1:]

            for g in to_drop:
                # Non rimuovere l'ego o il secondo veicolo se mai capitassero in un gruppo (per sicurezza)
                if not (
                    g.tag == "ScenarioObject"
                    and g.attrib.get("name") in {EGO_NAME, second_vehicle}
                ):
                    node.remove(g)

        # Ricorsione sui figli rimasti
        for ch in list(node):
            reduce_node(ch)

    reduce_node(root)

    # Serializza (senza pretty-print per semplicità; aggiungibile se vuoi)
    return ET.tostring(root, encoding="unicode")

def reduce_assistant(xosc_text: str, mode: str) -> str:
    if mode == "minify":
        return reduce_xosc(xosc_text)
    # "none"
    return xosc_text

# -----
# Main
# -----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="config/lora-codellama13b.yaml")
    p.add_argument("--jsonl_path", default="")
    p.add_argument("--reduce_mode", choices=["none","minify"], default="minify")
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
        assistant = reduce_assistant(ex["assistant"].strip(), args.reduce_mode)
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
