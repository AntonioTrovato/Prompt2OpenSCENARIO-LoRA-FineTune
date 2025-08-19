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

# ---------------------------
# Feature extractor (fornita)
# ---------------------------
def extract_features_from_xosc(xosc_text: str) -> dict:
    feat = {
        "map": None,
        "time_of_day": None,
        "weather": {"cloud_state": None, "precipitation": None, "fog": None, "wind": None},
        "speed_limits": [],
        "entities": [],
        "initial_positions": [],
        "events": [],
        "notes": []
    }
    try:
        root = ET.fromstring(xosc_text)
    except Exception:
        feat["notes"].append("xml_parse_failed")
        return feat

    rn = root.find(".//RoadNetwork/LogicFile")
    if rn is not None:
        feat["map"] = rn.attrib.get("filepath") or rn.attrib.get("file") or None

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

    for sl in root.findall(".//SpeedLimitAction") + root.findall(".//SpeedAction"):
        maxkph = sl.attrib.get("max") or sl.attrib.get("target")
        if maxkph:
            feat["speed_limits"].append(maxkph)

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

    for priv in root.findall(".//Init/Actions/Private"):
        name = priv.attrib.get("entityRef")
        wp = priv.find(".//WorldPosition")
        if wp is not None:
            feat["initial_positions"].append({
                "entity": name,
                "x": wp.attrib.get("x"), "y": wp.attrib.get("y"),
                "z": wp.attrib.get("z"), "h": wp.attrib.get("h")
            })

    for ev in root.findall(".//Storyboard//Event"):
        ev_name = ev.attrib.get("name")
        act = ev.find(".//Action")
        trig = ev.find(".//StartTrigger") or ev.find(".//ConditionGroup")
        desc = []
        if ev_name: desc.append(ev_name)
        if trig is not None:
            for cond in trig.findall(".//ByValueCondition/SimulationTimeCondition"):
                delay = cond.attrib.get("value")
                if delay:
                    desc.append(f"after {delay}s")
        if act is not None:
            atag = next((c.tag for c in list(act) if isinstance(c.tag, str)), None)
            if atag:
                desc.append(atag)
        if desc:
            feat["events"].append(" ".join(desc))
    return feat

# ---------------------------------
# Riduzione on-the-fly dell'XML gold
# ---------------------------------
def minify_xml(x: str) -> str:
    try:
        #parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
        #root = etree.fromstring(x.encode("utf-8"), parser=parser)
        #return etree.tostring(root, encoding="unicode", pretty_print=False)
        return reduce_xosc(x)
    except Exception:
        return x

from lxml import etree

def build_minimal_xosc_from_features(feat: dict) -> str:
    # Fallbacks
    map_path = feat.get("map") or "Town01.xodr"
    time_of_day = feat.get("time_of_day") or "2025-01-01T12:00:00"
    weather = feat.get("weather") or {}
    cloud = weather.get("cloud_state") or "free"
    precip = weather.get("precipitation") or "dry"

    entities = feat.get("entities") or []
    if not entities:
        entities = [{"name": "ego", "type": "vehicle"}]

    init_pos_by_entity = {}
    for ip in (feat.get("initial_positions") or []):
        init_pos_by_entity[ip.get("entity")] = {
            "x": ip.get("x") or "0", "y": ip.get("y") or "0",
            "z": ip.get("z") or "0", "h": ip.get("h") or "0"
        }

    def mk_pos(name):
        p = init_pos_by_entity.get(name) or {"x":"0","y":"0","z":"0","h":"0"}
        return p["x"], p["y"], p["z"], p["h"]

    E = etree.Element
    root = E("OpenScenario")

    # FileHeader
    fh = E("FileHeader", revMajor="1", revMinor="0",
           date="2025-01-01T00:00:00", description="Minimal scenario")
    root.append(fh)

    # RoadNetwork
    rn = E("RoadNetwork")
    rn.append(E("LogicFile", filepath=map_path))
    root.append(rn)

    # Empty but required
    root.append(E("ParameterDeclarations"))
    root.append(E("CatalogLocations"))

    # Entities
    ents = E("Entities")
    for ent in entities:
        nm = ent.get("name") or "ego"
        typ = ent.get("type") or "vehicle"
        if typ == "pedestrian":
            obj = E("Pedestrian", name=nm, mass="70", model="ped",
                    pedestrianCategory="pedestrian")
            bb = E("BoundingBox")
            bb.append(E("Center", x="0", y="0", z="0.9"))
            bb.append(E("Dimensions", width="0.5", length="0.5", height="1.8"))
            obj.append(bb)
        elif typ == "misc":
            obj = E("MiscObject", name=nm, mass="100",
                    miscObjectCategory="obstacle")
        else:
            obj = E("Vehicle", name=nm, vehicleCategory="car")
            bb = E("BoundingBox")
            bb.append(E("Center", x="0", y="0", z="0.9"))
            bb.append(E("Dimensions", width="2.0", length="4.5", height="1.5"))
            obj.append(bb)
        ents.append(obj)
    root.append(ents)

    # Storyboard
    sb = E("Storyboard")

    # Init: position + environment
    init = E("Init")
    actions = E("Actions")

    # Private Actions: teleport each entity
    for ent in entities:
        nm = ent.get("name") or "ego"
        priv = E("Private", entityRef=nm)
        actions_priv = E("Actions")
        act = E("Action")
        tp = E("TeleportAction")
        pos = E("Position")
        x,y,z,h = mk_pos(nm)
        pos.append(E("WorldPosition", x=x, y=y, z=z, h=h))
        tp.append(pos)
        act.append(tp)
        actions_priv.append(act)
        priv.append(actions_priv)
        actions.append(priv)

    # GlobalAction: Environment
    gact = E("GlobalAction")
    envact = E("EnvironmentAction")
    env = E("Environment", name="default_env")
    env.append(E("TimeOfDay", dateTime=time_of_day))
    env.append(E("Weather", cloudState=cloud, precipitationType=precip))
    envact.append(env)
    gact.append(envact)
    actions.append(gact)

    init.append(actions)
    sb.append(init)

    # Main Story
    story = E("Story", name="main_story")
    act = E("Act", name="act_1")
    for ent in entities:
        nm = ent.get("name") or "ego"
        mg = E("ManeuverGroup", name=f"mg_{nm}")
        man = E("Maneuver", name=f"man_{nm}")
        ev = E("Event", name=f"ev_{nm}", priority="overwrite")
        ev_action = E("Action")
        ev_action.append(E("ControllerAction"))
        ev.append(ev_action)
        # Event StartTrigger
        st = E("StartTrigger")
        cg = E("ConditionGroup")
        c = E("Condition", delay="0", conditionEdge="rising")
        byv = E("ByValueCondition")
        byv.append(E("SimulationTimeCondition", value="0", rule="greaterThan"))
        c.append(byv)
        cg.append(c)
        st.append(cg)
        ev.append(st)
        man.append(ev)
        mg.append(man)
        act.append(mg)

    # Act StartTrigger
    st_act = E("StartTrigger")
    cg_act = E("ConditionGroup")
    c_act = E("Condition", delay="0", conditionEdge="rising")
    byv_act = E("ByValueCondition")
    byv_act.append(E("SimulationTimeCondition", value="0", rule="greaterThan"))
    c_act.append(byv_act)
    cg_act.append(c_act)
    st_act.append(cg_act)
    act.append(st_act)

    story.append(act)
    sb.append(story)
    root.append(sb)

    return etree.tostring(root, encoding="unicode", pretty_print=False)

def reduce_assistant(xosc_text: str, mode: str) -> str:
    if mode == "minify_only":
        return minify_xml(xosc_text)
    if mode == "features_skeleton":
        try:
            feats = extract_features_from_xosc(xosc_text)
            return build_minimal_xosc_from_features(feats)
        except Exception:
            return minify_xml(xosc_text)
    # "none"
    return xosc_text

# -----
# Main
# -----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="config/lora-codellama13b.yaml")
    p.add_argument("--jsonl_path", default="")
    p.add_argument("--use_feature_hints", action="store_true")
    p.add_argument("--reduce_mode", choices=["none","minify_only","features_skeleton"], default="minify_only")
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
        if args.use_feature_hints:
            try:
                feats = extract_features_from_xosc(ex["assistant"])
                user_with_sys += "\n\n<HINTS>\n" + json.dumps(feats, ensure_ascii=False) + "\n</HINTS>\n"
            except Exception:
                pass

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
