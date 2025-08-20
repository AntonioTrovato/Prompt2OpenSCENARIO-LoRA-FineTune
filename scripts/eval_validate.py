import os, json, argparse, re
import time
from typing import List, Dict
from openai import OpenAI

import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from lxml import etree
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import sacrebleu
from rouge_score.rouge_scorer import RougeScorer
import xml.etree.ElementTree as ET
import psutil
import math
from collections import Counter
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
except Exception:
    NVML_HANDLE = None

STOP_STR = "</OpenScenario>"

class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tok, substrings, start_len=0):
        self.tok = tok
        self.substrings = substrings
        self.start_len = start_len  # numero di token di prompt

    def set_start_len(self, n):
        self.start_len = n

    def __call__(self, input_ids, scores, **kw):
        gen_ids = input_ids[0, self.start_len:].tolist()
        if not gen_ids:
            return False
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        return any(s in text for s in self.substrings)

EGO_NAME = "ego_vehicle"


# Usa l'estrattore definito in train (stessa logica delle feature)
#from scripts.train import extract_features_from_xosc, minify_xml, build_minimal_xosc_from_features
# ---------------------------
# Feature extractor (fornita)
# ---------------------------
def extract_features_from_xosc(xosc_text: str) -> dict:
    """
    Estrae feature salienti dall'OpenSCENARIO 1.0 (best-effort, robusta a mancanze).
    Non include speed_limits né notes.
    """
    feat = {
        "map": None,
        "time_of_day": None,
        "weather": {"cloud_state": None, "precipitation": None, "fog": None, "wind": None},
        "entities": [],   # [{"name":..., "type": "ego|vehicle|pedestrian|misc"}]
        "initial_positions": [],  # [{"entity": name, "x":.., "y":.., "z":.., "h":..}]
        "events": []      # descrizioni brevi di azioni/trigger
    }
    try:
        root = ET.fromstring(xosc_text)
    except Exception:
        return feat

    # RoadNetwork / LogicFile
    rn = root.find(".//RoadNetwork/LogicFile")
    if rn is not None:
        feat["map"] = rn.attrib.get("filepath") or rn.attrib.get("file") or None

    # Environment / TimeOfDay / Weather
    tod = root.find(".//Environment/TimeOfDay")
    if tod is not None:
        # Preferisci dateTime; se assente, metti animation (true/false) come fallback
        feat["time_of_day"] = tod.attrib.get("dateTime") or tod.attrib.get("animation") or None

    w = root.find(".//Environment/Weather")
    if w is not None:
        feat["weather"]["cloud_state"] = w.attrib.get("cloudState")

        # <Precipitation intensity="..." precipitationType="rain|snow|..." />
        prec = w.find("./Precipitation")
        if prec is not None:
            ptype = prec.attrib.get("precipitationType")
            pint  = prec.attrib.get("intensity")
            if ptype and pint:
                feat["weather"]["precipitation"] = f"{ptype} (intensity={pint})"
            elif ptype:
                feat["weather"]["precipitation"] = ptype
            elif pint:
                feat["weather"]["precipitation"] = f"intensity={pint}"

        fog = w.find("./Fog")
        if fog is not None:
            feat["weather"]["fog"] = fog.attrib.get("visualRange")

        # Alcuni XOSC hanno <Wind direction="..." speed="..."/>
        wind = w.find("./Wind")
        if wind is not None:
            direction = wind.attrib.get("direction")
            speed = wind.attrib.get("speed")
            if direction and speed:
                feat["weather"]["wind"] = f"dir={direction}, speed={speed}"
            else:
                feat["weather"]["wind"] = direction or speed

    # Entities (ScenarioObject -> Vehicle/Pedestrian/MiscObject)
    for ent in root.findall(".//Entities/ScenarioObject"):
        name = ent.attrib.get("name") or ent.attrib.get("nameRef")
        etype = "vehicle"  # default
        if ent.find("./Pedestrian") is not None:
            etype = "pedestrian"
        elif ent.find("./MiscObject") is not None:
            etype = "misc"
        elif ent.find("./Vehicle") is not None:
            etype = "vehicle"

        if name == "ego_vehicle":
            etype = "ego"

        if name:
            feat["entities"].append({"name": name, "type": etype})

    # Initial positions: SOLO TeleportAction/WorldPosition (non includere AcquirePositionAction)
    for priv in root.findall(".//Init/Actions/Private"):
        name = priv.attrib.get("entityRef")
        wp = priv.find("./PrivateAction/TeleportAction/Position/WorldPosition")
        if wp is not None:
            feat["initial_positions"].append({
                "entity": name,
                "x": wp.attrib.get("x"), "y": wp.attrib.get("y"),
                "z": wp.attrib.get("z"), "h": wp.attrib.get("h")
            })

    # Events & Triggers
    for ev in root.findall(".//Storyboard//Event"):
        ev_name = ev.attrib.get("name")
        # Può esserci più di un <Action> per evento
        actions = ev.findall("./Action")
        start_trig = ev.find("./StartTrigger")
        desc_bits = []
        if ev_name:
            desc_bits.append(ev_name)

        # Estrai condizioni di start: ByValueCondition/SimulationTimeCondition
        if start_trig is not None:
            for cond in start_trig.findall(".//ByValueCondition/SimulationTimeCondition"):
                val = cond.attrib.get("value")
                rule = cond.attrib.get("rule")
                if val:
                    if rule:
                        desc_bits.append(f"when sim_time {rule} {val}s")
                    else:
                        desc_bits.append(f"after {val}s")

        # Tipi di azione (foglia più informativa, es. ActivateControllerAction, SpeedAction, ecc.)
        action_types = []
        for act in actions:
            # cerca la prima foglia significativa dentro <Action>
            stack = list(act)
            leaf_type = None
            while stack:
                node = stack.pop(0)
                children = list(node)
                if not children:
                    # usa il tag della foglia
                    if isinstance(node.tag, str):
                        leaf_type = node.tag
                        break
                else:
                    stack.extend(children)
            if leaf_type is None:
                # fallback: primo figlio diretto di Action (PrivateAction/GlobalAction/..)
                child = next((c for c in list(act) if isinstance(c.tag, str)), None)
                if child is not None:
                    leaf_type = child.tag
            if leaf_type:
                action_types.append(leaf_type)

        if action_types:
            desc_bits.append("actions=" + ",".join(action_types))

        if desc_bits:
            feat["events"].append(" ".join(desc_bits))

    return feat

# ---------------------------------
# Riduzione on-the-fly dell'XML gold
# ---------------------------------

# Opzionali
try:
    import evaluate
    meteor_metric = evaluate.load("meteor")
except Exception:
    meteor_metric = None

try:
    bertscore = evaluate.load("bertscore")
except Exception:
    bertscore = None

def norm_edit_distance(a: str, b: str) -> float:
    n, m = len(a), len(b)
    if n == 0 and m == 0: return 1.0
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev, dp[0] = dp[0], i
        for j in range(1, m+1):
            cur = min(dp[j] + 1, dp[j-1] + 1, prev + (a[i-1] != b[j-1]))
            prev, dp[j] = dp[j], cur
    dist = dp[m]
    return 1.0 - (dist / max(n, m))

def extract_xml_tags(x: str):
    return set(re.findall(r"<\s*([A-Za-z_][\w:\.-]*)\b", x))

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0

def bool_f1(y_true, y_pred):
    keys = sorted(set(y_true.keys()) | set(y_pred.keys()))
    yt = [bool(y_true.get(k, False)) for k in keys]
    yp = [bool(y_pred.get(k, False)) for k in keys]
    return (
        precision_score(yt, yp, zero_division=0),
        recall_score(yt, yp, zero_division=0),
        f1_score(yt, yp, zero_division=0),
        accuracy_score(yt, yp),
    )

def xml_ok(x):
    try:
        etree.fromstring(x.encode("utf-8"))
        return True, None
    except Exception as e:
        return False, str(e)

def xsd_ok(x, xsd_path):
    if not xsd_path or not os.path.isfile(xsd_path):
        return None, "no_xsd"
    try:
        schema = etree.XMLSchema(etree.parse(xsd_path))
        xml = etree.fromstring(x.encode("utf-8"))
        return schema.validate(xml), None
    except Exception as e:
        return False, str(e)

def build_prompt(sys_tmpl: str, system: str, user: str) -> str:
    prompt = sys_tmpl.format(system=system.strip(), user=(user.strip()))
    if prompt.endswith("[/INST]"):
        prompt += "\n"
    return prompt

def generate(model, tok, prompt, max_new_tokens, do_sample, temperature, top_p):
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]

    stopper = StopOnSubstrings(tok, [STOP_STR])
    stopper.set_start_len(prompt_len)

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            stopping_criteria=StoppingCriteriaList([stopper]),
        )

    # 🔑 Decodifica SOLO i token generati (escludi il prompt)
    gen_ids = out[0][prompt_len:]
    txt = tok.decode(gen_ids, skip_special_tokens=True)

    m = re.search(r"<OpenScenario\b.*</OpenScenario>", txt, flags=re.DOTALL)
    if m:
        return m.group(0).strip()

    # fallback: se ha incluso testo extra prima del tag
    if "</OpenScenario>" in txt:
        return (txt.split("</OpenScenario>")[0].split("<OpenScenario")[-1].rpartition("<")[0] \
                and "<OpenScenario" + txt.split("<OpenScenario",1)[-1].split("</OpenScenario>")[0] + "</OpenScenario>") \
            or (txt.split("</OpenScenario>")[0].strip() + "</OpenScenario>")

    return txt.strip()

def compute_ppl(model, tok, prompt: str, gold: str) -> float:
    # Assicurati che il gold termini con il tag di chiusura (come nel training)
    if not gold.strip().endswith("</OpenScenario>"):
        gold = gold.strip() + "</OpenScenario>"

    enc_prompt = tok(prompt, return_tensors="pt")
    enc_full = tok(prompt + gold, return_tensors="pt")

    k = enc_prompt.input_ids.shape[1]
    input_ids = enc_full.input_ids.to(model.device)
    attention_mask = enc_full.attention_mask.to(model.device)

    labels = input_ids.clone()
    labels[:, :k] = -100  # maschera la parte di prompt: loss solo sull'assistant

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = float(out.loss.item())
    return loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/lora-codellama13b.yaml")
    ap.add_argument("--split", default="validation", choices=["train","validation"])
    ap.add_argument("--limit", type=int, default=47)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))

    # Carica il tokenizer dalla repo LoRA (così prendi anche chat_template.jinja)
    try:
        tok = AutoTokenizer.from_pretrained(cfg["hf_model_repo"], use_fast=True)
    except Exception:
        # fallback al base model se la LoRA non contiene il tokenizer
        tok = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # allineato al training

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
    model = PeftModel.from_pretrained(model, cfg["hf_model_repo"])
    model.eval()

    # Allinea gli id per la generazione (alcuni modelli ne hanno di diversi)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id

    sys_tmpl = open(cfg["sys_template"], "r", encoding="utf-8").read()

    ds = load_dataset(cfg["hf_dataset_repo"], split="train")
    split = ds.train_test_split(test_size=cfg["val_split_ratio"], seed=42)
    subset = split["test"] if args.split == "validation" else split["train"]
    if args.limit:
        subset = subset.shuffle(seed=42)
        subset = subset.select(range(min(args.limit, len(subset))))

    preds, golds, rows = [], [], []
    #bleu_refs, bleu_hyps = [], []
    #rouge = RougeScorer(["rougeL"], use_stemmer=True)
    gen_times, gpu_utils, vram_gbs, ram_gbs = [], [], [], []
    ppl_losses = []

    total = len(subset)
    gen_records = []
    t0 = time.time()

    for i, ex in enumerate(subset, start=1):
        gold = ex["assistant"].strip()

        prompt = build_prompt(sys_tmpl, ex["system"], ex["user"])
        t_step_start = time.time()
        pred = generate(model, tok, prompt, cfg["max_length"], cfg["do_sample"], cfg["gen_temperature"], cfg["gen_top_p"])
        gen_times.append(time.time() - t_step_start)
        #print(pred)

        # Perplexity Losses
        ppl_losses.append(compute_ppl(model, tok, prompt, gold))

        # Metriche base
        well_pred, _ = xml_ok(pred)
        xsd_pred, _ = xsd_ok(pred, cfg["xsd_path"])
        exact = int(pred.strip() == gold.strip())

        # BLEU / ROUGE
        #bleu_hyps.append(pred)
        #bleu_refs.append([gold])
        #r = rouge.score(gold, pred)["rougeL"].fmeasure

        # Slot F1 (da feature extractor)
        p_slots, g_slots = extract_features_from_xosc(pred), extract_features_from_xosc(gold)

        def slots_to_bools(s, closing_text):
            return {
                # presenza mappa
                "has_map": bool(s.get("map")),
                # tempo del giorno
                "has_time": bool(s.get("time_of_day")),
                # condizioni meteo
                "has_clouds": bool((s.get("weather") or {}).get("cloud_state")),
                "has_precipitation": bool((s.get("weather") or {}).get("precipitation")),
                "has_fog": bool((s.get("weather") or {}).get("fog")),
                # entità
                "has_ego": any(e for e in s.get("entities", []) if e.get("type") == "ego"),
                # posizioni iniziali
                "has_init_positions": len(s.get("initial_positions", [])) > 0,
                # eventi/storyboard
                "has_events": len(s.get("events", [])) > 0,
                # chiusura corretta
                "ends_close": closing_text.strip().endswith("</OpenScenario>")
            }

        p_bools, g_bools = slots_to_bools(p_slots, pred), slots_to_bools(g_slots, gold)
        prec, rec, f1, acc = bool_f1(g_bools, p_bools)

        # chrF++
        #chrf = sacrebleu.corpus_chrf([pred], [[gold]]).score / 100.0

        # METEOR / BERTScore
        #meteor = None
        #if meteor_metric is not None:
        #    try:
        #        meteor = meteor_metric.compute(predictions=[pred], references=[gold])["meteor"]
        #    except Exception:
        #        meteor = None

        #bert_f1 = None
        #if bertscore is not None:
        #    try:
        #        bs = bertscore.compute(predictions=[pred], references=[gold], lang="en")
        #        bert_f1 = float(bs["f1"][0])
        #    except Exception:
        #        bert_f1 = None

        # Similarità/edit/jaccard/len
        #edit_sim = norm_edit_distance(pred, gold)
        #jac_tags = jaccard(extract_xml_tags(pred), extract_xml_tags(gold))
        #len_ratio = (len(pred) / len(gold)) if len(gold) else 0.0

        rows.append({
            "wellformed_pred": well_pred,
            "xsd_valid_pred": xsd_pred,
            "exact_match": exact,
            #"rougeL": r,
            "slot_precision": prec, "slot_recall": rec, "slot_f1": f1, "slot_acc": acc,
            #"chrfpp": chrf,
            #"meteor": meteor,
            #"bertscore_f1": bert_f1,
            #"edit_sim": edit_sim,
            #"jaccard_tags": jac_tags,
            #"len_ratio": len_ratio
        })

        preds.append(pred); golds.append(gold)

        if torch.cuda.is_available():
            try:
                if NVML_HANDLE:
                    u = pynvml.nvmlDeviceGetUtilizationRates(NVML_HANDLE)
                    m = pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE)
                    gpu_utils.append(float(u.gpu))  # %
                    vram_gbs.append(m.used / 1e9)  # GB
                else:
                    vram_gbs.append(torch.cuda.memory_allocated() / 1e9)
            except Exception:
                pass
        ram_gbs.append(psutil.Process().memory_info().rss / 1e9)

        gen_records.append({
            "id": i,
            "system": ex["system"],
            "user": ex["user"],
            "gold": gold,
            "prediction": pred
        })

        if True:
            avg = (time.time() - t0) / i
            eta = avg * (total - i)
            print(f"{i}/{total}  avg={avg:.1f}s/it  ETA={eta/60:.1f}m", flush=True)

    #refs = list(map(list, zip(*bleu_refs))) if bleu_refs else [[]]
    #bleu = sacrebleu.corpus_bleu(bleu_hyps, refs)

    import numpy as np
    def mean(key):
        vals = [row[key] for row in rows if row[key] is not None]
        return float(np.nanmean(vals)) if vals else 0.0

    total_gen_time = float(np.sum(gen_times)) if gen_times else 0.0
    throughput_scen_per_s = (len(gen_times) / total_gen_time) if total_gen_time > 0 else None

    avg_ppl = math.exp(float(np.mean(ppl_losses))) if ppl_losses else None

    metrics = {
        "count": len(rows),
        "xml_wellformed_rate": mean("wellformed_pred"),
        "xsd_valid_rate": mean("xsd_valid_pred") if cfg["xsd_path"] else None,
        "exact_match": mean("exact_match"),
        "perplexity": avg_ppl,
        #"bleu": float(bleu.score),
        #"rougeL_f1": mean("rougeL"),
        #"chrfpp": mean("chrfpp"),
        #"meteor": mean("meteor"),
        #"bertscore_f1": mean("bertscore_f1"),
        "slot_precision": mean("slot_precision"),
        "slot_recall": mean("slot_recall"),
        "slot_f1": mean("slot_f1"),
        "slot_accuracy": mean("slot_acc"),
        #"edit_similarity": mean("edit_sim"),
        #"jaccard_xml_tags": mean("jaccard_tags"),
        #"length_ratio_avg": mean("len_ratio"),
        "avg_generation_time_s": float(np.mean(gen_times)) if gen_times else None,
        "throughput_scen_per_s": throughput_scen_per_s,
        "avg_gpu_util_percent": float(np.mean(gpu_utils)) if gpu_utils else None,
        "avg_vram_gb": float(np.mean(vram_gbs)) if vram_gbs else None,
        "avg_ram_gb": float(np.mean(ram_gbs)) if ram_gbs else None
    }

    os.makedirs("runs/eval", exist_ok=True)
    with open("runs/eval/metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    with open("runs/eval/predictions.jsonl", "w", encoding="utf-8") as f:
        for rec in gen_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
