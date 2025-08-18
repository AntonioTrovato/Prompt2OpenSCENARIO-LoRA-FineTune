import os, json, argparse, re
import time

import torch
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

class StopOnSequence(StoppingCriteria):
    def __init__(self, stop_ids):
        super().__init__()
        self.stop_ids = stop_ids
        self.k = len(stop_ids)
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < self.k:
            return False
        return (input_ids[0, -self.k:].tolist() == self.stop_ids)

# Usa l'estrattore definito in train (stessa logica delle feature)
#from scripts.train import extract_features_from_xosc, minify_xml, build_minimal_xosc_from_features
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
        parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
        root = etree.fromstring(x.encode("utf-8"), parser=parser)
        return etree.tostring(root, encoding="unicode", pretty_print=False)
    except Exception:
        return x

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

    fh = E("FileHeader", revMajor="1", revMinor="0", date="2025-01-01T00:00:00", description="Minimal scenario")
    root.append(fh)

    rn = E("RoadNetwork")
    rn.append(E("LogicFile", filepath=map_path))
    root.append(rn)

    root.append(E("ParameterDeclarations"))
    root.append(E("CatalogLocations"))

    ents = E("Entities")
    for ent in entities:
        nm = ent.get("name") or "ego"
        typ = ent.get("type") or "vehicle"
        if typ == "pedestrian":
            obj = E("Pedestrian", name=nm)
        elif typ == "misc":
            obj = E("MiscObject", name=nm)
        else:
            obj = E("Vehicle", name=nm)
        ents.append(obj)
    root.append(ents)

    sb = E("Storyboard")

    init = E("Init")
    actions = E("Actions")
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
    init.append(actions)
    sb.append(init)

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
        st = E("StartTrigger")
        cg = E("ConditionGroup")
        c = E("Condition", delay="0", conditionEdge="rising")
        byv = E("ByValueCondition")
        byv.append(E("SimulationTimeCondition", value="0", rule="greaterThan"))
        c.append(byv); cg.append(c); st.append(c)
        ev.append(st)
        man.append(ev); mg.append(man)
        act.append(mg)
    st_act = E("StartTrigger")
    cg_act = E("ConditionGroup")
    c_act = E("Condition", delay="0", conditionEdge="rising")
    byv_act = E("ByValueCondition")
    byv_act.append(E("SimulationTimeCondition", value="0", rule="greaterThan"))
    c_act.append(byv_act); cg_act.append(c_act); st_act.append(c_act)
    act.append(st_act)
    story.append(act)
    sb.append(story)
    root.append(sb)

    return etree.tostring(root, encoding="unicode", pretty_print=False)

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

def reduce_gold(xosc_text: str, mode: str) -> str:
    if mode == "minify_only":
        return minify_xml(xosc_text)
    if mode == "features_skeleton":
        try:
            feats = extract_features_from_xosc(xosc_text)
            return build_minimal_xosc_from_features(feats)
        except Exception:
            return minify_xml(xosc_text)
    return xosc_text  # "none"

def build_prompt(sys_tmpl: str, system: str, user: str, gold_assistant: str, use_feature_hints: bool) -> str:
    hints_block = ""
    if use_feature_hints:
        try:
            feats = extract_features_from_xosc(gold_assistant)
            hints_block = "\n\n<HINTS>\n" + json.dumps(feats, ensure_ascii=False) + "\n</HINTS>\n"
        except Exception:
            pass
    prompt = sys_tmpl.format(system=system.strip(), user=(user.strip() + hints_block))
    if prompt.endswith("[/INST]"):
        prompt += "\n"
    return prompt

def generate(model, tok, prompt, max_new_tokens, temperature, top_p, stops):
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.inference_mode():            # <<< qui
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            stopping_criteria=stops,
        )
    txt = tok.decode(out[0], skip_special_tokens=True)
    m = re.search(r"<OpenScenario\b.*</OpenScenario>", txt, flags=re.DOTALL)
    if m:
        return m.group(0).strip()
    # fallback: prendi dal prompt in poi
    rest = txt.split(prompt)[-1].strip()
    if "</OpenScenario>" in rest:
        return rest.split("</OpenScenario>")[0].strip() + "</OpenScenario>"
    return rest

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
    ap.add_argument("--base_model", default="codellama/CodeLlama-13b-Instruct-hf")
    ap.add_argument("--lora_repo", required=True)
    ap.add_argument("--hf_dataset_repo", required=True)
    ap.add_argument("--split", default="validation", choices=["train","validation"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--xsd_path", default="")
    ap.add_argument("--sys_template", default="prompt_templates/codellama_inst.txt")
    ap.add_argument("--use_feature_hints", action="store_true")
    ap.add_argument("--reduce_gold", choices=["none","minify_only","features_skeleton"], default="minify_only")
    ap.add_argument("--gen_max_new_tokens", type=int, default=1500)
    ap.add_argument("--gen_temperature", type=float, default=0.1)
    ap.add_argument("--gen_top_p", type=float, default=0.9)
    args = ap.parse_args()

    # Carica il tokenizer dalla repo LoRA (così prendi anche chat_template.jinja)
    try:
        tok = AutoTokenizer.from_pretrained(args.lora_repo, use_fast=True)
    except Exception:
        # fallback al base model se la LoRA non contiene il tokenizer
        tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # allineato al training

    stop_ids = tok("</OpenScenario>", add_special_tokens=False).input_ids
    stops = StoppingCriteriaList([StopOnSequence(stop_ids)])

    from transformers import BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # ok anche float16 se preferisci
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.lora_repo)
    model.eval()

    # Allinea gli id per la generazione (alcuni modelli ne hanno di diversi)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id

    sys_tmpl = open(args.sys_template, "r", encoding="utf-8").read()

    ds = load_dataset(args.hf_dataset_repo, split="train")
    split = ds.train_test_split(test_size=0.1, seed=42)
    subset = split["test"] if args.split == "validation" else split["train"]
    if args.limit:
        subset = subset.shuffle(seed=42)
        subset = subset.select(range(min(args.limit, len(subset))))

    preds, golds, rows = [], [], []
    bleu_refs, bleu_hyps = [], []
    rouge = RougeScorer(["rougeL"], use_stemmer=True)
    gen_times, gpu_utils, vram_gbs, ram_gbs = [], [], [], []
    ppl_losses = []

    total = len(subset)
    t0 = time.time()

    if args.use_feature_hints:
        print("Using features hints")

    for i, ex in enumerate(subset, start=1):

        gold_full = ex["assistant"].strip()
        # gold ridotto secondo policy scelta (coerente col training)
        gold = reduce_gold(gold_full, args.reduce_gold)

        prompt = build_prompt(sys_tmpl, ex["system"], ex["user"], gold_full, args.use_feature_hints)
        t_step_start = time.time()
        pred = generate(model, tok, prompt, args.gen_max_new_tokens, args.gen_temperature, args.gen_top_p, stops)
        gen_times.append(time.time() - t_step_start)

        # Perplexity Losses
        #ppl_losses.append(compute_ppl(model, tok, prompt, gold))

        # Metriche base
        well_pred, _ = xml_ok(pred)
        xsd_pred, _ = xsd_ok(pred, args.xsd_path)
        exact = int(pred.strip() == gold.strip())

        # BLEU / ROUGE
        #bleu_hyps.append(pred)
        #bleu_refs.append([gold])
        #r = rouge.score(gold, pred)["rougeL"].fmeasure

        # Slot F1 (da feature extractor)
        p_slots, g_slots = extract_features_from_xosc(pred), extract_features_from_xosc(gold)
        def slots_to_bools(s, closing_text):
            return {
                "has_map": bool(s.get("map")),
                "has_time": bool(s.get("time_of_day")),
                "has_weather": any(v for v in (s.get("weather") or {}).values() if v),
                "veh_ge1": sum(1 for e in s.get("entities", []) if e.get("type") != "pedestrian") >= 1,
                "veh_ge2": sum(1 for e in s.get("entities", []) if e.get("type") != "pedestrian") >= 2,
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
    throughput_scen_per_min = (throughput_scen_per_s * 60.0) if throughput_scen_per_s else None

    #avg_ppl = math.exp(float(np.mean(ppl_losses))) if ppl_losses else None

    metrics = {
        "count": len(rows),
        "xml_wellformed_rate": mean("wellformed_pred"),
        "xsd_valid_rate": mean("xsd_valid_pred") if args.xsd_path else None,
        "exact_match": mean("exact_match"),
        #"perplexity": avg_ppl,
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
        "throughput_scen_per_min": throughput_scen_per_min,
        "avg_gpu_util_percent": float(np.mean(gpu_utils)) if gpu_utils else None,
        "avg_vram_gb": float(np.mean(vram_gbs)) if vram_gbs else None,
        "avg_ram_gb": float(np.mean(ram_gbs)) if ram_gbs else None
    }

    os.makedirs("runs/eval", exist_ok=True)
    with open("runs/eval/metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
