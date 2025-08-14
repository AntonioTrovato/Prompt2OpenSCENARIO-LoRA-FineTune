import os, json, argparse, re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from lxml import etree
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import sacrebleu
from rouge_score.rouge_scorer import RougeScorer
from collections import Counter

from scripts.train import extract_features_from_xosc

try:
    import evaluate
    meteor_metric = evaluate.load("meteor")  # può scaricare asset al primo run
except Exception:
    meteor_metric = None

# BERTScore (opzionale)
try:
    bertscore = evaluate.load("bertscore")  # scarica modelli al primo run
except Exception:
    bertscore = None

def norm_edit_distance(a: str, b: str) -> float:
    """Normalized Levenshtein distance in [0,1]; 1=identico."""
    # DP semplice (char-level)
    n, m = len(a), len(b)
    if n == 0 and m == 0: return 1.0
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev, dp[0] = dp[0], i
        for j in range(1, m+1):
            cur = min(
                dp[j] + 1,            # delete
                dp[j-1] + 1,          # insert
                prev + (a[i-1] != b[j-1])  # substitute
            )
            prev, dp[j] = dp[j], cur
    dist = dp[m]
    maxlen = max(n, m)
    return 1.0 - (dist / maxlen)

def extract_xml_tags(x: str):
    # prende i nomi dei tag (apertura) senza slash, namespace semplice
    import re
    return set(re.findall(r"<\s*([A-Za-z_][\w:\.-]*)\b", x))

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def extract_slots(xosc_text:str):
    """Estrai slot chiave per F1 (semplificato ma efficace)."""
    slots = {}
    # mappa
    m = re.search(r"<LogicFile\s+filepath=\"([^\"]+)\"", xosc_text)
    if m: slots["map"] = m.group(1).lower()
    # presenza environment/time/weather
    slots["has_env"] = "EnvironmentAction" in xosc_text
    slots["has_time"] = "TimeOfDay" in xosc_text
    slots["has_weather"] = "Weather" in xosc_text
    # numero veicoli dichiarati
    vehs = re.findall(r"<Vehicle\b", xosc_text)
    slots["num_vehicles_ge1"] = len(vehs) >= 1
    slots["num_vehicles_ge2"] = len(vehs) >= 2
    # close tag
    slots["ends_close"] = xosc_text.strip().endswith("</OpenSCENARIO>")
    return slots

def bool_f1(y_true, y_pred):
    # converte dict booleani in liste parallele
    keys = sorted(set(y_true.keys()) | set(y_pred.keys()))
    yt = [bool(y_true.get(k, False)) for k in keys]
    yp = [bool(y_pred.get(k, False)) for k in keys]
    return (precision_score(yt, yp), recall_score(yt, yp), f1_score(yt, yp), accuracy_score(yt, yp))

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

def generate(model, tok, prompt, eos):
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        ** enc,
        max_new_tokens = 1800, temperature = 0.3, top_p = 0.9,
        pad_token_id = tok.pad_token_id,
        eos_token_id = tok.eos_token_id  # eos "normale", NON lo stop custom
    )
    txt = tok.decode(out[0], skip_special_tokens=True)
    # estrai SOLO l'XML, dal primo <OpenSCENARIO ...> fino a </OpenSCENARIO>
    m = re.search(r"<OpenSCENARIO\b.*</OpenSCENARIO>", txt, flags=re.DOTALL)
    if m:
        return m.group(0).strip()
    # fallback: taglia dopo la stop sequence se presente
    if eos in txt:
        return txt.split(eos)[0].strip() + eos
    # ultimo fallback: rimuovi il prompt e restituisci il resto
    return txt.split(prompt)[-1].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="codellama/CodeLlama-13b-Instruct-hf")
    ap.add_argument("--lora_repo", required=True)
    ap.add_argument("--hf_dataset_repo", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--xsd_path", default="")
    ap.add_argument("--sys_template", default="../prompt_templates/codellama_inst.txt")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(model, args.lora_repo)
    model.eval()

    sys_tmpl = open(args.sys_template, "r", encoding="utf-8").read()
    eos = "</OpenSCENARIO>"

    ds = load_dataset(args.hf_dataset_repo, split="train")
    # ricrea split locale per coerenza con train.py: usiamo il 10% tail come val
    n = len(ds)
    cut = int(n*0.9)
    val = ds.select(range(cut, n)) if args.split=="validation" else ds.select(range(0, cut))
    if args.limit: val = val.select(range(min(args.limit, len(val))))

    preds, golds = [], []
    rows = []
    bleu_refs, bleu_hyps = [], []
    rouge = RougeScorer(["rougeL"], use_stemmer=True)

    for ex in val:
        prompt = sys_tmpl.format(system=ex["system"].strip(), user=ex["user"].strip())
        pred = generate(model, tok, prompt, eos)
        gold = ex["assistant"].strip()

        # metriche base
        well_pred, _ = xml_ok(pred)
        well_gold, _ = xml_ok(gold)
        xsd_pred, _ = xsd_ok(pred, args.xsd_path)
        exact = int(pred.strip() == gold.strip())

        # BLEU / ROUGE
        bleu_hyps.append(pred)
        bleu_refs.append([gold])
        r = rouge.score(gold, pred)["rougeL"].fmeasure

        # slot F1 (booleane)
        p_slots, g_slots = extract_features_from_xosc(pred), extract_features_from_xosc(gold)
        # converti in boolean slots sintetici per F1 (come prima)
        def slots_to_bools(s):
            return {
                "has_map": bool(s.get("map")),
                "has_time": bool(s.get("time_of_day")),
                "has_weather": any(v for v in (s.get("weather") or {}).values() if v),
                "veh_ge1": sum(1 for e in s.get("entities", []) if e.get("type") != "pedestrian") >= 1,
                "veh_ge2": sum(1 for e in s.get("entities", []) if e.get("type") != "pedestrian") >= 2,
                "ends_close": pred.strip().endswith("</OpenSCENARIO>")
            }
        p_slots, g_slots = slots_to_bools(p_slots), slots_to_bools(g_slots)
        prec, rec, f1, acc = bool_f1(g_slots, p_slots)

        # --- ADD: chrF++ ---
        chrf = sacrebleu.corpus_chrf([pred], [[gold]]).score / 100.0  # [0..1]

        # --- ADD: METEOR (opzionale) ---
        if meteor_metric is not None:
            try:
                meteor = meteor_metric.compute(predictions=[pred], references=[gold])["meteor"]
            except Exception:
                meteor = None
        else:
            meteor = None

        # --- ADD: BERTScore (opzionale) ---
        if bertscore is not None:
            try:
                bs = bertscore.compute(predictions=[pred], references=[gold], lang="en")
                bert_f1 = float(bs["f1"][0])  # tipicamente ~[0..1]
            except Exception:
                bert_f1 = None
        else:
            bert_f1 = None

        # --- ADD: Normalized edit similarity ---
        edit_sim = norm_edit_distance(pred, gold)  # [0..1], 1 meglio

        # --- ADD: Jaccard sui tag XML ---
        jac_tags = jaccard(extract_xml_tags(pred), extract_xml_tags(gold))

        # --- ADD: length ratio ---
        len_ratio = (len(pred) / len(gold)) if len(gold) else 0.0

        rows.append({
            "wellformed_pred": well_pred,
            "xsd_valid_pred": xsd_pred,
            "exact_match": exact,
            "rougeL": r,
            "slot_precision": prec, "slot_recall": rec, "slot_f1": f1, "slot_acc": acc,
            "chrfpp": chrf,
            "meteor": meteor,
            "bertscore_f1": bert_f1,
            "edit_sim": edit_sim,
            "jaccard_tags": jac_tags,
            "len_ratio": len_ratio
        })

        preds.append(pred); golds.append(gold)

    refs = list(map(list, zip(*bleu_refs)))  # shape: [n_refs][n_hyps]
    bleu = sacrebleu.corpus_bleu(bleu_hyps, refs)
    # aggregazioni
    import numpy as np, json, os
    def mean(key): return float(np.nanmean([row[key] for row in rows if row[key] is not None]))

    metrics = {
        "count": len(rows),
        "xml_wellformed_rate": mean("wellformed_pred"),
        "xsd_valid_rate": mean("xsd_valid_pred") if args.xsd_path else None,
        "exact_match": mean("exact_match"),

        # string-match
        "bleu": float(bleu.score),  # 0..100
        "rougeL_f1": mean("rougeL"),  # 0..1
        "chrfpp": mean("chrfpp"),  # 0..1
        "meteor": mean("meteor"),  # 0..1 (None se non calcolato)
        "bertscore_f1": mean("bertscore_f1"),  # 0..1 (None se non calcolato)

        # struttura/qualità
        "slot_precision": mean("slot_precision"),
        "slot_recall": mean("slot_recall"),
        "slot_f1": mean("slot_f1"),
        "slot_accuracy": mean("slot_acc"),

        # similarità superficiale/strutturale
        "edit_similarity": mean("edit_sim"),  # 0..1
        "jaccard_xml_tags": mean("jaccard_tags"),
        "length_ratio_avg": mean("len_ratio")
    }

    os.makedirs("runs/eval", exist_ok=True)
    json.dump(metrics, open("runs/eval/metrics.json","w"), indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
