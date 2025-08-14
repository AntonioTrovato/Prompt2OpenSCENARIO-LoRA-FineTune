import os, json, argparse, re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from lxml import etree
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import sacrebleu
from rouge_score.rouge_scorer import RougeScorer

# Usa l'estrattore definito in train (stessa logica delle feature)
from scripts.train import extract_features_from_xosc, minify_xml, build_minimal_xosc_from_features

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

def generate(model, tok, prompt, max_new_tokens, temperature, top_p):
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id
    )
    txt = tok.decode(out[0], skip_special_tokens=True)
    m = re.search(r"<OpenSCENARIO\b.*</OpenSCENARIO>", txt, flags=re.DOTALL)
    if m:
        return m.group(0).strip()
    # fallback: prendi dal prompt in poi
    rest = txt.split(prompt)[-1].strip()
    if "</OpenSCENARIO>" in rest:
        return rest.split("</OpenSCENARIO>")[0].strip() + "</OpenSCENARIO>"
    return rest

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
    ap.add_argument("--gen_max_new_tokens", type=int, default=4000)
    ap.add_argument("--gen_temperature", type=float, default=0.3)
    ap.add_argument("--gen_top_p", type=float, default=0.9)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(model, args.lora_repo)
    model.eval()

    sys_tmpl = open(args.sys_template, "r", encoding="utf-8").read()

    ds = load_dataset(args.hf_dataset_repo, split="train")
    # split 90/10 come in train.py
    n = len(ds); cut = int(n * 0.9)
    subset = ds.select(range(cut, n)) if args.split == "validation" else ds.select(range(0, cut))
    if args.limit:
        subset = subset.select(range(min(args.limit, len(subset))))

    preds, golds, rows = [], [], []
    bleu_refs, bleu_hyps = [], []
    rouge = RougeScorer(["rougeL"], use_stemmer=True)

    for ex in subset:
        gold_full = ex["assistant"].strip()
        # gold ridotto secondo policy scelta (coerente col training)
        gold = reduce_gold(gold_full, args.reduce_gold)

        prompt = build_prompt(sys_tmpl, ex["system"], ex["user"], gold_full, args.use_feature_hints)
        pred = generate(model, tok, prompt, args.gen_max_new_tokens, args.gen_temperature, args.gen_top_p)

        # Metriche base
        well_pred, _ = xml_ok(pred)
        xsd_pred, _ = xsd_ok(pred, args.xsd_path)
        exact = int(pred.strip() == gold.strip())

        # BLEU / ROUGE
        bleu_hyps.append(pred)
        bleu_refs.append([gold])
        r = rouge.score(gold, pred)["rougeL"].fmeasure

        # Slot F1 (da feature extractor)
        p_slots, g_slots = extract_features_from_xosc(pred), extract_features_from_xosc(gold)
        def slots_to_bools(s, closing_text):
            return {
                "has_map": bool(s.get("map")),
                "has_time": bool(s.get("time_of_day")),
                "has_weather": any(v for v in (s.get("weather") or {}).values() if v),
                "veh_ge1": sum(1 for e in s.get("entities", []) if e.get("type") != "pedestrian") >= 1,
                "veh_ge2": sum(1 for e in s.get("entities", []) if e.get("type") != "pedestrian") >= 2,
                "ends_close": closing_text.strip().endswith("</OpenSCENARIO>")
            }
        p_bools, g_bools = slots_to_bools(p_slots, pred), slots_to_bools(g_slots, gold)
        prec, rec, f1, acc = bool_f1(g_bools, p_bools)

        # chrF++
        chrf = sacrebleu.corpus_chrf([pred], [[gold]]).score / 100.0

        # METEOR / BERTScore
        meteor = None
        if meteor_metric is not None:
            try:
                meteor = meteor_metric.compute(predictions=[pred], references=[gold])["meteor"]
            except Exception:
                meteor = None

        bert_f1 = None
        if bertscore is not None:
            try:
                bs = bertscore.compute(predictions=[pred], references=[gold], lang="en")
                bert_f1 = float(bs["f1"][0])
            except Exception:
                bert_f1 = None

        # Similarità/edit/jaccard/len
        edit_sim = norm_edit_distance(pred, gold)
        jac_tags = jaccard(extract_xml_tags(pred), extract_xml_tags(gold))
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

    refs = list(map(list, zip(*bleu_refs))) if bleu_refs else [[]]
    bleu = sacrebleu.corpus_bleu(bleu_hyps, refs)

    import numpy as np
    def mean(key):
        vals = [row[key] for row in rows if row[key] is not None]
        return float(np.nanmean(vals)) if vals else 0.0

    metrics = {
        "count": len(rows),
        "xml_wellformed_rate": mean("wellformed_pred"),
        "xsd_valid_rate": mean("xsd_valid_pred") if args.xsd_path else None,
        "exact_match": mean("exact_match"),
        "bleu": float(bleu.score),
        "rougeL_f1": mean("rougeL"),
        "chrfpp": mean("chrfpp"),
        "meteor": mean("meteor"),
        "bertscore_f1": mean("bertscore_f1"),
        "slot_precision": mean("slot_precision"),
        "slot_recall": mean("slot_recall"),
        "slot_f1": mean("slot_f1"),
        "slot_accuracy": mean("slot_acc"),
        "edit_similarity": mean("edit_sim"),
        "jaccard_xml_tags": mean("jaccard_tags"),
        "length_ratio_avg": mean("len_ratio")
    }

    os.makedirs("runs/eval", exist_ok=True)
    with open("runs/eval/metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
