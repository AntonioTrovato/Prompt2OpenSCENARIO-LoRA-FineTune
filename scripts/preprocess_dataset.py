# scripts/preprocess_dataset.py
import argparse, os, re, json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from lxml import etree

def minify_xml(x: str) -> str:
    try:
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.fromstring(x.encode("utf-8"), parser=parser)
        return etree.tostring(root, encoding="unicode", pretty_print=False)
    except Exception:
        return x  # se fallisce, torna com'è

def skeletonize_xml(x: str) -> str:
    """
    Best-effort: rimuove o compatta elementi noti super lunghi.
    Esempio: traiettorie con centinaia di waypoint, cataloghi pesanti, ecc.
    Adatta i tag al tuo dominio reale.
    """
    try:
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.fromstring(x.encode("utf-8"), parser=parser)

        # Rimuovi punti intermedi e lascia solo pochi keypoints (esempio generico)
        for traj in root.findall(".//Trajectory"):
            # taglia i Vertex in eccesso
            verts = traj.findall(".//Vertex")
            if len(verts) > 20:
                keep = [*verts[:5], *verts[-5:]]  # primi 5 + ultimi 5
                for v in verts:
                    if v not in keep:
                        p = v.getparent(); p.remove(v)

        # Eventuali cataloghi ingombranti: rimuovi descrizioni/Comment tag
        for cm in root.findall(".//Comment"):
            cm.getparent().remove(cm)

        return etree.tostring(root, encoding="unicode", pretty_print=False)
    except Exception:
        return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dataset_repo", required=True)
    ap.add_argument("--base_model", default="codellama/CodeLlama-13b-Instruct-hf")
    ap.add_argument("--max_target_tokens", type=int, default=3500)
    ap.add_argument("--apply_skeleton", action="store_true")
    ap.add_argument("--out_jsonl", default="runs/preprocessed_dataset.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset(args.hf_dataset_repo, split="train")
    kept, skipped = 0, 0
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for ex in ds:
            sys, usr, asst = ex["system"], ex["user"], ex["assistant"]
            # 1) minify
            asst_min = minify_xml(asst)
            # 2) skeleton (opzionale)
            if args.apply_skeleton:
                asst_min = skeletonize_xml(asst_min)
            # 3) calcola token dell'OUTPUT (assistant)
            tgt_ids = tok(asst_min, add_special_tokens=False)["input_ids"]
            if len(tgt_ids) <= args.max_target_tokens:
                f.write(json.dumps({"system": sys, "user": usr, "assistant": asst_min}, ensure_ascii=False)+"\n")
                kept += 1
            else:
                skipped += 1
    print(f"Kept: {kept}, Skipped: {skipped}, saved -> {args.out_jsonl}")

if __name__ == "__main__":
    main()
