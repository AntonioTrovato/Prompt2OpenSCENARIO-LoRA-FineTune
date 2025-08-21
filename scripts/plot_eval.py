import json, os, argparse
import matplotlib.pyplot as plt
import yaml

def main():
    parser = argparse.ArgumentParser(description="Plot evaluation summary from metrics.json")
    parser.add_argument("--cfg", default="./configlora-codellama13b.yaml", help="Path to YAML config.")
    parser.add_argument("--input_path", required=True, help="Path to metrics.json")
    args = parser.parse_args()

    # Load config
    if not os.path.isfile(args.cfg):
        raise FileNotFoundError(f"Config file not found: {args.cfg}")
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    plots_dir = cfg.get("plots_dir", "runs/plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load metrics
    with open(args.input_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    labels = [
        "XML ok","XSD ok","Exact","Perplexity","BLEu","ROUGE-L F1","CHRFPP","METEOR","BERT Score F1",
        "Slot Prec","Slot Rec","Slot Acc","Slot F1","Edit Similarity","Jaccard XML Tags","Length Ratio Avg",
        "Avg Generation Time (s)", "Throughput Scenarios per Gen (s)","Avg GPU Util (%)", "Avg VRAM GBs", "Avg RAM GBs"
    ]

    values = [
        m.get("xml_wellformed_rate", 0) or 0,                                  # XML ok
        (m.get("xsd_valid_rate", 0) or 0) if m.get("xsd_valid_rate") is not None else 0,   # XSD ok
        m.get("exact_match", 0) or 0,                                          # Exact
        m.get("perplexity", 0) or 0,                                           # Perplexity
        (m.get("bleu", 0) or 0) / 100.0 if m.get("bleu") is not None else 0,   # BLEU
        m.get("rougeL_f1", 0) or 0,                                            # ROUGE-L F1
        m.get("chrfpp", 0) or 0,                                               # CHRFPP
        m.get("meteor", 0) or 0,                                               # METEOR
        m.get("bertscore_f1", 0) or 0,                                         # BERT Score F1
        m.get("slot_precision", 0) or 0,                                       # Slot Prec
        m.get("slot_recall", 0) or 0,                                          # Slot Rec
        m.get("slot_accuracy", 0) or 0,                                        # Slot Acc
        m.get("slot_f1", 0) or 0,                                              # Slot F1
        m.get("edit_similarity", 0) or 0,                                      # Edit Similarity
        m.get("jaccard_xml_tags", 0) or 0,                                     # Jaccard XML Tags
        m.get("length_ratio_avg", 0) or 0,                                     # Length Ratio Avg
        m.get("avg_generation_time_s", 0) or 0,                                # Avg Generation Time (s)
        m.get("throughput_scen_per_s", 0) or 0,                                # Throughput Scenarios per Gen (s)
        m.get("avg_gpu_util_percent", 0) or 0,                                 # Avg GPU Util (%)
        m.get("avg_vram_gb", 0) or 0,                                          # Avg VRAM GBs
        m.get("avg_ram_gb", 0) or 0                                            # Avg RAM GBs
    ]

    plt.figure()
    plt.bar(labels, values)
    plt.ylim(0, 1)  # mantiene il comportamento originale (alcune metriche potrebbero superare 1)
    plt.title("Evaluation summary")
    for i, v in enumerate(values):
        try:
            txt = f"{float(v):.2f}"
        except Exception:
            txt = str(v)
        plt.text(i, (v if isinstance(v, (int, float)) else 0) + 0.01, txt, ha="center", rotation=0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = os.path.join(plots_dir, "eval_summary.png")
    plt.savefig(out_path, dpi=160)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
