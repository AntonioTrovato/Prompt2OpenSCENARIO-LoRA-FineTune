import json, os, argparse
import matplotlib.pyplot as plt
import yaml

def _num(x, scale=1.0, default=0.0):
    """Converte in float gestendo None e stringhe. Applica uno scale opzionale."""
    if x is None:
        return default
    try:
        return float(x) * scale
    except Exception:
        return default

def _bar(ax, labels, values, title, ylim01=False):
    ax.bar(labels, values)
    vmax = max(values) if values else 1.0
    if ylim01:
        # aggiungo 10% di margine ma senza superare 1.1
        ax.set_ylim(0, min(1.1, vmax * 1.1))
    else:
        ax.set_ylim(0, vmax * 1.1 if vmax > 0 else 1.0)

    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v + (vmax * 0.02), f"{v:.2f}", ha="center", rotation=0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")

def main():
    parser = argparse.ArgumentParser(description="Plot evaluation summary (grouped bar charts)")
    parser.add_argument("--cfg", default="./config/lora-codellama13b.yaml", help="Path to YAML config.")
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

    # ---- 1) XML wellformed + XSD valid (0..1)
    labels = ["XML ok", "XSD ok"]
    values = [
        _num(m.get("xml_wellformed_rate")),
        _num(m.get("xsd_valid_rate")),
    ]
    fig, ax = plt.subplots()
    _bar(ax, labels, values, "XML wellformed & XSD valid", ylim01=True)
    fig.tight_layout(); fig.savefig(os.path.join(plots_dir, "eval_xml_xsd.png"), dpi=160); plt.close(fig)

    # ---- 2) BLEU (/100) + ROUGE-L F1 (0..1)
    labels = ["BLEU", "ROUGE-L F1"]
    values = [
        _num(m.get("bleu"), 1.0/100.0),
        _num(m.get("rougeL_f1")),
    ]
    fig, ax = plt.subplots()
    _bar(ax, labels, values, "BLEU (÷100) & ROUGE-L F1", ylim01=True)
    fig.tight_layout(); fig.savefig(os.path.join(plots_dir, "eval_bleu_rouge.png"), dpi=160); plt.close(fig)

    # ---- 3) CHRFPP + METEOR
    labels = ["CHRF++", "METEOR"]
    values = [
        _num(m.get("chrfpp")),
        _num(m.get("meteor")),
    ]
    fig, ax = plt.subplots()
    _bar(ax, labels, values, "CHRF++ & METEOR", ylim01=True)
    fig.tight_layout(); fig.savefig(os.path.join(plots_dir, "eval_chrfpp_meteor.png"), dpi=160); plt.close(fig)

    # ---- 4) Slot metrics
    labels = ["Slot Prec", "Slot Rec", "Slot Acc", "Slot F1"]
    values = [
        _num(m.get("slot_precision")),
        _num(m.get("slot_recall")),
        _num(m.get("slot_accuracy")),
        _num(m.get("slot_f1")),
    ]
    fig, ax = plt.subplots()
    _bar(ax, labels, values, "Slot Metrics", ylim01=True)
    fig.tight_layout(); fig.savefig(os.path.join(plots_dir, "eval_slot_metrics.png"), dpi=160); plt.close(fig)

    # ---- 5) Edit/Jaccard/Length
    labels = ["Edit Sim",
              "Jaccard Tags",
              "Length Ratio Avg"]
    values = [
        _num(m.get("edit_similarity")),
        _num(m.get("jaccard_xml_tags")),
        _num(m.get("length_ratio_avg")),
    ]
    fig, ax = plt.subplots()
    _bar(ax, labels, values, "Edit"
                             "/Jaccard"
                             "/Length Ratio", ylim01=True)
    fig.tight_layout(); fig.savefig(os.path.join(plots_dir, "eval"
                                                            "_jaccard"
                                                            "_edit_length.png"), dpi=160); plt.close(fig)

    # ---- 6) Memory usage
    labels = ["Avg VRAM (GB)", "Avg RAM (GB)"]
    values = [
        _num(m.get("avg_vram_gb")),
        _num(m.get("avg_ram_gb")),
    ]
    fig, ax = plt.subplots()
    _bar(ax, labels, values, "Memory Usage")
    ax.set_ylabel("GB")
    fig.tight_layout(); fig.savefig(os.path.join(plots_dir, "eval_memory.png"), dpi=160); plt.close(fig)

if __name__ == "__main__":
    main()
