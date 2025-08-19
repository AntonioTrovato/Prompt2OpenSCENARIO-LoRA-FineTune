import json, matplotlib.pyplot as plt, os
m = json.load(open("runs/eval/metrics.json"))
labels = ["XML ok","XSD ok","Exact","Perplexity","BLEu","ROUGE-L F1","CHRFPP","METEOR","BERT Score F1", "Slot Prec","Slot Rec","Slot Acc","Slot F1",
          "Edit Similarity","Jaccard XML Tags","Length Ratio Avg","Avg Generation Time (s)", "Throughput Scenarios per Gen (s)",
          "Avg GPU Util (%)", "Avg VRAM GBs", "Avg RAM GBs"]
values = [
    m["xml_wellformed_rate"] or 0,                                  # XML ok
    (m["xsd_valid_rate"] or 0) if m.get("xsd_valid_rate") else 0,   # XSD ok
    m["exact_match"] or 0,                                          # Exact
    #m.get("perplexity", 0) or 0,                                    # Perplexity
    #(m["bleu"] or 0) / 100.0 if m.get("bleu") is not None else 0,   # BLEU
    #m.get("rougeL_f1", 0) or 0,                                     # ROUGE-L F1
    #m.get("chrfpp", 0) or 0,                                        # CHRFPP
    #m.get("meteor", 0) or 0,                                        # METEOR
    #m.get("bertscore_f1", 0) or 0,                                  # BERT Score F1
    m["slot_precision"] or 0,                                       # Slot Prec
    m["slot_recall"] or 0,                                          # Slot Rec
    m["slot_accuracy"] or 0,                                        # Slot Acc
    m["slot_f1"] or 0,                                              # Slot F1
    #m.get("edit_similarity", 0) or 0,                               # Edit Similarity
    #m.get("jaccard_xml_tags", 0) or 0,                              # Jaccard XML Tags
    #m.get("length_ratio_avg", 0) or 0,                              # Length Ratio Avg
    m["avg_generation_time_s"] or 0,                                # Avg Generation Time (s)
    m["throughput_scen_per_s"] or 0,                                # Throughput Scenarios per Gen (s)
    m["avg_gpu_util_percent"] or 0,                                 # Avg GPU Util (%)
    m["avg_vram_gb"] or 0,                                          # Avg VRAM GBs
    m["avg_ram_gb"] or 0                                            # Avg RAM GBs
]
plt.figure()
plt.bar(labels, values)
plt.ylim(0,1)
plt.title("Evaluation summary")
for i,v in enumerate(values):
    plt.text(i, v+0.01, f"{v:.2f}", ha="center")
os.makedirs("runs/plots", exist_ok=True)
plt.savefig("runs/plots/eval_summary.png", dpi=160)
print("Saved runs/plots/eval_summary.png")
