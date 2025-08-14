import json, matplotlib.pyplot as plt, os
m = json.load(open("runs/eval/metrics.json"))
labels = ["XML ok","XSD ok","Exact","ROUGE-L F1","Slot F1","Slot Acc","BLEU"]
values = [
    m["xml_wellformed_rate"] or 0,
    (m["xsd_valid_rate"] or 0) if m.get("xsd_valid_rate") is not None else 0,
    m["exact_match"] or 0,
    m["rougeL_f1"] or 0,
    m["slot_f1"] or 0,
    m["slot_accuracy"] or 0,
    (m["bleu"] or 0)/100.0
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
