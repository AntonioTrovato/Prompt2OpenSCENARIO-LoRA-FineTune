import json, sys, os
import numpy as np
import matplotlib.pyplot as plt

state = json.load(open(sys.argv[1], "r", encoding="utf-8"))
log_history = state["log_history"]

steps_tr, loss_tr, ppl_tr = [], [], []
steps_ev, loss_ev, ppl_ev = [], [], []

for e in log_history:
    if "step" in e and "loss" in e:
        steps_tr.append(e["step"])
        loss_tr.append(e["loss"])
        # Perplexity approssimata dal training loss (meno affidabile)
        ppl_tr.append(float(np.exp(np.clip(e["loss"], None, 20.0))))  # clip per stabilità

    if "step" in e and "eval_loss" in e:
        steps_ev.append(e["step"])
        loss_ev.append(e["eval_loss"])
        ppl_ev.append(float(np.exp(np.clip(e["eval_loss"], None, 20.0))))

os.makedirs("runs/plots", exist_ok=True)

# 1) Loss
plt.figure()
if steps_tr:
    plt.plot(steps_tr, loss_tr, label="train loss")
if steps_ev:
    plt.plot(steps_ev, loss_ev, label="eval loss")
plt.xlabel("Global step"); plt.ylabel("Loss")
plt.title("Training/Eval loss")
plt.legend()
plt.tight_layout()
plt.savefig("runs/plots/training_eval_loss.png", dpi=160)

# 2) Perplexity
plt.figure()
if steps_tr:
    plt.plot(steps_tr, ppl_tr, label="train ppl")
if steps_ev:
    plt.plot(steps_ev, ppl_ev, label="eval ppl")
plt.xlabel("Global step"); plt.ylabel("Perplexity")
plt.title("Training/Eval perplexity")
plt.yscale("log")  # spesso utile
plt.legend()
plt.tight_layout()
plt.savefig("runs/plots/training_eval_perplexity.png", dpi=160)

print("Saved:",
      "runs/plots/training_eval_loss.png",
      "runs/plots/training_eval_perplexity.png")
