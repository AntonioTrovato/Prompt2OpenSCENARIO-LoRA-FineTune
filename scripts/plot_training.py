import json, sys
import matplotlib.pyplot as plt
import os
state = json.load(open(sys.argv[1], "r", encoding="utf-8"))  # es: runs/codellama13b-lora/trainer_state.json
log_history = state["log_history"]
steps, losses = [], []
for e in log_history:
    if "loss" in e and "step" in e:
        steps.append(e["step"]); losses.append(e["loss"])
plt.figure()
plt.plot(steps, losses)
plt.xlabel("Global step"); plt.ylabel("Training loss")
plt.title("Training loss curve")
os.makedirs("runs/plots", exist_ok=True)
plt.savefig("runs/plots/training_loss.png", dpi=160)
print("Saved runs/plots/training_loss.png")
