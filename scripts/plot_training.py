import json, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml

def main():
    parser = argparse.ArgumentParser(description="Plot training/eval loss & perplexity from log_history.")
    parser.add_argument("--log_history", required=True, help="Path to JSON file containing state['log_history'].")
    parser.add_argument("--cfg", default="./configlora-codellama13b.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    # Load config
    if not os.path.isfile(args.cfg):
        raise FileNotFoundError(f"Config file not found: {args.cfg}")
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    plots_dir = cfg.get("plots_dir", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load log history
    with open(args.log_history, "r", encoding="utf-8") as f:
        state = json.load(f)
    log_history = state.get("log_history", [])

    steps_tr, loss_tr, ppl_tr = [], [], []
    steps_ev, loss_ev, ppl_ev = [], [], []

    for e in log_history:
        if "step" in e and "loss" in e:
            steps_tr.append(e["step"])
            loss_tr.append(e["loss"])
            ppl_tr.append(float(np.exp(np.clip(e["loss"], None, 20.0))))  # approx perplexity, clipped for stability
        if "step" in e and "eval_loss" in e:
            steps_ev.append(e["step"])
            loss_ev.append(e["eval_loss"])
            ppl_ev.append(float(np.exp(np.clip(e["eval_loss"], None, 20.0))))

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
    out_loss = os.path.join(plots_dir, "training_eval_loss.png")
    plt.savefig(out_loss, dpi=160)

    # 2) Perplexity
    plt.figure()
    if steps_tr:
        plt.plot(steps_tr, ppl_tr, label="train ppl")
    if steps_ev:
        plt.plot(steps_ev, ppl_ev, label="eval ppl")
    plt.xlabel("Global step"); plt.ylabel("Perplexity")
    plt.title("Training/Eval perplexity")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    out_ppl = os.path.join(plots_dir, "training_eval_perplexity.png")
    plt.savefig(out_ppl, dpi=160)

    print("Saved:", out_loss, out_ppl)

if __name__ == "__main__":
    main()
