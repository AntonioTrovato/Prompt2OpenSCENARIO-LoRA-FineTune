import torch, argparse, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--base_model", required=True)
ap.add_argument("--lora_repo", required=True)
ap.add_argument("--out_dir", default="merged_model_fp16")
ap.add_argument("--dtype", choices=["fp16","bf16"], default="fp16")
ap.add_argument("--cpu", action="store_true")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
dtype = torch.float16 if args.dtype=="fp16" else torch.bfloat16
device_map = None if args.cpu else "auto"
model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=dtype,
    device_map=device_map,
    low_cpu_mem_usage=not args.cpu
)
model = PeftModel.from_pretrained(model, args.lora_repo)
# Merge: applica i delta LoRA ai pesi del base
model = model.merge_and_unload()
os.makedirs(args.out_dir, exist_ok=True)
model.save_pretrained(args.out_dir, safe_serialization=True)
tok.save_pretrained(args.out_dir)
print(f"Merged model saved to {args.out_dir}")
