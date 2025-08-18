import re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
from peft import PeftModel

BASE = r"C:\Users\sesal\Documents\AntonioTrovato\CodeLlama-13b-Instruct-hf"
LORA = "anto0699/Prompt2OpenSCENARIO-CodeLlama13B-LoRA"
TEMPLATE = r"prompt_templates\codellama_inst.txt"
STOP_STR = "</OpenScenario>"

class StopOnString(StoppingCriteria):
    def __init__(self, tok, s): self.ids = tok(s, add_special_tokens=False).input_ids
    def __call__(self, input_ids, scores, **kw):
        k=len(self.ids);
        return input_ids.shape[1] >= k and input_ids[0, -k:].tolist() == self.ids

# === scegli una delle due opzioni ===
# Opzione A: VRAM ampia, niente offload
#tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
#tok.padding_side="right"
#if tok.pad_token is None: tok.pad_token = tok.eos_token
#model = AutoModelForCausalLM.from_pretrained(
#    BASE, torch_dtype=torch.bfloat16, device_map={"":0}, attn_implementation="sdpa"
#)

# Opzione B: VRAM limitata, tutto GPU in 4-bit
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.padding_side="right"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map="auto", attn_implementation="sdpa"
)

model = PeftModel.from_pretrained(model, LORA)
model.config.use_cache = True
model.eval()

# warm-up (facoltativo ma utile)
#_ = model.generate(**tok("Hi", return_tensors="pt").to(model.device), max_new_tokens=8)

def generate_xosc(system: str, user: str, max_new: int = 1500) -> str:
    tmpl = open(TEMPLATE, "r", encoding="utf-8").read()
    prompt = tmpl.format(system=system, user=user)
    if prompt.endswith("[/INST]"): prompt += "\n"
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False, temperature=0.1, top_p=0.9,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnString(tok, STOP_STR)])
        )
    txt = tok.decode(out[0], skip_special_tokens=True)
    m = re.search(r"<OpenScenario\b.*</OpenScenario>", txt, flags=re.DOTALL)
    return m.group(0) if m else txt

# esempio
system = "Act as an OpenSCENARIO 1.0 generator for ADS testing in CARLA. I will give you a scene description in English and you must return one valid `.xosc` file, XML only, encoded in UTF-8, starting with `<OpenScenario>` and ending with `</OpenScenario>`. The file must be deterministic, schema-compliant, and executable in CARLA without modifications. It must always specify the map, time of day, weather, any speed limits, all entities with unique names, their initial positions, and the storyboard with relevant events and triggers. Use realistic defaults if details are missing, but never omit these features. No comments or extra text, only the `.xosc`."
user = "On the ARG_Carcarana-10_1_T-1 road network at noon on 2023-03-20, the scene runs under clear skies with essentially no fog (very long visibility). The ego car starts at (21.56, 91.14, h≈-1.034) while NPC vehicles are preplaced at, for example, Npc31 (-3.07, 93.26, h≈-0.206), Npc32 (64.81, 82.66, h≈-3.347), Npc38 (24.53, 134.65, h≈-1.774), Npc312 (40.40, 127.22, h≈-1.743), and Npc318 (20.06, 102.29, h≈-1.774). At t=0.0 s, every NPC begins moving along its predefined trajectory. No pedestrians are present, and additional NPCs begin their paths immediately as well."
print(generate_xosc(system, user))
