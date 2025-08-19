import re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
from peft import PeftModel

BASE = r"C:\Users\sesal\Documents\AntonioTrovato\CodeLlama-13b-Instruct-hf"
LORA = "anto0699/Prompt2OpenSCENARIO-CodeLlama13B-LoRA"
TEMPLATE = r"prompt_templates\codellama_inst.txt"
STOP_STR = "</OpenScenario>"

class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tok, substrings, start_len=0):
        self.tok = tok
        self.substrings = substrings
        self.start_len = start_len  # numero di token di prompt

    def set_start_len(self, n):
        self.start_len = n

    def __call__(self, input_ids, scores, **kw):
        gen_ids = input_ids[0, self.start_len:].tolist()
        if not gen_ids:
            return False
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        return any(s in text for s in self.substrings)

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
tok = AutoTokenizer.from_pretrained(LORA, use_fast=True)
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

def generate_xosc(system: str, user: str, max_new: int = 4000) -> str:
    tmpl = open(TEMPLATE, "r", encoding="utf-8").read()
    prompt = tmpl.format(system=system, user=user)
    if prompt.endswith("[/INST]"): prompt += "\n"
    enc = tok(prompt, return_tensors="pt").to(model.device)
    stopper = StopOnSubstrings(tok, [STOP_STR])
    stopper.set_start_len(enc["input_ids"].shape[1])
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            stopping_criteria=StoppingCriteriaList([stopper]),
            return_dict_in_generate = True
        )
    gen_ids = out.sequences[0, enc["input_ids"].shape[1]:]
    txt = tok.decode(gen_ids, skip_special_tokens=True)
    m = re.search(r"<OpenScenario\b.*?</OpenScenario>", txt, flags=re.DOTALL)
    return m.group(0).strip() if m else txt

# esempio
system = "Act as an OpenSCENARIO 1.0 generator for ADS testing in CARLA. I will give you a scene description in English and you must return one valid `.xosc` file, XML only, encoded in UTF-8, starting with `<OpenScenario>` and ending with `</OpenScenario>`. The file must end with </OpenScenario> tag, be deterministic, schema-compliant, and executable in CARLA without modifications. It must always specify the map, time of day, weather, any speed limits, all entities with unique names, their initial positions, and the storyboard with relevant events and triggers. Use realistic defaults if details are missing, but never omit these features. No comments or extra text, only the `.xosc`."
user = "Write me a scenario with a pedestrian in OpenScenario 1.0 and in Town02 in a rainy day"
print(generate_xosc(system, user))
