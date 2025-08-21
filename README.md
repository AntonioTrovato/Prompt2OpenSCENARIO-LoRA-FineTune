# Prompt2OpenSCENARIO: Fine-Tuning CodeLlama-13B for Scenario Generation

This repository provides the full pipeline for **fine-tuning, validating, and evaluating a Large Language Model (LLM)** on the task of **generating OpenSCENARIO 1.0 (`.xosc`) files from natural language descriptions of driving scenes**.  

The work leverages **CodeLlama-13B-Instruct** as the base model and applies **LoRA/QLoRA parameter-efficient fine-tuning**. The resulting adapter weights are released on [Hugging Face Hub](https://huggingface.co/anto0699/Prompt2OpenSCENARIO-CodeLlama13B-LoRA).

---

## Repository Structure

- **`train.py`**  
  Fine-tunes the base model on a JSONL/Hugging Face dataset.  
  - LoRA configuration is read from `config/lora-codellama13b.yaml`.  
  - Training uses `trl.SFTTrainer` with `assistant_only_loss` for stable single-turn alignment.  
  - Supports mixed precision (`bfloat16` / `fp16`) and 4-bit quantization.

- **`eval_validate.py`**  
  Performs **systematic evaluation of the fine-tuned model**:
  - Generation of `.xosc` files given system and user prompts.  
  - Validation of XML well-formedness and XSD compliance.  
  - Multiple evaluation metrics: BLEU, ROUGE-L, chrF++, METEOR, BERTScore, edit similarity, jaccard overlap of XML tags, slot-based precision/recall/F1, and perplexity.  
  - Logs memory (RAM/VRAM) usage and throughput.  
  - Produces:
    - `metrics.json` (aggregated scores)  
    - `predictions.jsonl` (per-sample generations)

- **`plot_training.py`**  
  Parses Hugging Face `Trainer` log history and plots:
  - Training vs. evaluation **loss**  
  - Training vs. evaluation **perplexity** (log-scaled)

- **`plot_eval.py`**  
  Visualizes evaluation metrics from `metrics.json` in grouped bar charts:
  - XML/XSD validity  
  - BLEU & ROUGE  
  - chrF++ & METEOR  
  - Slot metrics (precision/recall/accuracy/F1)  
  - Edit similarity, jaccard, and length ratio  
  - Memory usage (VRAM/RAM)

- **`config/lora-codellama13b.yaml`**  
  Central configuration file controlling training, quantization, optimizer settings, evaluation parameters, and Hugging Face Hub integration.

- **`prompt_templates/codellama_inst.txt`**  
  Instruction-style prompt template aligned with CodeLlama’s expected format.

---

## Workflow

1. **Dataset Preparation**  
   Training data is stored in JSONL format with three fields:
   - `system` → System instruction (e.g., *"Act as an OpenSCENARIO generator for CARLA..."*)  
   - `user` → Natural language description of a driving scenario  
   - `assistant` → Gold-standard `.xosc` file  

   Alternatively, a Hugging Face dataset repository can be specified in the config file.

2. **Fine-Tuning**  
   ```bash
   python train.py 
   ```

The script automatically handles dataset splitting, tokenization, LoRA adapter setup, and checkpointing.

3. **Evaluation**
   ```bash
   python eval_validate.py --cfg config/lora-codellama13b.yaml --use_jsonl --jsonl_path ./datasets/test.jsonl 
   ```
This produces:
- runs/coherence/metrics.json
- runs/coherence/predictions.jsonl
4. **Plotting**
- Training curves:
   ```bash
   python plot_training.py --log_history runs/prompt2openscenario/codellama13b-lora/trainer_state.json
   ```
- Evaluation metrics:
    ```bash
    python plot_eval.py --input_path runs/coherence/metrics.json
     ```

## Key Features
Parameter-efficient tuning (LoRA/QLoRA) for large-scale models.
Comprehensive evaluation pipeline including structural XML validation against the OpenSCENARIO 1.0 XSD.
Multi-metric assessment combining NLP quality metrics (BLEU, ROUGE, chrF++, METEOR, BERTScore) with domain-specific slot-based metrics.
Visualization scripts for reproducible analysis.
Direct Hugging Face Hub integration for model push/pull.

## Citation
If you use this work, please cite:
@misc{prompt2openscenario2025,
  title        = {Empty Title},
  author       = {No Authors},
  year         = {2025},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/anto0699/Prompt2OpenSCENARIO-CodeLlama13B-LoRA}}
}