"""
finetune_full.py
================

LoRA fine-tuning of BLIP-3 (xGen-MM-Phi3-mini) on the VQAv2 dataset.

This script implements the *LoRA full* experiment described in:
    - Chapter 7, Section 7.5  "Fine-Tuning with Low-Rank Adaptation"
    - Chapter 8, Section 8.4  "Full Fine-Tuning Experiment" (results)

Two runs were performed in the thesis:
    LoRA small : 10,000 samples, 1 epoch  -> Section 7.5 / Table 8.2
    LoRA full  : 50,000 samples, 3 epochs -> Section 8.4 / Tables 8.4, 8.5

The configuration in this file corresponds to the *full* run. To reproduce
the *small* run, change MAX_SAMPLES = 10000 and NUM_EPOCHS = 1.

Hardware  : NVIDIA H200 SXM (139.8 GB VRAM), HPC PERUN cluster
Precision : bfloat16
Trainable : 0.082% of total parameters (LoRA adapters only)
Duration  : ~222 minutes for the full run (Section 8.4.1, Table 8.4)
"""

import os, json, torch, time
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import torch.optim as optim
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Paths (all relative to the PERUN home directory)
# ----------------------------------------------------------------------------
# checkpoints/xgen-mm-phi3 holds the unmodified BLIP-3 weights downloaded
# from HuggingFace (Salesforce/xgen-mm-phi3-mini-instruct-r-v1).
model_path = "/mnt/data/home/arpoke644/blip3_thesis/checkpoints/xgen-mm-phi3"

# data/vqa contains the COCO images and the LLaVA-formatted JSON produced
# by convert_vqa_to_llava.py (referenced in Section 7.4.2).
data_dir   = "/mnt/data/home/arpoke644/blip3_thesis/data/vqa"

# Output directory for the trained LoRA adapter.
# Only the adapter is saved (~13 MB), not the full model (~18 GB).
output_dir = "/mnt/data/home/arpoke644/blip3_thesis/checkpoints/blip3-lora-full"

train_json = f"{data_dir}/vqa_train_llava_format.json"
log_path   = "/mnt/data/home/arpoke644/blip3_thesis/results/training_log.json"

# ----------------------------------------------------------------------------
# Hyperparameters (Section 7.5.2, Table 7.1)
# ----------------------------------------------------------------------------
NUM_EPOCHS  = 3        # multiple epochs revealed overfitting -> Section 8.4.3
LR          = 2e-4     # standard LoRA learning rate (Hu et al., 2022)
MAX_SAMPLES = 50000    # subset of the 443,757 VQAv2 training pairs
GRAD_ACCUM  = 4        # effective batch size = 4 (per-step batch is 1)
LOG_EVERY   = 100      # log loss every 100 global steps

os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# 1. MODEL LOADING
# ----------------------------------------------------------------------------
# Reference: Section 7.1 (Experimental Environment), Section 7.2 (Model Selection)
#
# Compatibility fixes required on PERUN (documented in Chapter 7):
#   (a) open_clip 3.3.0 -> downgraded to 2.24.0
#       (newer API broke the EVA-CLIP loader inside xGen-MM)
#   (b) modeling_xgenmm.py patched: vision encoder loaded on CPU first to
#       avoid CUDA OOM during initialisation, then moved to GPU
#   (c) open_clip/factory.py patched to handle 'meta' tensors used by
#       low_cpu_mem_usage=True (otherwise raises NotImplementedError)
#   (d) bfloat16 instead of float16 -- numerically stable on H200, prevents
#       loss=NaN that we saw with fp16
# ============================================================================

print("Loading model...")
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    trust_remote_code=True,         # required: xGen-MM ships custom modeling code
    dtype=torch.bfloat16,           # see fix (d) above
    low_cpu_mem_usage=False         # see fix (c): True triggers meta-tensor bug
)
model = model.to('cuda')

# The Phi-3 tokenizer; use_fast=False/legacy=False match the xGen-MM defaults.
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, use_fast=False, legacy=False
)

# Image processor handles the any-resolution tiling described in Section 6.3.1
# (input image -> 1 global view + 4 quadrant patches at 378x378).
image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

# Adds special tokens (<|image|>, <|end|>, ...) used by the Phi-3 chat template.
tokenizer = model.update_special_tokens(tokenizer)

# Right-padding is required for training (the loss must be aligned with labels).
tokenizer.padding_side = "right"
print("Model loaded!")

# ============================================================================
# 2. LoRA INJECTION
# ----------------------------------------------------------------------------
# Reference: Section 7.5.2 "LoRA Configuration", Table 7.1
#
# LoRA is applied ONLY to the language backbone (Phi-3-mini), NOT to the vision
# encoder or the Perceiver Resampler. This matches the description in
# Section 7.5.2: "the LoRA adapters are inserted into the four projection
# matrices of every self-attention layer of the language model".
#
# Numbers reported in the thesis:
#   r = 16, alpha = 32, dropout = 0.05
#   target_modules = q_proj, k_proj, v_proj, o_proj  (all 4 attention matrices)
#   total trainable parameters: 3.15M (0.082% of full model)
# ============================================================================

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                                                  # rank of the LoRA matrices
    lora_alpha=32,                                         # scaling factor (alpha/r = 2)
    lora_dropout=0.05,                                     # regularisation
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # see Section 7.5.2
    bias="none",                                           # do not adapt bias terms
)

# Wrap ONLY the language model with PEFT. The vision encoder and Perceiver
# Resampler remain untouched -- this is the key point of the architecture
# diagram in Figure 7.1.
model.vlm.lang_model = get_peft_model(model.vlm.lang_model, lora_config)
model.vlm.lang_model.print_trainable_parameters()  # prints the 0.082% figure

# Hard freeze: every parameter without 'lora_' in its name is frozen.
# This is a defensive double-check on top of PEFT's own freezing logic.
for name, param in model.named_parameters():
    if 'lora_' not in name:
        param.requires_grad = False

# ============================================================================
# 3. DATA LOADING
# ----------------------------------------------------------------------------
# Reference: Section 7.4 "Dataset Preparation"
# Format: produced by convert_vqa_to_llava.py (Section 7.4.2)
# Each item:
#   { "id": "...",
#     "image": "train2014/COCO_train2014_000000XXXXXX.jpg",
#     "conversations": [ {"from": "human", "value": "<image>\n<question>"},
#                        {"from": "gpt",   "value": "<answer>"} ] }
# ============================================================================

with open(train_json) as f:
    data = json.load(f)[:MAX_SAMPLES]   # truncate to 50K for the full run

# Standard AdamW. We pass only the parameters that have requires_grad=True,
# i.e. only the LoRA A/B matrices -- ~3.15M parameters, not the full 3.8B.
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
model.train()


def apply_prompt_template(question, answer):
    """
    Phi-3 instruction-tuning chat template. Critically, the <image> token must
    be placed in the user turn -- the xGen-MM forward pass scans the input_ids
    for this token and substitutes it with the (M=64-128) visual tokens
    produced by the Perceiver Resampler (Section 6.4).
    """
    return (
        '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
        "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
        f'<|user|>\n<image>\n{question}<|end|>\n<|assistant|>\n{answer}<|end|>'
    )


# ============================================================================
# 4. TRAINING LOOP
# ----------------------------------------------------------------------------
# Reference: Section 7.5.3 "Training Procedure"; Section 8.4.1 (loss curve)
# ============================================================================

training_log = {"epochs": [], "steps": [], "losses": []}
global_step = 0
t_start = time.time()

print(f"Training on {len(data)} samples x {NUM_EPOCHS} epochs...")

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    epoch_steps = 0
    skipped = 0
    optimizer.zero_grad()

    for step, item in enumerate(tqdm(data, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
        try:
            # ----- Load image and parse the conversation -----
            img_path = os.path.join(data_dir, item['image'])
            image = Image.open(img_path).convert('RGB')
            question = item['conversations'][0]['value'].replace('<image>\n', '')
            answer = item['conversations'][1]['value']

            # ----- Image preprocessing (AnyRes tiling) -----
            # Output shape: [1, 1, 5, 3, 378, 378]
            # (batch, num_images, num_patches=5, channels, H, W)
            # The 5 patches are: 1 global overview + 4 spatial quadrants.
            # This is what Figure 8.5 visualises.
            pixel_values = image_processor([image], return_tensors="pt", image_aspect_ratio='anyres')
            vision_x = pixel_values['pixel_values'].to(torch.bfloat16).cuda()

            # ----- Text tokenisation -----
            prompt = apply_prompt_template(question, answer)
            tokenized = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=256)
            input_ids = tokenized['input_ids'].cuda()
            attention_mask = tokenized['attention_mask'].cuda()

            # Causal LM loss: labels are the same as input_ids, with padding
            # positions masked by -100 so they do not contribute to the loss.
            # Note: we do NOT mask the question portion here -- the model is
            # trained on full next-token prediction across the entire prompt.
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            # ----- Forward pass through the full VLM -----
            # vision_x -> EVA-CLIP ViT -> Perceiver Resampler -> visual tokens
            # lang_x   -> Phi-3 embedding -> token sequence with <image> placeholders
            # The model substitutes visual tokens at <image> positions and runs
            # unified self-attention (Section 6.6.2).
            outputs = model.vlm(
                vision_x=vision_x,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                image_size=[image.size],   # needed for AnyRes positional encoding
            )

            # Loss is divided here so that .backward() accumulates the average,
            # not the sum, across GRAD_ACCUM micro-batches.
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            # Optimiser step only every GRAD_ACCUM micro-batches.
            # Effective batch size = GRAD_ACCUM = 4.
            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Multiply back so that the *logged* loss is the per-sample loss.
            loss_val = loss.item() * GRAD_ACCUM
            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1

            if global_step % LOG_EVERY == 0:
                elapsed = time.time() - t_start
                print(f"Epoch {epoch+1} | Step {step} | Loss: {loss_val:.4f} | "
                      f"Elapsed: {elapsed/60:.1f}min", flush=True)
                training_log["steps"].append(global_step)
                training_log["losses"].append(round(loss_val, 4))

        except Exception as e:
            # Skip individual broken samples (corrupt JPEG, missing file, etc.)
            # but keep training. Logged so we know the rate.
            skipped += 1
            continue

    # ----- End-of-epoch bookkeeping -----
    avg_loss = epoch_loss / max(1, epoch_steps)
    print(f"\n=== Epoch {epoch+1} complete | Avg loss: {avg_loss:.4f} | Skipped: {skipped} ===\n")
    training_log["epochs"].append({"epoch": epoch+1, "avg_loss": round(avg_loss, 4), "skipped": skipped})

    # Per-epoch checkpoint so we can evaluate intermediate models if needed.
    epoch_dir = f"{output_dir}/epoch_{epoch+1}"
    os.makedirs(epoch_dir, exist_ok=True)
    model.vlm.lang_model.save_pretrained(epoch_dir)
    print(f"Saved epoch {epoch+1} checkpoint to {epoch_dir}")

# ============================================================================
# 5. SAVE FINAL ADAPTER AND LOG
# ----------------------------------------------------------------------------
# Only the LoRA adapter (~13 MB) is saved. This is the central practical
# advantage discussed in Section 8.6: the base model can be shared once and
# users only download the small adapter to obtain the fine-tuned behaviour.
# ============================================================================

model.vlm.lang_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Saved final model to {output_dir}")

with open(log_path, 'w') as f:
    json.dump(training_log, f, indent=2)
print(f"Saved training log to {log_path}")

total_time = (time.time() - t_start) / 60
print(f"\nTotal training time: {total_time:.1f} minutes")
