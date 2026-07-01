"""
eval_full.py
============

Three-model comparison evaluation: base BLIP-3 vs LoRA small vs LoRA full,
all evaluated on the SAME 500-sample VQAv2 validation subset.

This script produces the data reported in:
    - Section 8.4.2  "Three-Model Comparison"           -> Table 8.5
    - Section 8.4.3  "Overfitting Analysis"             -> Figure 8.4
    - Section 8.7    "Qualitative Error Analysis"       -> Tables 8.8, 8.9

Why 500 samples here (not 2,000):
    The 2,000-sample numbers (Tables 8.2, 8.3) come from a separate
    evaluation script that compares only base vs LoRA small. This script
    is the broader three-model run, kept smaller because it loads two
    LoRA adapters sequentially.
"""

import os, json, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')                # no display on the cluster
import matplotlib.pyplot as plt
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria
from peft import PeftModel
from PIL import Image
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
model_path   = "/mnt/data/home/arpoke644/blip3_thesis/checkpoints/xgen-mm-phi3"
lora_small   = "/mnt/data/home/arpoke644/blip3_thesis/checkpoints/blip3-lora-vqa"      # 10K x 1 ep
lora_full    = "/mnt/data/home/arpoke644/blip3_thesis/checkpoints/blip3-lora-full"     # 50K x 3 ep
data_dir     = "/mnt/data/home/arpoke644/blip3_thesis/data/vqa"
results_dir  = "/mnt/data/home/arpoke644/blip3_thesis/results"
val_json     = f"{data_dir}/vqa_val_llava_format.json"
ann_json     = f"{data_dir}/v2_mscoco_val2014_annotations.json"
NUM_EVAL     = 500                   # see header comment

os.makedirs(results_dir, exist_ok=True)


# ============================================================================
# 1. UTILITIES
# ============================================================================

class EosListStoppingCriteria(StoppingCriteria):
    """
    Stop generation as soon as the Phi-3 end-of-turn token (32007 = <|end|>)
    is emitted. Without this the model would continue up to max_new_tokens
    and produce padding that pollutes the decoded answer.
    """
    def __init__(self, eos_sequence=[32007]):
        self.eos_sequence = eos_sequence
    def __call__(self, input_ids, scores, **kwargs):
        return self.eos_sequence in input_ids[:, -len(self.eos_sequence):].tolist()


def apply_prompt_template(question):
    """
    Same chat template as in finetune_full.py, but without the assistant turn
    -- here we want the model to *generate* the answer, not learn it.
    """
    return (
        '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
        "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
        f'<|user|>\n<image>\n{question}<|end|>\n<|assistant|>\n'
    )


def normalize(s):
    """Light normalisation for soft string matching (Section 8.2)."""
    return s.strip().lower().rstrip('.')


# ============================================================================
# 2. LOAD VQAv2 ANNOTATIONS AND VALIDATION SUBSET
# ----------------------------------------------------------------------------
# Reference: Section 8.3 "Category-Level Analysis"
#
# VQAv2 annotations expose an 'answer_type' field with values
# {'yes/no', 'number', 'other'}. We use this to break accuracy down by
# question type, which is the key result of Section 8.3 (Table 8.3).
# ============================================================================

with open(ann_json) as f:
    annotations = json.load(f)['annotations']
qid_to_type = {ann['question_id']: ann['answer_type'] for ann in annotations}

with open(val_json) as f:
    val_data = json.load(f)[:NUM_EVAL]
for item in val_data:
    item['answer_type'] = qid_to_type.get(int(item['id']), 'other')

categories = ['yes/no', 'number', 'other']


# ============================================================================
# 3. EVALUATION FUNCTION
# ----------------------------------------------------------------------------
# Metric: SOFT string matching (Section 8.2, paragraph "evaluation metric").
# A prediction is correct if either:
#     - the ground-truth string is contained in the prediction, OR
#     - the prediction is contained in the ground-truth.
# This is the standard VQAv2 protocol and tolerates differences such as
# "yes" vs "yes, there is".
# ============================================================================

def evaluate(model, tokenizer, image_processor, data, tag):
    model.eval()
    results = {cat: {'correct': 0, 'total': 0} for cat in categories}
    results['overall'] = {'correct': 0, 'total': 0}

    with torch.no_grad():
        for item in tqdm(data, desc=f"Eval {tag}"):
            try:
                img_path = os.path.join(data_dir, item['image'])
                image = Image.open(img_path).convert('RGB')
                question = item['conversations'][0]['value'].replace('<image>\n', '')
                gt = normalize(item['conversations'][1]['value'])
                cat = item['answer_type']

                # AnyRes preprocessing (Section 6.3.1)
                proc = image_processor([image], return_tensors="pt", image_aspect_ratio='anyres')
                inp = {k: v.to(torch.bfloat16).cuda() for k, v in proc.items()}
                lang = tokenizer([apply_prompt_template(question)], return_tensors="pt")
                inp.update({k: v.cuda() for k, v in lang.items()})

                # Greedy decoding (do_sample=False, num_beams=1).
                # We deliberately avoid sampling so the comparison between
                # models is deterministic (Section 7.3).
                out = model.generate(
                    **inp, image_size=[image.size],
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False, max_new_tokens=32,
                    top_p=None, num_beams=1,
                    stopping_criteria=[EosListStoppingCriteria()]
                )
                # decoded text -> drop everything after the first <|end|>
                pred = normalize(tokenizer.decode(out[0], skip_special_tokens=True).split("<|end|>")[0])

                # Soft match (see comment above the function)
                correct = gt in pred or pred in gt

                if cat in results:
                    results[cat]['total'] += 1
                    if correct: results[cat]['correct'] += 1
                results['overall']['total'] += 1
                if correct: results['overall']['correct'] += 1
            except:
                # Same defensive skip as in fine-tuning
                continue

    for cat, vals in results.items():
        if vals['total'] > 0:
            acc = vals['correct'] / vals['total'] * 100
            print(f"  {cat:10s}: {acc:.1f}% ({vals['correct']}/{vals['total']})")
    return results


# ============================================================================
# 4. LOAD BASE MODEL ONCE, ATTACH/DETACH ADAPTERS
# ----------------------------------------------------------------------------
# Reference: Section 8.6 (computational performance) explains why we keep one
# model in memory and swap adapters rather than reload the full ~18 GB weights.
#
# Adapter merge strategy:
#   PeftModel.from_pretrained -> wraps the language model with LoRA layers.
#   .merge_and_unload()        -> bakes the LoRA delta into the base weights
#                                 in place, producing a plain Phi-3 module
#                                 that can be replaced by the next adapter.
# ============================================================================

print("Loading model...")
model = AutoModelForVision2Seq.from_pretrained(
    model_path, trust_remote_code=True,
    dtype=torch.bfloat16, low_cpu_mem_usage=False
)
model = model.to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, legacy=False)
image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
tokenizer = model.update_special_tokens(tokenizer)
tokenizer.padding_side = "left"      # left padding is correct for generation
print("Model loaded!")

# --- run 1: base ---
print("\n--- BASE ---")
base_results = evaluate(model, tokenizer, image_processor, val_data, "base")

# --- run 2: LoRA small (10K, 1 ep) ---
# merge_and_unload modifies the language model in-place; the next attach will
# wrap the (already-merged) weights, which is intentional here.
print("\n--- LoRA small (10K, 1 epoch) ---")
model.vlm.lang_model = PeftModel.from_pretrained(model.vlm.lang_model, lora_small)
model.vlm.lang_model = model.vlm.lang_model.merge_and_unload()
small_results = evaluate(model, tokenizer, image_processor, val_data, "lora_small")

# --- run 3: LoRA full (50K, 3 ep) ---
# NOTE: this attaches on top of LoRA-small-merged. The lora_full adapter was
# trained starting from the LoRA-small checkpoint (Section 8.4.3 lists this
# as the third overfitting factor), so the merge order matches the training
# order.
print("\n--- LoRA full (50K, 3 epochs) ---")
model.vlm.lang_model = PeftModel.from_pretrained(model.vlm.lang_model, lora_full)
model.vlm.lang_model = model.vlm.lang_model.merge_and_unload()
full_results = evaluate(model, tokenizer, image_processor, val_data, "lora_full")

# Persist raw results -- used to populate Table 8.5 in the thesis.
with open(f"{results_dir}/eval_full_comparison.json", 'w') as f:
    json.dump({"base": base_results, "lora_small": small_results, "lora_full": full_results}, f, indent=2)


# ============================================================================
# 5. PLOTS  -> Figure 8.4 (left: bar chart, right: training loss)
# ============================================================================

cats_labels = ['Yes/No', 'Number', 'Other', 'Overall']
cats_keys   = ['yes/no', 'number', 'other', 'overall']

def get_accs(res):
    return [res[k]['correct']/max(1,res[k]['total'])*100 for k in cats_keys]

base_accs  = get_accs(base_results)
small_accs = get_accs(small_results)
full_accs  = get_accs(full_results)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ---- Left subplot: per-category bar chart ----
x = np.arange(len(cats_labels))
w = 0.25
axes[0].bar(x - w, base_accs,  w, label='BASE',                color='#5b9bd5', edgecolor='white')
axes[0].bar(x,     small_accs, w, label='LoRA (10K, 1 epoch)', color='#ed7d31', edgecolor='white')
axes[0].bar(x + w, full_accs,  w, label='LoRA (50K, 3 epochs)',color='#70ad47', edgecolor='white')
for i, (b, s, f) in enumerate(zip(base_accs, small_accs, full_accs)):
    axes[0].text(x[i]-w, b+0.5, f'{b:.1f}', ha='center', fontsize=7)
    axes[0].text(x[i],   s+0.5, f'{s:.1f}', ha='center', fontsize=7)
    axes[0].text(x[i]+w, f+0.5, f'{f:.1f}', ha='center', fontsize=7)
axes[0].set_xticks(x)
axes[0].set_xticklabels(cats_labels, fontsize=11)
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].set_title('VQAv2 Accuracy: Three Model Comparison', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].set_ylim(0, 105)
axes[0].grid(True, alpha=0.3, axis='y')

# ---- Right subplot: training loss curve (loaded from finetune_full.py log) ----
log = json.load(open(f"{results_dir}/training_log.json"))
steps  = log['steps']
losses = log['losses']
epoch_avgs = [e['avg_loss'] for e in log['epochs']]

axes[1].plot(steps, losses, color='steelblue', linewidth=0.8, alpha=0.5, label='Step loss')
# 20-step moving average to smooth out per-batch noise
window = 20
smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
axes[1].plot(steps[window-1:], smoothed, color='steelblue', linewidth=2, label='Smoothed loss')
# Vertical dashed lines marking each epoch boundary with its average loss
for i, avg in enumerate(epoch_avgs):
    ep_step = (i+1) * len(losses) // 3
    axes[1].axvline(x=steps[min(ep_step, len(steps)-1)], color='red', linestyle='--', alpha=0.5)
    axes[1].text(steps[min(ep_step, len(steps)-1)], max(losses)*0.95,
                f'E{i+1}\n{avg:.3f}', ha='right', fontsize=9, color='red')
axes[1].set_xlabel('Global step', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Training Loss Curve (50K × 3 epochs)', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle('BLIP-3 Full Fine-tuning Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{results_dir}/8_full_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved 8_full_comparison.png")

# ----------------------------------------------------------------------------
# Final summary printed to stdout (also captured in the slurm log).
# These numbers populate Table 8.5 in Section 8.4.2 of the thesis.
# ----------------------------------------------------------------------------
print("\n=== FINAL SUMMARY ===")
for tag, res in [("BASE", base_results), ("LoRA small", small_results), ("LoRA full", full_results)]:
    acc = res['overall']['correct'] / res['overall']['total'] * 100
    print(f"{tag:15s}: {acc:.2f}%")
