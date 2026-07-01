"""
analyze_pipeline.py
===================

Internal pipeline analysis of BLIP-3 on a single representative image.
Produces five figures used in Chapter 8:

    1_input.png              -> Figure 8.1  (Section 8.1)
    2_vision_patches.png     -> Figure 8.5  (Section 8.5.1, AnyRes decomposition)
    3_vision_embeddings.png  -> Figure 8.6  (Section 8.5.2, vision encoder output)
    4_attention_maps.png     -> Figure 8.7  (Section 8.5.3, cross-modal attention)
    5_results_comparison.png -> Figure 8.2  (Section 8.2, base vs LoRA bar chart)

The script is intentionally a single self-contained file so the analysis
can be reproduced by running one command on the cluster.

NOTE on attention extraction: Phi-3 by default uses FlashAttention, which
returns attention weights as None. We force eager attention via
attn_implementation='eager' both at load time AND on the language model
afterwards (the second call is a workaround for a transformers issue
where the load-time flag is not honoured for nested modules).
"""

import os
import json
import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoImageProcessor,
    StoppingCriteria
)
from PIL import Image

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
model_path = (
    "/mnt/data/home/arpoke644/blip3_thesis/checkpoints/xgen-mm-phi3"
)
results_dir = (
    "/mnt/data/home/arpoke644/blip3_thesis/results"
)

# The food-web diagram shipped with xGen-MM's test_samples.
# Chosen because it has a non-trivial spatial structure (multiple labelled
# entities with arrows) that produces interpretable attention maps.
test_img = (
    "/mnt/data/home/arpoke644/blip3_thesis/checkpoints/xgen-mm-phi3/"
    "test_samples/images/26302.jpg"
)
os.makedirs(results_dir, exist_ok=True)


# ----------------------------------------------------------------------------
# Same helpers as in the other scripts
# ----------------------------------------------------------------------------

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[32007]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids, scores, **kwargs):
        return self.eos_sequence in input_ids[:, -len(self.eos_sequence):].tolist()


def apply_prompt_template(question):
    return (
        '<|system|>\nA chat between a curious user and an artificial '
        'intelligence assistant. The assistant gives helpful, detailed, '
        'and polite answers to the user\'s questions.<|end|>\n'
        f'<|user|>\n<td>\n{question}<|end|>\n<|assistant|>\n'
    )


# ============================================================================
# MODEL LOADING (with eager attention so we can extract attention weights)
# ============================================================================

print("Loading model...")
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    attn_implementation='eager'           # required to get attention weights
)
model = model.to('cuda')
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, use_fast=False, legacy=False
)
image_processor = AutoImageProcessor.from_pretrained(
    model_path, trust_remote_code=True
)
tokenizer = model.update_special_tokens(tokenizer)
tokenizer.padding_side = "left"

# Workaround: the constructor flag does not always propagate down to the
# language model on a nested VLM. Force it again here.
model.vlm.lang_model.set_attn_implementation('eager')

model.eval()
print("Model loaded!")

# ----------------------------------------------------------------------------
# Prepare the single test sample
# ----------------------------------------------------------------------------
image = Image.open(test_img).convert('RGB')
question = "In the food web, what are the predators?"

# AnyRes preprocessing -> pixel_values shape [1, 1, 5, 3, 378, 378]
proc = image_processor(
    [image], return_tensors="pt", image_aspect_ratio='anyres'
)
inputs = {k: v.to(torch.bfloat16).cuda() for k, v in proc.items()}
prompt = apply_prompt_template(question)
lang_inputs = tokenizer([prompt], return_tensors="pt")
inputs.update({k: v.cuda() for k, v in lang_inputs.items()})


# ============================================================================
# FIGURE 1 — INPUT IMAGE + QUESTION  (Figure 8.1 in the thesis)
# ----------------------------------------------------------------------------
# Documentary plot: shows what the model receives. The right panel embeds
# the question text and model metadata so the figure is self-explanatory.
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(image)
axes[0].set_title('Input Image', fontsize=14)
axes[0].axis('off')
axes[1].text(
    0.1, 0.5,
    f"Question:\n{question}\n\nModel: BLIP-3\n(xGen-MM-Phi3-mini)",
    transform=axes[1].transAxes,
    fontsize=12,
    verticalalignment='center',
    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
)
axes[1].axis('off')
plt.tight_layout()
plt.savefig(f"{results_dir}/1_input.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 1_input.png")


# ============================================================================
# FIGURE 2 — ANY-RESOLUTION DECOMPOSITION  (Figure 8.5, Section 8.5.1)
# ----------------------------------------------------------------------------
# Visualises the 5 patches produced by AnyRes:
#   patch[0]    : global overview at 378x378 (entire image, downsampled)
#   patch[1..4] : the four spatial quadrants at the original resolution
# This is the EVA-CLIP input -- before the Perceiver Resampler compresses it
# to M=64-128 visual tokens (Section 6.4.1).
# ============================================================================

with torch.no_grad():
    pixel_values = inputs['pixel_values']   # [1, 1, 5, 3, 378, 378]
    patches = pixel_values[0, 0]            # [5, 3, 378, 378]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    patch_names = ['Full image', 'Patch 1', 'Patch 2', 'Patch 3', 'Patch 4']
    for i in range(5):
        # Convert CHW -> HWC, denormalise to [0,1] for matplotlib display.
        patch = patches[i].float().cpu().numpy()
        patch = np.transpose(patch, (1, 2, 0))
        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
        axes[i].imshow(patch)
        axes[i].set_title(patch_names[i], fontsize=11)
        axes[i].axis('off')
    plt.suptitle(
        'AnyRes Image Decomposition (Vision Encoder Input)', fontsize=14
    )
    plt.tight_layout()
    plt.savefig(
        f"{results_dir}/2_vision_patches.png", dpi=150, bbox_inches='tight'
    )
    plt.close()
    print("Saved 2_vision_patches.png")


# ============================================================================
# FIGURE 3 — VISION ENCODER OUTPUT EMBEDDING  (Figure 8.6, Section 8.5.2)
# ----------------------------------------------------------------------------
# Runs ONE patch through the EVA-CLIP ViT to inspect the 1024-dim global
# embedding. The thesis reports:
#   * shape:    [1024]
#   * mean ~ 0, std ~ 1.2 (consistent with contrastive pretraining)
#   * L2 norm ~ 18.4
# Visualisation:
#   - left:   embedding reshaped to 32x32 heatmap
#   - center: 1D plot of all 1024 values
#   - right:  histogram showing the near-Gaussian distribution
# ============================================================================

with torch.no_grad():
    vision_encoder = model.vlm.vision_encoder
    single_patch = patches[0:1].to(torch.bfloat16).cuda()  # [1, 3, 378, 378]
    vision_embed = vision_encoder(single_patch)
    # The encoder may return (embedding, intermediate_outputs); take the first.
    if isinstance(vision_embed, tuple):
        vision_embed = vision_embed[0]
    print(f"Vision embedding shape: {vision_embed.shape}")

    embed_np = vision_embed[0].float().cpu().numpy()             # [1024]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 32x32 heatmap -- purely visual, helps spot spatial-like clusters.
    im = axes[0].imshow(
        embed_np.reshape(32, 32), aspect='auto', cmap='viridis'
    )
    axes[0].set_title(
        f'Vision Embedding\n(reshaped 32×32)', fontsize=11
    )
    axes[0].set_xlabel('Dim (x32)')
    axes[0].set_ylabel('Dim (x32)')
    plt.colorbar(im, ax=axes[0])

    # 1D dimension-by-dimension plot.
    axes[1].plot(embed_np, color='steelblue', linewidth=0.8, alpha=0.8)
    axes[1].set_title(
        f'Vision Embedding Values\n(shape: {embed_np.shape})', fontsize=11
    )
    axes[1].set_xlabel('Dimension')
    axes[1].set_ylabel('Value')
    axes[1].grid(True, alpha=0.3)

    # Distribution.
    axes[2].hist(
        embed_np.flatten(), bins=50, color='steelblue',
        alpha=0.7, edgecolor='white'
    )
    axes[2].set_title('Vision Embedding\nValue Distribution', fontsize=11)
    axes[2].set_xlabel('Value')
    axes[2].set_ylabel('Count')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('BLIP-3 Vision Encoder: Embedding Analysis', fontsize=13)
    plt.tight_layout()
    plt.savefig(
        f"{results_dir}/3_vision_embeddings.png", dpi=150, bbox_inches='tight'
    )
    plt.close()
    print("Saved 3_vision_embeddings.png")


# ============================================================================
# FIGURE 4 — CROSS-MODAL ATTENTION  (Figure 8.7, Section 8.5.3)
# ----------------------------------------------------------------------------
# Approach: register forward hooks on the first 4 layers of the Phi-3
# language model. Each hook captures the attention tensor of that layer
# during a real generation pass. We then average over heads and plot.
#
# Why the first 4 layers (and not all 32):
#   Section 8.5.3 in the thesis discusses an extended layer 1/8/16/32
#   analysis (Figure 8.8). This script extracts only the first 4 layers
#   for the original Figure 8.7. The extended figure was produced by
#   a separate analysis run targeting layers [0, 7, 15, 31].
#
# attn tensor shape per layer: [batch=1, heads, seq_len, seq_len]
# After mean over heads: [seq_len, seq_len]
# ============================================================================

attention_weights = []


def hook_fn(module, input, output):
    """
    Forward hook on self-attention. Phi-3 self-attention returns
    (hidden_state, attn_weights, ...); we keep the attention if it is
    not None (FlashAttention returns None, eager attention returns a tensor).
    """
    if isinstance(output, tuple):
        if len(output) > 1 and output[1] is not None:
            attention_weights.append(output[1].detach().float().cpu())


# Attach hooks to the first 4 transformer layers of the language model.
hooks = []
for layer in model.vlm.lang_model.model.layers[:4]:
    h = layer.self_attn.register_forward_hook(hook_fn)
    hooks.append(h)

# Real generation pass -- this triggers the hooks. We use a small max_new_tokens
# because attention patterns of interest already form during the prefill step.
with torch.no_grad():
    generated = model.generate(
        **inputs,
        image_size=[image.size],
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        max_new_tokens=16,
        top_p=None,
        num_beams=1,
        output_attentions=True,
        stopping_criteria=[EosListStoppingCriteria()]
    )

# Always remove the hooks afterwards -- forgetting this leaks memory and
# breaks subsequent forward passes.
for h in hooks:
    h.remove()

prediction = tokenizer.decode(
    generated[0], skip_special_tokens=True
).split("<|end|>")[0]
print(f"Model answer: {prediction}")

if attention_weights:
    fig, axes = plt.subplots(
        1, min(4, len(attention_weights)), figsize=(20, 5)
    )
    if len(attention_weights) == 1:
        axes = [axes]
    for i, attn in enumerate(attention_weights[:4]):
        # attn shape: [batch=1, heads, seq, seq]
        # Take batch 0, then mean over the head dimension -> [seq, seq]
        attn_mean = attn[0].mean(0).numpy()
        # Crop to the first 50 tokens so the figure is readable -- this is
        # the slice that mostly covers system + user prompt tokens.
        seq_len = min(50, attn_mean.shape[0])
        im = axes[i].imshow(
            attn_mean[:seq_len, :seq_len], cmap='hot', aspect='auto'
        )
        axes[i].set_title(f'Layer {i+1} Attention\n(avg over heads)', fontsize=10)
        axes[i].set_xlabel('Key tokens')
        axes[i].set_ylabel('Query tokens')
        plt.colorbar(im, ax=axes[i])
    plt.suptitle(
        'Cross-Modal Attention Patterns (Language Model Layers)', fontsize=13
    )
    plt.tight_layout()
    plt.savefig(
        f"{results_dir}/4_attention_maps.png", dpi=150, bbox_inches='tight'
    )
    plt.close()
    print("Saved 4_attention_maps.png")
else:
    # Fallback if the attention extraction failed (e.g. eager flag not honoured).
    # We then just plot the input token sequence as a sanity-check artefact.
    print("No attention weights captured — saving token analysis instead")
    tokens = tokenizer.convert_ids_to_tokens(
        inputs['input_ids'][0].cpu().tolist()
    )
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(tokens[:40])), [1] * min(40, len(tokens)),
           color='steelblue', alpha=0.6)
    ax.set_xticks(range(min(40, len(tokens))))
    ax.set_xticklabels(tokens[:40], rotation=45, ha='right', fontsize=8)
    ax.set_title('Input Token Sequence (first 40 tokens)', fontsize=13)
    plt.tight_layout()
    plt.savefig(
        f"{results_dir}/4_token_sequence.png", dpi=150, bbox_inches='tight'
    )
    plt.close()
    print("Saved 4_token_sequence.png")


# ============================================================================
# FIGURE 5 — BASE vs FINE-TUNED ACCURACY  (Figure 8.2, Section 8.2)
# ----------------------------------------------------------------------------
# Uses results produced by a separate base-vs-LoRA evaluation script
# (eval_base.json, eval_finetuned.json). This figure is a snapshot of the
# 500-sample preview comparison; the statistically robust 2,000-sample
# numbers are in Tables 8.2 and 8.3 (a different evaluation run).
# ============================================================================

eval_base = json.load(open(f"{results_dir}/eval_base.json"))
eval_ft = json.load(open(f"{results_dir}/eval_finetuned.json"))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: side-by-side accuracy bars
models = ['BASE\nBLIP-3', 'Fine-tuned\nBLIP-3 (LoRA)']
accs = [eval_base['accuracy'], eval_ft['accuracy']]
colors = ['#5b9bd5', '#70ad47']
bars = axes[0].bar(
    models, accs, color=colors, width=0.4,
    edgecolor='white', linewidth=1.5
)
axes[0].set_ylim(0, 100)
axes[0].set_ylabel('VQA Accuracy (%)', fontsize=12)
axes[0].set_title('VQAv2 Accuracy: Base vs Fine-tuned', fontsize=13)
for bar, acc in zip(bars, accs):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2.,
        bar.get_height() + 1,
        f'{acc:.1f}%',
        ha='center', va='bottom',
        fontsize=13, fontweight='bold'
    )
axes[0].grid(True, alpha=0.3, axis='y')

# Right: absolute counts of correct vs incorrect
categories = ['Correct', 'Incorrect']
base_vals = [
    eval_base['correct'],
    eval_base['total'] - eval_base['correct']
]
ft_vals = [
    eval_ft['correct'],
    eval_ft['total'] - eval_ft['correct']
]
x = np.arange(2)
w = 0.35
axes[1].bar(x - w/2, base_vals, w, label='BASE',
            color='#5b9bd5', edgecolor='white')
axes[1].bar(x + w/2, ft_vals, w, label='Fine-tuned',
            color='#70ad47', edgecolor='white')
axes[1].set_xticks(x)
axes[1].set_xticklabels(categories, fontsize=12)
axes[1].set_ylabel('Number of samples', fontsize=12)
axes[1].set_title(
    f'Correct vs Incorrect\n(n={eval_base["total"]} samples)', fontsize=13
)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle('BLIP-3 LoRA Fine-tuning on VQAv2', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(
    f"{results_dir}/5_results_comparison.png", dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved 5_results_comparison.png")

print("\n=== PIPELINE ANALYSIS COMPLETE ===")
print(f"All figures saved to {results_dir}/")