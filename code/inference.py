"""
inference.py
============

Baseline (zero-shot) inference of BLIP-3 on the six test samples shipped
with the xGen-MM checkpoint.

This script produces the qualitative baseline reported in:
    - Section 8.1 "Baseline Inference Quality"
    - Table 8.1   "Baseline BLIP-3 responses on the six provided test samples"

The six samples were chosen by the model authors to cover diverse VQA
scenarios: meme interpretation, food-web reasoning, microscopy, seasonal
inference, family scene, and OCR-style document reading.

This is the FIRST script that ran successfully end-to-end on PERUN. Several
compatibility issues had to be resolved before it produced sensible output;
those fixes are documented in the comment block below and discussed in
Chapter 7 (Section 7.1, Software Environment).
"""

# ============================================================================
# COMPATIBILITY FIXES REQUIRED FOR BLIP-3 ON HPC PERUN
# ----------------------------------------------------------------------------
# (1) open_clip 3.3.0 -> downgraded to 2.24.0
#     The newer open_clip changed the EVA-CLIP loader signature; xGen-MM
#     calls into the old API and crashes on 3.3.0 with KeyError on weights.
#
# (2) modeling_xgenmm.py: vision encoder is now loaded on CPU first and then
#     moved to GPU. The original code allocated EVA-CLIP directly on CUDA
#     while the LLM was loading, which exceeded GPU memory at init time on
#     H200 in our environment.
#
# (3) open_clip/factory.py: patched to handle 'meta' tensors. PyTorch's
#     low_cpu_mem_usage=True uses meta tensors as placeholders, but
#     open_clip's load_state_dict does not understand the meta device and
#     raises NotImplementedError. We bypass this by setting
#     low_cpu_mem_usage=False, which is the simplest workaround.
#
# (4) bfloat16 instead of float16. With fp16 we observed NaN loss within
#     ~10 steps; bfloat16 is numerically stable on H200 (Hopper supports
#     it natively at full throughput).
# ============================================================================

from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria
import torch, json, PIL.Image, textwrap

model_path = "/mnt/data/home/arpoke644/blip3_thesis/checkpoints/xgen-mm-phi3"
test_json  = f"{model_path}/test_samples/test.json"

# ----------------------------------------------------------------------------
# Load model + tokenizer + image processor
# ----------------------------------------------------------------------------
print("Loading model...")
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    trust_remote_code=True,         # required for xGen-MM custom modeling code
    dtype=torch.bfloat16,           # see fix (4) above
    low_cpu_mem_usage=False         # see fix (3) above
)
model = model.to('cuda')

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, legacy=False)
image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
tokenizer = model.update_special_tokens(tokenizer)
model.eval()
tokenizer.padding_side = "left"   # left-padding is correct for generation
print("Model loaded!")


# ----------------------------------------------------------------------------
# Phi-3 chat template (Section 7.3). The <image> placeholder is consumed by
# the xGen-MM forward pass and substituted with the visual tokens produced
# by the Perceiver Resampler.
# ----------------------------------------------------------------------------
def apply_prompt_template(prompt):
    return (
        '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
        "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
        f'<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n'
    )


# Stop at <|end|> (token id 32007) so the decoded output is clean.
class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[32007]):
        self.eos_sequence = eos_sequence
    def __call__(self, input_ids, scores, **kwargs):
        return self.eos_sequence in input_ids[:, -len(self.eos_sequence):].tolist()


# ----------------------------------------------------------------------------
# Iterate over the six test samples.
# Each sample contains an image_path and a list of questions. We rewrite the
# image path so it points at the local checkpoint rather than the relative
# './test_samples' that ships with the model.
# ----------------------------------------------------------------------------
with open(test_json) as f:
    data = json.load(f)

for sample in data:
    img_path = sample['image_path'].replace('./test_samples', f'{model_path}/test_samples')
    img = PIL.Image.open(img_path).convert('RGB')

    # AnyRes preprocessing -> [1, 1, 5, 3, 378, 378]  (Section 6.3.1)
    inputs = image_processor([img], return_tensors="pt", image_aspect_ratio='anyres')

    # Move to GPU; cast the float tensors (pixel values) to bfloat16, keep
    # integer tensors (image_size etc.) as-is.
    inputs = {
        k: v.cuda().to(torch.bfloat16) if v.dtype == torch.float32 else v.cuda()
        for k, v in inputs.items()
    }

    print("="*80)
    print(f"Image: {img_path}")
    for query in sample['question']:
        prompt = apply_prompt_template(query)
        language_inputs = tokenizer([prompt], return_tensors="pt")
        inputs.update({k: v.cuda() for k, v in language_inputs.items()})

        # Greedy decoding -- matches the protocol used in the quantitative
        # evaluation (eval_full.py) for consistency across the thesis.
        generated_text = model.generate(
            **inputs, image_size=[img.size],
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False, max_new_tokens=768,
            top_p=None, num_beams=1,
            stopping_criteria=[EosListStoppingCriteria()]
        )
        prediction = tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0]
        print(f"User: {query}")
        print(f"Assistant: {textwrap.fill(prediction, width=100)}")
    print("-"*80)

print("Done!")
