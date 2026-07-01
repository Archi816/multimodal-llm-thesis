"""
convert_vqa_to_llava.py
=======================

Converts raw VQAv2 annotations into the LLaVA-style JSON format expected
by the xGen-MM training framework.

Reference: Chapter 7, Section 7.4.2 "Data Format Conversion".

The official VQAv2 release ships two separate files:
    v2_OpenEnded_mscoco_<split>2014_questions.json   -- one entry per question
    v2_mscoco_<split>2014_annotations.json           -- one entry per question
                                                         with the 10 human answers

xGen-MM expects ONE conversation per sample with the structure:
    {
      "id": "<question_id>",
      "image": "<split>2014/COCO_<split>2014_<imageid12>.jpg",
      "conversations": [
          {"from": "human", "value": "<image>\\n<question>"},
          {"from": "gpt",   "value": "<single answer>"}
      ]
    }

This script writes two files used by finetune_full.py and eval_full.py:
    vqa_train_llava_format.json  (50,000 samples, training)
    vqa_val_llava_format.json    (5,000 samples,   evaluation pool)
"""

import json

# Local directory holding the four official VQAv2 JSON files plus COCO images.
data_dir = "/mnt/data/home/arpoke644/blip3_thesis/data/vqa"

# ============================================================================
# 1. LOAD TRAINING QUESTIONS AND ANNOTATIONS
# ----------------------------------------------------------------------------
# Both files reference the same question_id. We build a dict keyed by id so
# the merge in step 2 is O(1) per annotation.
# ============================================================================

print("Loading questions...")
with open(f"{data_dir}/v2_OpenEnded_mscoco_train2014_questions.json") as f:
    questions = {q['question_id']: q for q in json.load(f)['questions']}

print("Loading annotations...")
with open(f"{data_dir}/v2_mscoco_train2014_annotations.json") as f:
    annotations = json.load(f)['annotations']


# ============================================================================
# 2. CONVERT TO LLaVA FORMAT  (50K training samples)
# ----------------------------------------------------------------------------
# Each VQAv2 question has TEN human-provided answers. The standard practice
# (used by LLaVA, BLIP-2 and others) is to train on the 'multiple_choice_answer'
# field, which is the single most frequent answer chosen by the annotators.
# Training on all 10 would multiply the dataset size and dilute the signal.
#
# We slice annotations[:50000] -- this gives us the first 50K question_ids
# in the order the file ships, which matches the training-set size discussed
# in Section 8.4 (LoRA full experiment).
# ============================================================================

print("Converting...")
llava_data = []
for ann in annotations[:50000]:
    qid = ann['question_id']
    if qid not in questions:
        # Defensive: an annotation without a matching question is malformed,
        # skip rather than crash.
        continue
    q = questions[qid]

    # COCO file naming convention: zero-padded 12-digit image id.
    image_file = f"train2014/COCO_train2014_{str(q['image_id']).zfill(12)}.jpg"

    answer = ann['multiple_choice_answer']

    llava_data.append({
        "id": str(qid),                       # kept as string -- xGen-MM expects str
        "image": image_file,
        "conversations": [
            {"from": "human", "value": f"<image>\n{q['question']}"},
            {"from": "gpt",   "value": answer}
        ]
    })

out_path = f"{data_dir}/vqa_train_llava_format.json"
with open(out_path, 'w') as f:
    json.dump(llava_data, f)
print(f"Saved {len(llava_data)} samples to {out_path}")


# ============================================================================
# 3. SAME CONVERSION FOR THE VALIDATION SPLIT  (5K samples)
# ----------------------------------------------------------------------------
# The validation pool here is larger than what we actually evaluate on
# (2,000 for the primary table, 500 for the three-model comparison). Keeping
# 5,000 ready makes it easy to re-run evaluations on different subsets
# without re-converting.
# ============================================================================

with open(f"{data_dir}/v2_OpenEnded_mscoco_val2014_questions.json") as f:
    val_questions = {q['question_id']: q for q in json.load(f)['questions']}

with open(f"{data_dir}/v2_mscoco_val2014_annotations.json") as f:
    val_annotations = json.load(f)['annotations']

val_data = []
for ann in val_annotations[:5000]:
    qid = ann['question_id']
    if qid not in val_questions:
        continue
    q = val_questions[qid]
    image_file = f"val2014/COCO_val2014_{str(q['image_id']).zfill(12)}.jpg"
    answer = ann['multiple_choice_answer']
    val_data.append({
        "id": str(qid),
        "image": image_file,
        "conversations": [
            {"from": "human", "value": f"<image>\n{q['question']}"},
            {"from": "gpt",   "value": answer}
        ]
    })

out_path_val = f"{data_dir}/vqa_val_llava_format.json"
with open(out_path_val, 'w') as f:
    json.dump(val_data, f)
print(f"Saved {len(val_data)} val samples to {out_path_val}")
print("Done!")
