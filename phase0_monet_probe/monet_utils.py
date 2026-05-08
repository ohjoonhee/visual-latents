"""Minimal subset of upstream Monet `src/utils.py` and `src/task.py`.

Only the helpers needed for trace construction + alignment-position discovery.
Vendored verbatim where possible so semantics match the upstream collator.
"""
from __future__ import annotations

import math
import os
from typing import List, Union

import torch
from PIL import Image


# ----------------- text munging -----------------

def process_multiple_question_img(question_str: str) -> str:
    if "<abs_vis_token></abs_vis_token>" in question_str:
        question_str = (
            question_str
            .replace("<|vision_start|><|image_pad|><|vision_end|>", "")
            .replace("<abs_vis_token></abs_vis_token>", "<|vision_start|><|image_pad|><|vision_end|>")
        )
    return question_str


def replace_latent_placeholder_with_img_pad(
    text: str,
    image_pad: str = "<|vision_start|><|image_pad|><|vision_end|>",
    latent_placeholder: str = "<abs_vis_token></abs_vis_token>",
    sep_token: str = "<|im_start|>assistant",
) -> str:
    parts = text.split(sep_token)
    res_text = process_multiple_question_img(parts[0])
    for assistant in parts[1:]:
        if latent_placeholder in assistant:
            assistant = assistant.replace(image_pad, "")
            assistant = assistant.replace(latent_placeholder, image_pad)
        res_text += sep_token + assistant
    return res_text


def add_latent_pad_after_auxiliary_img(
    texts: List[str],
    latent_size: int,
    latent_pad_str: str = "<abs_vis_token_pad>",
) -> List[str]:
    update_texts = []
    latent_pad_strs = latent_pad_str * latent_size
    for text in texts:
        turns = text.split("<|im_start|>assistant")
        upd = process_multiple_question_img(turns[0])
        for turn in turns[1:]:
            upd += "<|im_start|>assistant" + turn.replace(
                "<|vision_start|><|image_pad|><|vision_end|>",
                f"<|vision_start|><|image_pad|><|vision_end|><abs_vis_token>{latent_pad_strs}</abs_vis_token>",
            )
        update_texts.append(upd)
    return update_texts


# ----------------- image resize -----------------

def resize_by_token_budget(
    images,
    global_max_pixels: int = 2000 * 28 * 28,
    per_img_max_pixels: int = 1280 * 28 * 28,
    divisor: int = 28,
):
    total = sum(img.width * img.height for img in images)
    if total <= global_max_pixels:
        return images, None
    ratio = math.sqrt(global_max_pixels / total)
    processed, new_sizes = [], []
    for img in images:
        w, h = int(img.width * ratio), int(img.height * ratio)
        w = max(divisor, (w // divisor) * divisor)
        h = max(divisor, (h // divisor) * divisor)
        if w * h > per_img_max_pixels:
            r = math.sqrt(per_img_max_pixels / (w * h))
            w = max(divisor, int(w * r) // divisor * divisor)
            h = max(divisor, int(h * r) // divisor * divisor)
        processed.append(img.resize((w, h), Image.BICUBIC))
        new_sizes.append((w, h))
    return processed, new_sizes


# ----------------- alignment-position helpers -----------------

def find_subsequence(row: torch.Tensor, pattern, start: int = 0) -> int:
    seq_len = row.size(0)
    if isinstance(pattern, torch.Tensor):
        max_pat_len = pattern.size(0)
    else:
        max_pat_len = max(pat.size(0) for pat in pattern)
    for start_idx in range(start, seq_len - max_pat_len + 1):
        if isinstance(pattern, torch.Tensor):
            pl = pattern.size(0)
            if torch.all(row[start_idx:start_idx + pl] == pattern):
                return start_idx
        else:
            for pat in pattern:
                pl = pat.size(0)
                if torch.all(row[start_idx:start_idx + pl] == pat):
                    return start_idx
    return -1


def find_ids_poss(input_ids: torch.Tensor, ans_start_pattern: torch.Tensor, target_id_or_list) -> List[List[int]]:
    """For each row, find positions of `target_id_or_list` strictly *after* the first
    occurrence of `ans_start_pattern`.
    """
    batch_poss = []
    for i in range(input_ids.shape[0]):
        poss = []
        start_idx = find_subsequence(input_ids[i], ans_start_pattern, 0)
        while start_idx != -1:
            start_idx = find_subsequence(input_ids[i], target_id_or_list, start_idx + 1)
            if start_idx != -1:
                poss.append(start_idx)
        batch_poss.append(poss)
    return batch_poss


# ----------------- task preprocessing (verbatim from upstream task.py) -----------------

def Monet_single_input_images_preprocess_function(
    sample: dict,
    dataset_root: str = "",
    allow_no_observation: bool = False,
):
    """Same as upstream `task.Monet_single_input_images_preprocess_function`."""
    n_img_pad = 0
    n_img = 0
    conversations = sample["data"]
    seen_observation = False
    for i, step in enumerate(conversations):
        new_step = step.copy()
        if step["role"] == "system":
            new_step["content"][0]["text"] = "You are a helpful assistant."
        seen_assistant_image = False if step["role"] == "assistant" else None
        for j, content in enumerate(new_step["content"]):
            if content["type"] == "image":
                img_file_name = content.pop("image")
                if "kling_mm" in dataset_root:
                    img_file_name = img_file_name.replace("created_dataset/filtered_data/", "")
                content["image"] = os.path.join(dataset_root, img_file_name)
                if j > 0 and new_step["content"][j - 1]["type"] == "text" and step["role"] == "assistant":
                    if "<abs_vis_token></abs_vis_token>" not in new_step["content"][j - 1]["text"]:
                        return None
                if step["role"] == "assistant":
                    n_img += 1
                    seen_assistant_image = True
            elif content["type"] == "text":
                if step["role"] == "assistant":
                    n_img_pad += content["text"].count("<abs_vis_token></abs_vis_token>")
                    if "<observation>" in content.get("text", "") and not seen_assistant_image:
                        content["text"] = content["text"].replace("<observation>", "").replace("</observation>", "")
                    if "<observation>" in content.get("text", ""):
                        seen_observation = True
                elif step["role"] == "user":
                    img_key = "image"
                    if (
                        "Zebra_CoT_visual_search" not in new_step["content"][0][img_key]
                        and "Zebra_CoT_count" not in new_step["content"][0][img_key]
                    ):
                        content["text"] = content["text"].replace("\nPut your final answer within \\boxed{}.", "")
            new_step["content"][j] = content
        conversations[i] = new_step
    sample["data"] = conversations
    if n_img != n_img_pad:
        return None
    if not seen_observation and not allow_no_observation:
        return None
    return sample
