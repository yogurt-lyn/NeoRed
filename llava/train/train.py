# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os

import shutil
import argparse
from datetime import datetime

def save_code_to_checkpoint(code_file_path, checkpoint_dir):
    """
    将指定的代码文件复制到 checkpoint 目录
    """
    if not os.path.exists(code_file_path):
        print(f"警告: 代码文件不存在 {code_file_path}")
        return False
    
    # 确保 checkpoint 目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 构建目标文件路径
    dest_path = os.path.join(checkpoint_dir, "llava_llama.py")
    
    # 复制文件
    shutil.copy2(code_file_path, dest_path)
    # print(f代码文件已保存到: {dest_path}")
    
    # 可选：也保存一个带时间戳的备份
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(checkpoint_dir, f"llava_llama_{timestamp}.py")
    shutil.copy2(code_file_path, backup_path)
    print(f"备份已保存到: {backup_path}")
    
    return True


import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token, open_image_with_retry,get_image_root,process_images
from llava.utils import data_loaders

from PIL import Image, ImageFile
# https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
ImageFile.LOAD_TRUNCATED_IMAGES = True

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_config: Optional[str] = field(default=None)
    vision_tower_checkpoint: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

    output_hidden_states: bool = field(
        default=False,
        metadata={"help": "Whether to return hidden states from all transformer layers."}
    )
    output_attentions: bool = field(
        default=False,
        metadata={"help": "Whether to return attention weights from all transformer layers."}
    )
    

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, 
                                metadata={"help": "Path to the validation data."})
    loader: str = "default"
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'pad'
    image_grid_pinpoints: Optional[str] = field(default=None)
    max_image_num:Optional[int] = field(default=None)
    num_diseases: int = field(default=9)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    group_by_modality_length: bool = field(default=False)
    findings_loss_weight: float = field(default=1.0)
    diagnosis_loss_weight: float = field(default=1.0)
    align_loss_weight: float = field(default=1.0)
    llm_loss_weight: float = field(default=1.0)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return



def find_all_linear_names(model):
    linear_module_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_module_names.append(name)   
    return linear_module_names


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _mask_targets(target, tokenized_lens, speakers):
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources

def preprocess_report(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def visualize_token_level(
    input_ids,
    labels,
    I_mask,
    D_mask,
    tokenizer,
    IGNORE_INDEX=-100,
    b=0,
    start=None,
    end=None
):
    """
    逐 token 可视化：
    index | token_id | token_str | label | I_mask | D_mask
    """
    ids = input_ids[b]
    lbl = labels[b]
    im = I_mask[b]
    dm = D_mask[b]

    if start is None:
        start = 0
    if end is None:
        end = ids.size(0)

    print(
        f"{'idx':>4} {'id':>6} {'token':>15} "
        f"{'label':>7} {'I':>3} {'D':>3}"
    )
    print("-" * 50)

    for i in range(start, end):
        token_id = ids[i].item()
        token_str = tokenizer.convert_ids_to_tokens(token_id)

        print(
            f"{i:4d} {token_id:6d} {token_str:>15} "
            f"{int(lbl[i] != IGNORE_INDEX):7d} "
            f"{int(im[i] > 0):3d} {int(dm[i] > 0):3d}"
        )


def find_subsequence(sequence, subseq):
    """返回 subseq 在 sequence 中第一次出现的起始 index；否则 None"""
    n, m = len(sequence), len(subseq)
    for i in range(n - m + 1):
        if sequence[i:i + m] == subseq:
            return i
    return None

# def generate_I_D_mask_from_labels(input_ids, labels):
#     """
#     基于 labels 的有效区域生成 I_mask 和 D_mask，附加 debug 输出
#     """
#     bsz, seqlen = labels.shape
#     I_mask = torch.zeros_like(labels, dtype=torch.bool)
#     D_mask = torch.zeros_like(labels, dtype=torch.bool)

#     IMAGING_CONCLUSION_IDS = [15997, 29901]
#     DISEASE_DIAGNOSIS_IDS  = [559, 24876, 19263, 29901]

#     for b in range(bsz):
#         label = labels[b]
#         # 只看 label 覆盖部分
#         valid_pos = (label != IGNORE_INDEX).nonzero(as_tuple=True)[0]
#         if len(valid_pos) == 0:
#             print(f"[DEBUG] batch {b}: no valid label positions")
#             continue

#         start, end = valid_pos[0].item(), valid_pos[-1].item() + 1
#         label_ids = input_ids[b, start:end].tolist()
#         print(f"[DEBUG] batch {b}: valid label span {start}-{end}, length={end-start}")

#         # 找 marker 在 label_ids 中的位置
#         im_start = None
#         for i in range(len(label_ids) - len(IMAGING_CONCLUSION_IDS) + 1):
#             if label_ids[i:i+len(IMAGING_CONCLUSION_IDS)] == IMAGING_CONCLUSION_IDS:
#                 im_start = i
#                 break
#         if im_start is None:
#             print(f"[DEBUG] batch {b}: IMAGING_CONCLUSION_IDS not found in label_ids")
#         else:
#             print(f"[DEBUG] batch {b}: IMAGING_CONCLUSION_IDS found at relative index {im_start}")

#         d_start = None
#         for i in range(len(label_ids) - len(DISEASE_DIAGNOSIS_IDS) + 1):
#             if label_ids[i:i+len(DISEASE_DIAGNOSIS_IDS)] == DISEASE_DIAGNOSIS_IDS:
#                 d_start = i
#                 break
#         if d_start is None:
#             print(f"[DEBUG] batch {b}: DISEASE_DIAGNOSIS_IDS not found in label_ids")
#         else:
#             print(f"[DEBUG] batch {b}: DISEASE_DIAGNOSIS_IDS found at relative index {d_start}")

#         if im_start is None or d_start is None:
#             continue

#         # I_mask 覆盖区间：marker 后到 D_marker 前
#         im_content_start = start-4 + im_start + len(IMAGING_CONCLUSION_IDS)
#         im_content_end   = start + d_start -1 # 不包含 D marker
#         print(f"[DEBUG] batch {b}: I_mask span {im_content_start}-{im_content_end}")

#         # D_mask 覆盖区间：D_marker 后到 labels 末尾
#         d_content_start = start-6 + d_start + len(DISEASE_DIAGNOSIS_IDS)
#         d_content_end   = end
#         print(f"[DEBUG] batch {b}: D_mask span {d_content_start}-{d_content_end}")

#         # 填充 mask
#         I_mask[b, im_content_start:im_content_end] = True
#         D_mask[b, d_content_start:d_content_end] = True

#     return I_mask, D_mask

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    # # ====== DEBUG: 打印第一个样本的 token、ID、label ======
    # if len(conversations) > 0:
    #     print("\n" + "="*80)
    #     print("🔍 DEBUG: Token-Level Inspection (First Sample)")
    #     print("="*80)
        
    #     input_ids_0 = input_ids[0].tolist()
    #     targets_0 = targets[0].tolist()
        
    #     # 在 mask 之前先保存原始 tokens（用于对齐）
    #     vocab_size = len(tokenizer)
    #     safe_input_ids_0 = [
    #         idx if 0 <= idx < vocab_size else tokenizer.unk_token_id 
    #         for idx in input_ids_0
    #     ]
    #     tokens_0 = tokenizer.convert_ids_to_tokens(safe_input_ids_0)
        
    #     # 先做一次 mask（为了看到最终 label）
    #     # 注意：这里我们临时 copy 一份 target 来做 mask，避免影响原逻辑
    #     temp_target = targets[0].clone()
    #     total_len = int(temp_target.ne(tokenizer.pad_token_id).sum())
    #     conversation = conversations[0]
    #     cur_len = 1
    #     temp_target[:cur_len] = IGNORE_INDEX
    #     rounds = conversation.split(conv.sep2)
    #     for i, rou in enumerate(rounds):
    #         if rou == "":
    #             break
    #         parts = rou.split(sep)
    #         if len(parts) != 2:
    #             break
    #         parts[0] += sep
    #         if has_image:
    #             round_len = len(tokenizer_image_token(rou, tokenizer))
    #             instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
    #         else:
    #             round_len = len(tokenizer(rou).input_ids)
    #             instruction_len = len(tokenizer(parts[0]).input_ids) - 2
    #         temp_target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
    #         cur_len += round_len
    #     temp_target[cur_len:] = IGNORE_INDEX
    #     if cur_len < tokenizer.model_max_length and cur_len != total_len:
    #         temp_target[:] = IGNORE_INDEX

    #     # 打印 token 表格
    #     for i, (token, id_val, label_val) in enumerate(zip(tokens_0, input_ids_0, temp_target.tolist())):
    #         if id_val == tokenizer.pad_token_id:
    #             continue  # 跳过 pad
    #         status = "✅" if label_val != IGNORE_INDEX else "❌"
    #         print(f"{i:4d} | {status} | ID={id_val:5d} | Token={repr(token):20}")
    #     print("="*80 + "\n")

    return {
        "input_ids": input_ids,
        "labels": targets
    }
    

    # try:
    # 取第一个样本
        # b = 0
        # ids = input_ids[b].tolist()
        # d_start = find_subsequence(ids, DISEASE_DIAGNOSIS_IDS)

        # if d_start is not None:
        #     visualize_token_level(
        #         input_ids=input_ids,
        #         labels=targets,
        #         I_mask=I_mask,
        #         D_mask=D_mask,
        #         tokenizer=tokenizer,
        #         b=b,
        #         # start=max(d_start - 500, 0),
        #         # end=min(d_start, input_ids.size(1))
        #         start=max(d_start - 30, 0),
        #         end=min(d_start + 100, input_ids.size(1))
        #     )
    # except Exception as e:
    #     print(f"Visualization failed: {e}")

    # ids = input_ids[0].tolist()
    # marker = [1888, 6751, 15997, 29901]

    # # 打印滑动窗口看看能否找到连续序列
    # for i in range(len(ids) - len(marker) + 1):
    #     if ids[i:i+len(marker)] == marker:
    #         print("Found IMAGING_CONCLUSION_IDS at index", i)

def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

from PIL import Image


def load_images(image_paths, max_images=6):
    images = []
    limited_paths = image_paths[:max_images]
    for p in limited_paths:
        img = open_image_with_retry(p)
        images.append(img)
    return images

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = data_loaders[data_args.loader](data_path)
        self.data_path=data_path

        self.num_diseases = data_args.num_diseases


        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1
        sample = sources[0]

        # ===== 新增：多标签疾病 =====
        disease_labels = torch.zeros(self.num_diseases, dtype=torch.float32)
        for d in sample.get("disease_ids", []):
            disease_labels[d] = 1.0
        has_signal = torch.tensor(1.0 if "disease_ids" in sample else 0.0) 


        if 'image' in sample:
            image_file = sample['image']

            # if len(image_file) == 1:
            #     print("Only input one image!")

            if isinstance(image_file, str):
                image_file = [image_file]

            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            # category = self.data_path.split('/')[-2]
            category=sample['category']
            image_paths = []
            images=[]

            if category == "IU_XRAY":
                # print("loading IU_XRAY..")
                root = os.path.join(image_folder, "IU_XRAY", "IU_XRAY", "images")
                image_paths = [os.path.join(root, img) for img in image_file]
                images = load_images(image_paths=image_paths, max_images=self.data_args.max_image_num)
                
            elif category == "MIMIC_CXR":
                # print("loading MIMIC_CXR..")
                subject_id = sample.get("subject_id")
                sid = sample.get("id")
                root = os.path.join(image_folder, "MIMIC_CXR", "MIMIC_CXR","data", "wcl",
                                    "physionet.org", "files", "mimic-cxr-jpg", "2.0.0", "files")
                image_paths = [
                    os.path.join(root, f"p{str(subject_id)[:2]}", f"p{subject_id}", f"s{sid}", img + ".jpg")
                    for img in image_file
                ]
                # print(*image_paths, sep='\n')  # 每个元素一行
                images = load_images(image_paths=image_paths, max_images=self.data_args.max_image_num)

            elif category =="VQA_RAD":
                # print("loading VQA_RAD..")
                
                images = sample.get("image")

            elif category == "SLAKE":
                # print("loading SLAKE..")
                img_path = sample.get("image")
                image_paths = [os.path.join(image_folder, "SLAKE", "SLAKE", "imgs", img_path)]
                images = load_images(image_paths=image_paths, max_images=self.data_args.max_image_num)

            elif category == "PATH_VQA":
                # print("loading PATH_VQA..")
                images = sample.get("image")

            elif category == "NeoCXR":
                # img_path = sample.get("image")
                # image_paths = [os.path.join(image_folder,img_path)]
                image_folder = "/data_lyn/data/NeoCXR/images"
                image_paths = [
                    os.path.join(image_folder, p.strip())
                    for p in sample["image"].split(",")
                    if p.strip()
                ]
                # print(image_paths)
                images = load_images(image_paths=image_paths, max_images=self.data_args.max_image_num)

            else:
                root = os.path.join(image_folder, category, category)
                image_paths = [os.path.join(root, img) for img in image_file]
                images = load_images(image_paths=image_paths)

            image_tensor = process_images(images, processor, self.data_args)

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             disease_labels=disease_labels, has_signal=has_signal,modal_token_spans=None)

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image_tensor
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            print("crop_size:",crop_size)
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        disease_labels = None
        if "disease_labels" in instances[0]:
            disease_labels = torch.stack(
                [inst["disease_labels"] for inst in instances]
            )

        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if "has_signal" in instances[0]:
            batch["has_signal"] = torch.stack([inst["has_signal"] for inst in instances])

        if "modal_token_spans" in instances[0]:
            batch["modal_token_spans"] = [
                inst["modal_token_spans"] for inst in instances
            ]

        if disease_labels is not None:
            batch["disease_labels"] = disease_labels

        # if I_mask is not None:
        #     batch["I_mask"] = I_mask

        if 'image' in instances[0]:
            images = [inst['image'] for inst in instances]

            if all(isinstance(img, torch.Tensor) and img.dim() == 3 for img in images):
                batch["images"] = torch.stack(images)
        
            else:
                batch["images"] = images

        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    eval_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.eval_data_path,     
        data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


def debug_clinical_tokenization_detailed(tokenizer):
    clinical_tags = ["<dev>", "</dev>", "<peri>", "</peri>", "<phys>", "</phys>"]
    
    print("\n" + "="*70)
    print("DETAILED Clinical Tokenization Analysis")
    print("="*70)
    
    original_vocab_size = 32000  # LLaMA 原始 vocab size，可根据你的模型调整
    
    for tag in clinical_tags:
        print(f"\n--- Analyzing: {tag} ---")
        
        # 获取 token IDs（不加 special tokens，避免 <s>/</s> 干扰）
        token_ids = tokenizer.encode(tag, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        print(f"Full string: {repr(tag)}")
        print(f"Token IDs   : {token_ids}")
        print(f"Tokens      : {tokens}")
        print(f"Num tokens  : {len(tokens)}")
        
        # 逐个分析每个 token
        for i, (tid, tok) in enumerate(zip(token_ids, tokens)):
            is_special = tok in tokenizer.all_special_tokens
            in_original_vocab = tid < original_vocab_size if tid >= 0 else False
            print(f"  [{i}] ID={tid:5} | token={repr(tok):15} | special={is_special} | in_original_vocab={in_original_vocab}")
        
        if len(tokens) == 1 and tokens[0] == tag:
            print("Perfect: treated as a single special token")
        elif len(tokens) > 1:
            print("WARNING: split into multiple subwords!")
        else:
            print("Unexpected case")
    
    print("="*70 + "\n")


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    print("Loading LLaVA from:", model_args.model_name_or_path)
    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            config = LlavaConfig.from_pretrained(model_args.model_name_or_path)
            config.findings_loss_weight=training_args.findings_loss_weight
            config.llm_loss_weight=training_args.llm_loss_weight
            config.align_loss_weight=training_args.align_loss_weight
            config.diagnosis_loss_weight=training_args.diagnosis_loss_weight
            config.output_hidden_states=model_args.output_hidden_states
            config.output_attentions=model_args.output_attentions
            config.num_diseases=data_args.num_diseases
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                config=config,
                **bnb_model_from_pretrained_args
            )
            # print("config",config)
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # target_modules=find_all_linear_names(model),
    names = find_all_linear_names(model)
    # print(names)
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=names,
            modules_to_save=["disease_source_weights"],
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
         
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
                        
    model.disease_source_weights.requires_grad_(True)

    # 在训练结束前保存代码文件

    if training_args.local_rank == 0 or training_args.local_rank == -1:

        code_file = "/data_lyn/Neo_Red/llava/model/language_model/llava_llama.py"
        checkpoint_dir = training_args.output_dir  # 您的 checkpoint 输出目录
        
        print("="*50)
        print("保存模型代码文件到 checkpoint 目录...")
        print("="*50)
        save_code_to_checkpoint(code_file, checkpoint_dir)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    
    # # ===== 调试：检查 disease_labels 是否进 batch =====

    # batch = next(iter(trainer.get_train_dataloader()))
    # print(batch.keys())
    # print(batch["disease_labels"].shape, batch["disease_labels"].dtype)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
    # model = model.merge_and_unload()
    # model.save_pretrained(training_args.output_dir + "_merged")

    



if __name__ == "__main__":
    train()
