#    Copyright 2023 Haotian Liu
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
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


# def load_tokenizer_prefer_model_path(model_path, model_base, use_fast=False):
#     if model_path is not None and os.path.isdir(model_path):
#         try:
#             tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
#             print(f"[Tokenizer] loaded from model_path: {model_path}")
#             return tokenizer
#         except Exception as e:
#             print(f"[Tokenizer] failed to load from model_path: {e}")

#     if model_base is not None:
#         tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=use_fast)
#         print(f"[Tokenizer] loaded from model_base: {model_base}")
#         return tokenizer

#     raise RuntimeError("Failed to load tokenizer")


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    
    if 'llava' in model_name.lower():
        # Load LLaVA model
        if ('lora' in model_name.lower() or 'llavarad' in model_name.lower()) and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if ('lora' in model_name.lower() or 'llavarad' in model_name.lower()) and model_base is not None:
            # 使用 LlavaConfig 直接加载，而不是 AutoConfig，确保 'llava' 类型被识别
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            # tokenizer = load_tokenizer_prefer_model_path(
            #     model_path=model_path,
            #     model_base=model_base,
            #     use_fast=False
            # )

            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # 尝试从本地常见位置查找
                alt_paths = [
                    os.path.join(os.path.dirname(model_path), 'non_lora_trainables.bin'),
                    os.path.join('data', 'models', 'llava-rad', 'non_lora_trainables.bin'),
                    os.path.join(os.getcwd(), 'non_lora_trainables.bin')
                ]
                found = False
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        non_lora_trainables = torch.load(alt_path, map_location='cpu')
                        found = True
                        break
                if not found:
                    raise FileNotFoundError(f"找不到文件 non_lora_trainables.bin。请确保文件存在于以下位置之一:\n"
                                         f"  - {os.path.join(model_path, 'non_lora_trainables.bin')}\n"
                                         f"  - {os.path.join('data', 'models', 'llava-rad', 'non_lora_trainables.bin')}")
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            # if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
            #     mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            # else: 
            #     raise FileNotFoundError(f"找不到文件 mm_projector.bin")
            
            # mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            # model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None
    
    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)

        if mm_use_im_start_end:
            special_tokens = []
            if DEFAULT_IM_START_TOKEN not in tokenizer.get_vocab():
                special_tokens.append(DEFAULT_IM_START_TOKEN)
            if DEFAULT_IM_END_TOKEN not in tokenizer.get_vocab():
                special_tokens.append(DEFAULT_IM_END_TOKEN)

            if len(special_tokens) > 0:
                tokenizer.add_tokens(special_tokens, special_tokens=True)
                
        if mm_use_im_patch_token:
            special_tokens = []
            if DEFAULT_IMAGE_PATCH_TOKEN not in tokenizer.get_vocab():
                special_tokens.append(DEFAULT_IMAGE_PATCH_TOKEN)
            if len(special_tokens) > 0:
                tokenizer.add_tokens(special_tokens, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))


        # if mm_use_im_patch_token:
        #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        # if mm_use_im_start_end:
        #     tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
