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

MARKER_ID_PATTERNS = {
    "dev": {
        "start": [29966,3359, 29958],  
        "end":   [829,3359, 29958],
    },
    "peri": {
        "start": [29966, 546, 29875, 29958],
        "end":   [829,546, 29875, 29958],
    },
    "phys": {
        "start": [29966,14017, 29958],
        "end":   [829,14017, 29958],
    },
}

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,DEFAULT_DEV_START_TOKEN,DEFAULT_DEV_END_TOKEN,DEFAULT_PERI_START_TOKEN,DEFAULT_PERI_END_TOKEN,DEFAULT_PHYS_START_TOKEN,DEFAULT_PHYS_END_TOKEN

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.mm_vision_tower_config = model_args.vision_tower_config
        self.config.mm_vision_tower_checkpoint = model_args.vision_tower_checkpoint

        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        # self.config.mm_hidden_size = 4096
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    @staticmethod
    def find_subseq_ids(seq, pattern):
        """
        seq: List[int] or 1D tensor
        pattern: List[int]
        return: (start_idx, end_idx) or None
        """
        m = len(pattern)
        for i in range(len(seq) - m + 1):
            if seq[i:i + m] == pattern:
                return i, i + m
        return None

    @staticmethod
    def generate_I_D_mask_from_labels(input_ids, labels):
        """
        基于 labels 的有效区域生成 I_mask 和 D_mask，附加 debug 输出
        """
        bsz, seqlen = labels.shape
        I_mask = torch.zeros_like(labels, dtype=torch.bool)
        D_mask = torch.zeros_like(labels, dtype=torch.bool)

        IMAGING_CONCLUSION_IDS = [6751,15997, 29901]
        DISEASE_DIAGNOSIS_IDS  = [559, 24876, 19263, 29901]

        for b in range(bsz):
            label = labels[b]

            valid_pos = (label != IGNORE_INDEX).nonzero(as_tuple=True)[0]
            if len(valid_pos) == 0:

                continue

            start, end = valid_pos[0].item(), valid_pos[-1].item() + 1
            label_ids = input_ids[b, start:end].tolist()

            im_start = None
            for i in range(len(label_ids) - len(IMAGING_CONCLUSION_IDS) + 1):
                if label_ids[i:i+len(IMAGING_CONCLUSION_IDS)] == IMAGING_CONCLUSION_IDS:
                    im_start = i
                    break


            d_start = None
            for i in range(len(label_ids) - len(DISEASE_DIAGNOSIS_IDS) + 1):
                if label_ids[i:i+len(DISEASE_DIAGNOSIS_IDS)] == DISEASE_DIAGNOSIS_IDS:
                    d_start = i
                    break


            if im_start is None or d_start is None:
                continue

            # I_mask 覆盖区间：marker 后到 D_marker 前
            im_content_start = start-4 + im_start + len(IMAGING_CONCLUSION_IDS)
            im_content_end   = start + d_start -1 
        

            # D_mask 覆盖区间：D_marker 后到 labels 末尾
            d_content_start = start-6 + d_start + len(DISEASE_DIAGNOSIS_IDS)
            d_content_end   = end


            # 填充 mask
            I_mask[b, im_content_start:im_content_end] = True
            D_mask[b, d_content_start:d_content_end] = True
            
        I_mask = [I_mask[b] for b in range(bsz)]
        D_mask = [D_mask[b] for b in range(bsz)]
        return I_mask, D_mask


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            batch_size = input_ids.shape[0]
        
            return input_ids, attention_mask, past_key_values, None, labels, None, None, None
        
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        modal_token_spans=[]

        I_mask=None
        D_mask=None
        if self.training:
            I_mask, D_mask = self.generate_I_D_mask_from_labels(input_ids, labels)
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
                    
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

            # ============================================================
            # 计算 modal_token_spans（embedding 级别）
            # ============================================================

            if self.training:
            # if 1==1:
                
                cur_input_ids_raw = input_ids[batch_idx]
                cur_input_ids_tmp = cur_input_ids_raw.clone()

                spans = {
                    "image": [],
                    "dev": None,
                    "peri": None,
                    "phys": None
                }

                cur_pos = 0
                cur_image_idx_for_span = 0

                # -------- image offset 计算 --------
                num_images = (cur_input_ids_raw == IMAGE_TOKEN_INDEX).sum().item()
                total_offset = 0
                if num_images > 0:
                    img_len = image_features[0].shape[0]
                    total_offset = num_images * (img_len - 1)

                if total_offset > 0:
                    cur_I_mask = I_mask[batch_idx]   # shape: [L_tok]
                    cur_D_mask = D_mask[batch_idx]

                    # 对当前样本做 shift（embedding-level mask）
                    if total_offset > 0:
                        new_len = cur_I_mask.shape[0] + total_offset
                        shifted_I = torch.zeros(new_len, dtype=cur_I_mask.dtype, device=cur_I_mask.device)
                        shifted_D = torch.zeros(new_len, dtype=cur_D_mask.dtype, device=cur_D_mask.device)
                        
                        shifted_I[total_offset:] = cur_I_mask
                        shifted_D[total_offset:] = cur_D_mask
                        
                        I_mask[batch_idx] = shifted_I
                        D_mask[batch_idx] = shifted_D
                    else:
                        pass

                # -------- image span --------
                while (cur_input_ids_tmp == IMAGE_TOKEN_INDEX).any():
                    image_indices = torch.where(cur_input_ids_tmp == IMAGE_TOKEN_INDEX)[0]
                    token_idx = image_indices[0].item()

                    start = cur_pos + token_idx
                    img_len = image_features[cur_image_idx_for_span].shape[0]
                    end = start + img_len

                    spans["image"].append((start, end))

                    cur_pos = end
                    cur_input_ids_tmp = cur_input_ids_tmp[token_idx + 1:]
                    cur_image_idx_for_span += 1

                if len(spans["image"]) > 0:
                    spans["image"] = (spans["image"][0][0], spans["image"][-1][1])
                else:
                    spans["image"] = None
                # -------- dev / peri / phys span（token id 级，不包含 marker） --------
                input_id_list = cur_input_ids_raw.tolist()

                for key, pat in MARKER_ID_PATTERNS.items():
                    start_pos = self.find_subseq_ids(input_id_list, pat["start"])
                    end_pos   = self.find_subseq_ids(input_id_list, pat["end"])

                    if start_pos is None or end_pos is None:
                        spans[key] = None
                        continue

                    start_marker_end = start_pos[1]   # <dev> 结束位置
                    end_marker_start = end_pos[0]     # </dev> 起始位置

                    # 核心原则：只取 marker 中间内容
                    if start_marker_end < end_marker_start:
                        spans[key] = (
                            start_marker_end + total_offset,
                            end_marker_start + total_offset
                        )
                    else:
                        spans[key] = None
                #     # -------- 新增打印逻辑 --------
                #     if spans[key] is not None:
                #         s_idx, e_idx = spans[key]
                #         # 注意：如果 spans 里的索引包含了 total_offset，
                #         # 在对当前 input_id_list 切片时要减去它
                #         local_s = s_idx - total_offset
                #         local_e = e_idx - total_offset
                        
                #         # 提取该 span 对应的 ID 段
                #         span_ids = input_id_list[local_s : local_e]
                #         # 解码成文字（用于肉眼核对）

                #         # span_text = self.tokenizer.decode(span_ids)
                        
                #         print(f"Field: 【{key}】")
                #         print(f"   - Span Range: [{s_idx}, {e_idx})")
                #         print(f"   - Token IDs:  {span_ids}")
                #         # print(f"   - Content:    \"{span_text}\"")
                #         print("-" * 40)
                #     else:
                #         print(f"Field: 【{key}】 - NOT FOUND (or invalid markers)")
                # print("input_id:",cur_input_ids_raw)
                # print(f"\n[ModalSpan][Batch {batch_idx}]")
                # print(f"  input_ids length      : {cur_input_ids_raw.shape[0]}")
                # print(f"  num_images             : {num_images}")
                # print(f"  total_offset           : {total_offset}")
                # print(f"  spans (embed idx)      : {spans}")

                modal_token_spans.append(spans)
            else:
                # 如果不在 training 模式，也要保证 modal_token_spans 与 batch 对齐
                # 可以 append None 或空结构，根据你后续使用方式决定
                modal_token_spans.append({
                    "image": None,
                    "dev": None,
                    "peri": None,
                    "phys": None
                })
        
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape

            if I_mask is not None and D_mask is not None:
                I_mask_align = []
                for cur_I_mask in I_mask:
                    cur_I_mask = torch.cat((cur_I_mask, torch.zeros((max_len - cur_I_mask.shape[0],), dtype=cur_I_mask.dtype, device=cur_I_mask.device)), dim=0)
                    I_mask_align.append(cur_I_mask)
                I_mask = torch.stack(I_mask_align, dim=0)

                D_mask_align = []
                for cur_D_mask in D_mask:
                    cur_D_mask = torch.cat((cur_D_mask, torch.zeros((max_len - cur_D_mask.shape[0],), dtype=cur_D_mask.dtype, device=cur_D_mask.device)), dim=0)
                    D_mask_align.append(cur_D_mask)
                D_mask = torch.stack(D_mask_align, dim=0)
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if I_mask is not None and D_mask is not None:
                I_mask = torch.stack(I_mask, dim=0)
                D_mask = torch.stack(D_mask, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels,modal_token_spans,I_mask, D_mask
    # def prepare_inputs_labels_for_multimodal(
    #     self, input_ids, attention_mask, past_key_values, labels, images
    # ):
    #     vision_tower = self.get_vision_tower()
    #     # 1. 生成原始 Mask (Batch, Seq_Len)
    #     I_mask, D_mask = self.generate_I_D_mask_from_labels(input_ids, labels)
        
    #     if vision_tower is None or images is None or input_ids.shape[1] == 1:
    #         if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
    #             attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            
    #         # [Fix] 即使没有图像，也要保存原始 Mask 以防后续 Loss 调用报错
    #         self.I_mask = I_mask
    #         self.D_mask = D_mask
    #         self.modal_token_spans = None # 或者生成空的 spans
    #         return input_ids, attention_mask, past_key_values, None, labels

    #     if type(images) is list or images.ndim == 5:
    #         concat_images = torch.cat([image for image in images], dim=0)
    #         image_features = self.encode_images(concat_images)
    #         split_sizes = [image.shape[0] for image in images]
    #         image_features = torch.split(image_features, split_sizes, dim=0)
    #         image_features = [x.flatten(0, 1) for x in image_features]
    #     else:
    #         image_features = self.encode_images(images)

    #     new_input_embeds = []
    #     new_labels = [] if labels is not None else None
        
    #     # [Fix] 移到循环外：用于收集处理后的 Mask 和 Spans
    #     new_I_masks = []
    #     new_D_masks = []
    #     modal_token_spans = [] 

    #     cur_image_idx = 0
        
    #     for batch_idx, cur_input_ids in enumerate(input_ids):
    #         # 获取当前样本的原始 mask (Seq_Len,)
    #         cur_I_mask_raw = I_mask[batch_idx]
    #         cur_D_mask_raw = D_mask[batch_idx]
        

    #         if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:

    #             half_len = cur_input_ids.shape[0] // 2
    #             cur_image_features = image_features[cur_image_idx]
    #             cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
    #             cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
    #             cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
    #             new_input_embeds.append(cur_input_embeds)
    #             if labels is not None:
    #                 new_labels.append(labels[batch_idx])
                
    #             # [Fix] 无图像时，直接添加原始 Mask 和空 Span
    #             new_I_masks.append(cur_I_mask_raw)
    #             new_D_masks.append(cur_D_mask_raw)
    #             modal_token_spans.append({}) # 保持列表长度与 Batch 一致
                
    #             cur_image_idx += 1
    #             continue

    #         # --- 有图像样本的处理 ---
    #         image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
    #         cur_new_input_embeds = []
    #         if labels is not None:
    #             cur_labels = labels[batch_idx]
    #             cur_new_labels = []
    #             assert cur_labels.shape == cur_input_ids.shape
            
    #         # ... (原有 Embeddings 拼接逻辑保持不变) ...
    #         while image_token_indices.numel() > 0:
    #             cur_image_features = image_features[cur_image_idx]
    #             image_token_start = image_token_indices[0]
    #             if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
    #                 cur_new_input_embeds.append(cur_image_features)
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
    #                 if labels is not None:
    #                     cur_new_labels.append(cur_labels[:image_token_start])
    #                     cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
    #                     cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
    #                     cur_labels = cur_labels[image_token_start+2:]
    #             else:
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
    #                 cur_new_input_embeds.append(cur_image_features)
    #                 if labels is not None:
    #                     cur_new_labels.append(cur_labels[:image_token_start])
    #                     cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
    #                     cur_labels = cur_labels[image_token_start+1:]
    #             cur_image_idx += 1
    #             if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
    #                 cur_input_ids = cur_input_ids[image_token_start+2:]
    #             else:
    #                 cur_input_ids = cur_input_ids[image_token_start+1:]
    #             image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            
    #         if cur_input_ids.numel() > 0:
    #             if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
    #             else:
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
    #             if labels is not None:
    #                 cur_new_labels.append(cur_labels)
            
    #         cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
    #         cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
    #         new_input_embeds.append(cur_new_input_embeds)
            
    #         if labels is not None:
    #             cur_new_labels = torch.cat(cur_new_labels, dim=0)
    #             new_labels.append(cur_new_labels)

    #         cur_input_ids_raw = input_ids[batch_idx]
            
    #         # [Fix] 使用真实的长度差来计算 offset，比乘法更鲁棒
    #         final_len = cur_new_input_embeds.shape[0]
    #         orig_len = cur_input_ids_raw.shape[0]
    #         total_offset = final_len - orig_len 

    #         # 创建新的 Mask (长度为 final_len)
    #         cur_I_mask_new = torch.zeros((final_len,), dtype=cur_I_mask_raw.dtype, device=cur_I_mask_raw.device)
    #         cur_D_mask_new = torch.zeros((final_len,), dtype=cur_D_mask_raw.dtype, device=cur_D_mask_raw.device)
            
    #         if total_offset >= 0:
    #             cur_I_mask_new[total_offset:] = cur_I_mask_raw[:orig_len] 
    #             cur_D_mask_new[total_offset:] = cur_D_mask_raw[:orig_len]
            
    #         new_I_masks.append(cur_I_mask_new)
    #         new_D_masks.append(cur_D_mask_new)

    #         # ============================================================
    #         # [Refine] Spans 计算逻辑
    #         # ============================================================
    #         spans = {
    #             "image": [],
    #             "dev": None,
    #             "peri": None,
    #             "phys": None
    #         }
            
    #         # -------- image span --------
    #     while (cur_input_ids_tmp == IMAGE_TOKEN_INDEX).any():
    #         image_indices = torch.where(cur_input_ids_tmp == IMAGE_TOKEN_INDEX)[0]
    #         token_idx = image_indices[0].item()

    #         start = cur_pos + token_idx
    #         img_len = image_features[cur_image_idx_for_span].shape[0]
    #         end = start + img_len

    #         spans["image"].append((start, end))

    #         cur_pos = end
    #         cur_input_ids_tmp = cur_input_ids_tmp[token_idx + 1:]
    #         cur_image_idx_for_span += 1

    #     if len(spans["image"]) > 0:
    #         spans["image"] = (spans["image"][0][0], spans["image"][-1][1])
    #     else:
    #         spans["image"] = None

    #         # -------- dev / peri / phys span --------
    #         input_id_list = cur_input_ids_raw.tolist()
    #         for key, pat in MARKER_ID_PATTERNS.items():
    #             start_pos = self.find_subseq_ids(input_id_list, pat["start"])
    #             end_pos   = self.find_subseq_ids(input_id_list, pat["end"])

    #             if start_pos is None or end_pos is None:
    #                 spans[key] = None
    #                 continue

    #             start_marker_end = start_pos[1]
    #             end_marker_start = end_pos[0]

    #             if start_marker_end < end_marker_start:
    #                 spans[key] = (
    #                     start_marker_end + total_offset,
    #                     end_marker_start + total_offset
    #                 )
    #             else:
    #                 spans[key] = None
            
    #         modal_token_spans.append(spans) # [Fix] 现在这是在外层列表 append

    #     # ============================================================
    #     # Padding 和 Stacking
    #     # ============================================================
    #     if any(x.shape[0] != new_input_embeds[0].shape[0] for x in new_input_embeds):
    #         max_len = max(x.shape[0] for x in new_input_embeds)

    #         new_input_embeds_align = []
    #         new_labels_align = []
            
    #         # [Fix] Mask 也需要 Align
    #         new_I_masks_align = []
    #         new_D_masks_align = []

    #         for i in range(len(new_input_embeds)):
    #             # Embeds Padding
    #             cur_embed = new_input_embeds[i]
    #             pad_len = max_len - cur_embed.shape[0]
    #             new_input_embeds_align.append(
    #                 torch.cat((cur_embed, torch.zeros((pad_len, cur_embed.shape[1]), dtype=cur_embed.dtype, device=cur_embed.device)), dim=0)
    #             )

    #             # Labels Padding
    #             if labels is not None:
    #                 cur_label = new_labels[i]
    #                 new_labels_align.append(
    #                     torch.cat((cur_label, torch.full((pad_len,), IGNORE_INDEX, dtype=cur_label.dtype, device=cur_label.device)), dim=0)
    #                 )
                
    #             # [Fix] Masks Padding (补 0)
    #             cur_I = new_I_masks[i]
    #             cur_D = new_D_masks[i]
    #             new_I_masks_align.append(torch.cat((cur_I, torch.zeros((pad_len,), dtype=cur_I.dtype, device=cur_I.device)), dim=0))
    #             new_D_masks_align.append(torch.cat((cur_D, torch.zeros((pad_len,), dtype=cur_D.dtype, device=cur_D.device)), dim=0))

    #         new_input_embeds = torch.stack(new_input_embeds_align, dim=0)
    #         new_labels = torch.stack(new_labels_align, dim=0) if labels is not None else None
            
    #         # [Fix] Stack Masks
    #         final_I_mask = torch.stack(new_I_masks_align, dim=0)
    #         final_D_mask = torch.stack(new_D_masks_align, dim=0)

    #         if attention_mask is not None:
    #             new_attention_mask = []
    #             for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
    #                 new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
    #                 new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
    #                 cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
    #                 new_attention_mask.append(cur_new_attention_mask)
    #             attention_mask = torch.stack(new_attention_mask, dim=0)
    #             assert attention_mask.shape == new_labels.shape
    #     else:
    #         new_input_embeds = torch.stack(new_input_embeds, dim=0)
    #         new_labels = torch.stack(new_labels, dim=0) if labels is not None else None
    #         final_I_mask = torch.stack(new_I_masks, dim=0)
    #         final_D_mask = torch.stack(new_D_masks, dim=0)


    #         if attention_mask is not None:
    #             new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
    #             attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
    #             assert attention_mask.shape == new_input_embeds.shape[:2]

    #     I_mask = final_I_mask
    #     D_mask = final_D_mask

    #     return None, attention_mask, past_key_values, new_input_embeds, new_labels,modal_token_spans,I_mask, D_mask

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
