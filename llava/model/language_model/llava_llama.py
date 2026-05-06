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

from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM



class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 正样本权重
        self.gamma = gamma  # 聚焦系数，越大越关注难例
        self.reduction = reduction
    
    def forward(self, logits, targets):
        BCE = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-BCE)  # 预测概率（针对真实标签）
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg  # 负样本聚焦系数
        self.gamma_pos = gamma_pos  # 正样本聚焦系数
        self.clip = clip  # 负样本概率的硬截断阈值
        self.eps = eps
    
    def forward(self, logits, targets):
        # logits: [B, C], targets: [B, C] binary
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        
        # Asymmetric clipping: 对负样本的概率进行截断 (Shift)
        # 官方 ASL 论文中的做法是将极低概率的负样本直接置 0，防止负样本主导梯度
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg - self.clip).clamp(min=0)
        
        # 正确的 ASL/Focal 权重计算方式 (权重 * log(p))
        # 注意前面加上了负号，让 los_pos 和 los_neg 本身就是正的 Loss 值
        los_pos = - targets * (1.0 - xs_pos)**self.gamma_pos * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = - (1.0 - targets) * (1.0 - xs_neg)**self.gamma_neg * torch.log(xs_neg.clamp(min=self.eps))
        
        loss = los_pos + los_neg  # [B, C]
        
        # 关键修改：在类别维度 (dim=1) 上求平均，返回 [B] 以对齐 valid_mask_f
        return loss.mean(dim=1)
    
def dice_loss(logits, targets, eps=1e-6):
    """
    Multi-label Dice Loss
    logits: [B, C] 未 sigmoid 的 logits
    targets: [B, C] binary labels (0/1)
    返回：[B] 每个样本的 loss
    """
    probs = torch.sigmoid(logits)
    
    # 计算每个标签的 Dice
    intersection = (probs * targets).sum(dim=1)  # [B]
    union = probs.sum(dim=1) + targets.sum(dim=1)  # [B]
    
    dice = (2.0 * intersection + eps) / (union + eps)
    
    # Dice Loss = 1 - Dice
    loss_per_sample = 1.0 - dice
    
    return loss_per_sample  # [B]

class LlavaConfig(LlamaConfig):
    model_type = "llavarad"
    def __init__(
        self,
        num_diseases=9,
        llm_loss_weight=1,         
        findings_loss_weight=0, 
        diagnosis_loss_weight=0,    
        align_loss_weight=0,
        **kwargs                 
    ):
        super().__init__(**kwargs)  
        self.num_diseases = num_diseases
        self.findings_loss_weight = findings_loss_weight
        self.diagnosis_loss_weight = diagnosis_loss_weight
        self.align_loss_weight = align_loss_weight
        self.llm_loss_weight = llm_loss_weight

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        # Initialize weights and apply final processing
        self.post_init()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10)) 
        
        self.num_diseases = config.num_diseases  
        self.findings_loss_weight = config.findings_loss_weight
        self.diagnosis_loss_weight = config.diagnosis_loss_weight
        self.align_loss_weight = config.align_loss_weight
        self.llm_loss_weight = config.llm_loss_weight


        # self.llm_diagnosis_head = nn.Linear(config.hidden_size, self.num_diseases)  # num_diseases=8
       
        # 先验知识 1
        init_weights = torch.tensor([
            [0.2, 1.0, 1.0, 1.2],  # Pneumonia
            [1.0, 0.9, 1.0, 1.5],  # NRDS
            [0.5, 1.0, 0.9, 1.3],  # TTN
            [0.3, 0.3, 1.0, 1.6],  # Pneumothorax
            [1.0, 0.4, 0.8, 1.4],  # BPD
            [0.3, 0.3, 1.0, 1.5],  # Atelectasis
            [0.4, 0.8, 1.0, 1.3],  # Pleural Effusion
            [1.0, 1.0, 1.0, 1.0],  # Normal
        ])

        # ####6
        # init_weights = torch.tensor([
        #     [1.0, 1.0, 1.0, 1.2],  # Pneumonia
        #     [1.0, 0.9, 1.0, 1.5],  # NRDS
        #     [0.5, 1.0, 0.9, 1.3],  # TTN
        #     [0.3, 0.3, 1.0, 1.6],  # Pneumothorax
        #     [1.0, 0.4, 0.8, 1.4],  # BPD
        #     [0.3, 0.3, 1.0, 1.5],  # Atelectasis
        #     [0.4, 0.8, 1.0, 1.3],  # Pleural Effusion
        #     [1.0, 1.0, 1.0, 1.0],  # Normal
        # ])
        # # 先验知识（固定）4
        # init_weights = torch.tensor([
        #     [0.2, 1.0, 1.0, 1.2],  # Pneumonia
        #     [1.0, 0.8, 1.0, 1.5],  # NRDS
        #     [0.5, 1.5, 0.5, 1.5],  # TTN
        #     [0.2, 0.2, 1.0, 1.6],  # Pneumothorax
        #     [1.0, 1.4, 0.8, 1.4],  # BPD
        #     [0.2, 0.2, 1.0, 1.5],  # Atelectasis
        #     [0.4, 0.8, 1.0, 1.5],  # Pleural Effusion
        #     [1.0, 1.0, 1.0, 1.0],  # Normal
        # ])
        # 先验知识（固定）5
        # init_weights = torch.tensor([
        #     [0.2, 1.0, 1.0, 1.2],  # Pneumonia
        #     [1.5, 0.5, 0.5, 1.5],  # NRDS
        #     [0.5, 1.5, 0.5, 1.5],  # TTN
        #     [0.2, 0.2, 1.0, 1.6],  # Pneumothorax
        #     [1.0, 1.4, 0.8, 1.4],  # BPD
        #     [0.2, 0.2, 1.0, 1.5],  # Atelectasis
        #     [0.4, 0.8, 1.0, 1.5],  # Pleural Effusion
        #     [1.0, 1.0, 1.0, 1.0],  # Normal
        # ])

        # 'dev', 'peri', 'phys', 'img'
        #2
        # init_weights = torch.tensor([
        #     [0.5, 1.0, 1.0, 1.5],  # Pneumonia
        #     [1.5, 0.5, 0.5, 1.5],  # NRDS
        #     [1.0, 1.5, 0.5, 1.5],  # TTN
        #     [0.5, 1.0, 1.0, 1.5],  # Pneumothorax
        #     [1.0, 1.5, 1.0, 1.5],  # BPD
        #     [0.5, 0.5, 1.5, 1.5],  # Atelectasis
        #     [0.5, 1.0, 1.5, 1.5],  # Pleural Effusion
        #     [1.0, 1.0, 1.0, 1.0],  # Normal
        # ])
        self.disease_source_weights = nn.Parameter(init_weights.clone())
        self.register_buffer('disease_prior_anchor', init_weights.clone())
        self.asym_loss = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)

        self.disease_classifier = nn.Linear(config.hidden_size, 1)
        self.clinical_head=nn.Linear(config.hidden_size*3, 8)
        self.image_classifier = nn.Linear(config.hidden_size, 8)
        self.focal_criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction='none')

        # self.weight_generator = nn.Sequential(
        #     nn.Linear(config.hidden_size * 3, config.hidden_size // 4),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size // 4, self.num_diseases * 4) 
        # )
        # self.weight_generator.to(torch.bfloat16)
        # self.disease_classifier.to(torch.bfloat16)

        # 共享分类头
        self.first_token_classifier = nn.Linear(
            self.config.hidden_size,
            self.num_diseases
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        


        # # --- consistency heads ---
        # self.img_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # findings
        # self.diagnosis_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # diagnosis


    # def compute_layer_logits(self, hidden_states, modal_token_spans, batch_size):
    #     """
    #     输入某一层的 hidden_states，输出对应的疾病分类 logits [B, 9]
    #     """
    #     batch_stacked_feats = []
    #     for b in range(batch_size):
    #         # 注意：这里需要传入当前层的 hidden_states 
    #         # 假设 safe_span_pool 已经修改为可以接受外部 tensor 
    #         dev_feat  = self.safe_span_pool(b, 'dev',   hidden_states, modal_token_spans)
    #         peri_feat = self.safe_span_pool(b, 'peri',  hidden_states, modal_token_spans)
    #         phys_feat = self.safe_span_pool(b, 'phys',  hidden_states, modal_token_spans)
    #         img_feat  = self.safe_span_pool(b, 'image', hidden_states, modal_token_spans)

    #         batch_stacked_feats.append(torch.stack([dev_feat, peri_feat, phys_feat, img_feat]))

    #     # [B, 4, H]
    #     stacked_features = torch.stack(batch_stacked_feats)
        
    #     stacked_features = F.layer_norm(stacked_features, (stacked_features.shape[-1],))

    #     # 疾病特异性融合 [B, 9, H]
    #     normalized_weights = torch.softmax(self.disease_source_weights, dim=-1)
    #     disease_feats = torch.einsum('bfh,df->bdh', stacked_features, normalized_weights)

    #     # 分类头 [B, 9]
    #     logits = self.disease_classifier(disease_feats).squeeze(-1)
    #     return logits

    def get_model(self):
        return self.model

    # 定义内部提取函数 (直接利用 inputs_embeds 作为 hidden_states)
    def safe_span_pool(self,b, key, hidden_states,modal_token_spans):
        # 获取当前样本的 tensor 形状
        H = hidden_states.shape[-1]
        L_hidden = hidden_states.shape[1]
        device = hidden_states.device
        
        span = modal_token_spans[b].get(key, None)
        
        # 异常处理：如果没有 span 或者 span 越界
        if span is None or span[0] < 0 or span[1] > L_hidden or span[1] <= span[0]:
            # 返回一个全 0 向量，且带有梯度追踪（虽然不会更新模型，但保证计算图完整）
            # 这里的 trick 是：用 hidden_states * 0 来保留梯度连接，比直接 torch.zeros 更好
            dummy = hidden_states[b].sum(dim=0) * 0 
            return dummy
        
        s, e = span
        # Mean Pooling: 把该模态的所有 token 取平均，得到一个 [H] 向量
        return hidden_states[b, s:e].mean(dim=0)
    # def safe_span_all(self,b, key, hidden_states,modal_token_spans):
    #     # 获取当前样本的 tensor 形状
    #     H = hidden_states.shape[-1]
    #     L_hidden = hidden_states.shape[1]
    #     device = hidden_states.device
        
    #     span = modal_token_spans[b].get(key, None)
        
    #     # 异常处理：如果没有 span 或者 span 越界
    #     if span is None or span[0] < 0 or span[1] > L_hidden or span[1] <= span[0]:
    #         # 返回一个全 0 向量，且带有梯度追踪（虽然不会更新模型，但保证计算图完整）
    #         # 这里的 trick 是：用 hidden_states * 0 来保留梯度连接，比直接 torch.zeros 更好
    #         dummy = hidden_states[b].sum(dim=0) * 0 
    #         return dummy
        
        # s, e = span

        # return hidden_states[b, s:e]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        disease_labels=None,
        has_signal=None,
        modal_token_spans=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        global_step: Optional[int] = None,
        total_training_steps: Optional[int] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels, modal_token_spans,I_mask, D_mask = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
            # output_hidden_states=True, # 必须开启
            # return_dict=True           # 建议开启，方便用名字访问
        )
        
        hidden_states = outputs[0]
        llm_logits = self.lm_head(hidden_states)
        B, L_hidden, H = hidden_states.shape
        device = hidden_states.device

        llm_loss = 0.0
        if labels is not None:
            # Shift so that tokens < n predict n
            llm_shift_logits = llm_logits[..., :-1, :].contiguous()
            llm_shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            llm_loss_fct = CrossEntropyLoss()
            llm_shift_logits = llm_shift_logits.view(-1, self.config.vocab_size)
            llm_shift_labels = llm_shift_labels.view(-1)
            # Enable model/pipeline parallelism
            llm_shift_labels = llm_shift_labels.to(llm_shift_logits.device)
            llm_loss = llm_loss_fct(llm_shift_logits, llm_shift_labels)

        # total_loss = self.llm_loss_weight*llm_loss
        total_loss = llm_loss

        if self.training:
            
            ###############Mix版本
            # 1. 初始化 align_loss，确保在任何分支下 total_loss 都能正常加总
            ############### Mix 版本（DDP-safe）
            align_loss = 0.0
            # if self.training and self.align_loss_weight > 0 and has_signal is not None:

            #     # [B] bool
            #     valid_mask = has_signal.bool()
            #     batch_size = inputs_embeds.shape[0]

            #     # =====================================================
            #     # 特征提取阶段 —— 所有样本、所有 rank 一律参与
            #     # =====================================================
            #     batch_stacked_feats = []

            #     for b in range(batch_size):
            #         dev_feat  = self.safe_span_pool(b, 'dev',   inputs_embeds,modal_token_spans)
            #         peri_feat = self.safe_span_pool(b, 'peri',  inputs_embeds,modal_token_spans)
            #         phys_feat = self.safe_span_pool(b, 'phys',  inputs_embeds,modal_token_spans)
            #         img_feat  = self.safe_span_pool(b, 'image', inputs_embeds,modal_token_spans)

            #         batch_stacked_feats.append(
            #             torch.stack([dev_feat, peri_feat, phys_feat, img_feat])
            #         )

            #     # [B, 4, H]
            #     stacked_features = torch.stack(batch_stacked_feats)

            #     # LayerNorm：仍然对所有样本做（DDP-safe）
            #     stacked_features = F.layer_norm(
            #         stacked_features,
            #         (stacked_features.shape[-1],)
            #     )

            #     # =====================================================
            #     # 2️⃣ 疾病特异性融合（仍然是 [B, 8, H]）
            #     # =====================================================
            #     normalized_weights = torch.softmax(
            #         self.disease_source_weights, dim=-1
            #     )  # [9, 4]

            #     disease_feats = torch.einsum(
            #         'bfh,df->bdh',
            #         stacked_features,
            #         normalized_weights
            #     )  # [B, 8, H]

            #     # =====================================================
            #     # 3️⃣ 分类头（仍然是 [B, 8]）
            #     # =====================================================
            #     logits = self.disease_classifier(disease_feats).squeeze(-1)

            #     # self.disease_classifier = nn.Linear(config.hidden_size, 1)

            #     # =====================================================
            #     # 4️⃣ Loss —— 只在这里用 mask（🔥关键）
            #     # =====================================================
            #     # per-sample loss: [B]
            #     loss_per_sample = F.binary_cross_entropy_with_logits(
            #         logits,
            #         disease_labels.float(),
            #         reduction="none"
            #     ).mean(dim=1)

            #     # mask 掉无 disease 的样本
            #     valid_mask_f = valid_mask.float()

            #     num_valid = valid_mask_f.sum()

            #     # ⚠️ DDP-safe 写法
            #     loss_cls = torch.where(
            #         num_valid > 0,
            #         (loss_per_sample * valid_mask_f).sum() / num_valid,
            #         loss_per_sample.sum() * 0.0
            #     )
                
            #     align_loss = loss_cls 
            #     total_loss = total_loss + self.align_loss_weight * align_loss
            # if self.training and self.align_loss_weight > 0 and has_signal is not None:
            #     # [B] bool
            #     valid_mask = has_signal.bool()
            #     batch_size = inputs_embeds.shape[0]
            #     device = inputs_embeds.device

            #     # =====================================================
            #     # 1️⃣ 特征提取与融合
            #     # =====================================================
            #     batch_stacked_feats = []
            #     for b in range(batch_size):
            #         dev_feat  = self.safe_span_pool(b, 'dev',   inputs_embeds, modal_token_spans)
            #         peri_feat = self.safe_span_pool(b, 'peri',  inputs_embeds, modal_token_spans)
            #         phys_feat = self.safe_span_pool(b, 'phys',  inputs_embeds, modal_token_spans)
            #         img_feat  = self.safe_span_pool(b, 'image', inputs_embeds, modal_token_spans)
            #         batch_stacked_feats.append(torch.stack([dev_feat, peri_feat, phys_feat, img_feat]))

            #     # [B, 4, H] -> LayerNorm
            #     stacked_features = torch.stack(batch_stacked_feats)
            #     stacked_features = F.layer_norm(stacked_features, (stacked_features.shape[-1],))

            #     # 疾病特异性融合 [B, 8, H]
            #     normalized_weights = torch.softmax(self.disease_source_weights, dim=-1)
            #     disease_feats = torch.einsum('bfh,df->bdh', stacked_features, normalized_weights)

            #     # =====================================================
            #     # 2️⃣ 分类 Loss (Classification Loss)
            #     # =====================================================
            #     logits = self.disease_classifier(disease_feats).squeeze(-1) # [B, 8]
                
            #     loss_per_sample = F.binary_cross_entropy_with_logits(
            #         logits, disease_labels.float(), reduction="none"
            #     ).mean(dim=1)

            #     valid_mask_f = valid_mask.float()
            #     num_valid = valid_mask_f.sum()

            #     loss_cls = torch.where(
            #         num_valid > 0,
            #         (loss_per_sample * valid_mask_f).sum() / num_valid,
            #         loss_per_sample.sum() * 0.0
            #     )

            #     # =====================================================
            #     # 3️⃣ 语义对齐 Loss (Semantic Alignment Loss)
            #     # =====================================================
            #     loss_sem_h = torch.tensor(0.0, device=device)
                
            #     if D_mask is not None:
            #         # A. 计算视觉端疾病聚合特征 [B, H]
            #         # 只选取 disease_labels 为 1 的特征进行平均
            #         labels_weight = disease_labels.float().unsqueeze(-1)  # [B, 8, 1]
            #         target_feat_sum = (disease_feats * labels_weight).sum(dim=1)
            #         active_disease_count = labels_weight.sum(dim=1).clamp(min=1.0)
            #         target_feat = target_feat_sum / active_disease_count
                    
            #         # B. 计算文本端锚点特征 (Anchor) [B, H]
            #         mask_floats = D_mask.float().unsqueeze(-1)
            #         # 假设 outputs.hidden_states[0] 是 LLM 的最后一层输出
            #         anchor_feat_sum = (outputs.hidden_states[0] * mask_floats).sum(dim=1)
            #         token_count = D_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            #         anchor_feat = (anchor_feat_sum / token_count).detach() # 锚点通常不更新文本端
                    
            #         # C. 计算余弦相似度 Loss
            #         cos_sim = F.cosine_similarity(target_feat, anchor_feat, dim=-1)
            #         loss_sem_per_sample = 1.0 - cos_sim
                    
            #         # D. 健壮的 Mask 聚合
            #         # 语义对齐的前提：1. 样本有信号 (valid_mask) 2. 样本至少有一个阳性疾病标签
            #         has_any_disease = (disease_labels.sum(dim=1) > 0).float()
            #         semantic_valid_mask = valid_mask_f * has_any_disease
            #         num_sem_valid = semantic_valid_mask.sum()
                    
            #         loss_sem_h = torch.where(
            #             num_sem_valid > 0,
            #             (loss_per_sample * semantic_valid_mask).sum() / num_sem_valid,
            #             (target_feat.sum() + anchor_feat.sum()) * 0.0 # 保持梯度链
            #         )

            #     # =====================================================
            #     # 4️⃣ 总对齐 Loss 汇总
            #     # =====================================================
            #     # 你可以根据需要调整两者的比例，目前假设 1:1
            #     align_loss = loss_cls + loss_sem_h
            #     total_loss = total_loss + self.align_loss_weight * align_loss

            #     if global_step is not None and global_step % 10 == 0:
            #         print(
            #             f"[Align Loss] Samples: {int(num_valid.item())}/{batch_size} | "
            #             f"Cls: {loss_cls.item():.4f}"
            #             f"sem: {loss_sem_h.item():.4f}"
            #         )
            # if self.training and self.align_loss_weight > 0:
            #     # 1. 获取目标数据类型 (BFloat16)
            #     target_dtype = hidden_states.dtype 
            #     batch_size = hidden_states.shape[0]
            #     device = hidden_states.device
                
            #     # Mask 处理
            #     safe_has_signal = has_signal.bool() if has_signal is not None else torch.zeros(batch_size, device=device).bool()
            #     valid_mask_f = safe_has_signal.float()
            #     num_valid = valid_mask_f.sum()

            #     # =====================================================
            #     # 1️⃣ 特征提取
            #     # =====================================================
            #     batch_stacked_feats = []
            #     for b in range(batch_size):
            #         feats = [
            #             self.safe_span_pool(b, m, hidden_states, modal_token_spans) 
            #             for m in ['dev', 'peri', 'phys', 'image']
            #         ]
            #         batch_stacked_feats.append(torch.stack(feats)) 

            #     # [B, 4, H]
            #     stacked_features = torch.stack(batch_stacked_feats)
            #     stacked_features = F.layer_norm(stacked_features, (self.config.hidden_size,))

            #     # =====================================================
            #     # 2️⃣ 主干逻辑 (融合与分类)
            #     # =====================================================
            #     context_feat = stacked_features[:, :3, :].flatten(1) # [B, 3*H]
            #     image_feat = stacked_features[:, 3, :]               # [B, H]
            
            #     dynamic_weights = self.disease_source_weights.unsqueeze(0)
            #     normalized_weights = torch.softmax(dynamic_weights, dim=-1)

            #     # 融合特征
            #     disease_feats = torch.einsum('bdf,bfh->bdh', normalized_weights, stacked_features)

            #     # 主分类 Logits
            #     logits_fusion = self.disease_classifier(disease_feats).squeeze(-1) # [B, 8]
            #     logits_clinical = self.clinical_head(context_feat) # [B, 8]
            #     logits_image = self.image_classifier(image_feat)   # [B, 8]
                
            #     if num_valid > 0:
            #         # --- A. 主任务 Loss ---
            #         loss_main_per = F.binary_cross_entropy_with_logits(
            #             logits_fusion.float(), disease_labels.float(), reduction="none"
            #         ).mean(dim=1)
            #         l_main = (loss_main_per * valid_mask_f).sum() / num_valid

            #         # # --- B. 临床锚定纠错 Loss (Anchor Loss) ---
            #         # anchor_weights = self.image_classifier.weight # [8, H]
            #         # img_norm = F.normalize(image_feat.float(), p=2, dim=1)        
            #         # w_norm = F.normalize(anchor_weights.float(), p=2, dim=1)   
                    
            #         # sim_matrix = torch.matmul(img_norm, w_norm.t()) 

            #         # num_labels = disease_labels.sum(dim=1).clamp(min=1)
            #         # pos_sim = (sim_matrix * disease_labels.float()).sum(dim=1) / num_labels
            #         # # 最难负样本
            #         # masked_sim = sim_matrix.masked_fill(disease_labels.bool(), -1e4)
            #         # img_neg_sim, img_neg_indices = masked_sim.max(dim=1)

            #         # # 获取临床分支对【真实疾病】的预测 Logits (取平均)
            #         # clin_pos_logits = (logits_clinical * disease_labels.float()).sum(dim=1) / num_labels
                    
            #         # # clinical_probs = torch.sigmoid(logits_clinical.float())
            #         # # target_clinical_conf = (clinical_probs * disease_labels.float()).sum(dim=1) / num_labels

            #         # clin_margin=1.0

            #         # clin_neg_logits = logits_clinical.gather(1, img_neg_indices.unsqueeze(1)).squeeze(1)
            #         # raw_clin_push_loss = F.relu(clin_neg_logits - clin_pos_logits + clin_margin)
                    
            #         # # 高级技巧：如果影像真的非常混淆 (img_neg_sim 很大)，临床就需要被推得更狠！
            #         # # 我们用影像的混淆程度作为权重 (detach 截断梯度，只作为标量权重)
            #         # image_confusion_weight = torch.exp(img_neg_sim.detach()) 
                    
            #         # l_cross_modal_mining = (raw_clin_push_loss * image_confusion_weight * valid_mask_f).sum() / num_valid

            #         # # 此时的 L_anchor 不再是限制影像，而是跨模态挖掘的 L_cross_modal_mining
            #         # l_anchor = l_cross_modal_mining
                    
            #         # margin = 0.2
            #         # raw_pull_loss = F.relu(neg_sim - pos_sim + margin) * target_clinical_conf
            #         # l_anchor = (raw_pull_loss * valid_mask_f).sum() / num_valid

            #         # --- C. 辅助头训练 Loss ---
            #         loss_aux_c = F.binary_cross_entropy_with_logits(logits_clinical.float(), disease_labels.float(), reduction="none").mean(dim=1)
            #         loss_aux_i = F.binary_cross_entropy_with_logits(logits_image.float(), disease_labels.float(), reduction="none").mean(dim=1)
            #         l_aux = ((loss_aux_c + loss_aux_i) * valid_mask_f).sum() / (2.0 * num_valid)

            #     else:
            #         dummy_zero = (logits_fusion.abs().sum() + logits_clinical.abs().sum() + logits_image.abs().sum()) * 0.0
            #         l_main = dummy_zero
            #         # l_anchor = dummy_zero
            #         l_aux = dummy_zero

            #     # =====================================================
            #     # 4️⃣ 最终加权与打印
            #     # =====================================================
            #     # align_loss = l_main + 0.1 * l_anchor + 1.0 * l_aux
            #     align_loss = l_main + 0.1 * l_aux
                
            #     total_loss = total_loss + (self.align_loss_weight * align_loss).to(total_loss.dtype)

            #     # 监控打印 (每 10 步)
            #     if global_step is not None and global_step % 10 == 0:
            #         print(
            #             f"[Align] Main: {l_main.item():.4f} | "
            #             # f"Anchor: {l_anchor.item():.4f} | "
            #             f"Aux: {l_aux.item():.4f} | Num: {int(num_valid.item())}"
            #         )

            if self.training and self.align_loss_weight > 0:
                # 1. 获取目标数据类型 (BFloat16)
                
                batch_size = hidden_states.shape[0]
                device = hidden_states.device
                
                # Mask 处理
                safe_has_signal = has_signal.bool() if has_signal is not None else torch.zeros(batch_size, device=device).bool()
                valid_mask_f = safe_has_signal.float()
                num_valid = valid_mask_f.sum()

                # 简洁的特征提取写法
                modals = ['dev', 'peri', 'phys', 'image']
                batch_stacked_feats = torch.stack([
                    torch.stack([self.safe_span_pool(b, m, hidden_states, modal_token_spans) for m in modals])
                    for b in range(batch_size)
                ]) 

                # 标准化
                stacked_features = F.layer_norm(batch_stacked_feats, (self.config.hidden_size,))

                # 解包赋值
                dev_feat, peri_feat, phys_feat, image_feat = stacked_features.unbind(dim=1)
            
                dynamic_weights = self.disease_source_weights.unsqueeze(0)
                normalized_weights = torch.softmax(dynamic_weights, dim=-1)

                # 融合特征
                disease_feats = torch.einsum('bdf,bfh->bdh', normalized_weights, stacked_features)

                context_feat = stacked_features[:, :3, :].flatten(1)
                logits_clinc= self.clinical_head(context_feat)

                # logits_dev   = self.clinical_head(dev_feat)   # [B, 8]
                # logits_peri  = self.clinical_head(peri_feat)  # [B, 8]
                # logits_phys  = self.clinical_head(phys_feat)  # [B, 8]
                logits_image = self.image_classifier(image_feat) # [B, 8]
                
                # 主分类 Logits (保持融合逻辑不变)
                logits_fusion = self.disease_classifier(disease_feats).squeeze(-1) # [B, 8]

                #######myloss
                
                if num_valid > 0:
                    # --- A. 主任务 Loss ---
                    loss_main_per = F.binary_cross_entropy_with_logits(
                        logits_fusion.float(), disease_labels.float(), reduction="none"
                    ).mean(dim=1)
                    l_main = (loss_main_per * valid_mask_f).sum() / num_valid

                    # --- C. 辅助头训练 Loss (拆分为 3+1 个) ---
                    # 计算临床三个模态的独立 Loss
                    
                    l_clinic  = F.binary_cross_entropy_with_logits(logits_clinc.float(), disease_labels.float(), reduction="none").mean(dim=1)
                    # l_dev  = F.binary_cross_entropy_with_logits(logits_dev.float(), disease_labels.float(), reduction="none").mean(dim=1)
                    # l_peri = F.binary_cross_entropy_with_logits(logits_peri.float(), disease_labels.float(), reduction="none").mean(dim=1)
                    # l_phys = F.binary_cross_entropy_with_logits(logits_phys.float(), disease_labels.float(), reduction="none").mean(dim=1)
                    
                    # 计算图像模态的 Loss
                    l_image = F.binary_cross_entropy_with_logits(logits_image.float(), disease_labels.float(), reduction="none").mean(dim=1)
                    
                    # 汇总辅助 Loss (这里对 4 个单模态 Loss 取平均)
                    # l_aux = ((l_image + l_clinic) * valid_mask_f).sum() / (2.0 * num_valid)
                    l_aux = (( l_clinic) * valid_mask_f).sum() / (1.0 * num_valid)
                    # l_aux = (( l_image) * valid_mask_f).sum() / (1.0 * num_valid)
                    # l_aux = ((l_dev + l_peri + l_phys + l_image) * valid_mask_f).sum() / (4.0 * num_valid)
                    # l_aux = ((l_dev + l_peri + l_phys + 0.0*l_image) * valid_mask_f).sum() / (3.0 * num_valid)
                    # l_aux = ((0.0*l_dev + 0.0*l_peri + 0.0*l_phys + l_image) * valid_mask_f).sum() / (1.0 * num_valid)

                else:
                    # # 更新 dummy_zero 包含所有新定义的 logits
                    # combined_logits = logits_fusion.abs().sum() + logits_dev.abs().sum() + \
                    #                   logits_peri.abs().sum() + logits_phys.abs().sum() + \
                    #                   logits_image.abs().sum()

                    combined_logits = logits_fusion.abs().sum() + logits_clinc.abs().sum() + \
                                      logits_image.abs().sum()
                    
                    dummy_zero = combined_logits * 0.0
                    l_main = dummy_zero
                    l_aux = dummy_zero

                align_loss = l_main + 0.1*l_aux

                # ##############dice#########
                # if num_valid > 0:
                #     # --- A. 主任务 Loss (Dice) ---
                #     loss_main_per = dice_loss(logits_fusion, disease_labels)  # [B]
                #     l_main = (loss_main_per * valid_mask_f).sum() / num_valid

                #     # --- C. 辅助头训练 Loss (拆分为 3+1 个，Dice) ---
                #     l_dev  = dice_loss(logits_dev, disease_labels)   # [B]
                #     l_peri = dice_loss(logits_peri, disease_labels)  # [B]
                #     l_phys = dice_loss(logits_phys, disease_labels)  # [B]

                #     # 计算图像模态的 Loss
                #     l_image = dice_loss(logits_image, disease_labels)  # [B]
                    
                #     # 如果需要加权求和（根据你原有的逻辑）
                #     l_aux = (l_dev + l_peri + l_phys + l_image) / 4.0
                #     l_aux = (l_aux * valid_mask_f).sum() / num_valid

                # else:
                #     # 更新 dummy_zero 包含所有新定义的 logits
                #     combined_logits = logits_fusion.abs().sum() + logits_dev.abs().sum() + \
                #                       logits_peri.abs().sum() + logits_phys.abs().sum() + \
                #                       logits_image.abs().sum()
                #     dummy_zero = combined_logits * 0.0
                #     l_main = dummy_zero
                #     l_aux = dummy_zero
                
                #####################3
                # if num_valid > 0:
                #     # --- A. 主任务 Loss (Asymmetric Loss) ---
                #     loss_main_per = self.asym_loss(logits_fusion, disease_labels.float())  # [B]
                #     l_main = (loss_main_per * valid_mask_f).sum() / num_valid

                #     # --- C. 辅助头训练 Loss (拆分为 3+1 个，Asymmetric Loss) ---
                #     l_dev  = self.asym_loss(logits_dev, disease_labels.float())   # [B]
                #     l_peri = self.asym_loss(logits_peri, disease_labels.float())  # [B]
                #     l_phys = self.asym_loss(logits_phys, disease_labels.float())  # [B]

                #     # 计算图像模态的 Loss
                #     l_image = self.asym_loss(logits_image, disease_labels.float())  # [B]
                    
                #     # 辅助头加权求和，然后应用 valid_mask
                #     l_aux_per = (l_dev + l_peri + l_phys + l_image) / 4.0
                #     l_aux = (l_aux_per * valid_mask_f).sum() / num_valid

                # else:
                #     # 更新 dummy_zero 包含所有新定义的 logits，保持分布式训练不卡死
                #     combined_logits = logits_fusion.abs().sum() + logits_dev.abs().sum() + \
                #                     logits_peri.abs().sum() + logits_phys.abs().sum() + \
                #                     logits_image.abs().sum()
                #     dummy_zero = combined_logits * 0.0
                #     l_main = dummy_zero
                #     l_aux = dummy_zero

                # if num_valid > 0:
                #     mask = valid_mask_f.view(disease_labels.size(0), -1) 

                #     # --- A. 主任务 Loss ---
                #     loss_main_per = self.focal_criterion(logits_fusion, disease_labels)  # 假设返回 [B, C]
                #     # 使用 mask 过滤并求平均
                #     l_main = (loss_main_per * mask).sum() / num_valid

                #     # --- C. 辅助头训练 Loss ---
                #     l_dev_per  = self.focal_criterion(logits_dev, disease_labels)
                #     l_peri_per = self.focal_criterion(logits_peri, disease_labels)
                #     l_phys_per = self.focal_criterion(logits_phys, disease_labels)
                #     l_image_per = self.focal_criterion(logits_image, disease_labels)

                #     l_aux_per = (l_dev_per + l_peri_per + l_phys_per + l_image_per) / 4.0
                #     l_aux = (l_aux_per * mask).sum() / num_valid
                
                # else:
                #     # 更新 dummy_zero 包含所有新定义的 logits
                #     combined_logits = logits_fusion.abs().sum() + logits_dev.abs().sum() + \
                #                       logits_peri.abs().sum() + logits_phys.abs().sum() + \
                #                       logits_image.abs().sum()
                #     dummy_zero = combined_logits * 0.0
                #     l_main = dummy_zero
                #     l_aux = dummy_zero
                    
                #     # 总 Loss
                ###############ALSloss#########
                
                total_loss = total_loss + (self.align_loss_weight * align_loss).to(total_loss.dtype)

                # 监控打印 (每 10 步)
                if global_step is not None and global_step % 10 == 0:
                    print(
                        f"[Align] Main: {l_main.item():.4f} | "
                        # f"Anchor: {l_anchor.item():.4f} | "
                        f"Aux: {l_aux.item():.4f} | Num: {int(num_valid.item())}"
                    )

################中级涨点 但故事不好
            if self.training and self.diagnosis_loss_weight > 0:
                # 1. 预准备：确保所有 Rank 都拥有基础变量
                batch_size = hidden_states.shape[0]
                device = hidden_states.device
                
                # 即使 D_mask 或 has_signal 为 None，也要统一处理成全 0 或全 False
                safe_has_signal = has_signal.bool() if has_signal is not None else torch.zeros(batch_size, device=device).bool()
                valid_mask_f = safe_has_signal.float()
                num_valid = valid_mask_f.sum()
                
                batch_stacked_feats_h = []
                for b in range(batch_size):
                    # 提取特征
                    dev_feat_h = self.safe_span_pool(b, 'dev', hidden_states, modal_token_spans)
                    peri_feat_h = self.safe_span_pool(b, 'peri', hidden_states, modal_token_spans)
                    phys_feat_h = self.safe_span_pool(b, 'phys', hidden_states, modal_token_spans)
                    img_feat_h = self.safe_span_pool(b, 'image', hidden_states, modal_token_spans)
                    batch_stacked_feats_h.append(torch.stack([dev_feat_h, peri_feat_h, phys_feat_h, img_feat_h]))
                
                stacked_features_h = torch.stack(batch_stacked_feats_h)  # [B, 4, H]
                stacked_features_h = F.layer_norm(stacked_features_h, (stacked_features_h.shape[-1],))
                
                # 获取融合权重与疾病特征
                normalized_weights = torch.softmax(self.disease_source_weights, dim=-1)  # [8, 4]
                disease_feats_h = torch.einsum('bfh,df->bdh', stacked_features_h, normalized_weights)  # [B, 8, H]
                
                logits_fusion = self.disease_classifier(disease_feats_h).squeeze(-1)  # [B, 8]
                
                cls_feat = hidden_states[:, 0, :]
                logits_cls = self.first_token_classifier(cls_feat)  # [B, 8]

                
                loss_cls_main = F.binary_cross_entropy_with_logits(
                    logits_cls, disease_labels.float(), reduction="none"
                ).mean(dim=1)

                loss_fusion_aux = F.binary_cross_entropy_with_logits(
                    logits_fusion, disease_labels.float(), reduction="none"
                ).mean(dim=1)
                
                probs_p = torch.sigmoid(logits_cls)     # Global view
                probs_q = torch.sigmoid(logits_fusion)  # Local view
                m = 0.5 * (probs_p + probs_q)           # Average distribution
                
                # 数值稳定常数
                eps = 1e-8

                def compute_binary_kl(p, target):
                    """计算二元分布的 KL 散度: p * log(p/target) + (1-p) * log((1-p)/(1-target))"""
                    # 限制范围防止 log(0)
                    p = torch.clamp(p, eps, 1.0 - eps)
                    target = torch.clamp(target, eps, 1.0 - eps)
                    return p * torch.log(p / target) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - target))

                # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
                kl_pm = compute_binary_kl(probs_p, m)
                kl_qm = compute_binary_kl(probs_q, m)
                js_score = 0.5 * (kl_pm + kl_qm)  # [B, 8]
                
                # 对所有标签取平均，得到每个样本的一致性 Loss
                loss_consistency_per_sample = js_score.mean(dim=1) # [B]

                # =====================================================
                # 4️⃣ 健壮的 Loss 聚合 (处理分布式死锁)
                # =====================================================
                if num_valid > 0:
                    # 聚合有效样本的 Loss
                    l_main = (loss_cls_main * valid_mask_f).sum() / num_valid
                    l_aux = (loss_fusion_aux * valid_mask_f).sum() / num_valid
                    l_consist = (loss_consistency_per_sample * valid_mask_f).sum() / num_valid
                else:
                    # 伪造 0 Loss 以保持计算图完整 (DeepSpeed/DDP 需要)
                    dummy_zero = (logits_cls.sum() + logits_fusion.sum()) * 0.0
                    l_main = dummy_zero
                    l_aux = dummy_zero
                    l_consist = dummy_zero
                
                # alpha, beta, gamma = 1.0, 0.5, 0.1

                alpha, beta, gamma = 1.0, 0.5, 0.1
                
                diagnosis_loss = (alpha * l_main) + (beta * l_consist) + (gamma * l_aux)

                self.temp_captured_cls_feat = cls_feat.detach().cpu()
                self.temp_captured_disease_feats = disease_feats_h.detach().cpu()
                
                total_loss = total_loss + self.diagnosis_loss_weight * diagnosis_loss
                if global_step is not None and global_step % 10 == 0:
                    print(
                        f"Cls: {l_main.item():.4f} | "
                        f"l_consist: {l_consist.item():.4f} | "
                        f"feafusion: {l_aux.item():.4f}"
                    )

            # if self.training and self.diagnosis_loss_weight > 0:
            #     # 1. 预准备：确保所有 Rank 都拥有基础变量
            #     batch_size = hidden_states.shape[0]
            #     device = hidden_states.device
                
            #     # 即使 D_mask 或 has_signal 为 None，也要统一处理成全 0 或全 False
            #     safe_has_signal = has_signal.bool() if has_signal is not None else torch.zeros(batch_size, device=device).bool()
            #     valid_mask_f = safe_has_signal.float()
            #     num_valid = valid_mask_f.sum()
                
            #     batch_stacked_feats_h = []
            #     for b in range(batch_size):
            #         # 提取特征
            #         dev_feat_h = self.safe_span_pool(b, 'dev', hidden_states, modal_token_spans)
            #         peri_feat_h = self.safe_span_pool(b, 'peri', hidden_states, modal_token_spans)
            #         phys_feat_h = self.safe_span_pool(b, 'phys', hidden_states, modal_token_spans)
            #         img_feat_h = self.safe_span_pool(b, 'image', hidden_states, modal_token_spans)
            #         batch_stacked_feats_h.append(torch.stack([dev_feat_h, peri_feat_h, phys_feat_h, img_feat_h]))
                
            #     stacked_features_h = torch.stack(batch_stacked_feats_h)  # [B, 4, H]
            #     stacked_features_h = F.layer_norm(stacked_features_h, (stacked_features_h.shape[-1],))
                
            #     # 获取融合权重与疾病特征
            #     normalized_weights = torch.softmax(self.disease_source_weights, dim=-1)  # [8, 4]
            #     disease_feats_h = torch.einsum('bfh,df->bdh', stacked_features_h, normalized_weights)  # [B, 8, H]
                
            #     logits_fusion = self.disease_classifier(disease_feats_h).squeeze(-1)  # [B, 8]
                
            #     cls_feat = hidden_states[:, 0, :]
            #     logits_cls = self.first_token_classifier(cls_feat)  # [B, 8]


            #     loss_cls_main = self.asym_loss(logits_cls, disease_labels.float())

            #     loss_fusion_aux = self.asym_loss(logits_fusion, disease_labels.float())
            #     # loss_cls_main = dice_loss(
            #     #     logits_cls, disease_labels.float()
            #     # )
            #     # loss_fusion_aux = dice_loss(
            #     #     logits_fusion, disease_labels.float()
            #     # )
                
            #     probs_p = torch.sigmoid(logits_cls)     # Global view
            #     probs_q = torch.sigmoid(logits_fusion)  # Local view
            #     m = 0.5 * (probs_p + probs_q)           # Average distribution
                
            #     # 数值稳定常数
            #     eps = 1e-8

            #     def compute_binary_kl(p, target):
            #         """计算二元分布的 KL 散度: p * log(p/target) + (1-p) * log((1-p)/(1-target))"""
            #         # 限制范围防止 log(0)
            #         p = torch.clamp(p, eps, 1.0 - eps)
            #         target = torch.clamp(target, eps, 1.0 - eps)
            #         return p * torch.log(p / target) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - target))

            #     # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
            #     kl_pm = compute_binary_kl(probs_p, m)
            #     kl_qm = compute_binary_kl(probs_q, m)
            #     js_score = 0.5 * (kl_pm + kl_qm)  # [B, 8]
                
            #     # 对所有标签取平均，得到每个样本的一致性 Loss
            #     loss_consistency_per_sample = js_score.mean(dim=1) # [B]

            #     # =====================================================
            #     # 4️⃣ 健壮的 Loss 聚合 (处理分布式死锁)
            #     # =====================================================
            #     if num_valid > 0:
            #         # 聚合有效样本的 Loss
            #         l_main = (loss_cls_main * valid_mask_f).sum() / num_valid
            #         l_aux = (loss_fusion_aux * valid_mask_f).sum() / num_valid
            #         l_consist = (loss_consistency_per_sample * valid_mask_f).sum() / num_valid
            #     else:
            #         # 伪造 0 Loss 以保持计算图完整 (DeepSpeed/DDP 需要)
            #         dummy_zero = (logits_cls.sum() + logits_fusion.sum()) * 0.0
            #         l_main = dummy_zero
            #         l_aux = dummy_zero
            #         l_consist = dummy_zero
                
            #     alpha, beta, gamma = 1.0, 0.5, 0.1
                
            #     diagnosis_loss = (alpha * l_main) + (beta * l_consist) + (gamma * l_aux)
                
            #     total_loss = total_loss + self.diagnosis_loss_weight * diagnosis_loss
            #     if global_step is not None and global_step % 10 == 0:
            #         print(
            #             f"Cls: {l_main.item():.4f} | "
            #             f"l_consist: {l_consist.item():.4f} | "
            #             f"feafusion: {l_aux.item():.4f}"
            #         )

            #####focal loss

            # if self.training and self.diagnosis_loss_weight > 0:
            #     # 1. 预准备：确保所有 Rank 都拥有基础变量
            #     batch_size = hidden_states.shape[0]
            #     device = hidden_states.device
                
            #     # 即使 D_mask 或 has_signal 为 None，也要统一处理成全 0 或全 False
            #     safe_has_signal = has_signal.bool() if has_signal is not None else torch.zeros(batch_size, device=device).bool()
            #     valid_mask_f = safe_has_signal.float()  # [B]
            #     num_valid = valid_mask_f.sum()
                
            #     batch_stacked_feats_h = []
            #     for b in range(batch_size):
            #         # 提取特征
            #         dev_feat_h = self.safe_span_pool(b, 'dev', hidden_states, modal_token_spans)
            #         peri_feat_h = self.safe_span_pool(b, 'peri', hidden_states, modal_token_spans)
            #         phys_feat_h = self.safe_span_pool(b, 'phys', hidden_states, modal_token_spans)
            #         img_feat_h = self.safe_span_pool(b, 'image', hidden_states, modal_token_spans)
            #         batch_stacked_feats_h.append(torch.stack([dev_feat_h, peri_feat_h, phys_feat_h, img_feat_h]))
                
            #     stacked_features_h = torch.stack(batch_stacked_feats_h)  # [B, 4, H]
            #     stacked_features_h = F.layer_norm(stacked_features_h, (stacked_features_h.shape[-1],))
                
            #     # 获取融合权重与疾病特征
            #     normalized_weights = torch.softmax(self.disease_source_weights, dim=-1)  # [8, 4]
            #     disease_feats_h = torch.einsum('bfh,df->bdh', stacked_features_h, normalized_weights)  # [B, 8, H]
                
            #     logits_fusion = self.disease_classifier(disease_feats_h).squeeze(-1)  # [B, 8]
                
            #     cls_feat = hidden_states[:, 0, :]
            #     logits_cls = self.first_token_classifier(cls_feat)  # [B, 8]

            #     # =====================================================
            #     # 替换点：使用 Focal Loss 替代 BCE
            #     # 注意：self.focal_criterion 必须初始化为 reduction='none'
            #     # =====================================================
            #     loss_cls_main = self.focal_criterion(
            #         logits_cls, disease_labels.float()
            #     ).mean(dim=1)  # 从 [B, 8] 取平均变为 [B]

            #     loss_fusion_aux = self.focal_criterion(
            #         logits_fusion, disease_labels.float()
            #     ).mean(dim=1)  # 从 [B, 8] 取平均变为 [B]
            #     # =====================================================
                
            #     probs_p = torch.sigmoid(logits_cls)     # Global view
            #     probs_q = torch.sigmoid(logits_fusion)  # Local view
            #     m = 0.5 * (probs_p + probs_q)           # Average distribution
                
            #     # 数值稳定常数
            #     eps = 1e-8

            #     def compute_binary_kl(p, target):
            #         """计算二元分布的 KL 散度: p * log(p/target) + (1-p) * log((1-p)/(1-target))"""
            #         # 限制范围防止 log(0)
            #         p = torch.clamp(p, eps, 1.0 - eps)
            #         target = torch.clamp(target, eps, 1.0 - eps)
            #         return p * torch.log(p / target) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - target))

            #     # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
            #     kl_pm = compute_binary_kl(probs_p, m)
            #     kl_qm = compute_binary_kl(probs_q, m)
            #     js_score = 0.5 * (kl_pm + kl_qm)  # [B, 8]
                
            #     # 对所有标签取平均，得到每个样本的一致性 Loss
            #     loss_consistency_per_sample = js_score.mean(dim=1) # [B]

            #     # =====================================================
            #     # 健壮的 Loss 聚合 (处理分布式死锁)
            #     # =====================================================
            #     if num_valid > 0:
            #         # 聚合有效样本的 Loss，因为前面加了 .mean(dim=1)，这里 [B] * [B] 完全合法
            #         l_main = (loss_cls_main * valid_mask_f).sum() / num_valid
            #         l_aux = (loss_fusion_aux * valid_mask_f).sum() / num_valid
            #         l_consist = (loss_consistency_per_sample * valid_mask_f).sum() / num_valid
            #     else:
            #         # 伪造 0 Loss 以保持计算图完整 (DeepSpeed/DDP 需要)
            #         dummy_zero = (logits_cls.sum() + logits_fusion.sum()) * 0.0
            #         l_main = dummy_zero
            #         l_aux = dummy_zero
            #         l_consist = dummy_zero
                
            #     alpha, beta, gamma = 1.0, 0.5, 0.1
                
            #     diagnosis_loss = (alpha * l_main) + (beta * l_consist) + (gamma * l_aux)
                
            #     total_loss = total_loss + self.diagnosis_loss_weight * diagnosis_loss
            #     if global_step is not None and global_step % 10 == 0:
            #         print(
            #             f"Cls: {l_main.item():.4f} | "
            #             f"l_consist: {l_consist.item():.4f} | "
            #             f"feafusion: {l_aux.item():.4f}"
            #         )

            #####################clipstyle nlg涨了一点###########################

            if self.findings_loss_weight > 0 and has_signal is not None:

                batch_size = hidden_states.shape[0]
                device = hidden_states.device

                valid_mask = has_signal.bool()

                if valid_mask.sum() == 0:
                    ica_loss = torch.tensor(
                        0.0, device=device, dtype=hidden_states.dtype
                    )

                else:

                    # =====================================================
                    # image span pooling（safe_span_pool 已经是 mean）
                    # =====================================================
                    img_feats = []

                    for b in range(batch_size):
                        # 返回 [H]
                        img_feat = self.safe_span_pool(
                            b, 'image', hidden_states, modal_token_spans
                        )
                        img_feats.append(img_feat)

                    img_feats = torch.stack(img_feats, dim=0)  # [B, H]

                    # =====================================================
                    # 3️⃣ findings text span pooling（用 I_mask）
                    # =====================================================
                    # inputs_embeds: [B, L, H]
                    text_mask = I_mask.float().unsqueeze(-1)  # [B, L, 1]

                    text_sum = (hidden_states * text_mask).sum(dim=1)
                    text_count = text_mask.sum(dim=1).clamp(min=1.0)

                    text_feats = text_sum / text_count  # [B, H]

                    # =====================================================
                    # 4️⃣ normalize（cosine space）
                    # =====================================================
                    img_feats = F.normalize(img_feats, p=2, dim=-1)
                    text_feats = F.normalize(text_feats, p=2, dim=-1)

                    # =====================================================
                    # 5️⃣ CLIP-style bidirectional contrastive
                    # =====================================================
                    temperature = 0.07
                    text_feats = text_feats.to(img_feats.dtype)

                    logits = torch.matmul(img_feats, text_feats.t()) / temperature  # [B,B]

                    labels = torch.arange(batch_size, device=device)

                    loss_i2t = F.cross_entropy(logits, labels)
                    loss_t2i = F.cross_entropy(logits.t(), labels)

                    ica_loss_all = (loss_i2t + loss_t2i) / 2.0

                    # =====================================================
                    # disease gating（保持你原设计）
                    # =====================================================
                    has_any_disease = (disease_labels.sum(dim=1) > 0).float()
                    final_valid_mask = valid_mask.float() * has_any_disease

                    num_final_valid = final_valid_mask.sum()

                    ica_loss = torch.where(
                        num_final_valid > 0,
                        ica_loss_all * final_valid_mask.mean(),
                        torch.tensor(0.0, device=device, dtype=hidden_states.dtype)
                    )

                # =====================================================
                # 加入 total loss
                # =====================================================
                total_loss = total_loss + self.findings_loss_weight * ica_loss

        if not return_dict:
            output = (llm_logits,) + outputs[1:]
            return (total_loss,) + output if total_loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=llm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llavarad", LlavaConfig) 
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)


# def safe_span_pool(b, key):
#                 span = modal_token_spans[b].get(key, None)
#                 if span is None:
#                     # 创建可微分的零向量
#                     dummy = torch.zeros(H, device=device)
#                     dummy.requires_grad_(True) 
#                     return dummy
#                 s, e = span
#                 if s < 0 or e < 0 or s >= L_hidden or e > L_hidden or e <= s:
#                     # 同样处理异常情况
#                     dummy = torch.zeros(H, device=device)
#                     dummy.requires_grad_(True)
#                     return dummy
#                 return hidden_states[b, s:e].mean(dim=0)
#             def safe_span_pool2(b, key):
#                 span = modal_token_spans[b].get(key, None)
#                 if span is None:
#                     # 创建可微分的零向量
#                     dummy = torch.zeros(H, device=device)
#                     dummy.requires_grad_(True) 
#                     return dummy
#                 s, e = span
#                 if s < 0 or e < 0 or s >= L_hidden or e > L_hidden or e <= s:
#                     # 同样处理异常情况
#                     dummy = torch.zeros(H, device=device)
#                     dummy.requires_grad_(True)
#                     return dummy
#                 return hidden_states[b, s:e]
            
#             # def safe_span_pool(b, key):
#             #     span = modal_token_spans[b].get(key, None)
#             #     if span is None:
#             #         print(f"[DEBUG] b={b} key={key}: span=None")
#             #         return torch.zeros(H, device=device)
#             #     s, e = span
#             #     if e <= s:
#             #         return torch.zeros(H, device=device)
#             #     return hidden_states[b, s:e].mean(dim=0)

#             dev_feat  = torch.stack([safe_span_pool(b, "dev") for b in range(B)], dim=0)
#             peri_feat = torch.stack([safe_span_pool(b, "peri") for b in range(B)], dim=0)
#             phys_feat = torch.stack([safe_span_pool(b, "phys") for b in range(B)], dim=0)
#             img_feat =torch.stack([safe_span_pool(b, "image") for b in range(B)], dim=0)
#             image_feats_all =torch.stack([safe_span_pool2(b, "image") for b in range(B)], dim=0)  # [B, L_img,1369, H]
#             ############################################# findings_loss2 ##########################################



# ############################################# align_loss ##########################################
#             if self.align_loss_weight > 0:
#                 D_align_loss = torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
                
#                 # 只有当 D_mask 存在且启用了动态对齐权重时才计算
#                 if D_mask is not None:
#                     device = fusion_logits.device
#                     # T = getattr(self.config, 'align_temperature', 2.0)
#                     T = 2.0  # 你固定用 2.0
#                     B = fusion_logits.shape[0]
                    
#                     # =========================================================
#                     # (1) 计算 correctness 权重 [B]
#                     # =========================================================
#                     with torch.no_grad():
#                         probs = torch.sigmoid(fusion_logits)                    # [B, 8]
#                         targets = disease_labels.to(device).float()                # [B, 8]
#                         per_sample_bce = F.binary_cross_entropy(
#                             probs, targets, reduction='none'
#                         ).mean(dim=-1)                                             # [B]
#                         correctness = torch.clamp(1.0 - per_sample_bce, min=0.0)   # [B]

#                     # =========================================================
#                     # (2) 准备 LLM 的 hidden states 和 mask
#                     # =========================================================
#                     shift_hidden = hidden_states[..., :-1, :].contiguous()        # [B, L-1, H]
#                     shift_D_mask = D_mask[..., 1:].bool()                         # [B, L-1]

#                     total_kl = 0.0
#                     total_weight = 0.0

#                     # =========================================================
#                     # (3) 逐样本计算 KL 对齐损失
#                     # =========================================================
#                     for b in range(B):
#                         mask = shift_D_mask[b]  # [L-1]
#                         # 跳过无诊断文本或预测极差的样本
#                         if not mask.any() or correctness[b] <= 1e-4:
#                             continue

#                         # 提取 diagnosis 区域的 hidden states
#                         diag_hiddens = shift_hidden[b][mask]                      # [N, H]
#                         llm_disease_logits = self.llm_diagnosis_head(diag_hiddens) # [N, 8]
#                         llm_disease_logits_mean = llm_disease_logits.mean(dim=0)  # [8]

#                         # 专家分布（来自多源融合诊断分支），带温度软化
#                         with torch.no_grad():
#                             expert_probs = torch.sigmoid(fusion_logits[b] / T)  # [8]

#                         # LLM 的预测分布（log-probs，带温度）
#                         llm_log_probs = F.logsigmoid(llm_disease_logits_mean / T) # [8]

#                         # KL(p || q) = sum(p * (log p - log q))
#                         kl = F.kl_div(
#                             llm_log_probs,       # log_q
#                             expert_probs,        # p
#                             reduction='sum'      # sum over 8 diseases
#                         )

#                         # 加权累加
#                         weight = correctness[b]
#                         total_kl += weight * kl
#                         total_weight += weight

#                     # =========================================================
#                     # (4) 归一化并补偿温度缩放
#                     # =========================================================
#                     if total_weight > 0:
#                         D_align_loss = (total_kl / total_weight) * (T ** 2)
#                     else:
#                         D_align_loss = torch.tensor(0.0, device=device, requires_grad=True)

#                 # =========================================================
#                 # (5) 计算 I_align_loss（你原有的影像结论对齐）
#                 # =========================================================
#                 # shift_logits_llm = llm_logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
#                 # align_mask = I_shift_mask * (shift_labels != -100).float()
                
#                 # valid_logits_llm = shift_logits_llm[align_mask.bool()]
#                 # valid_logits_img = shift_logits[align_mask.bool()]
                
#                 # T_img = getattr(self.config, 'align_temperature', 1.0)  # 注意：这里你原用 T=1.0
#                 # p_expert = torch.nn.functional.softmax(valid_logits_img.detach() / T_img, dim=-1)
#                 # log_q_llm = torch.nn.functional.log_softmax(valid_logits_llm / T_img, dim=-1)
                
#                 # I_align_loss = torch.nn.functional.kl_div(
#                 #     log_q_llm, p_expert, reduction='batchmean'
#                 # ) * (T_img ** 2)

#                 # =========================================================
#                 # (6) 合并总对齐损失
#                 # =========================================================
#                 # align_loss = I_align_loss + D_align_loss
#                 total_loss = total_loss + self.align_loss_weight * D_align_loss


            # # 当前训练进度 [0, 1]
            #     if global_step is not None and total_training_steps is not None:
            #         progress = global_step / total_training_steps
            #         # print("decay_weight start!")
            #     else:
            #         progress = 0.0  # fallback：等价于 full weight
                    
            #     tau = 0.03

            #     # soft decay（指数衰减）
            #     decay_weight = torch.exp(
            #         - torch.tensor(progress / tau, device=hidden_states.device)
            #     )
            #     # 防止数值太小（可选）
            #     decay_weight = torch.clamp(decay_weight, min=0.0, max=1.0)
            #     # =========================================================
            #     # 0. 准备工作：特征提取
                # =========================================================


                #    ############################################# diagnosis_loss ##########################################
#             diagnosis_loss = 0.0
#             if self.diagnosis_loss_weight > 0:
#                 cls_feat = hidden_states[:, 0, :]  # [B, H]
                
#                 stacked_features = torch.stack([dev_feat, peri_feat, phys_feat, img_feat], dim=1)  # [B, 4, H]
                
#                 # 获取可学习权重 [9, 4]
#                 # learnable_weights = self.disease_source_weights.weight  # [9, 4]
#                 normalized_weights = torch.softmax(self.disease_source_weights, dim=-1)  # [9, 4]
#                 # print(normalized_weights)
#                 # print("Requires grad:", self.disease_source_weights.requires_grad)
                
#                 # 加权融合: [B,4,H] * [9,4] -> [B,9,H]
#                 disease_specific_feats = torch.einsum('bfh,df->bdh', stacked_features, normalized_weights)
                
#                 B, D, H = disease_specific_feats.shape
#                 feats_flat = disease_specific_feats.reshape(-1, H)  # [B*D, H]
#                 logits_flat = self.disease_classifier(feats_flat)   # [B*D, 1]
#                 fusion_logits = logits_flat.reshape(B, D)           # [B, 9]
                
#                 # 首 token 也需要为每个疾病分类
#                 first_token_logits = self.first_token_classifier(cls_feat)
            
#                 # 损失计算
#                 target_labels = disease_labels.float()  # [B, 9]
#                 bce_fusion = F.binary_cross_entropy_with_logits(fusion_logits, target_labels)
#                 bce_first = F.binary_cross_entropy_with_logits(first_token_logits, target_labels)

#                 diagnosis_loss = 0*bce_fusion + bce_first
#                 total_loss = total_loss + self.diagnosis_loss_weight * diagnosis_loss

#                 print(f"BCE<s> Loss: {bce_first.item():.4f},BCE Loss: {bce_fusion.item():.4f}")




 # # =================================================================
            # # [核心模块] 知识蒸馏 (Distill) 与 语义分类 (Classify)
            # # =================================================================
            # distill_loss = 0.0
            # cls_loss = 0.0
            # if self.diagnosis_loss_weight > 0 :
                
            #     # --- 1. 获取 Student (Decoder 最后一层的 CLS Token) ---
            #     # 这是模型当前的"自主理解"
            #     # hidden_states: [B, Seq_Len, H]
            #     student_feat = hidden_states[:, 0, :]  # [B, H]
                
            #     # --- 2. 构建 Teacher (基于 Label 的完美特征) ---
            #     # 这是基于专家先验构建的"标准答案"
                
            #     # A. 准备专家权重 [8, 4] (假设 label 是 8 维)
            #     # softmax 确保每种病的 4 个模态权重和为 1
            #     expert_weights = torch.softmax(self.disease_source_weights, dim=-1)
                
            #     # B. 计算样本级权重 [B, 4]
            #     # disease_labels: [B, 8]
            #     # 直接通过矩阵乘法，根据 Label 自动组合出该样本应有的模态关注度
            #     # 既然没有全 0 的情况，这里不需要任何 mask 或填充
            #     weight_dtype = expert_weights.dtype

            #     # 将 disease_labels 转为相同 dtype
            #     sample_modality_weights = torch.matmul(
            #         disease_labels.to(weight_dtype),  # 不用 .float()，改用 .to(dtype)
            #         expert_weights
            #     )
            #     # sample_modality_weights = torch.matmul(disease_labels.float(), expert_weights)
                
            #     # C. 归一化 (处理多标签)
            #     # 如果一个样本同时有两个标签 (e.g. 肺炎+呼吸窘迫)，权重相加会 >1
            #     # 除以标签数量，保持特征的数值稳定性
            #     label_counts = disease_labels.sum(dim=1, keepdim=True)
            #     final_weights = sample_modality_weights / label_counts # [B, 4]
            #     # 容器：[B, 4, H]
            #     batch_stacked_feats = []
            #     batch_size = hidden_states.shape[0]
                
            #     for b in range(batch_size):
            #         # 依次提取 4 个模态 (请确保你的 span key 和这里一致)
            #         # 假设 span key 是: 'dev', 'peri', 'phys', 'img'
            #         dev_feat = safe_span_pool(b, 'dev', hidden_states)
            #         peri_feat = safe_span_pool(b, 'peri', hidden_states)
            #         phys_feat = safe_span_pool(b, 'phys', hidden_states)
            #         img_feat = safe_span_pool(b, 'image', hidden_states) 

            #         # 堆叠单样本特征 [4, H]
            #         batch_stacked_feats.append(torch.stack([dev_feat, peri_feat, phys_feat, img_feat]))

            #     # 转为 Batch Tensor: [B, 4, H]
            #     stacked_features = torch.stack(batch_stacked_feats)
                
            #     # --- 关键步骤 1: LayerNorm (特征对齐) ---
            #     # 强迫 Image 和 Text 在同一个数值量级上竞争
            #     stacked_features = F.layer_norm(stacked_features, stacked_features.shape[1:])
                
            #     # D. 合成 Teacher 特征
            #     # stacked_features: [B, 4, H] (来自 Encoder)
            #     # 加权融合 -> [B, H]
            #     teacher_feat = (stacked_features * final_weights.unsqueeze(-1)).sum(dim=1)
                
            #     # [关键] 切断梯度! 
            #     # 老师是标杆，不能动。我们要训练的是 Student (CLS)
            #     teacher_feat = teacher_feat.detach()
                
            #     # --- 3. 计算 Loss A: 特征对齐 (MSE) ---
            #     # 强迫 CLS Token 的语义向量，去逼近 Teacher 的融合向量
            #     # 作用：修正 Attention 机制，让模型学会"该看哪里"
            #     distill_loss = F.mse_loss(student_feat, teacher_feat)
                
            #     # --- 4. 计算 Loss B: 语义分类 (BCE) ---
            #     # 强迫 CLS Token 包含足够的信息来预测 Label
            #     # 作用：保证特征包含具体的病理语义
            #     # self.first_token_classifier 需在 init 中定义: nn.Linear(H, 8)
            #     cls_logits = self.first_token_classifier(student_feat) # [B, 8]
            #     cls_loss = F.binary_cross_entropy_with_logits(cls_logits, disease_labels.float())
            #     diagnosis_loss=cls_loss + 2.0 * distill_loss
            #     total_loss = total_loss+self.diagnosis_loss_weight *diagnosis_loss



            # # =================================================================
            # # [Step 2] 诊断一致性 Loss (diagnosis_loss Loss) - 8维 Label 直接引导
            # # =================================================================
            # # 只要有 D_mask 且在训练时
            # if self.diagnosis_loss_weight > 0 and D_mask is not None and has_signal is not None:
            #     # [B] bool
            #     valid_mask = has_signal.bool()
            #     batch_size = hidden_states.shape[0]

            #     # =====================================================
            #     # 1️⃣ 特征提取与融合 (基于 hidden_states)
            #     # =====================================================
            #     batch_stacked_feats_h = []
            #     for b in range(batch_size):
            #         dev_feat_h = safe_span_pool(b, 'dev', hidden_states)
            #         peri_feat_h = safe_span_pool(b, 'peri', hidden_states)
            #         phys_feat_h = safe_span_pool(b, 'phys', hidden_states)
            #         img_feat_h = safe_span_pool(b, 'image', hidden_states) 
            #         batch_stacked_feats_h.append(torch.stack([dev_feat_h, peri_feat_h, phys_feat_h, img_feat_h]))
    
            #     stacked_features_h = torch.stack(batch_stacked_feats_h) # [B, 4, H]
            #     stacked_features_h = F.layer_norm(stacked_features_h, (stacked_features_h.shape[-1],))
                
            #     # 获取融合权重并生成疾病预测特征
            #     normalized_weights = torch.softmax(self.disease_source_weights, dim=-1) # [8, 4]
            #     disease_feats_h = torch.einsum('bfh,df->bdh', stacked_features_h, normalized_weights) # [B, 8, H]
                
            #     # =====================================================
            #     # 2️⃣ 分类预测与 Loss (带 valid_mask 约束)
            #     # =====================================================
            #     logits_h = self.disease_classifier(disease_feats_h).squeeze(-1) # [B, 8]
                
            #     loss_cls_per_sample = F.binary_cross_entropy_with_logits(
            #         logits_h, disease_labels.float(), reduction="none"
            #     ).mean(dim=1)

            #     valid_mask_f = valid_mask.float()
            #     num_valid = valid_mask_f.sum()

            #     loss_cls_h = torch.where(
            #         num_valid > 0,
            #         (loss_cls_per_sample * valid_mask_f).sum() / num_valid,
            #         loss_cls_per_sample.sum() * 0.0
            #     )

            #     # =====================================================
            #     # 语义对齐 Loss (修正版)
            #     # =====================================================
                
            #     # A. 预测端特征 (保持不变)
            #     labels_weight = disease_labels.float().unsqueeze(-1)
            #     target_feat_sum = (disease_feats_h * labels_weight).sum(dim=1) 
            #     active_disease_count = labels_weight.sum(dim=1).clamp(min=1.0)
            #     target_feat = target_feat_sum / active_disease_count 

            #     # B. 文本端锚点 (Anchor Feature) -> 【这里改了！】
            #     # 错误做法: 使用 inputs_embeds (静态向量)
            #     # anchor_feat_sum = (inputs_embeds * mask_floats).sum(dim=1)
                
            #     # 正确做法: 使用 hidden_states (上下文语义)
            #     # 我们希望融合后的特征，能逼近模型在这个语境下对“肺炎”的理解
            #     mask_floats = D_mask.float().unsqueeze(-1)
                
            #     # 直接复用 Section 1 里用的那个 hidden_states 变量
            #     # 建议加上 .detach()，防止语义 Loss 破坏语言模型的原有知识
            #     anchor_feat_sum = (hidden_states * mask_floats).sum(dim=1)
                
            #     token_count = D_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            #     anchor_feat = anchor_feat_sum / token_count # [B, H]
                
            #     # 【建议】加上 detach
            #     # 逻辑：让“预测头”去凑“文本”，而不是为了凑相似度把“文本理解”给带偏了
            #     anchor_feat = anchor_feat.detach()

            #     # C. 计算余弦相似度
            #     cos_sim = F.cosine_similarity(target_feat, anchor_feat, dim=-1)
            #     loss_sem_per_sample = 1.0 - cos_sim

            #     # D. 【关键约束】双重掩码过滤
            #     # 1. 样本必须在 valid_mask 中 (有信号)
            #     # 2. 样本必须至少有一种疾病标签 (disease_labels.sum > 0)
            #     has_any_disease = (disease_labels.sum(dim=1) > 0).float()
            #     semantic_valid_mask = valid_mask_f * has_any_disease
                
            #     num_sem_valid = semantic_valid_mask.sum()

            #     # DDP-safe 的均值计算
            #     loss_sem_h = torch.where(
            #         num_sem_valid > 0,
            #         (loss_sem_per_sample * semantic_valid_mask).sum() / num_sem_valid,
            #         loss_sem_per_sample.sum() * 0.0
            #     )

            #     # =====================================================
            #     # 最终合并
            #     # =====================================================
            #     diagnosis_loss = loss_cls_h + loss_sem_h
            #     total_loss = total_loss + self.diagnosis_loss_weight * diagnosis_loss

            #     if global_step is not None and global_step % 10 == 0:
            #         print(
            #             f"[Diagnosis Loss] Valid Samples: {int(num_sem_valid.item())} | "
            #             f"Cls: {loss_cls_h.item():.4f} | Sem: {loss_sem_h.item():.4f}"
            #         )
            # if self.diagnosis_loss_weight > 0 and D_mask is not None:
            #     batch_size = hidden_states.shape[0]

            #     batch_stacked_feats_h=[]
            #     for b in range(batch_size):
            #         dev_feat_h = safe_span_pool(b, 'dev', hidden_states)
            #         peri_feat_h = safe_span_pool(b, 'peri', hidden_states)
            #         phys_feat_h = safe_span_pool(b, 'phys', hidden_states)
            #         img_feat_h = safe_span_pool(b, 'image', hidden_states) 
            #         # 堆叠单样本特征 [4, H]
            #         batch_stacked_feats_h.append(torch.stack([dev_feat_h, peri_feat_h, phys_feat_h, img_feat_h]))
    
            #     # 最后一层监督
            #     stacked_features_h = torch.stack(batch_stacked_feats_h)
                
            #     stacked_features_h = F.layer_norm(stacked_features_h, stacked_features_h.shape[1:])
                
            #     # --- 关键步骤 2: 获取先验权重 ---
            #     # self.disease_source_weights shape: [8, 4]
            #     normalized_weights = torch.softmax(self.disease_source_weights, dim=-1)
                
            #     # --- 关键步骤 3: 加权融合 ---
            #     # Einsum: [B, 4, H] x [8, 4] -> [B, 8, H]
            #     # 含义: Batch x (Dev/Peri/Phys/Img) x Hidden -> Batch x (9种疾病) x Hidden
            #     disease_feats_h = torch.einsum('bfh,df->bdh', stacked_features_h, normalized_weights)
                
            #     # --- 关键步骤 4: 分类预测 ---
            #     # [B, 8, H] -> [B, 8, 1] -> [B, 8]
            #     logits_h = self.disease_classifier(disease_feats_h).squeeze(-1)
                
            #     # --- 关键步骤 5: 计算 Loss ---
            #     # A. 诊断准确性 Loss
            #     loss_cls_h = F.binary_cross_entropy_with_logits(logits_h, disease_labels.float())
                
            #     # 1. 准备权重
            #     # disease_labels: [B, 8] (包含 7种病 + 1种 Normal)
            #     labels_weight = disease_labels.float().unsqueeze(-1) # [B, 8, 1]
                
            #     # 2. 计算 Target Feature (Encoder 端的期望特征)
            #     # 逻辑：根据 Label 选出对应的特征
            #     # case A (肺炎): 选出 feature_0
            #     # case B (正常): 选出 feature_7
            #     # disease_feats_h [B, 8, H] labels_weight:[B, 8, 1]
            #     target_feat_sum = (disease_feats_h * labels_weight).sum(dim=1) # [B, H]
                
            #     # 3. 归一化处理
            #     # 就算只有一个标签是1，sum 也是对的。
            #     # 如果是多标签 (Label有两个1)，除以2取平均，防止模长过大
            #     num_active = labels_weight.sum(dim=1).clamp(min=1.0) 
            #     target_feat = (target_feat_sum / num_active).detach() # [B, H] (切断梯度)
                
            #     # 4. 提取 Decoder 生成的文本特征
            #     # decoder_feat = outputs.hidden_states[-1] # [B, Seq, H]
                
            #     diagnosis_text_feats = []
                
            #     # --- 4. 提取 Decoder 生成的文本特征 (简化版) ---
            #     # decoder_feat: [B, Seq, H], D_mask: [B, Seq]
            #     mask_floats = D_mask.float().unsqueeze(-1)  # 扩展维度变成 [B, Seq, 1] 用于广播

            #     # 计算每个样本 D_mask 区域的平均特征
            #     # 分子：将 Mask 区域外的特征清零并求和 -> [B, H]
            #     # 分母：每个样本 Mask 中 1 的个数 -> [B, 1]
            #     # sum_features = (decoder_feat * mask_floats).sum(dim=1)
            #     sum_features = (inputs_embeds * mask_floats).sum(dim=1)
                
            #     mask_counts = D_mask.sum(dim=1, keepdim=True).clamp(min=1.0) 
            #     pred_feats = sum_features / mask_counts # 得到 [B, H]

            #     # --- 5. 计算 Loss ---
            #     # 因为每个样本都有 mask，所以直接对比全量 target_feat
            #     diagnosis_loss = F.mse_loss(pred_feats, target_feat)

            #     # Debug 打印
            #     print(f"Diagnosis_loss Loss: {diagnosis_loss.item():.4f}")
            #     total_loss = total_loss + self.diagnosis_loss_weight * diagnosis_loss

            #     total_loss = total_loss+self.diagnosis_loss_weight * diagnosis_loss



            
            
            # # findings_loss = 0.0

            # # if self.findings_loss_weight > 0 and D_mask is not None:
            # #     batch_size = hidden_states.shape[0]

            # #     batch_stacked_feats=[]
            # #     for b in range(batch_size):
                   
            # #         img_global = safe_span_all(b, 'image', hidden_states) 
            # #     # [B, T,H]

            # #     img_global_norm = F.normalize(img_global, p=2, dim=-1)

            # #     # [B] Mask 准备
            # #     # 确保 mask 维度匹配: [B, L] -> [B, L, 1]
            # #     mask_expanded = I_mask.unsqueeze(-1).to(hidden_states.dtype)
                
            # #     # 统计 Findings 有效长度
            # #     valid_token_counts = mask_expanded.sum(dim=1)
            # #     valid_token_counts = torch.clamp(valid_token_counts, min=1e-9)

            # #     # 1. 获取 Label 的静态 Embedding (不经过 LLM 层，纯粹的词义)
            # #     safe_labels = labels.clone()
            # #     safe_labels[safe_labels == -100] = 0 
            # #     label_embeds = self.get_model().embed_tokens(safe_labels) # [B, L, H]
                
            # #     # 2. 只取 Findings 部分并池化
            # #     masked_label_feats = label_embeds * mask_expanded
            # #     label_global = masked_label_feats.sum(dim=1) / valid_token_counts
            # #     label_global_norm = F.normalize(label_global, p=2, dim=-1)
               
                
            # #     # 1. 获取 LLM 输出的 Hidden States (包含上下文生成的意图)
            # #     masked_context_feats = hidden_states * mask_expanded
                
            # #     # 2. 池化得到 Findings 部分的生成状态
            # #     context_global = masked_context_feats.sum(dim=1) / valid_token_counts
            # #     context_global_norm = F.normalize(context_global, p=2, dim=-1)

            # #     # 3. 计算 Loss (强制 LLM 的生成状态 接近 Image)
            # #     loss_semantic_anchor = 1 - F.cosine_similarity(img_global_norm, label_global_norm, dim=-1).mean()
            # #     loss_generation_guide = 1 - F.cosine_similarity(img_global_norm, context_global_norm, dim=-1).mean()

            # #     I_llm_loss = 0.0

            # #     if labels is not None:

            # #         if I_mask is not None:
            # #             assert I_mask.shape == labels.shape, "I_mask and labels must have same shape"
    
            # #             I_masked_labels = torch.where(I_mask == 1, labels, -100)
            # #         else:
            # #             I_masked_labels = labels

            # #         llm_shift_logits = llm_logits[..., :-1, :].contiguous()
            # #         I_llm_shift_labels = I_masked_labels[..., 1:].contiguous()  

            # #         I_llm_loss_fct = CrossEntropyLoss()
            # #         llm_shift_logits = llm_shift_logits.view(-1, self.config.vocab_size)
            # #         I_llm_shift_labels = I_llm_shift_labels.view(-1)

            # #         I_llm_shift_labels = I_llm_shift_labels.to(llm_shift_logits.device)
            # #         I_llm_loss = I_llm_loss_fct(llm_shift_logits, I_llm_shift_labels)

            # #     findings_loss = loss_semantic_anchor + loss_generation_guide+ 0*I_llm_loss
                
            # #     print(f"Semantic Loss: {loss_semantic_anchor.item():.4f}, Guide Loss: {loss_generation_guide.item():.4f}")

            # #     total_loss = total_loss + self.findings_loss_weight * findings_loss


            ##########################align loss语义对齐##############

                # # =====================================================
                # # 5️⃣ 语义对齐 Loss (loss_sem) —— 逻辑闭环 + Mask 过滤
                # # =====================================================
                # # 1. 归一化特征
                # norm_disease_feats = F.normalize(disease_feats, p=2, dim=-1)
                # norm_inputs_embeds = F.normalize(inputs_embeds, p=2, dim=-1)

                # # 2. 计算点对点相似度矩阵: [B, 9, S]
                # sim_matrix = torch.einsum('bdh,bsh->bds', norm_disease_feats, norm_inputs_embeds)

                # # 3. 构建联合掩码 (Joint Mask)
                # # 只有 (该样本有效) 且 (该疾病存在) 且 (D_mask 标记的 token) 才计算
                # # valid_mask: [B] -> [B, 1, 1]
                # # disease_labels: [B, 9] -> [B, 9, 1]
                # # D_mask: [B, S] -> [B, 1, S]
                # v_mask = valid_mask.view(-1, 1, 1).float()
                # d_label_mask = disease_labels.unsqueeze(-1).float()
                # token_mask = D_mask.unsqueeze(1).float()

                # joint_mask = v_mask * d_label_mask * token_mask # [B, 9, S]

                # # 4. 计算语义损失
                # loss_sem_all = (1.0 - sim_matrix) * joint_mask
                
                # num_sem_pairs = joint_mask.sum()

                # # ⚠️ DDP-safe 写法：确保 loss_sem 在无有效点时不会变为 NaN
                # loss_sem = torch.where(
                #     num_sem_pairs > 0,
                #     loss_sem_all.sum() / num_sem_pairs,
                #     loss_sem_all.sum() * 0.0
                # )

                # # =====================================================
                # # 最终合并
                # # =====================================================


                ###################################################align_loss################################################### 
            # align_loss=0.0
            # if self.align_loss_weight > 0 and modal_token_spans is not None:
            #     batch_size = inputs_embeds.shape[0]
                
            #     # 容器：[B, 4, H]
            #     batch_stacked_feats = []
                
            #     for b in range(batch_size):
            #         # 依次提取 4 个模态 (请确保你的 span key 和这里一致)
            #         # 假设 span key 是: 'dev', 'peri', 'phys', 'img'
            #         dev_feat = safe_span_pool(b, 'dev', inputs_embeds)
            #         peri_feat = safe_span_pool(b, 'peri', inputs_embeds)
            #         phys_feat = safe_span_pool(b, 'phys', inputs_embeds)
            #         img_feat = safe_span_pool(b, 'image', inputs_embeds) 

            #         # 堆叠单样本特征 [4, H]
            #         batch_stacked_feats.append(torch.stack([dev_feat, peri_feat, phys_feat, img_feat]))

            #     # 转为 Batch Tensor: [B, 4, H]
            #     stacked_features = torch.stack(batch_stacked_feats)
                
            #     # --- 关键步骤 1: LayerNorm (特征对齐) ---
            #     # 强迫 Image 和 Text 在同一个数值量级上竞争
            #     stacked_features = F.layer_norm(stacked_features, stacked_features.shape[1:])
                
            #     # --- 关键步骤 2: 获取先验权重 ---
            #     # self.disease_source_weights shape: [9, 4]
            #     normalized_weights = torch.softmax(self.disease_source_weights, dim=-1)
                
            #     # --- 关键步骤 3: 加权融合 ---
            #     # Einsum: [B, 4, H] x [9, 4] -> [B, 9, H]
            #     # 含义: Batch x (Dev/Peri/Phys/Img) x Hidden -> Batch x (9种疾病) x Hidden
            #     disease_feats = torch.einsum('bfh,df->bdh', stacked_features, normalized_weights)
                
            #     # --- 关键步骤 4: 分类预测 ---
            #     # [B, 9, H] -> [B, 9, 1] -> [B, 9]
            #     logits = self.disease_classifier(disease_feats).squeeze(-1)
                
            #     # --- 关键步骤 5: 计算 Loss ---
            #     # A. 诊断准确性 Loss
            #     loss_cls = F.binary_cross_entropy_with_logits(logits, disease_labels.float())

            #     # B. 先验正则化 Loss (防止模型把权重学坏)
            #     # 强迫学习到的权重不要偏离专家设定太远
            #     target_probs = torch.softmax(self.disease_prior_anchor, dim=-1)
            #     loss_prior = F.mse_loss(normalized_weights, target_probs)
                
            #     # 组合 Loss (建议给 prior 一个较小的系数，比如 0.1 或 0.5)
            #     align_loss = loss_cls + 0.1 * loss_prior
            #     total_loss=total_loss+self.align_loss_weight*align_loss

            #     # Debug 打印 
            #     print(f"Align Loss: {align_loss.item():.4f} (Cls: {loss_cls.item():.4f}, Prior: {loss_prior.item():.4f})")
            ################################多层版##############################3
            # if self.training and self.align_loss_weight > 0 and has_signal is not None:
            #     valid_mask = has_signal.bool()
            #     valid_mask_f = valid_mask.float()
            #     num_valid = valid_mask_f.sum()
            #     batch_size = hidden_states.shape[0]

            #     # 1. 确保 loss 在正确的设备上初始化
            #     loss_cls_total = torch.tensor(0.0, device=hidden_states.device)
                
            #     layers_to_supervise = {
            #         0: outputs.hidden_states[0],
            #         16: outputs.hidden_states[16],
            #     }
            #     layer_weights = {0: 1.0, 16: 1.0}

            #     for idx, layer_hidden in layers_to_supervise.items():
            #         # 执行原版中的特征提取、LayerNorm、Einsum 和 Classifier 逻辑
            #         layer_logits = self.compute_layer_logits(layer_hidden, modal_token_spans, batch_size)
                    
            #         # 计算该层 Loss
            #         loss_per_sample = F.binary_cross_entropy_with_logits(
            #             layer_logits,
            #             disease_labels.float(),
            #             reduction="none"
            #         ).mean(dim=1)
                    
            #         # DDP-safe 平均值计算
            #         layer_loss = torch.where(
            #             num_valid > 0,
            #             (loss_per_sample * valid_mask_f).sum() / num_valid,
            #             loss_per_sample.sum() * 0.0
            #         )
                    
            #         loss_cls_total += layer_weights[idx] * layer_loss

            #     align_loss = loss_cls_total * self.align_loss_weight
            #     total_loss = llm_loss + align_loss

             # if self.diagnosis_loss_weight > 0 and D_mask is not None and has_signal is not None:
            #     valid_mask = has_signal.bool()
            #     batch_size = hidden_states.shape[0]
                
            #     if valid_mask.sum() == 0:  # 没有有效样本，直接跳过
            #         print("[Diagnosis Loss] No valid samples with signal, skipping...")
            #     else:
            #         # =====================================================
            #         # 1️特征提取与融合
            #         # =====================================================
            #         batch_stacked_feats_h = []
            #         for b in range(batch_size):
            #             # 确保每个特征提取不会失败
            #             dev_feat_h = self.safe_span_pool(b, 'dev', hidden_states,modal_token_spans)
            #             peri_feat_h = self.safe_span_pool(b, 'peri', hidden_states,modal_token_spans)
            #             phys_feat_h = self.safe_span_pool(b, 'phys', hidden_states,modal_token_spans)
            #             img_feat_h = self.safe_span_pool(b, 'image', hidden_states,modal_token_spans)
            #             batch_stacked_feats_h.append(torch.stack([dev_feat_h, peri_feat_h, phys_feat_h, img_feat_h]))
                    
            #         stacked_features_h = torch.stack(batch_stacked_feats_h)  # [B, 4, H]
            #         stacked_features_h = F.layer_norm(stacked_features_h, (stacked_features_h.shape[-1],))
                    
            #         # 获取融合权重
            #         normalized_weights = torch.softmax(self.disease_source_weights, dim=-1)  # [8, 4]
            #         disease_feats_h = torch.einsum('bfh,df->bdh', stacked_features_h, normalized_weights)  # [B, 8, H]
                    
            #         # =====================================================
            #         # 2️分类预测 Loss
            #         # =====================================================
            #         logits_h = self.disease_classifier(disease_feats_h).squeeze(-1)  # [B, 8]
                    
            #         # 只对有效样本计算损失
            #         loss_cls_per_sample = F.binary_cross_entropy_with_logits(
            #             logits_h, disease_labels.float(), reduction="none"
            #         ).mean(dim=1)

            #         cls_feat = hidden_states[:, 0, :]          # [B, H]
            #         cls_logits = self.first_token_classifier(cls_feat)  # [B, D]

            #         cls_loss=F.binary_cross_entropy_with_logits(
            #             cls_logits, disease_labels.float(), reduction="none"
            #         ).mean(dim=1)

            #         valid_mask_f = valid_mask.float()
            #         num_valid = valid_mask_f.sum().clamp(min=1.0)  # 防止除零
                    
            #         loss_cls_h = (loss_cls_per_sample * valid_mask_f).sum() / num_valid
            #         cls_loss_m=(cls_loss * valid_mask_f).sum() / num_valid
                    
            #         # =====================================================
            #         # 3️语义对齐 Loss (修复梯度问题)
            #         # =====================================================
            #         # A. 预测端特征 (保留梯度！)
            #         labels_weight = disease_labels.float().unsqueeze(-1)  # [B, 8, 1]
            #         target_feat_sum = (disease_feats_h * labels_weight).sum(dim=1)  # [B, H]
            #         active_disease_count = labels_weight.sum(dim=1).clamp(min=1.0)  # [B, 1]
            #         target_feat = target_feat_sum / active_disease_count  # [B, H]
                    
            #         # B. 文本端锚点特征
            #         mask_floats = D_mask.float().unsqueeze(-1)  # [B, Seq, 1]
                    
            #         # 使用 hidden_states 
            #         anchor_feat_sum = (outputs.hidden_states[0] * mask_floats).sum(dim=1)  # [B, H]
            #         token_count = D_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
            #         anchor_feat = anchor_feat_sum / token_count  # [B, H]
                    
            #         # 可以 detach 锚点特征，但不要 detach target_feat
            #         anchor_feat = anchor_feat.detach()  # 固定文本特征，让预测去匹配
                    
            #         # C. 计算余弦相似度损失
            #         cos_sim = F.cosine_similarity(target_feat, anchor_feat, dim=-1)  # [B]
            #         loss_sem_per_sample = 1.0 - cos_sim
                    
            #         # 双重掩码：有信号 + 有疾病标签
            #         has_any_disease = (disease_labels.sum(dim=1) > 0).float()  # [B]
            #         semantic_valid_mask = valid_mask_f * has_any_disease
            #         num_sem_valid = semantic_valid_mask.sum().clamp(min=1.0)
                    
            #         loss_sem_h = (loss_sem_per_sample * semantic_valid_mask).sum() / num_sem_valid
                    
            #         # =====================================================
            #         # 4️合并损失
            #         # =====================================================
            #         diagnosis_loss = loss_cls_h + cls_loss_m
                    

            #         total_loss = total_loss + self.diagnosis_loss_weight * diagnosis_loss
                    
                    # # 调试信息
                    # if global_step is not None and global_step % 10 == 0:
                    #     print(
                    #         f"[Diagnosis Loss] Step {global_step} | "
                    #         f"Valid: {int(num_valid.item())}/{batch_size} | "
                    #         f"Clsh: {loss_cls_h.item():.4f} | Sem: {loss_sem_h.item():.4f} | Clsm: {cls_loss_m.item():.4f}"
                    #         f"Total: {diagnosis_loss.item():.4f} | "
                    #     )
                        
                    # # 可选：检查梯度
                    # if global_step is not None and global_step % 100 == 0:
                    #     grad_norm = torch.nn.utils.clip_grad_norm_(
                    #         self.disease_classifier.parameters(), max_norm=1.0
                    #     )
                    #     print(f"[Grad Norm] disease_classifier: {grad_norm:.4f}")


         # if self.training and self.align_loss_weight > 0 and has_signal is not None:
            #     # 1. 获取目标数据类型 (BFloat16)
            #     target_dtype = hidden_states.dtype 
            #     batch_size = hidden_states.shape[0]
            #     device = hidden_states.device
                
            #     # 修正1：确保 valid_mask 即使是 None 也是 Tensor
            #     valid_mask = has_signal.bool() if has_signal is not None else torch.ones(batch_size, device=device).bool()
                
            #     # Mask 转 float 用于计算，这会产生 Float32，没关系，计算完转回去即可
            #     valid_mask_f = valid_mask.float()
            #     num_valid = valid_mask_f.sum()

            #     # =====================================================
            #     # 1️⃣ 特征提取与标准化
            #     # =====================================================
            #     batch_stacked_feats = []
            #     for b in range(batch_size):
            #         # 修正2：使用 hidden_states
            #         feats = [
            #             self.safe_span_pool(b, m, hidden_states, modal_token_spans) 
            #             for m in ['dev', 'peri', 'phys', 'image']
            #         ]
            #         batch_stacked_feats.append(torch.stack(feats)) 

            #     # stacked_features: [B, 4, H]
            #     stacked_features = torch.stack(batch_stacked_feats)
            #     stacked_features = F.layer_norm(stacked_features, (self.config.hidden_size,))

            #     # =====================================================
            #     # 2️⃣ 动态残差权重生成
            #     # =====================================================
            #     context_feat = stacked_features[:, :3, :].flatten(1) # [B, 3*H]
                
            #     res_weight = self.weight_generator(context_feat).view(batch_size, 8, 4)
            #     dynamic_weights = self.disease_source_weights.unsqueeze(0) + torch.tanh(res_weight) * 0.2
            #     normalized_weights = torch.softmax(dynamic_weights, dim=-1)

            #     disease_feats = torch.einsum('bdf,bfh->bdh', normalized_weights, stacked_features)

            #     # =====================================================
            #     # 3️⃣ 辅助损失函数 (核心修复区)
            #     # =====================================================
            #     # 临床锚定纠错 Loss (Clinical-Anchored Rectification Loss)
            #     # 我们需要构建一个动态的三角关系来训练影像特征：

            #     # 1. **锚点 (Anchor)**：**真实标签对应的权重向量** ()。这是我们希望影像特征最终去的地方。
            #     # 2. **干扰项 (Negative)**：**影像最容易误判的那个疾病的权重向量** ()。比如对于 NRDS 样本，干扰项通常是“肺炎”的权重。
            #     # 3. **推力 (Force)**：利用**临床信息**作为“置信度开关”。
            #     # * 如果临床信息能准确预测 ，说明临床很确定。
            #     # * 此时，如果影像特征跑到了  旁边，我们就狠狠地把它**拉回** ，并**推开** 。


            #     # =====================================================
            #     # 4️⃣ 分类与总损失
            #     # =====================================================
            #     logits = self.disease_classifier(disease_feats).squeeze(-1) # [B, 8]
                
            #     # BCE 为了数值稳定，通常在 Float32 下计算
            #     loss_per_sample = F.binary_cross_entropy_with_logits(
            #         logits.float(), disease_labels.float(), reduction="none"  # 输入转 float
            #     ).mean(dim=1)

            #     # 聚合后转回 BF16
            #     # # 核心修复：.to(target_dtype)
            #     loss_cls = ((loss_per_sample * valid_mask_f).sum() / (num_valid + 1e-6)).to(target_dtype)

            #     align_loss = loss_cls 
                
            #     # 加到总 Loss 上 (确保类型一致)
            #     total_loss = total_loss + (self.align_loss_weight * align_loss).to(total_loss.dtype)

            # #############这个不行
            # if self.training and self.align_loss_weight > 0 and has_signal is not None:
                
            #     align_loss = torch.tensor(
            #             0.0, device=device, dtype=inputs_embeds.dtype
            #         )
            #     valid_mask = has_signal.bool()
            #     batch_size = inputs_embeds.shape[0]
            #     device = inputs_embeds.device
            #     target_dtype = inputs_embeds.dtype
                
            #     # =====================================================
            #     # 1️ 多模态特征提取 (dev, peri, phys, image)
            #     # =====================================================
            #     batch_stacked_feats = []
            #     for b in range(batch_size):
            #         feats = [
            #             self.safe_span_pool(b, m, inputs_embeds, modal_token_spans) 
            #             for m in ['dev', 'peri', 'phys', 'image']
            #         ]
            #         batch_stacked_feats.append(torch.stack(feats))

            #     # [B, 4, H]
            #     stacked_features = torch.stack(batch_stacked_feats)
            #     stacked_features = F.layer_norm(stacked_features, (stacked_features.shape[-1],))

                # # =====================================================
                # # 2️⃣ 动态权重生成：解决“同影异病”
                # # =====================================================
                # # 决策背景：发育(0)、围产(1)、生理(2)
                # context_feat = stacked_features[:, :3, :].reshape(batch_size, -1)
                # # context_feat = stacked_features[:, :3, :].mean(dim=1) 
                
                # # dynamic_weights: [B, 8, 4]
                # dynamic_weights = self.weight_generator(context_feat).view(batch_size, 8, 4)
                # normalized_weights = torch.softmax(dynamic_weights, dim=-1)

                # # 得到疾病特征：[B, 8, H]
                # disease_feats = torch.einsum('bdf,bfh->bdh', normalized_weights, stacked_features)

            #     # =====================================================
            #     # 3️⃣ 任务一：特征级正交化约束 (Feature Orthogonality)
            #     # =====================================================
            

            #     feats_norm = F.normalize(disease_feats, p=2, dim=-1)
            #     gram_matrices = torch.bmm(feats_norm, feats_norm.transpose(1, 2))
                
            #     # 显式指定单位矩阵为 target_dtype (bf16)
            #     identity = torch.eye(8, device=device, dtype=target_dtype).unsqueeze(0).expand(batch_size, -1, -1)
                
            #     # 计算 MSE (输出可能是 float32，这没关系，后续会处理)
            #     loss_ortho_per_sample = F.mse_loss(gram_matrices, identity, reduction='none').mean(dim=(1, 2))
                
            #     valid_mask_f = valid_mask.to(target_dtype) # bool -> bf16
            #     num_valid = valid_mask_f.sum()
                
            #     # 【改动3】torch.where 的 else 分支必须也是 bf16
            #     loss_feat_ortho = torch.where(
            #         num_valid > 0,
            #         (loss_ortho_per_sample * valid_mask_f).sum() / num_valid,
            #         torch.tensor(0.0, device=device, dtype=target_dtype) 
            #     )

                # # =====================================================
                # # 4️⃣ 任务二：影像霸权惩罚 (保持不变)
                # # =====================================================
                # image_weights = normalized_weights[:, :, 3] # [B, 8]
                # loss_balance = torch.mean(image_weights ** 2)

            #     # =====================================================
            #     # 5️⃣ 分类 Loss
            #     # =====================================================
            #     logits = self.disease_classifier(disease_feats).squeeze(-1) # [B, 8]

            #     loss_per_sample = F.binary_cross_entropy_with_logits(
            #         logits, disease_labels.float(), reduction="none"
            #     ).mean(dim=1)

            #     loss_per_sample = loss_per_sample.to(target_dtype)

            #     loss_cls = torch.where(
            #         num_valid > 0,
            #         (loss_per_sample * valid_mask_f).sum() / num_valid,
            #         torch.tensor(0.0, device=device, dtype=target_dtype)
            #     )

            #     sum_fp32 = (
            #         loss_cls.float() + 
            #         0.5 * loss_feat_ortho.float() + 
            #         loss_balance.float()
            #     )
                
            #     # 2. 乘上权重 (假设权重是 float，这里乘完还是 float32)
            #     align_loss_weighted_fp32 = sum_fp32 * self.align_loss_weight
                
            #     # 3. 【关键】最后一步转回 bf16，加到 total_loss 上
            #     # 这样 DeepSpeed 看到的是: total_loss(bf16) + delta(bf16) -> 合法
            #     total_loss = total_loss + align_loss_weighted_fp32.to(target_dtype)

            #     if global_step is not None and global_step % 10 == 0:
            #         print(
            #             f"[Align Loss] Samples: {int(num_valid.item())}/{batch_size} | "
            #             f"Cls: {loss_cls.item():.4f} | "
            #             f"Feat_Ortho: {loss_feat_ortho.item():.4f} | "
            #             f"Balance: {loss_balance.item():.4f}"
            #         )
            
            # if self.training and self.diagnosis_loss_weight > 0:
            #     # 1. 预准备：确保所有 Rank 都拥有基础变量
            #     batch_size = hidden_states.shape[0]
            #     device = hidden_states.device
                
            #     # 即使 D_mask 或 has_signal 为 None，也要统一处理成全 0 或全 False
            #     safe_has_signal = has_signal.bool() if has_signal is not None else torch.zeros(batch_size, device=device).bool()
            #     valid_mask_f = safe_has_signal.float()
            #     num_valid = valid_mask_f.sum()
                
            #     # =====================================================
            #     # 1️⃣ 特征提取与融合 (无论是否有有效样本，都运行此逻辑)
            #     # =====================================================
            #     batch_stacked_feats_h = []
            #     for b in range(batch_size):
            #         # 提取特征
            #         dev_feat_h = self.safe_span_pool(b, 'dev', hidden_states, modal_token_spans)
            #         peri_feat_h = self.safe_span_pool(b, 'peri', hidden_states, modal_token_spans)
            #         phys_feat_h = self.safe_span_pool(b, 'phys', hidden_states, modal_token_spans)
            #         img_feat_h = self.safe_span_pool(b, 'image', hidden_states, modal_token_spans)
            #         batch_stacked_feats_h.append(torch.stack([dev_feat_h, peri_feat_h, phys_feat_h, img_feat_h]))
                
            #     stacked_features_h = torch.stack(batch_stacked_feats_h)  # [B, 4, H]
            #     stacked_features_h = F.layer_norm(stacked_features_h, (stacked_features_h.shape[-1],))
                
            #     # 获取融合权重与疾病特征
            #     normalized_weights = torch.softmax(self.disease_source_weights, dim=-1)  # [8, 4]
            #     disease_feats_h = torch.einsum('bfh,df->bdh', stacked_features_h, normalized_weights)  # [B, 8, H]
                
            #     # =====================================================
            #     # 2️⃣ 分类预测 (计算 Logits)
            #     # =====================================================
            #     # View A: 局部融合视角 (现在作为 Teacher，提取细粒度特征)
            #     logits_fusion = self.disease_classifier(disease_feats_h).squeeze(-1)  # [B, 8]
                
            #     # View B: 全局 CLS 视角 (现在作为 Student，被 Fusion 引导)
            #     cls_feat = hidden_states[:, 0, :]
            #     logits_cls = self.first_token_classifier(cls_feat)  # [B, 8]

            #     # =====================================================
            #     # 3️⃣ Loss 计算：分类 + 一致性 (Consistency)
            #     # =====================================================
                
            #     # A. 主任务 Loss (CLS/Student): 拟合真实标签
            #     loss_cls_main = F.binary_cross_entropy_with_logits(
            #         logits_cls, disease_labels.float(), reduction="none"
            #     ).mean(dim=1)

            #     # B. 辅助任务 Loss (Fusion/Teacher): 必须提供强监督，Teacher才能学好
            #     loss_fusion_aux = F.binary_cross_entropy_with_logits(
            #         logits_fusion, disease_labels.float(), reduction="none"
            #     ).mean(dim=1)

            #     # C. 一致性 Loss (Logits MSE): 局部引导全局
            #     # 截断 Fusion 的梯度，防止 CLS 的早期不稳定反向破坏融合特征
            #     teacher_logits = logits_fusion.detach()
            #     student_logits = logits_cls
                
            #     # 计算 MSE [B, 8]，量级比 JS Divergence 大，能提供充足梯度
            #     mse_score = F.mse_loss(student_logits, teacher_logits, reduction="none")
                
            #     # 对所有标签取平均，得到每个样本的一致性 Loss [B]
            #     loss_consistency_per_sample = mse_score.mean(dim=1)

            #     # =====================================================
            #     # 4️⃣ 健壮的 Loss 聚合 (处理分布式死锁)
            #     # =====================================================
            #     if num_valid > 0:
            #         # 聚合有效样本的 Loss
            #         l_main = (loss_cls_main * valid_mask_f).sum() / num_valid
            #         l_aux = (loss_fusion_aux * valid_mask_f).sum() / num_valid
            #         l_consist = (loss_consistency_per_sample * valid_mask_f).sum() / num_valid
            #     else:
            #         # 伪造 0 Loss 以保持计算图完整 (DeepSpeed/DDP 需要)
            #         dummy_zero = (logits_cls.sum() + logits_fusion.sum()) * 0.0
            #         l_main = dummy_zero
            #         l_aux = dummy_zero
            #         l_consist = dummy_zero

            #     # =====================================================
            #     # 5️⃣ 最终加权 (Hyperparameters)
            #     # =====================================================
            #     # 权重配置建议：
            #     # alpha (主任务/CLS): 1.0 - 维持基础分类能力
            #     # beta (一致性/MSE): 0.5 - MSE量级天然较大，0.5通常是个不错的起点
            #     # gamma (辅助任务/Fusion): 1.0 - 既然是Teacher，必须给足权重让它学到最好的特征
                
            #     alpha, beta, gamma = 1.0, 0.1, 1.0
                
            #     diagnosis_loss = (alpha * l_main) + (beta * l_consist) + (gamma * l_aux)
                
            #     total_loss = total_loss + self.diagnosis_loss_weight * diagnosis_loss
                
            #     if global_step is not None and global_step % 10 == 0:
            #         print(
            #             f"CLS(Student): {l_main.item():.4f} | "
            #             f"Consist(MSE): {l_consist.item():.4f} | "
            #             f"Fusion(Teacher): {l_aux.item():.4f}"
            #         )

            # if self.findings_loss_weight > 0 and has_signal is not None:
            #     batch_size = inputs_embeds.shape[0]
            #     device = inputs_embeds.device
            #     dtype = inputs_embeds.dtype

            #     # =====================================================
            #     # 1️⃣ valid mask
            #     # =====================================================
            #     valid_mask = has_signal.bool()

            #     if valid_mask.sum() == 0:
            #         findings_loss = torch.tensor(0.0, device=device, dtype=dtype)
            #     else:
            #         # =====================================================
            #         # 2️⃣ Findings Text Pooling (text_feats: [B, H])
            #         # =====================================================
            #         text_mask = I_mask.float().unsqueeze(-1)
            #         text_sum = (inputs_embeds * text_mask).sum(dim=1)
            #         text_count = text_mask.sum(dim=1).clamp(min=1.0)
            #         text_feats = text_sum / text_count 
            #         text_feats = F.normalize(text_feats, p=2, dim=-1)

            #         # =====================================================
            #         # 3️⃣ 双重 Top-K 视觉特征对齐 (使用 safe_span_all)
            #         # =====================================================
            #         K1, K2 = 512, 256
            #         sample_losses = []

            #         for b in range(batch_size):
            #             # 调用封装好的 safe_span_all 函数获取图像 tokens
            #             # 注意：如果原本 modal_token_spans[b]['image'] 是 [[s, e]] 嵌套格式，
            #             # 需要确保 safe_span_all 内部处理了 [0] 索引或传入正确格式。
            #             raw_img_tokens = self.safe_span_all(b, 'image', inputs_embeds, modal_token_spans)
                        
            #             # 安全检查：如果返回的是 dummy (ndim=1) 或者 token 数量太少
            #             # 判断 raw_img_tokens.ndim == 1 说明命中了 safe_span_all 的异常返回逻辑
            #             if raw_img_tokens.ndim < 2 or raw_img_tokens.size(0) == 0:
            #                 sample_losses.append(torch.tensor(0.0, device=device, dtype=dtype))
            #                 continue

            #             # --- 第一重筛选：按 Norm 选出有能量的区域 ---
            #             token_norms = raw_img_tokens.norm(p=2, dim=-1)  # [N]
            #             actual_k1 = min(K1, raw_img_tokens.size(0))
            #             _, topk1_idx = torch.topk(token_norms, k=actual_k1)
            #             candidate_tokens = raw_img_tokens[topk1_idx]  # [K1, H]
                        
            #             # --- 第二重筛选：按相似度筛选 ---
            #             # 确保 candidate_tokens_norm 和 text_feats[b] 都是同样的 dtype (BFloat16)
            #             candidate_tokens_norm = F.normalize(candidate_tokens, p=2, dim=-1).to(dtype=dtype)
            #             current_text_feat = text_feats[b].to(dtype=dtype)

            #             # [K1, H] @ [H] -> [K1]
            #             sims = torch.matmul(candidate_tokens_norm, current_text_feat)
                        
            #             actual_k2 = min(K2, candidate_tokens.size(0))
            #             topk2_sims, _ = torch.topk(sims, k=actual_k2)

            #             # --- 计算对齐损失 ---
            #             # 这里的 1.0 - mean 可以理解为最大化最相关区域的余弦相似度
            #             sample_loss = 1.0 - topk2_sims.mean()
            #             sample_losses.append(sample_loss)

            #         # 聚合 Loss
            #         findings_loss = torch.stack(sample_losses).mean()



        # ##############################output hiddenstate[0]Diagnosis部分与consist loss###########################
            # =================================================================
        # [Step 2] 诊断一致性 Loss (diagnosis_loss Loss)
        # =================================================================
        ##########cls轻微涨点

            # if self.training and self.diagnosis_loss_weight > 0:
            #     # 1. 预准备：确保所有 Rank 都拥有基础变量  outputs.hidden_states[16]
            #     batch_size = hidden_states.shape[0]
            #     device = hidden_states.device
                
            #     # 即使 D_mask 或 has_signal 为 None，也要统一处理成全 0 或全 False，避免分支歧义
            #     safe_has_signal = has_signal.bool() if has_signal is not None else torch.zeros(batch_size, device=device).bool()
            #     valid_mask_f = safe_has_signal.float()
            #     num_valid = valid_mask_f.sum()
                
            #     # =====================================================
            #     # 1️⃣ 特征提取与融合 (无论是否有有效样本，都运行此逻辑)
            #     # =====================================================
            #     batch_stacked_feats_h = []
            #     for b in range(batch_size):
            #         # 提取特征
            #         dev_feat_h = self.safe_span_pool(b, 'dev', hidden_states, modal_token_spans)
            #         peri_feat_h = self.safe_span_pool(b, 'peri', hidden_states, modal_token_spans)
            #         phys_feat_h = self.safe_span_pool(b, 'phys', hidden_states, modal_token_spans)
            #         img_feat_h = self.safe_span_pool(b, 'image', hidden_states, modal_token_spans)
            #         batch_stacked_feats_h.append(torch.stack([dev_feat_h, peri_feat_h, phys_feat_h, img_feat_h]))
                
            #     stacked_features_h = torch.stack(batch_stacked_feats_h)  # [B, 4, H]
            #     stacked_features_h = F.layer_norm(stacked_features_h, (stacked_features_h.shape[-1],))
                
            #     # 获取融合权重与疾病特征
            #     normalized_weights = torch.softmax(self.disease_source_weights, dim=-1)  # [8, 4]
            #     disease_feats_h = torch.einsum('bfh,df->bdh', stacked_features_h, normalized_weights)  # [B, 8, H]
                
            #     # =====================================================
            #     # 2️⃣ 分类预测 Loss (计算所有样本，最后用 mask 过滤)
            #     # =====================================================
            #     logits_h = self.disease_classifier(disease_feats_h).squeeze(-1)  # [B, 8]
            #     cls_feat = hidden_states[:, 0, :]  # [B, H]
            #     cls_logits = self.first_token_classifier(cls_feat)  # [B, D]

            #     # 计算逐样本损失 (Reduction="none")
            #     loss_cls_per_sample = F.binary_cross_entropy_with_logits(
            #         logits_h, disease_labels.float(), reduction="none"
            #     ).mean(dim=1)

            #     cls_loss_per_sample = F.binary_cross_entropy_with_logits(
            #         cls_logits, disease_labels.float(), reduction="none"
            #     ).mean(dim=1)

            #     # =====================================================
            #     # 4️⃣ 健壮的 Loss 聚合 (防止分布式死锁的关键)
            #     # =====================================================
      
            #     # 使用 torch.where 或 直接乘法来处理 num_valid 为 0 的情况
            #     # 即使 num_valid 为 0，也保持 loss 能够反向传播（虽然梯度为0）
            #     if num_valid > 0:
            #         loss_cls_h = (loss_cls_per_sample * valid_mask_f).sum() / num_valid
            #         cls_loss_m = (cls_loss_per_sample * valid_mask_f).sum() / num_valid
            #     else:
            #         # 如果当前 Rank 没有有效数据，创造一个与参数挂钩的 0 loss
            #         # 这样 DeepSpeed 依然能看到完整的计算图
            #         loss_cls_h = (logits_h.sum() + cls_logits.sum()) * 0.0
            #         cls_loss_m = torch.tensor(0.0, device=device, requires_grad=True)

            #     # 合并总损失
            #     # 确保 diagnosis_loss 始终是一个包含梯度信息的 Tensor
            #     diagnosis_loss = loss_cls_h + cls_loss_m
            #     total_loss = total_loss + self.diagnosis_loss_weight * diagnosis_loss
           

            ######################################### findings_loss ##########################################
            # if self.findings_loss_weight > 0:
            #     # [B] bool
            #     valid_mask = has_signal.bool()
            #     batch_size = hidden_states.shape[0]
            #     K = 512  # Top-K 显著性 Token 数量

            #     # =====================================================
            #     #  影像端：提取全量 Token 并筛选 Top-K 显著区域
            #     # =====================================================
            #     img_list = []

            #     for b in range(batch_size):
            #         # 假设 safe_span_all 返回 [N_img, H] (单个样本)
            #         # 如果它返回 [1, N, H]，记得 squeeze 掉第一维
            #         feat = self.safe_span_all(b, 'image', hidden_states,modal_token_spans)
            #         img_list.append(feat)
                
            #     # 堆叠成 [B, N_img, H]
            #     img_all = torch.stack(img_list, dim=0) 

            #     # 计算 Norm 并选取 Top-K
            #     token_norms = torch.norm(img_all, p=2, dim=-1) # [B, N_img]
            #     _, topk_indices = torch.topk(token_norms, k=K, dim=1) # [B, K]

            #     # 提取 Hidden States: [B, K, H]
            #     # 使用 gather 需要扩展索引维度
            #     indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, img_all.size(-1))
            #     selected_img_feats = torch.gather(img_all, 1, indices_expanded)
                
            #     img_feats_norm = F.normalize(selected_img_feats, p=2, dim=-1)

            #     full_seq_norm = F.normalize(inputs_embeds, p=2, dim=-1)

            #     sim_matrix = torch.bmm(img_feats_norm, full_seq_norm.transpose(1, 2))

            #     text_mask = I_mask.float().unsqueeze(1)
  

            #     text_counts = I_mask.sum(dim=1, keepdim=True).clamp(min=1.0) # [B, 1]

            #     sim_matrix_for_max = sim_matrix.clone()
            #     sim_matrix_for_max.masked_fill_(text_mask.bool() == False, -1e4)
                
            #     # 每个 Image patch 最匹配的那个词的相似度
            #     best_text_match_vals, _ = sim_matrix_for_max.max(dim=2) # [B, K]
            #     loss_i2t = 1.0 - best_text_match_vals.mean(dim=1) # [B]

            #     best_img_match_vals, _ = sim_matrix.max(dim=1) # [B, Seq_Len]
                
            #     # 只计算 mask 内的词的 Loss
            #     loss_t2i_sum = ( (1.0 - best_img_match_vals) * I_mask.float() ).sum(dim=1)
            #     loss_t2i = loss_t2i_sum / text_counts.squeeze(-1) # [B]

            #     findings_loss_per_sample = (loss_i2t + loss_t2i) / 2.0

            #     has_any_disease = (disease_labels.sum(dim=1) > 0).float()
            #     final_valid_mask = valid_mask.float() * has_any_disease
                
            #     num_final_valid = final_valid_mask.sum()

            #     findings_loss = torch.where(
            #         num_final_valid > 0,
            #         (findings_loss_per_sample * final_valid_mask).sum() / num_final_valid,
            #         torch.tensor(0.0, device=inputs_embeds.device, dtype=inputs_embeds.dtype) # 保持梯度图完整性
            #     )

            #     total_loss = total_loss + self.findings_loss_weight * findings_loss

            # ############################################ findings_loss ##########################################
            # if self.findings_loss_weight > 0 and I_mask is not None and has_signal is not None:
            #     # [B] bool
            #     valid_mask = has_signal.bool()
            #     batch_size = inputs_embeds.shape[0]
            #     K = 512  # Top-K 显著性 Token 数量

            #     # =====================================================
            #     # 1️⃣ 影像端：提取全量 Token 并筛选 Top-K 显著区域
            #     # =====================================================
            #     img_list = []
            #     for b in range(batch_size):
            #         # 假设 safe_span_all 返回 [N_img, H] (单个样本)
            #         # 如果它返回 [1, N, H]，记得 squeeze 掉第一维
            #         feat = self.safe_span_all(b, 'image', inputs_embeds,modal_token_spans)
            #         img_list.append(feat)
                
            #     # 堆叠成 [B, N_img, H]
            #     img_all = torch.stack(img_list, dim=0) 

            #     # 计算 Norm 并选取 Top-K
            #     token_norms = torch.norm(img_all, p=2, dim=-1) # [B, N_img]
            #     _, topk_indices = torch.topk(token_norms, k=K, dim=1) # [B, K]

            #     # 提取 Hidden States: [B, K, H]
            #     # 使用 gather 需要扩展索引维度
            #     indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, img_all.size(-1))
            #     selected_img_feats = torch.gather(img_all, 1, indices_expanded)
                
            #     img_feats_norm = F.normalize(selected_img_feats, p=2, dim=-1)

            #     # =====================================================
            #     # 2️⃣ 文本端：必须使用 Hidden States 而非 Embedding
            #     # =====================================================
            #     # 假设 hidden_states 包含了 [Image + Text] 的全序列
            #     # 我们需要利用 labels/I_mask 来定位哪些是 Findings 的文本 hidden state
                
            #     # 注意：这里不能重新 forward 一次 embed_tokens，直接用输入的 hidden_states
            #     # 这里的逻辑稍微复杂，假设你很难直接切分出纯文本 hidden states，
            #     # 我们可以直接用 I_mask 盖在全量的 hidden_states 上。
                
            #     # [B, Seq_Len, H] -> 归一化
            #     full_seq_norm = F.normalize(inputs_embeds, p=2, dim=-1)

            #     # =====================================================
            #     # 3️⃣ 计算双向逐点相似度矩阵
            #     # =====================================================
            #     # [B, K, H] x [B, H, Seq_Len] -> [B, K, Seq_Len]
            #     # 计算图像 Top-K patch 和 全序列 的相似度
            #     sim_matrix = torch.bmm(img_feats_norm, full_seq_norm.transpose(1, 2))
                
            #     # 应用 Findings Mask (I_mask)
            #     # I_mask: [B, Seq_Len] -> [B, 1, Seq_Len]
            #     # 只有属于 Findings 的文本位置才保留值，其他位置设为极小值(用于softmax)或0(用于relu/sum)
            #     text_mask = I_mask.float().unsqueeze(1)
                
            #     # 为了避免 mask 掉的部分影响计算（变成0），如果你后续只做 sum 且期望正向匹配，乘 mask 是 ok 的。
            #     # 但如果 sim 可能是负数，乘 0 会消除负数惩罚。这里假设我们要最大化相似度。
            #     sim_matrix_masked = sim_matrix * text_mask

            #     # =====================================================
            #     # 4️⃣ 双向 Loss 计算 (改为 Max/Mean 混合策略)
            #     # =====================================================
            #     # text_counts = 实际有效的 Findings Token 数
            #     text_counts = I_mask.sum(dim=1, keepdim=True).clamp(min=1.0) # [B, 1]

            #     # (1) 影像 -> 文本 (I2T): 
            #     # 逻辑：每个显著图像 Patch，应该能在 Findings 文本里找到至少一个对应的词
            #     # 做法：对文本维取 Max，然后对图像维取 Mean
            #     # sim: [B, K, Seq_Len] -> max(dim=2) -> [B, K]
            #     # 注意：直接 max 会取到非 mask 区域（如果 mask 是 0 且相似度全是负的）。
            #     # 建议：将 mask 为 0 的位置设为 -1e9
            #     sim_matrix_for_max = sim_matrix.clone()
            #     sim_matrix_for_max.masked_fill_(text_mask.bool() == False, -1e4)
                
            #     # 每个 Image patch 最匹配的那个词的相似度
            #     best_text_match_vals, _ = sim_matrix_for_max.max(dim=2) # [B, K]
            #     loss_i2t = 1.0 - best_text_match_vals.mean(dim=1) # [B]

            #     # (2) 文本 -> 影像 (T2I):
            #     # 逻辑：Findings 里的每个词，都应该能在 Top-K 图像里找到对应区域
            #     # 做法：对图像维取 Max (Multi-Instance Learning)，然后对文本维取 Mean
            #     # sim: [B, K, Seq_Len] -> max(dim=1) -> [B, Seq_Len]
            #     best_img_match_vals, _ = sim_matrix.max(dim=1) # [B, Seq_Len]
                
            #     # 只计算 mask 内的词的 Loss
            #     loss_t2i_sum = ( (1.0 - best_img_match_vals) * I_mask.float() ).sum(dim=1)
            #     loss_t2i = loss_t2i_sum / text_counts.squeeze(-1) # [B]

            #     findings_loss_per_sample = (loss_i2t + loss_t2i) / 2.0

            #     # =====================================================
            #     # 5️⃣ 逻辑闭环
            #     # =====================================================
            #     has_any_disease = (disease_labels.sum(dim=1) > 0).float()
            #     final_valid_mask = valid_mask.float() * has_any_disease
                
            #     num_final_valid = final_valid_mask.sum()

            #     findings_loss = torch.where(
            #         num_final_valid > 0,
            #         (findings_loss_per_sample * final_valid_mask).sum() / num_final_valid,
            #         torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype) # 保持梯度图完整性
            #     )

            #     total_loss = total_loss + self.findings_loss_weight * findings_loss


            # if self.findings_loss_weight > 0:
            #     # [B] bool
            #     valid_mask = has_signal.bool()
            #     batch_size = inputs_embeds.shape[0]
            #     K = 512  # Top-K 显著性 Token 数量

            #     # =====================================================
            #     # 1️⃣ 影像端：提取全量 Token 并筛选 Top-K 显著区域
            #     # =====================================================
            #     img_list = []
            #     for b in range(batch_size):
            #         # 假设 safe_span_all 返回 [N_img, H] (单个样本)
            #         # 如果它返回 [1, N, H]，记得 squeeze 掉第一维
            #         feat =  self.safe_span_all(b, 'image', inputs_embeds)
            #         img_list.append(feat)
            #     # 堆叠成 [B, N_img, H]
            #     img_all = torch.stack(img_list, dim=0)

            #     # 计算 Norm 并选取 Top-K
            #     token_norms = torch.norm(img_all, p=2, dim=-1)  # [B, N_img]
            #     _, topk_indices = torch.topk(token_norms, k=K, dim=1)  # [B, K]

            #     # 提取 Hidden States: [B, K, H]
            #     # 使用 gather 需要扩展索引维度
            #     indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, img_all.size(-1))
            #     selected_img_feats = torch.gather(img_all, 1, indices_expanded)
            #     img_feats_norm = F.normalize(selected_img_feats, p=2, dim=-1)

            #     # =====================================================
            #     # 2️⃣ 文本端：必须使用 Hidden States 而非 Embedding
            #     # =====================================================
            #     # 假设 hidden_states 包含了 [Image + Text] 的全序列
            #     # 我们需要利用 labels/I_mask 来定位哪些是 Findings 的文本 hidden state
            #     # 注意：这里不能重新 forward 一次 embed_tokens，直接用输入的 hidden_states
            #     # 这里的逻辑稍微复杂，假设你很难直接切分出纯文本 hidden states，
            #     # 我们可以直接用 I_mask 盖在全量的 hidden_states 上。
            #     # [B, Seq_Len, H] -> 归一化
            #     full_seq_norm = F.normalize(inputs_embeds, p=2, dim=-1)

            #     # =====================================================
            #     # 3️⃣ 计算双向逐点相似度矩阵
            #     # =====================================================
            #     # [B, K, H] x [B, H, Seq_Len] -> [B, K, Seq_Len]
            #     # 计算图像 Top-K patch 和 全序列 的相似度
            #     sim_matrix = torch.bmm(img_feats_norm, full_seq_norm.transpose(1, 2))
            #     # 应用 Findings Mask (I_mask)
            #     # I_mask: [B, Seq_Len] -> [B, 1, Seq_Len]
            #     # 只有属于 Findings 的文本位置才保留值，其他位置设为极小值(用于softmax)或0(用于relu/sum)
            #     text_mask = I_mask.float().unsqueeze(1)
            #     # 为了避免 mask 掉的部分影响计算（变成0），如果你后续只做 sum 且期望正向匹配，乘 mask 是 ok 的。
            #     # 但如果 sim 可能是负数，乘 0 会消除负数惩罚。这里假设我们要最大化相似度。
            #     sim_matrix_masked = sim_matrix * text_mask

            #     # =====================================================
            #     # 4️⃣ 双向 Loss 计算 (改为 Max/Mean 混合策略)
            #     # =====================================================
            #     # text_counts = 实际有效的 Findings Token 数
            #     text_counts = I_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]

            #     # (1) 影像 -> 文本 (I2T):
            #     # 逻辑：每个显著图像 Patch，应该能在 Findings 文本里找到至少一个对应的词
            #     # 做法：对文本维取 Max，然后对图像维取 Mean
            #     # sim: [B, K, Seq_Len] -> max(dim=2) -> [B, K]
            #     # 注意：直接 max 会取到非 mask 区域（如果 mask 是 0 且相似度全是负的）。
            #     # 建议：将 mask 为 0 的位置设为 -1e9
            #     sim_matrix_for_max = sim_matrix.clone()
            #     sim_matrix_for_max.masked_fill_(text_mask.bool() == False, -1e4)
            #     # 每个 Image patch 最匹配的那个词的相似度
            #     best_text_match_vals, _ = sim_matrix_for_max.max(dim=2)  # [B, K]
            #     loss_i2t = 1.0 - best_text_match_vals.mean(dim=1)  # [B]

            #     # (2) 文本 -> 影像 (T2I):
            #     # 逻辑：Findings 里的每个词，都应该能在 Top-K 图像里找到对应区域
            #     # 做法：对图像维取 Max (Multi-Instance Learning)，然后对文本维取 Mean
            #     # sim: [B, K, Seq_Len] -> max(dim=1) -> [B, Seq_Len]
            #     best_img_match_vals, _ = sim_matrix.max(dim=1)  # [B, Seq_Len]
            #     # 只计算 mask 内的词的 Loss
            #     loss_t2i_sum = ((1.0 - best_img_match_vals) * I_mask.float()).sum(dim=1)
            #     loss_t2i = loss_t2i_sum / text_counts.squeeze(-1)  # [B]

            #     findings_loss_per_sample = (loss_i2t + loss_t2i) / 2.0

            #     # =====================================================
            #     # 5️⃣ 逻辑闭环
            #     # =====================================================
            #     has_any_disease = (disease_labels.sum(dim=1) > 0).float()
            #     final_valid_mask = valid_mask.float() * has_any_disease
            #     num_final_valid = final_valid_mask.sum()

            #     findings_loss = torch.where(
            #         num_final_valid > 0,
            #         (findings_loss_per_sample * final_valid_mask).sum() / num_final_valid,
            #         torch.tensor(0.0, device=inputs_embeds.device, dtype=inputs_embeds.dtype)  # 保持梯度图完整性
            #     )

            #     total_loss = total_loss + self.findings_loss_weight * findings_loss

            #     # =====================================================
            #     # 5️影像端分类：基于 Top-K 显著 Token 的特征池化
            #     # =====================================================
            #     for b in range(batch_size):
            #         img_feat_h = self.safe_span_pool(b, 'image', hidden_states,modal_token_spans)

            #     # 2. 计算分类 Logits
            #     # 假设 self.disease_classifier 是一个能够处理 H 维输入并输出疾病类别的 Linear 层
            #     img_logits_h = self.disease_classifier(img_feat_h) # [B, Num_Classes]

            #     # 3. 计算分类 Loss (BCE)
            #     # [B, Num_Classes] -> [B]
            #     image_cls_per_sample = F.binary_cross_entropy_with_logits(
            #         img_logits_h, 
            #         disease_labels.float(), 
            #         reduction="none"
            #     ).mean(dim=1)

            #     # 4. 应用有效样本掩码 (valid_mask)
            #     valid_mask_f = valid_mask.float()
            #     num_valid = valid_mask_f.sum()

            #     image_loss_cls_h = torch.where(
            #         num_valid > 0,
            #         (image_cls_per_sample * valid_mask_f).sum() / num_valid,
            #         image_cls_per_sample.sum() * 0.0 # 保证梯度回传路径不中断
            #     )

            #     #+loss_cls_h
            # # ##########语义相似度的findloss在Ce上涨了一点 但是NLG没有涨########
            # if self.findings_loss_weight > 0 and I_mask is not None and has_signal is not None:
            
            #     findings_loss = 0.0

            #     batch_size = hidden_states.shape[0]
                
            #     # === 修复1: 正确提取每个样本的image全局特征 (原循环错误: 仅保留最后一个样本) ===
            #     img_globals = []
            #     for b in range(batch_size):
            #         img_global_b = self.safe_span_all(b, 'image', hidden_states)  # [1, H] or [H]
                    
            #         # 维度适配: 确保为 [1, H]
            #         if img_global_b.dim() == 1:
            #             img_global_b = img_global_b.unsqueeze(0)  # [H] -> [1, H]
            #         elif img_global_b.dim() == 3:
            #             img_global_b = img_global_b.mean(dim=1)   # [1, T, H] -> [1, H]
                    
            #         img_globals.append(img_global_b)
                
            #     img_global = torch.cat(img_globals, dim=0)  # [B, H]
            #     img_global_norm = F.normalize(img_global, p=2, dim=-1)  # [B, H]
                
            #     # === 准备Findings部分的特征 ===
            #     # I_mask: [B, L] 标记影像结论文本位置
            #     mask_expanded = I_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, L, 1]
            #     valid_token_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [B, 1]
                
            #     # 1. Label静态embedding (词义层面)
            #     safe_labels = labels.clone()
            #     safe_labels[safe_labels == -100] = 0
            #     label_embeds = self.get_model().embed_tokens(safe_labels)  # [B, L, H]
            #     masked_label_feats = label_embeds * mask_expanded
            #     label_global = masked_label_feats.sum(dim=1) / valid_token_counts  # [B, H]
            #     label_global_norm = F.normalize(label_global, p=2, dim=-1)
                
            #     # 2. LLM生成状态 (上下文感知)
            #     masked_context_feats = hidden_states * mask_expanded
            #     context_global = masked_context_feats.sum(dim=1) / valid_token_counts  # [B, H]
            #     context_global_norm = F.normalize(context_global, p=2, dim=-1)
                
            #     # === 修复2: 每样本loss计算 + valid_mask过滤 (仅新生儿数据参与loss) ===
            #     # 计算每个样本的cosine loss (避免先mean导致成人数据污染)
            #     loss_semantic_anchor_per_sample = 1.0 - F.cosine_similarity(
            #         img_global_norm, label_global_norm, dim=-1
            #     )  # [B]
                
            #     loss_generation_guide_per_sample = 1.0 - F.cosine_similarity(
            #         img_global_norm, context_global_norm, dim=-1
            #     )  # [B]
                
            #     # valid_mask: 仅新生儿数据 (has_signal=True) 参与loss
            #     valid_mask = has_signal.bool()  # [B], True=新生儿, False=成人(MIMIC等)
            #     valid_mask_f = valid_mask.float()
            #     num_valid = valid_mask_f.sum()
                
            #     # DDP-safe 加权平均 (仅对有效样本求平均)
            #     if num_valid > 0:
            #         loss_semantic_anchor = (loss_semantic_anchor_per_sample * valid_mask_f).sum() / num_valid
            #         loss_generation_guide = (loss_generation_guide_per_sample * valid_mask_f).sum() / num_valid
            #     else:
            #         # 无有效样本时返回0梯度 (避免DDP同步错误)
            #         loss_semantic_anchor = torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
            #         loss_generation_guide = torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
                
            #     findings_loss = loss_semantic_anchor + loss_generation_guide
            #     total_loss = total_loss + self.findings_loss_weight * findings_loss

            # if self.findings_loss_weight > 0 and has_signal is not None:

            #     device = hidden_states.device
                
            #     # =====================================================
            #     # 1️⃣ Early Gating：筛选真正有效的样本索引
            #     # =====================================================
            #     # 必须同时满足：1. 有信号；2. 包含至少一种疾病标记（减少噪声负样本）
            #     has_any_disease = (disease_labels.sum(dim=1) > 0)
            #     # 得到布尔掩码：[B]
            #     is_informative = has_signal.bool() & has_any_disease
                
            #     # 提取有效索引
            #     valid_indices = torch.where(is_informative)[0]
            #     num_valid = valid_indices.numel()

            #     # 对比学习至少需要 2 个样本才能构建负样本对
            #     if num_valid < 2:
            #         findings_loss = torch.tensor(0.0, device=device, dtype=hidden_states.dtype)
            #     else:
            #         # 只保留有意义的特征进行计算
            #         v_hidden_states = hidden_states[valid_indices]  # [N, L, H]
            #         v_I_mask = I_mask[valid_indices]               # [N, L]
            #         # 注意：modal_token_spans 是 list，需要按索引提取
            #         v_spans = [modal_token_spans[i.item()] for i in valid_indices]

            #         # =====================================================
            #         # 2️⃣ Image & Text Pooling (仅针对有效样本)
            #         # =====================================================
            #         img_feats = []
            #         for i in range(num_valid):
            #             # 重新映射索引：在 v_hidden_states 中索引是 i
            #             img_feat = self.safe_span_pool(
            #                 i, 'image', v_hidden_states, v_spans
            #             )
            #             img_feats.append(img_feat)
                    
            #         img_feats = torch.stack(img_feats, dim=0)  # [N, H]

            #         text_mask = v_I_mask.float().unsqueeze(-1) # [N, L, 1]
            #         text_sum = (v_hidden_states * text_mask).sum(dim=1)
            #         text_count = text_mask.sum(dim=1).clamp(min=1.0)
            #         text_feats = text_sum / text_count         # [N, H]

            #         # =====================================================
            #         # 3️⃣ 数值安全 Normalize (防止全零向量导致 NaN)
            #         # =====================================================
            #         eps = 1e-8
            #         img_feats = F.normalize(img_feats, p=2, dim=-1, eps=eps)
            #         text_feats = F.normalize(text_feats, p=2, dim=-1, eps=eps).to(img_feats.dtype)

            #         # =====================================================
            #         # 4️⃣ DCL (Decoupled Contrastive Loss) 
            #         # =====================================================
            #         temperature = 0.07
            #         # 此时 logits 矩阵大小为 [N, N]，N = num_valid
            #         logits = torch.matmul(img_feats, text_feats.t()) / temperature

            #         # 正样本：对角线
            #         pos_sim = torch.diag(logits)

            #         # 负样本：排除掉对角线
            #         mask = torch.eye(num_valid, device=device).bool()
                    
            #         # i2t 方向：logsumexp 仅计算非对角线元素
            #         neg_logits_i2t = logits.masked_fill(mask, -1e9)
            #         loss_i2t = -pos_sim + torch.logsumexp(neg_logits_i2t, dim=1)

            #         # t2i 方向：转置后计算
            #         neg_logits_t2i = logits.t().masked_fill(mask, -1e9)
            #         loss_t2i = -pos_sim + torch.logsumexp(neg_logits_t2i, dim=1)

            #         # 最终损失取均值（此时已经是针对 informative samples 的平均）
            #         findings_loss = (loss_i2t.mean() + loss_t2i.mean()) / 2.0

            #     # =====================================================
            #     # 5️⃣ 加入 Total Loss
            #     # =====================================================
            #     # 因为 findings_loss 已经是针对有效部分的 mean，
            #     # 按照惯例，我们需要乘以 batch 内有效样本的占比，防止该 loss 权重过大
            #     valid_ratio = num_valid / batch_size
            #     total_loss = total_loss + self.findings_loss_weight * findings_loss * valid_ratio


