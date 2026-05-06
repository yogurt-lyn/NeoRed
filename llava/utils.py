import datetime
import logging
import logging.handlers
import os
import sys
import json

import logging
import glob
import pandas as pd
from PIL import Image
from typing import List, Dict, Any, Optional

import glob

import requests
import pandas as pd

from llava.constants import LOGDIR

from llava.question_formats import get_judgement_prompt,get_open_ended_prompt, get_report_generation_prompt

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def data_loader_default(data_path):
    logging.info("using the default loader.")
    dataset = json.load(open(data_path, "r"))
    logging.info(f"loaded {len(dataset)} samples.")
    return dataset


import pprint

import io


def decode_image_from_hf_parquet_entry(image_data):

    if isinstance(image_data, dict) and 'bytes' in image_data and image_data['bytes']:
        image_bytes = image_data['bytes']
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    return None


def RG_convert_to_conversation_format(sample,print_sample=False,category=""):

    prompt=get_report_generation_prompt()

    images = sample.get("AP_image")

    if isinstance(images, list):
        image_placeholders = "".join(["<image>\n" for _ in images])
    elif isinstance(images, str):
        image_placeholders = "<image>\n"
    else:
        print("AP_image is null")
        image_placeholders = ""
    
    human_value = f"{image_placeholders}{prompt}"

    gpt_value = f"Findings: {sample['findings']} Impression: {sample['impression']}"
    subject_id=sample.get("subject_id")

    conversation = {
        "id": sample["id"],
        "category":category,
        "subject_id":subject_id,
        "image": sample["AP_image"],  
        "conversations": [
            {
                "from": "human",
                "value": human_value
            },
            {
                "from": "gpt", 
                "value": gpt_value
            }
        ]
    }
    if print_sample:

        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(conversation)
        print("=" * 50)
    return conversation

def VQA_convert_to_conversation_format(sample,print_sample,category):

    question = sample["question"]
    # image = sample["image"]
    image = sample.get("image") or sample.get("img_name")
    answer = sample["answer"]
    

    # is_reasoning = True if os.environ.get("REASONING", "False") == "True" else False
    is_reasoning = False
    answer = answer.lower()
    

    if answer in ["yes", "no"]:
        prompt = get_judgement_prompt(question, is_reasoning)
    else:
        prompt = get_open_ended_prompt(question, is_reasoning)
    
        # "imgs":sample.get("imgs"),
    conversation = {
        # "id": sample.get("id",f"vqa_rad_{hash(question)}"),  
        "id": sample.get("id") or sample.get("qid") or f"vqa_rad_{hash(question)}",
        "category":category,
        "image": image,  
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{prompt}"
            },
            {
                "from": "gpt",
                "value": answer
            }
        ]
    }
    
    if print_sample:
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(conversation)
        print("=" * 50)
    return conversation


import logging
import os
import json
from typing import List, Dict, Any

# def data_loader_NeoCXRNoneClinic(data_path: str) -> List[Dict[str, Any]]:
#     """
#     加载 NeoCXR 数据集，转换为标准 conversation 格式，使用外部 clinic_prompt.txt 模板 + structured clinical info。
    
#     Args:
#         data_path: JSON 文件路径（单个或 {path1, path2} 格式）
#     """
#     logging.info("using the NeoCXR loader (structured clinical + external prompt template).")

#     image_folder = "/data_lyn/data/NeoCXR/images"
#     prompt_clinic_path = "/data_lyn/MedEvalKit/utils/NeoCXR/prompt/clinic_prompt.txt"

#     # === 读取外部 prompt 模板 ===
#     try:
#         with open(prompt_clinic_path, 'r', encoding='utf-8') as f:
#             prompt_template = f.read().strip()
#         logging.info(f"Loaded prompt template from {prompt_clinic_path}")
#     except Exception as e:
#         logging.error(f"Failed to load prompt template: {e}")
#         raise

#     # 验证模板包含必要占位符
#     required_placeholders = [
#         "__IMAGE_TOKENS__",
#         "__DEV_FACTORS__",
#         "__PERINATAL_RISK__",
#         "__PHYS_STATUS__",
#     ]
#     for ph in required_placeholders:
#         if ph not in prompt_template:
#             raise ValueError(f"Prompt template missing required placeholder: {ph}")

#     # 解析 data_path
#     if data_path.startswith('{') and data_path.endswith('}'):
#         inner_paths = data_path[1:-1]
#         paths = [p.strip() for p in inner_paths.split(',')]
#         logging.info(f"Parsed multiple data paths: {paths}")
#     else:
#         paths = [data_path]
#         logging.info(f"Using single data path: {data_path}")

#     conversation_datas = []
#     loaded_counts = {}

#     for path in paths:
#         logging.info(f"Loading dataset from {path}")
#         category = "NeoCXR"
#         loaded_counts[category] = 0
#         skipped_image_count = 0

#         try:
#             with open(path, "r", encoding='utf-8') as f:
#                 dataset: List[Dict[str, Any]] = json.load(f)
#         except Exception as e:
#             logging.error(f"Error loading {path}: {e}")
#             continue

#         for original_data in dataset:
#             # === 1. 图像路径检查 ===
#             img_field = original_data.get("image", "").strip()
#             if not img_field:
#                 skipped_image_count += 1
#                 continue

#             image_paths = [p.strip() for p in img_field.split(",") if p.strip()]
#             full_image_paths = [os.path.join(image_folder, p) for p in image_paths]

#             if not all(os.path.exists(p) for p in full_image_paths):
#                 logging.warning(f"Missing image(s) for sample {original_data.get('id', 'N/A')}: {full_image_paths}. Skipping.")
#                 skipped_image_count += 1
#                 continue

#             # === 2. 构建 structured clinical blocks ===
#             clinical_structured = original_data.get("clinical_structured", {})
#             dev = clinical_structured.get("development_factors", {})
#             peri = clinical_structured.get("perinatal_risk", {})
#             phys = clinical_structured.get("physiological_status", {})

#             def format_section(d):
#                 if not d:
#                     return "  (not provided)"
#                 lines = []
#                 for k, v in d.items():
#                     if isinstance(v, dict):
#                         sub_items = []
#                         for sk, sv in v.items():
#                             sub_items.append(f"{sk}: {sv}")
#                         v_str = ", ".join(sub_items)
#                         lines.append(f"  {k}: {v_str}")
#                     else:
#                         lines.append(f"  {k}: {v}")
#                 return "\n".join(lines)

#             developmental_block = format_section(dev)
#             perinatal_block = format_section(peri)
#             physiological_block = format_section(phys)

#             # === 3. 插入模板 ===
#             image_tokens = "\n".join(["<image>"] * len(image_paths))
#             human_prompt = (
#                 prompt_template
#                 .replace("__IMAGE_TOKENS__", image_tokens)
#                 .replace("__DEV_FACTORS__", developmental_block)
#                 .replace("__PERINATAL_RISK__", perinatal_block)
#                 .replace("__PHYS_STATUS__", physiological_block)
#             )

#             # === 4. 构建 GPT 响应 ===
#             imaging_conclusion = original_data.get("imaging_conclusion_en", "").strip()
#             disease_diagnosis = original_data.get("disease_diagnosis_en", "").strip()
#             gpt_response = f"Imaging conclusion: {imaging_conclusion} Disease diagnosis: {disease_diagnosis}"

#             # === 5. 构造最终样本 ===
#             processed_data = {
#                 "id": original_data["id"],
#                 "patient_id": original_data["patient_id"],
#                 "image": original_data["image"],
#                 "data_type": original_data["data_type"],
#                 "category": "NeoCXR",
#                 "conversations": [
#                     {"from": "human", "value": human_prompt},
#                     {"from": "gpt", "value": gpt_response}
#                 ]
#             }

#             conversation_datas.append(processed_data)
#             loaded_counts[category] += 1

#         logging.info(f"Loaded {loaded_counts[category]} valid samples from {path}. Skipped {skipped_image_count} samples due to missing images.")

#     logging.info(f"Loaded a total of {len(conversation_datas)} samples.")
#     return conversation_datas

def data_loader_NeoCXRNoneClinic(data_path):
    """
    加载 NeoCXR 数据集，转换对话格式，添加类别标签，并跳过图像文件不存在的样本。
    
    Args:
        data_path: JSON 文件路径（可以是单个路径或 {} 括起来的多个路径）。
        image_folder: 包含所有图像文件的根目录路径。
    """
    logging.info("using the NeoCXR loader.")

    image_folder="/data_lyn/data/NeoCXR/images"
    
    # 路径解析部分保持不变
    if data_path.startswith('{') and data_path.endswith('}'):
        inner_paths = data_path[1:-1]
        paths = [p.strip() for p in inner_paths.split(',')]
        logging.info(f"Parsed multiple data paths: {paths}")
    else:
        paths = [data_path]
        logging.info(f"Using single data path: {data_path}")

    conversation_datas = []
    loaded_counts = {}

    for path in paths:
        logging.info(f"Loading dataset from {path}")

        category = "NeoCXR"
        loaded_counts[category] = 0
        skipped_image_count = 0 # 新增计数器

        try:
            with open(path, "r", encoding='utf-8') as f:
                dataset: List[Dict[str, Any]] = json.load(f)
        except Exception as e:
            logging.error(f"Error loading {path}: {e}")
            continue

        for original_data in dataset:
            
            img_path = original_data.get("image")
            
            # if img_path:

            #     full_image_path = os.path.join(image_folder, img_path)
                

            #     if not os.path.exists(full_image_path):
            #         logging.warning(f"Image file not found for sample {original_data.get('id', 'N/A')}: {full_image_path}. Skipping.")
            #         skipped_image_count += 1
            #         continue 
            # else:
            #     # logging.warning(f"Sample {original_data.get('id', 'N/A')} is missing 'image' key. Skipping.")
            #     skipped_image_count += 1
            #     continue

            img_path_field = original_data.get("image")
            if img_path_field:
                img_paths = [
                    os.path.join(image_folder, p.strip())
                    for p in img_path_field.split(",")
                    if p.strip()
                ]
                
                if not img_paths:
                    logging.warning(f"Sample {original_data.get('id', 'N/A')} has empty image field after split. Skipping.")
                    skipped_image_count += 1
                    continue
                full_image_paths = [os.path.join(image_folder, p) for p in img_paths]
                
                if not all(os.path.exists(p) for p in full_image_paths):
                    logging.warning(f"Missing image(s) for sample {original_data.get('id', 'N/A')}: {full_image_paths}. Skipping.")
                    skipped_image_count += 1
                    continue
            else:
                skipped_image_count += 1
                continue

            processed_data = original_data.copy()
            
            if 'conversation_en' in processed_data and isinstance(processed_data['conversation_en'], list):

                processed_data['conversations'] = processed_data['conversation_en']
                
                del processed_data['conversation_en']
                
                processed_data['category'] = category
                
                conversation_datas.append(processed_data)
                loaded_counts[category] += 1
            else:
                logging.warning(f"Sample {original_data.get('id', 'N/A')} is missing 'conversation_en' or it's invalid. Skipping.")
                
        logging.info(f"Loaded {loaded_counts[category]} valid samples from {path}. Skipped {skipped_image_count} samples due to missing images.")

    
    logging.info(f"Loaded a total of {len(conversation_datas)} samples.")
    return conversation_datas

def data_loader_NeoCXR(data_path):
    logging.info("using the NeoCXR loader.")
    if data_path.startswith('{') and data_path.endswith('}'):
        inner_paths = data_path[1:-1]
        paths = [p.strip() for p in inner_paths.split(',')]
        logging.info(f"Parsed multiple data paths: {paths}")
    else:
        paths = [data_path]
        logging.info(f"Using single data path: {data_path}")

    conversation_datas = []
    loaded_counts = {}

    for path in paths:
        logging.info(f"Loading dataset from {path}")

        category = "NeoCXR"
        loaded_counts[category] = 0

        try:
            with open(path, "r", encoding='utf-8') as f:
                dataset: List[Dict[str, Any]] = json.load(f)
        except Exception as e:
            logging.error(f"Error loading {path}: {e}")
            continue

        for original_data in dataset:
            
            processed_data = original_data.copy()
            
            if 'conversation_en' in processed_data and isinstance(processed_data['conversation_en'], list):

                processed_data['conversations'] = processed_data['conversation_en']
            
                del processed_data['conversation_en']

                processed_data['category'] = category
                
                conversation_datas.append(processed_data)
                loaded_counts[category] += 1
            else:
                logging.warning(f"Sample {original_data.get('id', 'N/A')} is missing 'conversation_en' or it's invalid. Skipping.")
                
        logging.info(f"Loaded {loaded_counts[category]} samples from {path}")


    return conversation_datas



import os
import json
import logging
from typing import List, Dict, Any


# DISEASE2ID = {
#     "Neonatal Pneumonia": 0,
#     "Neonatal Respiratory Distress Syndrome (NRDS)": 1,
#     "Neonatal Transient Tachypnea (TTN)": 2,
#     "Pneumothorax": 3,
#     "Bronchopulmonary Dysplasia(BPD)": 4,
#     "Atelectasis": 5,
#     "Pleural Effusion": 6,
#     "Fracture": 7,
#     "No obvious abnormalities": 8,
# }

DISEASE2ID = {
    "Neonatal Pneumonia": 0,
    "Neonatal Respiratory Distress Syndrome (NRDS)": 1,
    "Neonatal Transient Tachypnea (TTN)": 2,
    "Pneumothorax": 3,
    "Bronchopulmonary Dysplasia(BPD)": 4,
    "Atelectasis": 5,
    "Pleural Effusion": 6,
    "No obvious abnormalities": 7,
}
# ALIAS_MAP = {
#     "no obvious abnormality": "Normal",
# }

import re

DISEASE_PATTERNS = {
    re.compile(r"\b" + re.escape(name.lower()) + r"\b"): idx
    for name, idx in DISEASE2ID.items()
}

# ALIAS_MAP_LOWER = {k.lower(): v.lower() for k, v in ALIAS_MAP.items()}

def parse_disease_ids(disease_str: str):
    if not disease_str:
        return []

    s = disease_str.lower()

    # for k, v in ALIAS_MAP_LOWER.items():
    #     s = s.replace(k, v)

    disease_ids = set()

    for pattern, idx in DISEASE_PATTERNS.items():
        if pattern.search(s):
            disease_ids.add(idx)


    return sorted(disease_ids)



def data_loader_NeoCXRClinicToken(data_path):
    """
    加载 NeoCXR 数据集，转换为标准 conversation 格式，使用外部 clinic_prompt.txt 模板 + structured clinical info。
    
    Args:
        data_path: JSON 文件路径（单个或 {path1, path2} 格式）
    """
    logging.info("using the NeoCXR loader (structured clinical + external prompt template).")

    image_folder = "/data_lyn/data/NeoCXR/images"
    prompt_clinic_path = "/data_lyn/MedEvalKit/utils/NeoCXR/prompt/clinic_prompt_token.txt"
    

    # === 读取外部 prompt 模板 ===
    try:
        with open(prompt_clinic_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read().strip()
        logging.info(f"Loaded prompt template from {prompt_clinic_path}")
    except Exception as e:
        logging.error(f"Failed to load prompt template: {e}")
        raise

    # 验证模板包含必要占位符
    required_placeholders = [
        "__IMAGE_TOKENS__",
        "__DEV_FACTORS__",
        "__PERINATAL_RISK__",
        "__PHYS_STATUS__",
    ]
    for ph in required_placeholders:
        if ph not in prompt_template:
            raise ValueError(f"Prompt template missing required placeholder: {ph}")

    # 解析 data_path
    if data_path.startswith('{') and data_path.endswith('}'):
        inner_paths = data_path[1:-1]
        paths = [p.strip() for p in inner_paths.split(',')]
        logging.info(f"Parsed multiple data paths: {paths}")
    else:
        paths = [data_path]
        logging.info(f"Using single data path: {data_path}")

    conversation_datas = []
    loaded_counts = {}

    for path in paths:
        logging.info(f"Loading dataset from {path}")
        category = "NeoCXR"
        loaded_counts[category] = 0
        skipped_image_count = 0

        try:
            with open(path, "r", encoding='utf-8') as f:
                dataset: List[Dict[str, Any]] = json.load(f)
        except Exception as e:
            logging.error(f"Error loading {path}: {e}")
            continue

        for original_data in dataset:
            # === 1. 图像路径检查 ===
            img_field = original_data.get("image", "").strip()
            if not img_field:
                skipped_image_count += 1
                continue

            image_paths = [p.strip() for p in img_field.split(",") if p.strip()]
            full_image_paths = [os.path.join(image_folder, p) for p in image_paths]

            if not all(os.path.exists(p) for p in full_image_paths):
                logging.warning(f"Missing image(s) for sample {original_data.get('id', 'N/A')}: {full_image_paths}. Skipping.")
                skipped_image_count += 1
                continue

            # === 2. 构建 structured clinical blocks ===
            clinical_structured = original_data.get("clinical_structured", {})
            dev = clinical_structured.get("development_factors", {})
            peri = clinical_structured.get("perinatal_risk", {})
            phys = clinical_structured.get("physiological_status", {})

            
            def format_section(d):
                if not d:
                    return "  (not provided)"
                lines = []
                for k, v in d.items():
                    lines.append(f"  {k}: {v}")
                return "\n".join(lines)


            developmental_block = format_section(dev)
            perinatal_block = format_section(peri)
            physiological_block = format_section(phys)

            # === 3. 插入模板 ===
            image_tokens = "".join(["<image>"] * len(image_paths))
            human_prompt = (
                prompt_template
                .replace("__IMAGE_TOKENS__", image_tokens)
                .replace("__DEV_FACTORS__", developmental_block)
                .replace("__PERINATAL_RISK__", perinatal_block)
                .replace("__PHYS_STATUS__", physiological_block)
            )

            # === 4. 构建 GPT 响应 ===
            imaging_conclusion = original_data.get("imaging_conclusion_en", "").strip()
            disease_diagnosis = original_data.get("disease_diagnosis_en", "").strip()
            gpt_response = f"Imaging conclusion: {imaging_conclusion} Disease diagnosis: {disease_diagnosis}"

            # === 4.1 解析多标签疾病 ===
            disease_ids = parse_disease_ids(disease_diagnosis)

            if loaded_counts[category] < 10:  # 只打印前 10 个样本（避免刷屏）
                logging.info(f"Sample {original_data.get('id', 'N/A')}:")
                logging.info(f"Raw diagnosis: '{disease_diagnosis}'")
                logging.info(f"Parsed disease_ids: {disease_ids}")
                # 反向映射回疾病名
                id_to_disease = {v: k for k, v in DISEASE2ID.items()}
                parsed_names = [id_to_disease.get(idx, f"UNKNOWN({idx})") for idx in disease_ids]
                logging.info(f"  Parsed diseases: {parsed_names}")

            # === 5. 构造最终样本 ===
            processed_data = {
                "id": original_data["id"],
                "patient_id": original_data["patient_id"],
                "image": original_data["image"],
                "data_type": original_data["data_type"],
                "category": "NeoCXR",
                "disease_ids": disease_ids,
                "conversations": [
                    {"from": "human", "value": human_prompt},
                    {"from": "gpt", "value": gpt_response}
                ]
            }
            # print(processed_data)

            conversation_datas.append(processed_data)
            loaded_counts[category] += 1

        logging.info(f"Loaded {loaded_counts[category]} valid samples from {path}. Skipped {skipped_image_count} samples due to missing images.")

    logging.info(f"Loaded a total of {len(conversation_datas)} samples.")
    return conversation_datas

def data_loader_NeoCXRClinic(data_path):
    """
    加载 NeoCXR 数据集，转换为标准 conversation 格式，使用外部 clinic_prompt.txt 模板 + structured clinical info。
    
    Args:
        data_path: JSON 文件路径（单个或 {path1, path2} 格式）
    """
    logging.info("using the NeoCXR loader (structured clinical + external prompt template).")

    image_folder = "/data_lyn/data/NeoCXR/images"
    prompt_clinic_path = "/data_lyn/MedEvalKit/utils/NeoCXR/prompt/clinic_prompt.txt"
    

    # === 读取外部 prompt 模板 ===
    try:
        with open(prompt_clinic_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read().strip()
        logging.info(f"Loaded prompt template from {prompt_clinic_path}")
    except Exception as e:
        logging.error(f"Failed to load prompt template: {e}")
        raise

    # 验证模板包含必要占位符
    required_placeholders = [
        "__IMAGE_TOKENS__",
        "__DEV_FACTORS__",
        "__PERINATAL_RISK__",
        "__PHYS_STATUS__",
    ]
    for ph in required_placeholders:
        if ph not in prompt_template:
            raise ValueError(f"Prompt template missing required placeholder: {ph}")

    # 解析 data_path
    if data_path.startswith('{') and data_path.endswith('}'):
        inner_paths = data_path[1:-1]
        paths = [p.strip() for p in inner_paths.split(',')]
        logging.info(f"Parsed multiple data paths: {paths}")
    else:
        paths = [data_path]
        logging.info(f"Using single data path: {data_path}")

    conversation_datas = []
    loaded_counts = {}

    for path in paths:
        logging.info(f"Loading dataset from {path}")
        category = "NeoCXR"
        loaded_counts[category] = 0
        skipped_image_count = 0

        try:
            with open(path, "r", encoding='utf-8') as f:
                dataset: List[Dict[str, Any]] = json.load(f)
        except Exception as e:
            logging.error(f"Error loading {path}: {e}")
            continue

        for original_data in dataset:
            # === 1. 图像路径检查 ===
            img_field = original_data.get("image", "").strip()
            if not img_field:
                skipped_image_count += 1
                continue

            image_paths = [p.strip() for p in img_field.split(",") if p.strip()]
            full_image_paths = [os.path.join(image_folder, p) for p in image_paths]

            if not all(os.path.exists(p) for p in full_image_paths):
                logging.warning(f"Missing image(s) for sample {original_data.get('id', 'N/A')}: {full_image_paths}. Skipping.")
                skipped_image_count += 1
                continue

            # === 2. 构建 structured clinical blocks ===
            clinical_structured = original_data.get("clinical_structured", {})
            dev = clinical_structured.get("development_factors", {})
            peri = clinical_structured.get("perinatal_risk", {})
            phys = clinical_structured.get("physiological_status", {})

            def format_section(d):
                if not d:
                    return "  (not provided)"
                lines = []
                for k, v in d.items():
                    lines.append(f"  {k}: {v}")
                return "\n".join(lines)

            developmental_block = format_section(dev)
            perinatal_block = format_section(peri)
            physiological_block = format_section(phys)

            # === 3. 插入模板 ===
            image_tokens = "".join(["<image>"] * len(image_paths))
            human_prompt = (
                prompt_template
                .replace("__IMAGE_TOKENS__", image_tokens)
                .replace("__DEV_FACTORS__", developmental_block)
                .replace("__PERINATAL_RISK__", perinatal_block)
                .replace("__PHYS_STATUS__", physiological_block)
            )

            # === 4. 构建 GPT 响应 ===
            imaging_conclusion = original_data.get("imaging_conclusion_en", "").strip()
            disease_diagnosis = original_data.get("disease_diagnosis_en", "").strip()
            gpt_response = f"Imaging conclusion: {imaging_conclusion} Disease diagnosis: {disease_diagnosis}"

            # === 4.1 解析多标签疾病 ===
            disease_ids = parse_disease_ids(disease_diagnosis)

            if loaded_counts[category] < 10:  # 只打印前 10 个样本（避免刷屏）
                logging.info(f"Sample {original_data.get('id', 'N/A')}:")
                logging.info(f"  Raw diagnosis: '{disease_diagnosis}'")
                logging.info(f"  Parsed disease_ids: {disease_ids}")
                # 反向映射回疾病名（便于人工核对）
                id_to_disease = {v: k for k, v in DISEASE2ID.items()}
                parsed_names = [id_to_disease.get(idx, f"UNKNOWN({idx})") for idx in disease_ids]
                logging.info(f"  Parsed diseases: {parsed_names}")

            # === 5. 构造最终样本 ===
            processed_data = {
                "id": original_data["id"],
                "patient_id": original_data["patient_id"],
                "image": original_data["image"],
                "data_type": original_data["data_type"],
                "category": "NeoCXR",
                "disease_ids": disease_ids,
                "conversations": [
                    {"from": "human", "value": human_prompt},
                    {"from": "gpt", "value": gpt_response}
                ]
            }

            conversation_datas.append(processed_data)
            loaded_counts[category] += 1

        logging.info(f"Loaded {loaded_counts[category]} valid samples from {path}. Skipped {skipped_image_count} samples due to missing images.")

    logging.info(f"Loaded a total of {len(conversation_datas)} samples.")
    return conversation_datas





def data_loader_five(data_path: str) -> List[Dict[str, Any]]:
    logging.info("using the five loader.")
    
    if data_path.startswith('{') and data_path.endswith('}'):
        inner_paths = data_path[1:-1]
        paths = [p.strip() for p in inner_paths.split(',')]
        logging.info(f"Parsed multiple data paths: {paths}")
    else:
        paths = [data_path]
        logging.info(f"Using single data path: {data_path}")

    print_sample = False
    conversation_datas = []
    

    loaded_counts = {}

    for path in paths:
        logging.info(f"Loading dataset from {path}")
        category = path.split('/')[-2]

        loaded_counts[category] = 0
                
        
        if category in ["IU_XRAY", "MIMIC_CXR"]:
            print(f"loading {category} (Report Generation)..")
            with open(path, "r") as f:
                dataset = json.load(f)
                for data in dataset:

                    if data.get('AP_image') is None or data.get('impression') is None or data.get('findings') is None:
                        continue
                        
                    conversation_data = RG_convert_to_conversation_format(data, print_sample,category)

                    if print_sample:
                        print_sample = False

                    conversation_datas.append(conversation_data)

                    loaded_counts[category] += 1

        elif category in ["VQA_RAD", "SLAKE", "PATH_VQA"]:

            if category == "VQA_RAD":
                print("loading VQA_RAD..")
                dataset_dataframe = pd.read_parquet(path)
                
                dataset_dataframe['image'] = dataset_dataframe['image'].apply(decode_image_from_hf_parquet_entry)
                dataset = dataset_dataframe.to_dict('records')
                
                if dataset:
                    for sample in dataset:
                        if isinstance(sample['image'], Image.Image):
                            sample['image'] = [sample['image']] 
                        else: 
                            sample['image'] = []
                            
                for data in dataset:
       
                    conversation_data = VQA_convert_to_conversation_format(data, print_sample,category)
                    if print_sample:
                        print_sample = False
                    conversation_datas.append(conversation_data)

                    loaded_counts[category] += 1


            elif category == "PATH_VQA":
                print("loading PATH_VQA..")

                parent_dir = os.path.dirname(path)
                search_pattern = os.path.join(parent_dir, "train*.parquet")
                parquet_files = glob.glob(search_pattern)
                parquet_files.sort()
                
                dataset = []
                if parquet_files:
                    dfs = [pd.read_parquet(f) for f in parquet_files]
                    dataset_dataframe = pd.concat(dfs, ignore_index=True)
                    dataset_dataframe['image'] = dataset_dataframe['image'].apply(decode_image_from_hf_parquet_entry)
                    dataset = dataset_dataframe.to_dict('records')

                    if dataset:
                        for sample in dataset:
                            if isinstance(sample['image'], Image.Image):
                                sample['image'] = [sample['image']] 
                            else:
                                sample['image'] = []
                else:
                    print(f"Warning: No 'train*.parquet' files found in directory: {parent_dir}")
                    
                for data in dataset:
                    conversation_data = VQA_convert_to_conversation_format(data, print_sample,category)
                    if print_sample:
                        print_sample = False
                    conversation_datas.append(conversation_data)
                    loaded_counts[category] += 1

            elif category == "SLAKE":
                print("loading SLAKE..")
                with open(path, "r") as f:
                    dataset = json.load(f)

                for data in dataset:
                    conversation_data = VQA_convert_to_conversation_format(data, print_sample,category)
                    if print_sample:
                        print_sample = False
                    conversation_datas.append(conversation_data)

                    loaded_counts[category] += 1
        
        else:
            print(f"loading uncategorized data from {category}..")
            with open(path, "r") as f:
                dataset = json.load(f)

                conversation_datas.extend(dataset)
                loaded_counts[category] += len(dataset) 


    logging.info(f"Loaded a total of {len(conversation_datas)} samples from {len(paths)} file(s).")
    

    print("\n--- 有效样本加载统计 ---")
    total_loaded = 0
    for category, count in loaded_counts.items():
        print(f"  {category:<10}: {count} 条")
        total_loaded += count
    print(f"----------------------")
    print(f"  总计加载:   {total_loaded} 条 (与 logging.info 保持一致)")
    print("----------------------\n")
    
    return conversation_datas




def data_loader_NeoCXRClinicTokenMix(data_path: str) -> List[Dict[str, Any]]:
    logging.info("using the Mix loader.")

    def format_neocxr_section(d):
        if not d:
            return "  (not provided)"
        lines = []
        for k, v in d.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
    
    if data_path.startswith('{') and data_path.endswith('}'):
        inner_paths = data_path[1:-1]
        paths = [p.strip() for p in inner_paths.split(',')]
        logging.info(f"Parsed multiple data paths: {paths}")
    else:
        paths = [data_path]
        logging.info(f"Using single data path: {data_path}")

    print_sample = False
    conversation_datas = []
    

    loaded_counts = {}

    for path in paths:
        logging.info(f"Loading dataset from {path}")
        category = path.split('/')[-2]


        # [修正] 防止同一个category有多个文件时计数器被重置
        if category not in loaded_counts:
            loaded_counts[category] = 0

        # loaded_counts[category] = 0
                
        
        if category in ["IU_XRAY", "MIMIC_CXR"]:
            print(f"loading {category} (Report Generation)..")
            with open(path, "r") as f:
                dataset = json.load(f)
                for data in dataset:

                    if data.get('AP_image') is None or data.get('impression') is None or data.get('findings') is None:
                        continue
                        
                    conversation_data = RG_convert_to_conversation_format(data, print_sample,category)

                    if print_sample:
                        print_sample = False

                    conversation_datas.append(conversation_data)

                    loaded_counts[category] += 1

        elif category in ["NeoCXR"]:
            logging.info(f"Processing NeoCXR logic for {path}")
            
            # --- 1. 配置路径 (根据原始逻辑硬编码) ---
            image_folder = "/data_lyn/data/NeoCXR/images"
            prompt_clinic_path = "/data_lyn/MedEvalKit/utils/NeoCXR/prompt/clinic_prompt_token.txt"
            
            # --- 2. 加载 Prompt 模板 ---
            try:
                with open(prompt_clinic_path, 'r', encoding='utf-8') as f_p:
                    prompt_template = f_p.read().strip()
            except Exception as e:
                logging.error(f"Failed to load NeoCXR prompt template from {prompt_clinic_path}: {e}")
                # 如果模板加载失败，跳过该文件的处理
                continue 

            # --- 3. 加载数据集 JSON ---
            try:
                with open(path, "r", encoding='utf-8') as f:
                    dataset = json.load(f)
            except Exception as e:
                logging.error(f"Error loading json {path}: {e}")
                continue

            skipped_image_count = 0

            # --- 4. 遍历处理样本 ---
            for original_data in dataset:
                # A. 图像路径检查
                img_field = original_data.get("image", "").strip()
                if not img_field:
                    skipped_image_count += 1
                    continue

                image_paths = [p.strip() for p in img_field.split(",") if p.strip()]
                full_image_paths = [os.path.join(image_folder, p) for p in image_paths]

                if not all(os.path.exists(p) for p in full_image_paths):
                    # 只有在需要调试时取消注释，避免日志过多
                    # logging.warning(f"Missing image(s): {full_image_paths}. Skipping.")
                    skipped_image_count += 1
                    continue

                # B. 构建 structured clinical blocks
                clinical_structured = original_data.get("clinical_structured", {})
                dev = clinical_structured.get("development_factors", {})
                peri = clinical_structured.get("perinatal_risk", {})
                phys = clinical_structured.get("physiological_status", {})

                developmental_block = format_neocxr_section(dev)
                perinatal_block = format_neocxr_section(peri)
                physiological_block = format_neocxr_section(phys)

                # C. 插入模板 (Prompt Engineering) not provided
                # image_tokens = "".join(["<image>"] * len(image_paths))
                # try:
                #     human_prompt = (
                #         prompt_template
                #         .replace("__IMAGE_TOKENS__", image_tokens)
                #         .replace("__DEV_FACTORS__", developmental_block)
                #         .replace("__PERINATAL_RISK__", perinatal_block)
                #         .replace("__PHYS_STATUS__", physiological_block)
                #     )
                # except Exception as e:
                #     logging.error(f"Error constructing prompt for sample {original_data.get('id')}: {e}")
                #     continue

                image_tokens = "".join(["<image>"] * len(image_paths))
                try:
                    human_prompt = (
                        prompt_template
                        .replace("__IMAGE_TOKENS__", image_tokens)
                        .replace("__DEV_FACTORS__", "developmental_block")
                        .replace("__PERINATAL_RISK__", "perinatal_block")
                        .replace("__PHYS_STATUS__", "physiological_block")
                    )
                except Exception as e:
                    logging.error(f"Error constructing prompt for sample {original_data.get('id')}: {e}")
                    continue

                # D. 构建 GPT 响应
                imaging_conclusion = original_data.get("imaging_conclusion_en", "").strip()
                disease_diagnosis = original_data.get("disease_diagnosis_en", "").strip()
                gpt_response = f"Imaging conclusion: {imaging_conclusion} Disease diagnosis: {disease_diagnosis}"

                # E. 解析多标签疾病 (假设 parse_disease_ids 已定义)
                disease_ids = parse_disease_ids(disease_diagnosis)

                # F. 构造最终样本
                processed_data = {
                    "id": original_data.get("id"),
                    "patient_id": original_data.get("patient_id"),
                    "image": original_data.get("image"),
                    "data_type": original_data.get("data_type", "NeoCXR"),
                    "category": "NeoCXR",
                    "disease_ids": disease_ids,
                    "conversations": [
                        {"from": "human", "value": human_prompt},
                        {"from": "gpt", "value": gpt_response}
                    ]
                }

                conversation_datas.append(processed_data)
                loaded_counts[category] += 1
            
            logging.info(f"  NeoCXR: Loaded {loaded_counts[category]} valid samples. Skipped {skipped_image_count} missing images.")
   
        
        else:
            print(f"loading uncategorized data from {category}..")
            with open(path, "r") as f:
                dataset = json.load(f)

                conversation_datas.extend(dataset)
                loaded_counts[category] += len(dataset) 


    logging.info(f"Loaded a total of {len(conversation_datas)} samples from {len(paths)} file(s).")
    

    print("\n--- 有效样本加载统计 ---")
    total_loaded = 0
    for category, count in loaded_counts.items():
        print(f"  {category:<10}: {count} 条")
        total_loaded += count
    print(f"----------------------")
    print(f"  总计加载:   {total_loaded} 条 (与 logging.info 保持一致)")
    print("----------------------\n")
    
    return conversation_datas

def data_loader_mimic_cxr_all_frontal_findings(data_path):
    logging.info("using the MIMIC-CXR loader: all frontal findings.")
    with open(data_path) as f:
        dataset = json.load(f)
    ret = []
    for d in dataset:
        # Skip empty findings
        if not isinstance(d["conversations"][1]["value"], str):
            continue
        if d['view'] in ('AP', 'PA'):
            ret.append(d)
    logging.info(f"loaded {len(ret)}/{len(dataset)} samples.")
    return ret


def data_loader_mimic_cxr_all_views_findings(data_path):
    logging.info("using the MIMIC-CXR loader: all views findings.")
    with open(data_path) as f:
        dataset = json.load(f)
    ret = []
    for d in dataset:
        # Skip empty findings
        if not isinstance(d["conversations"][1]["value"], str):
            continue
        view = d["view"] if isinstance(d["view"], str) else "Unknown"
        d["conversations"][0]["value"] = f"<image>\nGiven the chest X-ray image from {view} view, describe the findings in the image: "
        ret.append(d)
    logging.info(f"loaded {len(ret)}/{len(dataset)} samples.")
    return ret


def data_loader_mimic_reason_findings(data_path, split):
    logging.info(f"using the MIMIC-CXR loader: MIMIC {split}.")
    with open(data_path) as f:
        dataset = json.load(f)
    ret = []
    for d in dataset:
        if split == 'test' and d['generate_method'] != 'rule-based':
            continue
        # Skip empty findings
        if not isinstance(d["conversations"][1]["value"], str):
            continue
        if d['view'] not in ('AP', 'PA'):
            continue
        if d['image'].startswith("mimic/"):
            d['image'] = d['image'][len('mimic/'):]
        if d['reason'] is not None:
            reason = d['reason'].replace('\n', ' ')
            d['conversations'][0]['value'] = f"<image>\nProvide a description of the findings in the radiology image given the following indication: {reason}"
        else:
            d['conversations'][0]['value'] = f"<image>\nProvide a description of the findings in the radiology image."
        ret.append(d)
    logging.info(f"loaded {len(ret)}/{len(dataset)} samples.")
    return ret


data_loaders = {
    "default": data_loader_default,
    "Five":data_loader_five,
    # "Five_test":data_loader_five(path,"test"),
    "NeoCXR":data_loader_NeoCXR,
    "NeoCXRNoneClinic":data_loader_NeoCXRNoneClinic,
    "NeoCXRClinic": data_loader_NeoCXRClinic,
    "NeoCXRClinicToken": data_loader_NeoCXRClinicToken,
    "NeoCXRClinicTokenMix": data_loader_NeoCXRClinicTokenMix,
    "mimic_train_findings": lambda x: data_loader_mimic_reason_findings(x, "train"),
    "mimic_test_findings": lambda x: data_loader_mimic_reason_findings(x, "test"),
    "mimic_cxr_all_frontal_findings": data_loader_mimic_cxr_all_frontal_findings,
    "mimic_cxr_all_views_findings": data_loader_mimic_cxr_all_views_findings,
}