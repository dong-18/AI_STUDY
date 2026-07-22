import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


# ============================================================
# 基本配置
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_FILE = "./train.jsonl"
OUTPUT_DIR = "./output/company-qwen-lora"

# GTX 1060 3GB 建议从 384 或 512 开始
MAX_LENGTH = 512

SYSTEM_PROMPT = """你是公司制度问答助手。

回答要求：
1. 只能根据用户提供的制度资料回答。
2. 不得编造制度中没有的金额、日期、条件或流程。
3. 回答时尽量说明制度名称或条款。
4. 如果资料中没有答案，明确说明“现有制度资料中没有找到相关规定”。
5. 回答应准确、简洁，不要加入与问题无关的内容。
"""


# ============================================================
# 读取 JSONL
# ============================================================

def load_jsonl(file_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到训练文件：{file_path}")

    records: List[Dict[str, str]] = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"train.jsonl 第 {line_number} 行不是合法 JSON：{exc}"
                ) from exc

            required_fields = {"context", "question", "answer"}
            missing_fields = required_fields - record.keys()

            if missing_fields:
                raise ValueError(
                    f"train.jsonl 第 {line_number} 行缺少字段："
                    f"{sorted(missing_fields)}"
                )

            records.append(
                {
                    "context": str(record["context"]).strip(),
                    "question": str(record["question"]).strip(),
                    "answer": str(record["answer"]).strip(),
                }
            )

    if not records:
        raise ValueError("训练文件为空，没有可用数据。")

    return records


# ============================================================
# Dataset
# ============================================================

class CompanyPolicyDataset(Dataset):
    """
    将每条数据转换为：

    system:
        回答规则

    user:
        制度资料 + 问题

    assistant:
        标准答案

    labels 中把 system 和 user 部分设成 -100，
    只让 assistant 答案参与 loss。
    """

    def __init__(
        self,
        records: List[Dict[str, str]],
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        self.samples: List[Dict[str, List[int]]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for index, record in enumerate(records):
            sample = self._encode_record(record)

            if sample is None:
                print(f"跳过第 {index + 1} 条：答案在截断后没有有效 token。")
                continue

            self.samples.append(sample)

        if not self.samples:
            raise ValueError(
                "没有生成有效训练样本。请增大 MAX_LENGTH，"
                "或者缩短 context。"
            )

    def _encode_record(
        self,
        record: Dict[str, str],
    ) -> Dict[str, List[int]] | None:

        user_content = (
            "请根据下面的公司制度资料回答问题。\n\n"
            f"【制度资料】\n{record['context']}\n\n"
            f"【员工问题】\n{record['question']}"
        )

        # 只有 system + user，用于计算回答开始的位置
        prompt_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        # 完整对话：system + user + assistant
        full_messages = [
            *prompt_messages,
            {
                "role": "assistant",
                "content": record["answer"],
            },
        ]

        # add_generation_prompt=True 会在结尾加入 assistant 起始标记
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 完整训练文本已经包含 assistant，不再添加生成提示
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        full_encoding = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = full_encoding["input_ids"]
        attention_mask = full_encoding["attention_mask"]

        # labels 初始等于 input_ids
        labels = input_ids.copy()

        # system、user、assistant 起始标记不计算 loss
        prompt_length = min(len(prompt_ids), len(labels))

        for position in range(prompt_length):
            labels[position] = -100

        # 若答案全部被截断，则跳过
        if all(label == -100 for label in labels):
            return None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.samples[index]


# ============================================================
# 动态 padding
# ============================================================

@dataclass
class CausalLMDataCollator:
    tokenizer: AutoTokenizer
    label_pad_token_id: int = -100

    def __call__(
        self,
        features: List[Dict[str, List[int]]],
    ) -> Dict[str, torch.Tensor]:

        input_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        max_length = batch["input_ids"].shape[1]

        padded_labels = []

        for feature in features:
            labels = feature["labels"]
            padding_length = max_length - len(labels)

            # Qwen通常使用右侧padding
            padded = labels + [self.label_pad_token_id] * padding_length
            padded_labels.append(padded)

        batch["labels"] = torch.tensor(
            padded_labels,
            dtype=torch.long,
        )

        return batch


# ============================================================
# 主训练流程
# ============================================================

def main() -> None:
    set_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "没有检测到 CUDA GPU。请检查 NVIDIA 驱动和 CUDA 版 PyTorch。"
        )

    print("CUDA设备：", torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    # 某些 causal LM 没有 pad_token，这里使用 eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # --------------------------------------------------------
    # FP16加载基础模型
    # --------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # 不使用 device_map="auto"，单GPU直接放到cuda
    model = model.to("cuda")

    # 训练开启gradient checkpointing时必须关闭cache
    model.config.use_cache = False

    # 降低激活值显存占用
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        }
    )

    # 输入embedding需要梯度传递，配合checkpointing更稳妥
    model.enable_input_require_grads()

    # --------------------------------------------------------
    # 配置LoRA
    # --------------------------------------------------------

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",

        # 低秩维度
        r=8,

        # LoRA缩放系数，实际缩放约为 alpha / r
        lora_alpha=16,

        lora_dropout=0.05,

        # 不训练原模型bias
        bias="none",

        # 对Qwen2注意力投影层添加LoRA
        # 为节省3GB显存，先只训练q_proj和v_proj
        target_modules=[
            "q_proj",
            "v_proj",
        ],
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    # --------------------------------------------------------
    # 数据
    # --------------------------------------------------------

    records = load_jsonl(TRAIN_FILE)

    train_dataset = CompanyPolicyDataset(
        records=records,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    print(f"原始数据数量：{len(records)}")
    print(f"有效训练样本：{len(train_dataset)}")

    data_collator = CausalLMDataCollator(
        tokenizer=tokenizer,
    )

    # --------------------------------------------------------
    # 训练参数
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # 3GB显存必须从1开始
        per_device_train_batch_size=1,

        # 累计8次后更新一次，等效batch约为8
        gradient_accumulation_steps=8,

        # LoRA常用学习率通常比全参数微调大
        learning_rate=2e-4,

        num_train_epochs=3,

        # 使用FP16混合精度训练
        fp16=True,
        bf16=False,

        # 不使用bitsandbytes优化器
        optim="adamw_torch",

        # 梯度裁剪
        max_grad_norm=1.0,

        # 学习率预热
        warmup_ratio=0.05,

        # 余弦学习率调度
        lr_scheduler_type="cosine",

        logging_steps=1,

        # 每个epoch保存一次
        save_strategy="epoch",

        # 最多保留两个checkpoint
        save_total_limit=2,

        # 数据集已自行处理字段
        remove_unused_columns=False,

        # 减少额外日志依赖
        report_to="none",

        # 与上面模型设置一致
        gradient_checkpointing=True,

        # Windows下建议0，避免多进程数据加载问题
        dataloader_num_workers=0,

        # 允许TF32仅适用于Ampere等较新显卡；
        # GTX 1060不支持，因此关闭
        tf32=False,

        seed=42,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # --------------------------------------------------------
    # 开始训练
    # --------------------------------------------------------

    print("开始训练……")

    train_result = trainer.train()

    print("训练完成。")
    print(train_result)

    # --------------------------------------------------------
    # 保存LoRA适配器和tokenizer
    # --------------------------------------------------------

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"LoRA适配器已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()