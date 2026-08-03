"""
评估原始 Qwen 模型和 LoRA 模型。

测试集默认读取 test.jsonl，每行格式示例：

{"id":"leave_001","question":"请事假要提前多久？","reference_answer":"员工请事假应至少提前一个工作日提交申请。","answerable":true,"keywords":["一个工作日"]}
{"id":"unknown_001","question":"婚假可以休几天？","reference_answer":"暂时无法确认该制度，请咨询人力资源部门。","answerable":false,"keywords":["无法确认","人力资源"]}

字段说明：
    question:         必填，模型需要回答的问题。
    reference_answer: 必填，人工编写的参考答案。
    answerable:       可选，制度是否包含答案，默认 true。
    keywords:         可选，正确答案应该包含的关键信息列表。
    id:               可选，样本编号；缺省时自动生成。

运行示例：
    # 同时评估原始模型与 LoRA
    python evaluate.py --models base lora

    # 只评估 LoRA，并指定测试集和结果文件
    python evaluate.py --models lora --test-file ./test.jsonl \
        --output-file ./evaluation_results.json

说明：
    自动指标只能帮助稳定地比较不同实验，不能完全判断事实正确性。
    正式项目还应抽样进行人工评审，重点检查事实错误与无依据编造。
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# 使用绝对基准目录，避免必须在 company_lora 目录中执行脚本。
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_LORA_PATH = SCRIPT_DIR / "output" / "company-qwen-lora"
DEFAULT_TEST_FILE = SCRIPT_DIR / "test.jsonl"
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "evaluation_results.json"

# 必须与训练、推理时的 system prompt 保持一致，否则对比结果不公平。
SYSTEM_PROMPT = """你是公司制度问答助手。

请根据训练中学习到的公司制度回答员工问题。
回答应准确、简洁。
当你无法确定答案时，回答：
“暂时无法确认该制度，请咨询人力资源部门。”
"""

# 用于判断模型是否选择了“拒答”。实际项目可根据业务话术继续扩展。
REFUSAL_PATTERNS = (
    "无法确认",
    "无法确定",
    "没有找到",
    "未找到",
    "不清楚",
    "不知道",
    "请咨询人力资源",
    "请咨询hr",
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="评估原始 Qwen 与 LoRA 公司制度问答模型。"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("base", "lora"),
        default=("base", "lora"),
        help="需要评估的模型，默认依次评估 base 和 lora。",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Hugging Face 基础模型名称或本地路径。",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=DEFAULT_LORA_PATH,
        help="LoRA 适配器目录。",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=DEFAULT_TEST_FILE,
        help="JSONL 格式的独立测试集。",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="保存完整评估结果的 JSON 文件。",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="推理设备。auto 会优先使用 CUDA。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="每个回答最多生成的 token 数。",
    )
    return parser.parse_args()


def load_test_data(file_path: Path) -> list[dict[str, Any]]:
    """读取并校验 JSONL 测试集，尽早暴露数据格式问题。"""
    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到测试集：{file_path}\n"
            "请先创建 test.jsonl，格式可参考 evaluate.py 顶部注释。"
        )

    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{file_path} 第 {line_number} 行不是合法 JSON：{exc}"
                ) from exc

            missing_fields = {"question", "reference_answer"} - record.keys()
            if missing_fields:
                raise ValueError(
                    f"{file_path} 第 {line_number} 行缺少字段："
                    f"{sorted(missing_fields)}"
                )

            question = str(record["question"]).strip()
            reference_answer = str(record["reference_answer"]).strip()
            if not question or not reference_answer:
                raise ValueError(
                    f"{file_path} 第 {line_number} 行的问题或参考答案为空。"
                )

            keywords = record.get("keywords", [])
            if not isinstance(keywords, list):
                raise ValueError(
                    f"{file_path} 第 {line_number} 行的 keywords 必须是列表。"
                )

            records.append(
                {
                    "id": str(record.get("id", f"sample_{line_number:04d}")),
                    "question": question,
                    "reference_answer": reference_answer,
                    "answerable": bool(record.get("answerable", True)),
                    "keywords": [
                        str(keyword).strip()
                        for keyword in keywords
                        if str(keyword).strip()
                    ],
                }
            )

    if not records:
        raise ValueError(f"测试集为空：{file_path}")

    return records


def normalize_text(text: str) -> str:
    """
    对文本做轻量归一化。

    中文答案经常只在空格、标点上不同。评估时去除这类差异，避免把
    “一个工作日。”和“一个工作日”错误地判断成完全不同。
    """
    normalized_chars = []
    for char in text.lower().strip():
        category = unicodedata.category(char)
        if char.isspace() or category.startswith(("P", "S")):
            continue
        normalized_chars.append(char)
    return "".join(normalized_chars)


def exact_match(prediction: str, reference: str) -> float:
    """归一化后的答案是否完全一致。"""
    return float(normalize_text(prediction) == normalize_text(reference))


def character_f1(prediction: str, reference: str) -> float:
    """
    计算字符级 F1。

    中文没有天然空格分词，字符级 F1 不依赖额外分词库，适合作为轻量基线。
    """
    prediction_chars = list(normalize_text(prediction))
    reference_chars = list(normalize_text(reference))

    if not prediction_chars or not reference_chars:
        return float(prediction_chars == reference_chars)

    common = Counter(prediction_chars) & Counter(reference_chars)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_chars)
    recall = overlap / len(reference_chars)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """用最长公共子序列计算字符级 ROUGE-L F1。"""
    prediction_text = normalize_text(prediction)
    reference_text = normalize_text(reference)

    if not prediction_text or not reference_text:
        return float(prediction_text == reference_text)

    # 一维动态规划可减少内存占用。
    dp = [0] * (len(reference_text) + 1)
    for prediction_char in prediction_text:
        previous_diagonal = 0
        for index, reference_char in enumerate(reference_text, start=1):
            previous_up = dp[index]
            if prediction_char == reference_char:
                dp[index] = previous_diagonal + 1
            else:
                dp[index] = max(dp[index], dp[index - 1])
            previous_diagonal = previous_up

    lcs_length = dp[-1]
    precision = lcs_length / len(prediction_text)
    recall = lcs_length / len(reference_text)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_recall(prediction: str, keywords: list[str]) -> float | None:
    """计算参考关键词中有多少出现在模型回答里；没有关键词时返回 None。"""
    if not keywords:
        return None

    normalized_prediction = normalize_text(prediction)
    hits = sum(
        normalize_text(keyword) in normalized_prediction for keyword in keywords
    )
    return hits / len(keywords)


def is_refusal(answer: str) -> bool:
    """根据拒答关键词判断模型是否拒绝回答。"""
    normalized_answer = re.sub(r"\s+", "", answer.lower())
    return any(pattern in normalized_answer for pattern in REFUSAL_PATTERNS)


def resolve_device(device_argument: str) -> torch.device:
    """确定实际推理设备。"""
    if device_argument == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前环境没有检测到可用 GPU。")
        return torch.device("cuda")
    if device_argument == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    model_variant: str,
    base_model_name: str,
    lora_path: Path,
    device: torch.device,
):
    """加载基础模型，必要时再挂载 LoRA 适配器。"""
    tokenizer_path = str(lora_path) if model_variant == "lora" else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CPU 不适合使用 FP16，因此回退到 FP32。
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)

    if model_variant == "lora":
        if not lora_path.exists():
            raise FileNotFoundError(f"找不到 LoRA 适配器：{lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path))

    model.eval()
    model.config.use_cache = True
    return tokenizer, model


def generate_answer(
    tokenizer,
    model,
    question: str,
    max_new_tokens: int,
) -> tuple[str, float]:
    """生成一个答案，同时返回本次生成耗时。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    model_inputs = {
        key: value.to(model.device) for key, value in model_inputs.items()
    }

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if model.device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            generation_config=generation_config,
        )

    if model.device.type == "cuda":
        torch.cuda.synchronize()
    latency_seconds = time.perf_counter() - start_time

    input_length = model_inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_length:]
    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()
    return answer, latency_seconds


def summarize_results(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总逐题指标，分别关注回答质量和拒答能力。"""
    keyword_scores = [
        sample["metrics"]["keyword_recall"]
        for sample in samples
        if sample["metrics"]["keyword_recall"] is not None
    ]
    answerable_samples = [sample for sample in samples if sample["answerable"]]
    unanswerable_samples = [
        sample for sample in samples if not sample["answerable"]
    ]

    # 对无答案问题，拒答是正确行为。
    refusal_accuracy = (
        mean(sample["metrics"]["is_refusal"] for sample in unanswerable_samples)
        if unanswerable_samples
        else None
    )

    # 对有答案问题，拒答属于错误行为。
    false_refusal_rate = (
        mean(sample["metrics"]["is_refusal"] for sample in answerable_samples)
        if answerable_samples
        else None
    )

    return {
        "sample_count": len(samples),
        "answerable_count": len(answerable_samples),
        "unanswerable_count": len(unanswerable_samples),
        "exact_match": mean(
            sample["metrics"]["exact_match"] for sample in samples
        ),
        "character_f1": mean(
            sample["metrics"]["character_f1"] for sample in samples
        ),
        "rouge_l": mean(sample["metrics"]["rouge_l"] for sample in samples),
        "keyword_recall": mean(keyword_scores) if keyword_scores else None,
        "refusal_accuracy": refusal_accuracy,
        "false_refusal_rate": false_refusal_rate,
        "average_latency_seconds": mean(
            sample["latency_seconds"] for sample in samples
        ),
    }


def evaluate_variant(
    model_variant: str,
    records: list[dict[str, Any]],
    base_model_name: str,
    lora_path: Path,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    """评估一种模型，并返回可序列化的完整结果。"""
    print(f"\n正在加载 {model_variant} 模型……")
    tokenizer, model = load_model(
        model_variant=model_variant,
        base_model_name=base_model_name,
        lora_path=lora_path,
        device=device,
    )

    samples: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        prediction, latency_seconds = generate_answer(
            tokenizer=tokenizer,
            model=model,
            question=record["question"],
            max_new_tokens=max_new_tokens,
        )
        refusal = is_refusal(prediction)

        sample_result = {
            **record,
            "prediction": prediction,
            "latency_seconds": latency_seconds,
            "metrics": {
                "exact_match": exact_match(
                    prediction, record["reference_answer"]
                ),
                "character_f1": character_f1(
                    prediction, record["reference_answer"]
                ),
                "rouge_l": rouge_l(
                    prediction, record["reference_answer"]
                ),
                "keyword_recall": keyword_recall(
                    prediction, record["keywords"]
                ),
                "is_refusal": refusal,
                "refusal_is_correct": refusal == (not record["answerable"]),
            },
        }
        samples.append(sample_result)
        print(
            f"[{index}/{len(records)}] {record['id']} | "
            f"F1={sample_result['metrics']['character_f1']:.3f} | "
            f"{latency_seconds:.2f}s"
        )

    result = {
        "model": model_variant,
        "summary": summarize_results(samples),
        "samples": samples,
    }

    # 两个模型依次评估时，主动释放显存，避免 GTX 1060 3GB 显存不足。
    del model
    del tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def print_summary(result: dict[str, Any]) -> None:
    """在终端打印便于快速比较的指标。"""
    summary = result["summary"]

    def format_metric(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    print(f"\n===== {result['model'].upper()} 评估结果 =====")
    print(f"样本数：          {summary['sample_count']}")
    print(f"Exact Match：     {format_metric(summary['exact_match'])}")
    print(f"字符级 F1：       {format_metric(summary['character_f1'])}")
    print(f"ROUGE-L：         {format_metric(summary['rouge_l'])}")
    print(f"关键词召回率：    {format_metric(summary['keyword_recall'])}")
    print(f"拒答准确率：      {format_metric(summary['refusal_accuracy'])}")
    print(f"错误拒答率：      {format_metric(summary['false_refusal_rate'])}")
    print(
        "平均生成延迟：    "
        f"{format_metric(summary['average_latency_seconds'])} 秒"
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    records = load_test_data(args.test_file.resolve())

    print(f"测试集：{args.test_file.resolve()}")
    print(f"样本数：{len(records)}")
    print(f"设备：{device}")

    results = {
        "config": {
            "base_model": args.base_model,
            "lora_path": str(args.lora_path.resolve()),
            "test_file": str(args.test_file.resolve()),
            "device": str(device),
            "max_new_tokens": args.max_new_tokens,
        },
        "evaluations": [],
    }

    for model_variant in args.models:
        evaluation = evaluate_variant(
            model_variant=model_variant,
            records=records,
            base_model_name=args.base_model,
            lora_path=args.lora_path.resolve(),
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        results["evaluations"].append(evaluation)
        print_summary(evaluation)

    output_file = args.output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(f"\n完整评估结果已保存到：{output_file}")


if __name__ == "__main__":
    main()