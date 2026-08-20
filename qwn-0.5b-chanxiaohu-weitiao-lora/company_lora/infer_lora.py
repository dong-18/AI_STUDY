import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_CHECKPOINT = (
    SCRIPT_DIR / "output" / "company-qwen-lora-v2" / "checkpoint-228"
)

SYSTEM_PROMPT = """你是公司制度问答助手。

回答要求：
1. 只能根据用户提供的制度资料回答。
2. 不得编造制度中没有的金额、日期、条件或流程。
3. 回答时尽量说明制度名称或条款。
4. 如果资料中没有答案，明确说明“现有制度资料中没有找到相关规定”。
5. 回答应准确、简洁，不要加入与问题无关的内容。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="加载指定 LoRA checkpoint，进行单条或交互式公司制度问答。"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"LoRA checkpoint 目录，默认：{DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="基础模型名称或本地目录。",
    )
    context_group = parser.add_mutually_exclusive_group()
    context_group.add_argument(
        "--context",
        help="直接在命令行提供制度资料。",
    )
    context_group.add_argument(
        "--context-file",
        type=Path,
        help="从 UTF-8 文本文件读取制度资料。",
    )
    parser.add_argument(
        "--question",
        help="要验证的问题；省略时进入交互模式。",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="推理设备，auto 优先使用 CUDA。",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=448,
        help="system、context 和 question 的最大输入 token 数。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="模型最多生成的 token 数。",
    )
    return parser.parse_args()


def resolve_device(device_argument: str) -> torch.device:
    if device_argument == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前环境没有检测到可用 GPU。")
        return torch.device("cuda")
    if device_argument == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_checkpoint(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"找不到 checkpoint 目录：{checkpoint}")
    adapter_config = checkpoint / "adapter_config.json"
    if not adapter_config.is_file():
        raise FileNotFoundError(
            f"目录中缺少 adapter_config.json，不是有效的 LoRA checkpoint：{checkpoint}"
        )
    return checkpoint


def load_model(
    checkpoint: Path,
    base_model_name: str,
    device: torch.device,
):
    tokenizer_source = (
        str(checkpoint)
        if (checkpoint / "tokenizer_config.json").is_file()
        else base_model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model = PeftModel.from_pretrained(base_model, str(checkpoint))
    model.eval()
    model.config.use_cache = True
    return tokenizer, model


def build_messages(context: str, question: str) -> list[dict[str, str]]:
    user_content = (
        "请根据下面的公司制度资料回答问题。\n\n"
        f"【制度资料】\n{context}\n\n"
        f"【员工问题】\n{question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def answer_question(
    tokenizer,
    model,
    context: str,
    question: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> tuple[str, int, int]:
    empty_context_ids = tokenizer.apply_chat_template(
        build_messages("", question),
        tokenize=True,
        add_generation_prompt=True,
    )
    context_budget = max_input_tokens - len(empty_context_ids)
    if context_budget <= 0:
        raise ValueError(
            "system prompt 和问题已超过 --max-input-tokens，请缩短问题或增大限制。"
        )

    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    if len(context_ids) > context_budget:
        context_ids = context_ids[:context_budget]
        context = tokenizer.decode(context_ids, skip_special_tokens=True)

    model_inputs = tokenizer.apply_chat_template(
        build_messages(context, question),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    while model_inputs["input_ids"].shape[1] > max_input_tokens:
        context_ids = context_ids[:-8]
        if not context_ids:
            raise ValueError("无法在保留完整问题的情况下压缩制度资料。")
        context = tokenizer.decode(context_ids, skip_special_tokens=True)
        model_inputs = tokenizer.apply_chat_template(
            build_messages(context, question),
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
    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            generation_config=generation_config,
        )

    input_tokens = model_inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_tokens:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer, input_tokens, len(generated_ids)


def read_context(args: argparse.Namespace) -> str | None:
    if args.context is not None:
        return args.context.strip()
    if args.context_file is not None:
        path = args.context_file.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到制度资料文件：{path}")
        return path.read_text(encoding="utf-8").strip()
    return None


def print_answer(
    tokenizer,
    model,
    context: str,
    question: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> None:
    answer, input_tokens, output_tokens = answer_question(
        tokenizer=tokenizer,
        model=model,
        context=context,
        question=question,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
    )
    print(f"\n回答：{answer}")
    print(f"[输入 token：{input_tokens}，生成 token：{output_tokens}]")


def interactive_loop(
    tokenizer,
    model,
    fixed_context: str | None,
    max_input_tokens: int,
    max_new_tokens: int,
) -> None:
    print("\n进入交互验证，输入 exit 退出。")
    if fixed_context:
        print("已使用命令行提供的固定制度资料。")

    while True:
        if fixed_context is None:
            context = input("\n制度资料：").strip()
            if context.lower() == "exit":
                return
            if not context:
                continue
        else:
            context = fixed_context

        question = input("员工问题：").strip()
        if question.lower() == "exit":
            return
        if not question:
            continue

        print_answer(
            tokenizer,
            model,
            context,
            question,
            max_input_tokens,
            max_new_tokens,
        )


def main() -> None:
    args = parse_args()
    if args.max_input_tokens <= 0 or args.max_new_tokens <= 0:
        raise ValueError("token 长度参数必须大于 0。")
    if args.question and not (args.context or args.context_file):
        raise ValueError("使用 --question 时，还必须提供 --context 或 --context-file。")

    checkpoint = validate_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    context = read_context(args)

    print(f"checkpoint：{checkpoint}")
    print(f"基础模型：{args.base_model}")
    print(f"设备：{device}")
    tokenizer, model = load_model(
        checkpoint=checkpoint,
        base_model_name=args.base_model,
        device=device,
    )
    print("模型加载完成。")

    if args.question:
        print_answer(
            tokenizer,
            model,
            context=context or "",
            question=args.question,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        interactive_loop(
            tokenizer,
            model,
            fixed_context=context,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
