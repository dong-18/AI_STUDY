import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from peft import PeftModel


BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "./output/company-qwen-lora"

# SYSTEM_PROMPT = """你是公司制度问答助手。
#
# 回答要求：
# 1. 只能根据用户提供的制度资料回答。
# 2. 不得编造制度中没有的金额、日期、条件或流程。
# 3. 回答时尽量说明制度名称或条款。
# 4. 如果资料中没有答案，明确说明“现有制度资料中没有找到相关规定”。
# 5. 回答应准确、简洁，不要加入与问题无关的内容。
# """
SYSTEM_PROMPT = """你是公司制度问答助手。

请根据训练中学习到的公司制度回答员工问题。
回答应准确、简洁。
当你无法确定答案时，回答：
“暂时无法确认该制度，请咨询人力资源部门。”
"""

def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到CUDA GPU。")

    tokenizer = AutoTokenizer.from_pretrained(
        LORA_PATH,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    base_model = base_model.to("cuda")

    # 加载训练得到的LoRA适配器
    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
    )

    model.eval()
    model.config.use_cache = True

    return tokenizer, model


# def answer_question(
#     tokenizer,
#     model,
#     context: str,
#     question: str,
# ) -> str:
#
#     user_content = (
#         "请根据下面的公司制度资料回答问题。\n\n"
#         f"【制度资料】\n{context}\n\n"
#         f"【员工问题】\n{question}"
#     )
#
#     messages = [
#         {
#             "role": "system",
#             "content": SYSTEM_PROMPT,
#         },
#         {
#             "role": "user",
#             "content": user_content,
#         },
#     ]
#
#     model_inputs = tokenizer.apply_chat_template(
#         messages,
#         tokenize=True,
#         add_generation_prompt=True,
#         return_tensors="pt",
#         return_dict=True,
#     )
#
#     model_inputs = {
#         key: value.to(model.device)
#         for key, value in model_inputs.items()
#     }
#
#     generation_config = GenerationConfig(
#         max_new_tokens=256,
#
#         # 制度问答追求稳定，不随机采样
#         do_sample=False,
#
#         repetition_penalty=1.05,
#
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#     )
#
#     with torch.inference_mode():
#         output_ids = model.generate(
#             **model_inputs,
#             generation_config=generation_config,
#         )
#
#     # 去掉输入prompt，只解码新生成的部分
#     input_length = model_inputs["input_ids"].shape[1]
#     generated_ids = output_ids[0, input_length:]
#
#     answer = tokenizer.decode(
#         generated_ids,
#         skip_special_tokens=True,
#     )
#
#     return answer.strip()
def answer_question(
    tokenizer,
    model,
    question: str,
) -> str:

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    model_inputs = {
        key: value.to(model.device)
        for key, value in model_inputs.items()
    }

    generation_config = GenerationConfig(
        max_new_tokens=256,
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

    input_length = model_inputs["input_ids"].shape[1]

    generated_ids = output_ids[
        0,
        input_length:
    ]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return answer.strip()

def main():
    tokenizer, model = load_model()

    context = """《差旅管理制度》第五条：
普通员工在一线城市出差，住宿标准最高为每晚500元；
其他城市最高为每晚350元。
超出标准的部分原则上由员工自行承担。
"""

    while True:
        question = input("\n请输入问题，输入 exit 退出：").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        # answer = answer_question(
        #     tokenizer=tokenizer,
        #     model=model,
        #     context=context,
        #     question=question,
        # )
        answer = answer_question(
            tokenizer=tokenizer,
            model=model,
            question=question,
        )
        print("\n回答：")
        print(answer)


if __name__ == "__main__":
    main()