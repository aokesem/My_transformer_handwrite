from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer

# 1. 定义模型路径
# 这里直接使用 LoRA 微调后保存的目录
# vLLM 会自动处理 LoRA 适配器的加载与合并
model_path = "./qwen3_gsm8k_lora_merged"

# 2. 加载模型（dtype 与训练保持一致）
print(f"Loading model from {model_path} with vLLM...")
llm = LLM(
    model=model_path,
    dtype=torch.bfloat16,  # 使用 torch.bfloat16，而不是字符串
    gpu_memory_utilization=0.6,  # 默认0.9
    max_model_len=1024, # 默认40960
    max_num_seqs = 1,
)

# 3. 定义采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256,
    stop=["<|im_end|>"],  # Qwen 模型的停止 token
)

# 4. 准备输入问题（可同时展示 vLLM 的批处理能力）
prompts = [
    (
        "Natalia sold 48/2 = 24 clips in May. She sold half of the clips in the first week "
        "and the other half in the second week. In the first week, she sold some clips for "
        "$1.50 each and the rest for $2.00 each. If she made $18 in the first week, how many "
        "clips did she sell for $1.50 each?"
    ),
    "The perimeter of a rectangle is 20 cm. If its length is 6 cm, what is its width?",
    "A car travels at 60 miles per hour. How long does it take to travel 180 miles?",
]

# 5. 使用 tokenizer 手动格式化 Qwen 聊天模板
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)

formatted_prompts = []
for prompt_text in prompts:
    messages = [{"role": "user", "content": prompt_text}]
    formatted_prompts.append(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # 告诉模型开始回答
        )
    )

# 6. 生成回答
print("Generating responses with vLLM...")
outputs = llm.generate(formatted_prompts, sampling_params)

# 7. 打印结果
print("\n--- vLLM Generated Responses ---")
for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text
    print(f"Prompt {i + 1}:")
    print(formatted_prompts[i])
    print("Generated text:")
    print(generated_text)
    print()
