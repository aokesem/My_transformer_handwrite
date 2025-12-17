import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载你微调后的模型和分词器
model_path = "./qwen3_gsm8k_finetuned"
model_name = "Qwen/Qwen3-0.6B"#原始模型

#定义LoRA adapter的路径
lora_adapter_path = "./qwen3_gsm8k_lora_finetuned"

#加载分词器和基础模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

fine_tune_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

#加载lora adapter并将其附加到模型上
lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

#可以选择合并到模型上
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained("./qwen3_gsm8k_merged")
# tokenizer.save_pretrained("./qwen3_gsm8k_merged")

model = base_model

# 准备一个问题
question = (
    # "Natalia sold 48/2 = 24 clips in May. She sold half of the clips in the first week "
    # "and the other half in the second week. In the first week, she sold some clips for "
    # "$1.50 each and the rest for $2.00 each. If she made $18 in the first week, how many "
    # "clips did she sell for $1.50 each?"
    "hello, what's your name"
)

# 使用聊天模板格式化输入
messages = [
    {"role": "user", "content": question}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# 生成回答
outputs = model.generate(
    input_ids,
    max_new_tokens=256,  # 最大生成长度
    do_sample=True,      # 启用采样
    temperature=0.7,     # 随机性控制
    top_k=50,
    top_p=0.95,
)

# 解码并打印结果
response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
print("Model's Answer:")
print(response)
