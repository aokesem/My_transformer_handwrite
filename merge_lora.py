import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 定义路径 ---
base_model_path = "Qwen/Qwen3-0.6B"
lora_adapter_path = "./qwen3_gsm8k_lora_deepspeed_finetuned"  # 训练好的 LoRA 适配器目录
merged_model_path = "./qwen3_gsm8k_lora_merged"                    # 合并后完整模型的保存路径

print(f"Loading base model from: {base_model_path}")

# 1. 加载原始基础模型（建议先放在 CPU，避免显存冲突）
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)

print(f"Loading LoRA adapter from: {lora_adapter_path}")

# 2. 加载 LoRA 适配器并合并
# PeftModel.from_pretrained 会将 adapter 权重加载到 base_model 上
model = PeftModel.from_pretrained(
    base_model,
    lora_adapter_path,
)

# merge_and_unload 会真正执行权重合并，
# 并返回一个普通的、不再依赖 peft 的 Transformer 模型
merged_model = model.merge_and_unload()

print(f"Saving merged model to: {merged_model_path}")

# 3. 保存合并后的完整模型和 tokenizer
merged_model.save_pretrained(merged_model_path)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True,
)
tokenizer.save_pretrained(merged_model_path)

print("Done!")
