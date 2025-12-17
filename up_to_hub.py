from huggingface_hub import HfApi, create_repo
import os

# --- 配置 ---

# 1. 设置你的 Hugging Face 用户名
username = "aokesem"

# 2. 定义本地模型路径和要创建的仓库 ID
local_model_path = "./qwen3_gsm8k_lora_deepspeed_finetuned_offload"
repo_id = f"{username}/qwen3_gsm8k_lora"

# --- 上传流程 ---

print(f"Creating repository: {repo_id}")

# 3. 创建远程仓库（如果已存在则忽略）
create_repo(
    repo_id,
    private=False,
    exist_ok=True,
)

print(f"Uploading files from: {local_model_path}")

# 4. 上传整个模型文件夹
api = HfApi()
api.upload_folder(
    folder_path=local_model_path,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload My test model",
)

print("Upload complete!")
print(f"You can find your model at: https://huggingface.co/{repo_id}")
