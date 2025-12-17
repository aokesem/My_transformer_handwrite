import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "Qwen/Qwen3-0.6B"

# 1.加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

#一个优化点，确保pad token存在
#如果模型没有 padding token，就将其设为 EOS token。这是为了让 data collator 知道用哪个数字来填补空位。
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Setting pad_token to {tokenizer.pad_token}")

# 加载模型（全量微调）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,   # 使用 bfloat16 节省显存
    device_map="auto",            # 自动分配硬件
    trust_remote_code=True
)

#Lora微调代替全量微调
from peft import get_peft_model,LoraConfig
peft_config = LoraConfig(
    r=8,                               # LoRA 秩
    lora_alpha=16,                     # 缩放因子
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 应用 LoRA 的模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config) #使用PEFT包装模型
model.print_trainable_parameters()

#——————————————————————————————————————————————————————————

#2.加载并处理数据集
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k","main")
max_seq_length = 512
def preprocess_function(examples): #构造大模型的对话模板
    inputs = []
    for q,a in zip(examples["question"],examples["answer"]):
        messages = [
            {"role":"user","content":q},
            {"role":"assistant","content":a}
        ]
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt=False
        )
        inputs.append(formatted_text)
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True, #批处理
    remove_columns=["question", "answer"],  # 处理完后，把原始的文本删掉，只留数字，省内存
)

#————————————————————————————————————————————————————————————————————————————————————————————

#3.配置训练器

#引入DataCollator,它会自动 padding input_ids，并且把 labels 中的 padding 部分设为 -100
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model = model,padding=True)#它会在trainer中传入


from transformers import TrainingArguments, Trainer

# 定义训练参数
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./qwen3_gsm8k_lora_deepspeed_finetuned_offload",  # DeepSpeed 训练建议使用新目录
    num_train_epochs=0.01,#试运行

    # ⚠️ DeepSpeed 会从 ds_config.json 中接管这些参数
    # per_device_train_batch_size=2,
    # gradient_accumulation_steps=16,

    learning_rate=2e-5,

    logging_dir="./logs_deepspeed_offload",  # 建议使用新的日志目录
    logging_steps=10,

    save_steps=500,
    save_total_limit=2,

    bf16=True,
    report_to="tensorboard",

    dataloader_num_workers=0,
    optim="adamw_torch",

    # ⚠️ 使用 DeepSpeed ZeRO 时通常不需要手动开启
    # gradient_checkpointing=True,
    # gradient_checkpointing_kwargs={"use_reentrant": False},

    # --- 启用 DeepSpeed ---
    deepspeed="ds_config_stage3_offload.json",
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],  # 默认有 test，若不想评估可以设为 None
    tokenizer=tokenizer,
    data_collator= data_collator
)


# 启动训练
print("Starting training...")
trainer.train()
print("Training finished!")

# 保存最终模型
trainer.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)  # 也要保存分词器

print(f"Model and tokenizer saved to {training_args.output_dir}")