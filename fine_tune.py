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
    # full_text = [
    #     f"<|im_start|>user\n{q}<|im_end|>\n"
    #     f"<|im_start|>assistant\n{a}<|im_end|>"
    #     for q, a in zip(examples["question"], examples["answer"])
    # ]

    # tokenized_inputs = tokenizer(
    #     full_text,
    #     truncation=True,  #截断到最大长度（512）
    #     max_length=max_seq_length,
    #     padding="max_length",  # 如果一句话不足最大长度，则补齐它/但这会导致出现很多0，占用显存，动态填充是更高效的做法
    #     return_tensors="pt",  # 返回 PyTorch 张量
    # )
    # tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()#将题目和答案inputs复制给答案labels，给模型的考卷

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
training_args = TrainingArguments(
    output_dir="./qwen3_gsm8k_lora_finetuned",  # 模型和日志输出目录
    num_train_epochs=1,                    # 训练 1 个 epoch
    per_device_train_batch_size=2,         # 每个 GPU 的批大小
    gradient_accumulation_steps=16,        # 梯度累积步数，模拟大批次 (4 * 8 = 32)
    learning_rate=2e-5,                    # 学习率
    logging_dir="./logs",                  # 日志目录
    logging_steps=10,                      # 每 10 步记录日志
    save_strategy="steps",                 #可选值: "no" (不存), "epoch" (每跑完一轮存一次), "steps" (按步数存)
                                           #“训练结束时总是保存最后状态作为一个 Checkpoint”。
    save_steps=500,                        # 每 500 步保存模型
    save_total_limit=2,                    # 最多保存 2 个 checkpoint
    bf16=True,                             # 启用 bfloat16 混合精度
    report_to="tensorboard",                      # 不报告到外部服务 (如 WandB)
    dataloader_num_workers=0,              # 设为 0！也就是只用主进程加载数据，不开启多进程，防止内存爆炸
    optim="adamw_torch",                   # 使用 PyTorch 原生优化器，更省内存
    gradient_checkpointing=True,           # 加入这个参数可以降低显存占用，虽然对0.6B模型来说不必须
    gradient_checkpointing_kwargs={'use_reentrant': False},
    # evaluation_strategy="steps",         # 如果启用评估，这两行一起用
    # eval_steps=500,
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