#第一阶段
model.py是我们用pytorch手写的transformer，它包含了全套的transformer组件，以及训练的部分。运行它会启动训练过程，并打印出文本。
input.txt是一个小型的莎士比亚数据集，也是我们用来训练的数据集。

#第二阶段
fine_tune.py是使用hugging face库进行全量/LoRA微调，包括了训练的全过程，执行它会获得一个文件夹，包含模型权重，检查点等。
inference.py是根据hugging face库进行推理，使用上一步获得的模型生成文本。全量微调模型和LoRA模型过程稍有不同。

#第三阶段
fine_tune_deepspeed.py是通过deepspeed库进行微调，不能直接运行，需要执行deepspeed --num_gpus=1 fine_tune_deepspeed.py，相比fine_tune它修改了training_args,启用了deepspeed。
ds_config_stage3_offload.json是上一步使用的deepspeed相关参数，Zero stage3，启用offload。
vllm_inference是通过vllm库进行推理，取代了inference.py.

#其他代码
merge_lora.py会将lora adapter合并到基础模型上，合并之后使用上等同于微调模型，推理时无需再配置adapter。
up_to_hub.py会将本地模型文件夹上传到制定hugging face仓库中。