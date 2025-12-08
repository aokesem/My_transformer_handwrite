import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu' # 你的显卡就在这里登场

#训练/优化时的超参数
max_iters = 3000     # 训练轮数：一共要把模型扔进锅里练 3000 次
learning_rate = 1e-2 # 学习率：每次根据错误调整参数的幅度
eval_interval = 300  # 评估间隔：每训练 300 步，停下来检查一下“学习成绩”
eval_iter = 200      # 评估迭代次数：每次检查时，随机抽 200 道题来算平均分

#模型结构与计算参数
#x.shape = [B,T,D], B=batch_size; T=sequence_length，D = n_embed
batch_size = 32     # 并行数：模型同时处理的独立样本数量/张量的第一个维度 B，影响GPU显存的主要因素
block_size = 8      # 上下文长度：进行自注意力计算时考虑的历史信息长度/张量的第二个维度 T，决定注意力矩阵的大小（T*T）
n_embed = 32        # 嵌入维度：每个token被编码成的向量维度/张量的第三个维度 D，模型表示信息的能力和容量
head_size = 16      # 每个注意力头维度/注意力计算中的内积维度: q @ k^T = [B,T,H] * [B,H,T]，单个注意力头的特征复杂度

#机制参数
theta = 10000 #RoPE的基频

class RMSnorm(nn.Module):
    def __init__(self,dim,eps=1e-6):
        super().__init__()

        #创建可学习的缩放参数，维度=dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self,x):
        # RMS = torch.sqrt(torch.mean(x^2,dim=-1)/dim + eps)
        # torch.mean()已经包含了除以维度的操作，不需要再除以dim，属性访问必须通过self
        RMS = torch.sqrt(torch.mean(x*x,dim=-1,keepdim=True)+self.eps)
        rmsnorm_x = x/RMS * self.weight

        return rmsnorm_x



class Head(nn.Module):
    def __init__(self):
        super().__init__() #记着super调用父类
        #q,k,v头
        self.query = nn.Linear(n_embed,head_size,bias=False)
        self.key = nn.Linear(n_embed, head_size,bias=False)
        self.value = nn.Linear(n_embed, head_size,bias=False)

        #上三角掩码
        tril = torch.tril(torch.ones(block_size,block_size))
        self.register_buffer("tril",tril)

        #RMSnorm参数
        self.scale = head_size ** -0.5

        #RoPE Part1
        self.inv_freq = 1/(theta ** (torch.arange(0,head_size,2)/head_size))
        self.pos = torch.arange(0,block_size)#不是block_size-1
        self.freqs = torch.outer(self.pos,self.inv_freq) #freqs.shape = [T,H/2] (block_size,head_size/2)

        self.extend_freqs = torch.repeat_interleave(self.freqs,repeats=2,dim=-1)
        self.cos_values = self.extend_freqs.cos()
        self.sin_values = self.extend_freqs.sin()
        self.register_buffer("cos_values",self.cos_values)
        self.register_buffer("sin_values",self.sin_values)

    #RoPE Part2
    def apply_rotary_emb(self,x):
        # x为q，k，shape = [B,T,H]
        #(q0, q1, q2, q3, q4, q5,…)变为(-q1, q0, -q3, q2, -q5, q4, ……)
        T = x.shape[1]

        x_2k = x[...,::2]
        x_2k1 = x[...,1::2]
        stacked = torch.stack((-x_2k1, x_2k), dim=-1)
        x_rot = torch.flatten(stacked,-2,-1)

        cos_T = self.cos_values[:T,...]
        sin_T = self.sin_values[:T,...]

        return x*cos_T + x_rot*sin_T


    def forward(self,x):
        # q,k,v形状[Batch_size,sequence_length,head_size]
        B,T,C = x.shape# Batch, Time (sequence length), Channels (n_embed) 从输入x的形状中获取参数
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        #RoPE Part2
        q = self.apply_rotary_emb(q)
        k = self.apply_rotary_emb(k)

        attention_score = q @ k.transpose(-1,-2) * self.scale

        # 将mask裁剪为[T,T]形状
        mask = (self.tril[:T, :T] == 0)
        mask_score = attention_score.masked_fill(mask, float('-inf'))
        score = F.softmax(mask_score,dim=-1)
        output = score @ v
        return output







