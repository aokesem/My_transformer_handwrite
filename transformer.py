import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu' # 你的显卡就在这里登场

#训练/优化时的超参数
max_iters = 3000    # 训练轮数：一共要把模型扔进锅里练 3000 次
learning_rate = 1e-2 # 学习率：每次根据错误调整参数的幅度
eval_interval = 300  # 评估间隔：每训练 300 步，停下来检查一下“学习成绩”
eval_iter = 200      # 评估迭代次数：每次检查时，随机抽 200 道题来算平均分

#模型结构与计算参数
#x.shape = [B,T,D], B=batch_size; T=sequence_length，D = n_embed
batch_size = 32     # 并行数：模型同时处理的独立样本数量/张量的第一个维度 B，影响GPU显存的主要因素
block_size = 8      # 上下文长度：进行自注意力计算时考虑的历史信息长度/张量的第二个维度 T，决定注意力矩阵的大小（T*T）
n_embed = 32        # 嵌入维度：每个token被编码成的向量维度/张量的第三个维度 D，模型表示信息的能力和容量
head_size = 8      # 每个注意力头维度/注意力计算中的内积维度: q @ k^T = [B,T,H] * [B,H,T]，单个注意力头的特征复杂度
num_heads = 4       # 多头注意力的头数, num_heads * head_size =n_embed(D)

#模型宏观参数
n_layers = 4       #Transformer_blocks堆叠的层数
vocab_size = 65    #词汇表大小/input层的大小
max_token = 1000   #推理时最大生成词数

#机制参数
theta = 10000 #RoPE的基频

class RMSnorm(nn.Module):
    def __init__(self,n_embed,eps=1e-6):
        super().__init__()

        #创建可学习的缩放参数，维度=dim
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.eps = eps

    def forward(self,x):
        # RMS = torch.sqrt(torch.mean(x^2,dim=-1)/dim + eps)
        # torch.mean()已经包含了除以维度的操作，不需要再除以dim，属性访问必须通过self
        RMS = torch.sqrt(torch.mean(x*x,dim=-1,keepdim=True)+self.eps)
        rmsnorm_x = x/RMS * self.weight

        return rmsnorm_x



class Head(nn.Module):
    def __init__(self,head_size):
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
        self.cos_values_tmp = self.extend_freqs.cos()
        self.sin_values_tmp = self.extend_freqs.sin()
        self.register_buffer("cos_values",self.cos_values_tmp)
        self.register_buffer("sin_values",self.sin_values_tmp)

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

class Multi_Head(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.layers = nn.ModuleList([Head(head_size) for i in range(num_heads)])
        self.proj = nn.Linear(n_embed,n_embed)#最后收尾的线性层

    def forward(self,x):
        output = [h(x) for h in self.layers]
        output_all = torch.cat(output,dim=-1)
        output_final = self.proj(output_all)

        return output_final

class Feedforward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        hidden_dim = 4 * n_embed
        self.w1 = nn.Linear(n_embed, hidden_dim,bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embed, bias=False)
        self.w3 = nn.Linear(n_embed, hidden_dim , bias=False)

    def forward(self,x):
        SwiGLU_x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return SwiGLU_x

class Transformer_Block(nn.Module):
    def __init__(self,n_embed,num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.multi_head = Multi_Head(num_heads,head_size)
        self.Feedforward = Feedforward(n_embed)
        self.RMSnorm1 = RMSnorm(n_embed)
        self.RMSnorm2 = RMSnorm(n_embed)

    def forward(self,x):
        x_1 = x + self.multi_head(self.RMSnorm1(x))
        x_output = x_1 + self.Feedforward(self.RMSnorm2(x_1))
        return x_output

class miniGPT(nn.Module):
    def __init__(self,vocab_size,n_embed):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,n_embed)#词嵌入层
        self.Transformer_Blocks = nn.ModuleList([Transformer_Block(n_embed,num_heads) for i in range(n_layers)])#堆叠的Transformer块
        self.RMSnorm = RMSnorm(n_embed)#最后的归一化层
        self.output = nn.Linear(n_embed,vocab_size,bias=False)#最后的线性输出，从n_embed返回到vocab_size

    def forward(self,idx,targets=None):
        x = self.embedding(idx)
        for block in self.Transformer_Blocks:
            x = block(x)
        x = self.RMSnorm(x)
        logits = self.output(x)
        if targets is None:
            loss = None
        else:
            #cross_entropy期望的输入是(N,C),N是样本总数,C是类别数
            B,T,C = logits.shape
            logits = logits.view(B*T,C) # 展平成 (B*T, C)
            #targets的形状是(B,T)
            targets = targets.flatten() # 展平成 (B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss

    def generate(self,idx,max_token):
        # idx的形状是[B,T]
        for i in range(max_token):
            idx_cond = idx[:,-block_size:]#只取最后block_size个token给模型看
            logits,loss = self.forward(idx_cond)
            last_logits = logits[:,-1,:]
            prob = F.softmax(last_logits,dim=-1)
            id_next = torch.multinomial(prob,num_samples=1)
            idx = torch.cat((idx,id_next),dim=-1)
        return idx


# 读取文件
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()
# 找出所有出现过的字符（去重、排序）
# set()去重，sorted(list())排序
chars = sorted(list(set(text)))
vocab_size = len(chars)
# 建立映射表
stoi = { ch:i for i,ch in enumerate(chars) } # 查字典：a -> 1, b -> 2
itos = { i:ch for i,ch in enumerate(chars) } # 反查：1 -> a, 2 -> b
# 定义转换函数 (Lambda表达式)
encode = lambda s: [stoi[c] for c in s]#根据字符串获取整数序列
decode = lambda l: ''.join([itos[i] for i in l])#根据整数序列获取字符串
# 把由整数组成的列表，转换成 PyTorch 的 Tensor (张量)
data = torch.tensor(encode(text), dtype=torch.long)
# 切分训练集和验证集
n = int(0.9*len(data))
train_data = data[:n]  # 90% 用来训练
val_data = data[n:]    # 10% 用来考试
# 随机出题器
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # 1. 随机选 32 个起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 2. 抓取输入 x (题目)
    x = torch.stack([data[i:i + block_size] for i in ix])
    # 3. 抓取目标 y (答案)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    # 4. 搬到显卡上
    x, y = x.to(device), y.to(device)
    return x, y

model = miniGPT(vocab_size,n_embed)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # 切换到评估模式
    for split in ['train', 'val']: # 分别计算训练集和验证集的 Loss
        losses = torch.zeros(eval_iter) # 准备一个容器，比如存 200 个 batch 的 loss
        for k in range(eval_iter):  # 循环多次 (eval_iter 是个超参数，比如 200)
            X, Y = get_batch(split) # 随机抽取一个 batch 的数据
            logits, loss = model(X, Y) #前向传播
            losses[k] = loss.item() #记录loss
        out[split] = losses.mean()  #取平均
    model.train()
    return out

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb,yb = get_batch('train')
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)#清除梯度/节省内存读写，减少反向传播计算
    loss.backward()#反向传播
    optimizer.step()#更新梯度

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(context, max_token=500)[0].tolist())
print(generated_text)














