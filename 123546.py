import torch
import torch.nn as nn
import time


# 传统可变形注意力模块
class DeformableAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(DeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # 预测 8 个采样点的偏移量
        self.offset_proj = nn.Linear(embed_dim, 2 * 8)

    def forward(self, q, k, v, mask=None):
        # 预测偏移量
        offsets = self.offset_proj(q)
        offsets = offsets.view(*offsets.shape[:-1], 8, 2)

        batch_size, seq_len, _ = q.shape
        all_sampled_k = []
        all_sampled_v = []

        for b in range(batch_size):
            sampled_k = []
            sampled_v = []
            for t in range(seq_len):
                for i in range(8):
                    offset = offsets[b, t, i]
                    # 简单模拟采样点的获取
                    index = min(max(int(t + offset[0].item()), 0), seq_len - 1)
                    sampled_k.append(k[b, index].unsqueeze(0))
                    sampled_v.append(v[b, index].unsqueeze(0))
            sampled_k = torch.cat(sampled_k, dim=0).view(seq_len, 8, -1)
            sampled_v = torch.cat(sampled_v, dim=0).view(seq_len, 8, -1)
            all_sampled_k.append(sampled_k)
            all_sampled_v.append(sampled_v)

        all_sampled_k = torch.stack(all_sampled_k, dim=0)
        all_sampled_v = torch.stack(all_sampled_v, dim=0)

        all_sampled_k = all_sampled_k.view(-1, 8, self.embed_dim).transpose(0, 1)
        all_sampled_v = all_sampled_v.view(-1, 8, self.embed_dim).transpose(0, 1)
        q = q.reshape(-1, self.embed_dim).unsqueeze(0)

        q_att, _ = self.attention(q, all_sampled_k, all_sampled_v, key_padding_mask=mask)
        q_att = q_att.reshape(batch_size, seq_len, -1)
        return q_att


# 利用曼哈顿距离的注意力模块
class ManhattanAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(ManhattanAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.shape
        all_sampled_k = []
        all_sampled_v = []

        for b in range(batch_size):
            sampled_k = []
            sampled_v = []
            for t in range(seq_len):
                # 定义 8 个采样点的相对位置
                relative_positions = [i for i in range(-4, 4) if 0 <= t + i < seq_len]
                while len(relative_positions) < 8:
                    if t + len(relative_positions) < seq_len:
                        relative_positions.append(t + len(relative_positions))
                    else:
                        relative_positions.insert(0, t - (8 - len(relative_positions)))
                for pos in relative_positions:
                    sampled_k.append(k[b, pos].unsqueeze(0))
                    sampled_v.append(v[b, pos].unsqueeze(0))
            sampled_k = torch.cat(sampled_k, dim=0).view(seq_len, 8, -1)
            sampled_v = torch.cat(sampled_v, dim=0).view(seq_len, 8, -1)
            all_sampled_k.append(sampled_k)
            all_sampled_v.append(sampled_v)

        all_sampled_k = torch.stack(all_sampled_k, dim=0)
        all_sampled_v = torch.stack(all_sampled_v, dim=0)

        all_sampled_k = all_sampled_k.view(-1, 8, self.embed_dim).transpose(0, 1)
        all_sampled_v = all_sampled_v.view(-1, 8, self.embed_dim).transpose(0, 1)
        q = q.reshape(-1, self.embed_dim).unsqueeze(0)

        q_att, _ = self.attention(q, all_sampled_k, all_sampled_v, key_padding_mask=mask)
        q_att = q_att.reshape(batch_size, seq_len, -1)
        return q_att


# 示例输入
B, T, C = 64, 16, 160
x = torch.randn(B, T, C)
z = torch.randn(B, T, C)
num_heads = 8
dropout = 0.1

# 创建传统可变形注意力模型
deformable_model = DeformableAttention(C, num_heads, dropout)
# 创建曼哈顿注意力模型
manhattan_model = ManhattanAttention(C, num_heads, dropout)

# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


deformable_params = count_parameters(deformable_model)
manhattan_params = count_parameters(manhattan_model)
param_reduction = (deformable_params - manhattan_params) / deformable_params * 100

# 测量运行时间
num_runs = 10
deformable_total_time = 0
manhattan_total_time = 0

for _ in range(num_runs):
    # 传统可变形注意力模型运行时间
    start_time = time.time()
    output = deformable_model(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
    end_time = time.time()
    deformable_total_time += end_time - start_time

    # 曼哈顿注意力模型运行时间
    start_time = time.time()
    output = manhattan_model(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
    end_time = time.time()
    manhattan_total_time += end_time - start_time

deformable_avg_time = deformable_total_time / num_runs
manhattan_avg_time = manhattan_total_time / num_runs
time_reduction = (deformable_avg_time - manhattan_avg_time) / deformable_avg_time * 100

print(f"参数量降低: {param_reduction:.2f}%")
print(f"运行时间降低: {time_reduction:.2f}%")
    