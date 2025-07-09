import torch
from deformable_attention import DeformableAttention
from deformable_attention import DeformableAttention1D
from deformable_attention import DeformableAttention3D

attn_1D = DeformableAttention1D(
    dim = 128,
    downsample_factor = 4,
    offset_scale = 2,
    offset_kernel_size = 6
)

x = torch.randn(1, 128, 512)
result=attn_1D(x) # (1, 128, 512)
print('1D',result.shape)


attn_2D = DeformableAttention(
    dim = 512,                   # feature dimensions
    dim_head = 64,               # dimension per head
    heads = 8,                   # attention heads
    dropout = 0.,                # dropout
    downsample_factor = 4,       # downsample factor (r in paper)
    offset_scale = 4,            # scale of offset, maximum offset
    offset_groups = None,        # number of offset groups, should be multiple of heads
    offset_kernel_size = 6,      # offset kernel size
)

x = torch.randn(1, 512, 64, 64)
result=attn_2D(x) # (1, 512, 64, 64)
print('2D',result.shape)


attn_3D = DeformableAttention3D(
    dim = 512,                          # feature dimensions
    dim_head = 64,                      # dimension per head
    heads = 8,                          # attention heads
    dropout = 0.,                       # dropout
    downsample_factor = (2, 8, 8),      # downsample factor (r in paper)
    offset_scale = (2, 8, 8),           # scale of offset, maximum offset
    offset_kernel_size = (4, 10, 10),   # offset kernel size
)

x = torch.randn(1, 512, 10, 32, 32)
result=attn_3D(x) # (1, 512, 64, 64)
print('3D',result.shape)