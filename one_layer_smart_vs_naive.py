import ttnn
import torch
import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch.utils._pytree as pytree
from torch._dynamo.output_graph import FakeRootModule
import sys
import time
import torch.nn as nn

from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    profiler,
)

ttnn_ROW_MAJOR_LAYOUT = ttnn.ROW_MAJOR_LAYOUT
ttnn_TILE_LAYOUT = ttnn.TILE_LAYOUT
ttnn_bfloat16 = ttnn.bfloat16
ttnn_uint32 = ttnn.uint32
ttnn_decorators_ttnn_from_torch = ttnn.from_torch
ttnn_decorators_ttnn_layer_norm = ttnn.layer_norm
ttnn_decorators_ttnn_add = ttnn.add
ttnn_decorators_ttnn_multiply = ttnn.multiply
ttnn_decorators_ttnn_reshape = ttnn.reshape
ttnn_decorators_ttnn_permute = ttnn.permute 
ttnn_decorators_ttnn_transpose = ttnn.transpose
ttnn_decorators_ttnn_matmul = ttnn.matmul
ttnn_decorators_ttnn_transformer_attention_softmax_ = ttnn.transformer.attention_softmax_
ttnn_decorators_ttnn_linear = ttnn.linear
ttnn_decorators_ttnn_typecast = ttnn.typecast
ttnn_decorators_ttnn_rsub = ttnn.rsub
ttnn_decorators_ttnn_to_layout = ttnn.to_layout
ttnn_decorators_ttnn_slice = ttnn.slice
ttnn_decorators_ttnn_embedding = ttnn.embedding
ttnn_decorators_ttnn_experimental_view = ttnn.experimental.view
ttnn_decorators_ttnn_to_torch = ttnn.to_torch
ttnn_decorators_ttnn_squeeze = ttnn.squeeze
ttnn_decorators_ttnn_split = ttnn.split
ttnn_decorators_ttnn_softmax = ttnn.softmax
ttnn_decorators_ttnn_gelu = ttnn.gelu
ttnn_decorators_ttnn_transformer_scaled_dot_product_attention = ttnn.transformer.scaled_dot_product_attention
ttnn_decorators_ttnn_concat = ttnn.concat
ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads = ttnn.transformer.split_query_key_value_and_split_heads
ttnn_decorators_ttnn_transformer_concatenate_heads = ttnn.transformer.concatenate_heads


#repo means:
# the latest version in github 
# https://github.com/tenstorrent/pytorch2.0_ttnn/commit/0d04533d13254e55551db5042e49d5da27ba2e62
def repo(ttnn_Specified_Device, i, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1):
    ttnn_from_torch = ttnn_decorators_ttnn_from_torch(arg25_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_uint32);  arg25_1 = None
    ttnn_reshape = ttnn_decorators_ttnn_reshape(ttnn_from_torch, [1, 1, 256]);  ttnn_from_torch = None
    ttnn_reshape_1 = ttnn_decorators_ttnn_reshape(ttnn_reshape, [1, 1, 1, 256]);  ttnn_reshape = None
    ttnn_typecast = ttnn_decorators_ttnn_typecast(ttnn_reshape_1, ttnn_bfloat16);  ttnn_reshape_1 = None
    ttnn_rsub = ttnn_decorators_ttnn_rsub(ttnn_typecast, 1.0);  ttnn_typecast = None
    ttnn_multiply = ttnn_decorators_ttnn_multiply(ttnn_rsub, -3.3895313892515355e+38);  ttnn_rsub = None
    ttnn_from_torch_1 = ttnn_decorators_ttnn_from_torch(arg23_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg23_1 = None
    ttnn_slice = ttnn_decorators_ttnn_slice(ttnn_from_torch_1, [0, 0], [1, 256]);  ttnn_from_torch_1 = None
    ttnn_from_torch_2 = ttnn_decorators_ttnn_from_torch(arg24_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg24_1 = None
    ttnn_from_torch_3 = ttnn_decorators_ttnn_from_torch(arg0_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg0_1 = None
    ttnn_embedding = ttnn_decorators_ttnn_embedding(ttnn_from_torch_2, ttnn_from_torch_3, layout = ttnn_TILE_LAYOUT);  ttnn_from_torch_2 = ttnn_from_torch_3 = None
    ttnn_from_torch_4 = ttnn_decorators_ttnn_from_torch(arg26_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg26_1 = None
    ttnn_from_torch_5 = ttnn_decorators_ttnn_from_torch(arg1_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg1_1 = None
    ttnn_embedding_1 = ttnn_decorators_ttnn_embedding(ttnn_from_torch_4, ttnn_from_torch_5, layout = ttnn_TILE_LAYOUT);  ttnn_from_torch_4 = ttnn_from_torch_5 = None
    ttnn_add_7 = ttnn_decorators_ttnn_add(ttnn_embedding, ttnn_embedding_1);  ttnn_embedding = ttnn_embedding_1 = None
    ttnn_to_layout = ttnn_decorators_ttnn_to_layout(ttnn_slice, ttnn_ROW_MAJOR_LAYOUT);  ttnn_slice = None
    ttnn_from_torch_6 = ttnn_decorators_ttnn_from_torch(arg2_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg2_1 = None
    ttnn_embedding_2 = ttnn_decorators_ttnn_embedding(ttnn_to_layout, ttnn_from_torch_6, layout = ttnn_TILE_LAYOUT);  ttnn_to_layout = ttnn_from_torch_6 = None
    ttnn_add_8 = ttnn_decorators_ttnn_add(ttnn_add_7, ttnn_embedding_2);  ttnn_add_7 = ttnn_embedding_2 = None
    ttnn_from_torch_7 = ttnn_decorators_ttnn_from_torch(arg3_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg3_1 = None
    ttnn_from_torch_8 = ttnn_decorators_ttnn_from_torch(arg4_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg4_1 = None
    ttnn_layer_norm = ttnn_decorators_ttnn_layer_norm(ttnn_add_8, epsilon = 1e-12, weight = ttnn_from_torch_7, bias = ttnn_from_torch_8);  ttnn_add_8 = ttnn_from_torch_7 = ttnn_from_torch_8 = None
    ttnn_experimental_view = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm, [256, 1024])
    ttnn_from_torch_9 = ttnn_decorators_ttnn_from_torch(arg5_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg5_1 = None
    ttnn_from_torch_10 = ttnn_decorators_ttnn_from_torch(arg6_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg6_1 = None
    ttnn_linear = ttnn_decorators_ttnn_linear(ttnn_experimental_view, ttnn_from_torch_9, transpose_b = True, bias = ttnn_from_torch_10, activation = None);  ttnn_from_torch_9 = ttnn_from_torch_10 = None
    ttnn_experimental_view_1 = ttnn_decorators_ttnn_experimental_view(ttnn_linear, [1, 256, 1024]);  ttnn_linear = None
    ttnn_from_torch_11 = ttnn_decorators_ttnn_from_torch(arg7_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg7_1 = None
    ttnn_from_torch_12 = ttnn_decorators_ttnn_from_torch(arg8_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg8_1 = None
    ttnn_linear_1 = ttnn_decorators_ttnn_linear(ttnn_experimental_view, ttnn_from_torch_11, transpose_b = True, bias = ttnn_from_torch_12, activation = None);  ttnn_from_torch_11 = ttnn_from_torch_12 = None
    ttnn_experimental_view_3 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_1, [1, 256, 1024]);  ttnn_linear_1 = None
    ttnn_reshape_2 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_3, [1, 256, 16, 64]);  ttnn_experimental_view_3 = None
    ttnn_permute = ttnn_decorators_ttnn_permute(ttnn_reshape_2, [0, 2, 1, 3]);  ttnn_reshape_2 = None
    ttnn_from_torch_13 = ttnn_decorators_ttnn_from_torch(arg9_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg9_1 = None
    ttnn_from_torch_14 = ttnn_decorators_ttnn_from_torch(arg10_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg10_1 = None
    ttnn_linear_2 = ttnn_decorators_ttnn_linear(ttnn_experimental_view, ttnn_from_torch_13, transpose_b = True, bias = ttnn_from_torch_14, activation = None);  ttnn_experimental_view = ttnn_from_torch_13 = ttnn_from_torch_14 = None
    ttnn_experimental_view_5 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_2, [1, 256, 1024]);  ttnn_linear_2 = None
    ttnn_reshape_3 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_5, [1, 256, 16, 64]);  ttnn_experimental_view_5 = None
    ttnn_permute_1 = ttnn_decorators_ttnn_permute(ttnn_reshape_3, [0, 2, 1, 3]);  ttnn_reshape_3 = None
    ttnn_reshape_4 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_1, [1, 256, 16, 64]);  ttnn_experimental_view_1 = None
    ttnn_permute_2 = ttnn_decorators_ttnn_permute(ttnn_reshape_4, [0, 2, 1, 3]);  ttnn_reshape_4 = None
    ttnn_transpose_3 = ttnn_decorators_ttnn_transpose(ttnn_permute, 3, 2);  ttnn_permute = None
    ttnn_experimental_view_6 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_2, [16, 256, 64]);  ttnn_permute_2 = None
    ttnn_experimental_view_7 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_3, [16, 64, 256]);  ttnn_transpose_3 = None
    ttnn_matmul_3 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_6, ttnn_experimental_view_7);  ttnn_experimental_view_6 = ttnn_experimental_view_7 = None
    ttnn_experimental_view_8 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_3, [1, 16, 256, 256]);  ttnn_matmul_3 = None
    ttnn_transformer_attention_softmax_ = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_experimental_view_8, head_size = 64, attention_mask = ttnn_multiply);  ttnn_experimental_view_8 = ttnn_multiply = None
    ttnn_experimental_view_9 = ttnn_decorators_ttnn_experimental_view(ttnn_transformer_attention_softmax_, [16, 256, 256]);  ttnn_transformer_attention_softmax_ = None
    ttnn_experimental_view_10 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_1, [16, 256, 64]);  ttnn_permute_1 = None
    ttnn_matmul_4 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_9, ttnn_experimental_view_10);  ttnn_experimental_view_9 = ttnn_experimental_view_10 = None
    ttnn_experimental_view_11 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_4, [1, 16, 256, 64]);  ttnn_matmul_4 = None
    ttnn_permute_3 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_11, [0, 2, 1, 3]);  ttnn_experimental_view_11 = None
    ttnn_reshape_5 = ttnn_decorators_ttnn_reshape(ttnn_permute_3, [1, 256, 1024]);  ttnn_permute_3 = None
    ttnn_experimental_view_12 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_5, [256, 1024]);  ttnn_reshape_5 = None
    ttnn_from_torch_15 = ttnn_decorators_ttnn_from_torch(arg11_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg11_1 = None
    ttnn_from_torch_16 = ttnn_decorators_ttnn_from_torch(arg12_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg12_1 = None
    ttnn_linear_3 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_12, ttnn_from_torch_15, transpose_b = True, bias = ttnn_from_torch_16, activation = None);  ttnn_from_torch_15 = ttnn_from_torch_16 = None
    ttnn_experimental_view_13 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_3, [1, 256, 1024]);  ttnn_linear_3 = None
    ttnn_add_10 = ttnn_decorators_ttnn_add(ttnn_experimental_view_13, ttnn_layer_norm);  ttnn_experimental_view_13 = ttnn_layer_norm = None
    ttnn_from_torch_17 = ttnn_decorators_ttnn_from_torch(arg13_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg13_1 = None
    ttnn_from_torch_18 = ttnn_decorators_ttnn_from_torch(arg14_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg14_1 = None
    ttnn_layer_norm_1 = ttnn_decorators_ttnn_layer_norm(ttnn_add_10, epsilon = 1e-12, weight = ttnn_from_torch_17, bias = ttnn_from_torch_18);  ttnn_add_10 = ttnn_from_torch_17 = ttnn_from_torch_18 = None
    ttnn_experimental_view_14 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_1, [256, 1024])
    ttnn_from_torch_19 = ttnn_decorators_ttnn_from_torch(arg15_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg15_1 = None
    ttnn_from_torch_20 = ttnn_decorators_ttnn_from_torch(arg16_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg16_1 = None
    ttnn_linear_4 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_14, ttnn_from_torch_19, transpose_b = True, bias = ttnn_from_torch_20, activation = 'gelu');  ttnn_from_torch_19 = ttnn_from_torch_20 = None
    ttnn_experimental_view_16 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_4, [256, 4096]);  ttnn_linear_4 = None
    ttnn_from_torch_21 = ttnn_decorators_ttnn_from_torch(arg17_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg17_1 = None
    ttnn_from_torch_22 = ttnn_decorators_ttnn_from_torch(arg18_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg18_1 = None
    ttnn_linear_5 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_16, ttnn_from_torch_21, transpose_b = True, bias = ttnn_from_torch_22, activation = None);  ttnn_experimental_view_16 = ttnn_from_torch_21 = ttnn_from_torch_22 = None
    ttnn_experimental_view_17 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_5, [1, 256, 1024]);  ttnn_linear_5 = None
    ttnn_add_11 = ttnn_decorators_ttnn_add(ttnn_experimental_view_17, ttnn_layer_norm_1);  ttnn_experimental_view_17 = ttnn_layer_norm_1 = None
    ttnn_from_torch_23 = ttnn_decorators_ttnn_from_torch(arg19_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg19_1 = None
    ttnn_from_torch_24 = ttnn_decorators_ttnn_from_torch(arg20_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg20_1 = None
    ttnn_layer_norm_2 = ttnn_decorators_ttnn_layer_norm(ttnn_add_11, epsilon = 1e-12, weight = ttnn_from_torch_23, bias = ttnn_from_torch_24);  ttnn_add_11 = ttnn_from_torch_23 = ttnn_from_torch_24 = None
    ttnn_experimental_view_18 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_2, [256, 1024]);  ttnn_layer_norm_2 = None
    ttnn_from_torch_25 = ttnn_decorators_ttnn_from_torch(arg21_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg21_1 = None
    ttnn_from_torch_26 = ttnn_decorators_ttnn_from_torch(arg22_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg22_1 = None
    ttnn_linear_6 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_18, ttnn_from_torch_25, transpose_b = True, bias = ttnn_from_torch_26, activation = None);  ttnn_experimental_view_18 = ttnn_from_torch_25 = ttnn_from_torch_26 = None
    ttnn_experimental_view_19 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_6, [1, 256, 2]);  ttnn_linear_6 = None
    ttnn_to_layout_1 = ttnn_decorators_ttnn_to_layout(ttnn_experimental_view_19, ttnn_ROW_MAJOR_LAYOUT);  ttnn_experimental_view_19 = None
    ttnn_split = ttnn_decorators_ttnn_split(ttnn_to_layout_1, 1, 2);  ttnn_to_layout_1 = None
    getitem_3 = ttnn_split[0]
    getitem_4 = ttnn_split[1];  ttnn_split = None
    ttnn_to_layout_2 = ttnn_decorators_ttnn_to_layout(getitem_3, ttnn_TILE_LAYOUT);  getitem_3 = None
    ttnn_squeeze = ttnn_decorators_ttnn_squeeze(ttnn_to_layout_2, -1);  ttnn_to_layout_2 = None
    ttnn_to_layout_3 = ttnn_decorators_ttnn_to_layout(getitem_4, ttnn_TILE_LAYOUT);  getitem_4 = None
    ttnn_squeeze_1 = ttnn_decorators_ttnn_squeeze(ttnn_to_layout_3, -1);  ttnn_to_layout_3 = None
    ttnn_to_torch = ttnn_decorators_ttnn_to_torch(ttnn_squeeze, dtype = torch.bfloat16);  ttnn_squeeze = None
    ttnn_to_torch_1 = ttnn_decorators_ttnn_to_torch(ttnn_squeeze_1, dtype = torch.bfloat16);  ttnn_squeeze_1 = None
    return (ttnn_to_torch, ttnn_to_torch_1, ttnn_experimental_view_12)


#naive means:
# raw compiled code without any optimization nor fusion
def naive(ttnn_Specified_Device, i, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1):
    ttnn_from_torch = ttnn_decorators_ttnn_from_torch(arg25_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_uint32);  arg25_1 = None
    ttnn_reshape = ttnn_decorators_ttnn_reshape(ttnn_from_torch, [1, 1, 256]);  ttnn_from_torch = None
    ttnn_reshape_1 = ttnn_decorators_ttnn_reshape(ttnn_reshape, [1, 1, 1, 256]);  ttnn_reshape = None
    ttnn_typecast = ttnn_decorators_ttnn_typecast(ttnn_reshape_1, ttnn_bfloat16);  ttnn_reshape_1 = None
    ttnn_rsub = ttnn_decorators_ttnn_rsub(ttnn_typecast, 1.0);  ttnn_typecast = None
    ttnn_multiply = ttnn_decorators_ttnn_multiply(ttnn_rsub, -3.3895313892515355e+38);  ttnn_rsub = None
    ttnn_from_torch_1 = ttnn_decorators_ttnn_from_torch(arg23_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg23_1 = None
    ttnn_slice = ttnn_decorators_ttnn_slice(ttnn_from_torch_1, [0, 0], [1, 256]);  ttnn_from_torch_1 = None
    ttnn_from_torch_2 = ttnn_decorators_ttnn_from_torch(arg24_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg24_1 = None
    ttnn_from_torch_3 = ttnn_decorators_ttnn_from_torch(arg0_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg0_1 = None
    ttnn_embedding = ttnn_decorators_ttnn_embedding(ttnn_from_torch_2, ttnn_from_torch_3, layout = ttnn_TILE_LAYOUT);  ttnn_from_torch_2 = ttnn_from_torch_3 = None
    ttnn_from_torch_4 = ttnn_decorators_ttnn_from_torch(arg26_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg26_1 = None
    ttnn_from_torch_5 = ttnn_decorators_ttnn_from_torch(arg1_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg1_1 = None
    ttnn_embedding_1 = ttnn_decorators_ttnn_embedding(ttnn_from_torch_4, ttnn_from_torch_5, layout = ttnn_TILE_LAYOUT);  ttnn_from_torch_4 = ttnn_from_torch_5 = None
    ttnn_add_7 = ttnn_decorators_ttnn_add(ttnn_embedding, ttnn_embedding_1);  ttnn_embedding = ttnn_embedding_1 = None
    ttnn_to_layout = ttnn_decorators_ttnn_to_layout(ttnn_slice, ttnn_ROW_MAJOR_LAYOUT);  ttnn_slice = None
    ttnn_from_torch_6 = ttnn_decorators_ttnn_from_torch(arg2_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg2_1 = None
    ttnn_embedding_2 = ttnn_decorators_ttnn_embedding(ttnn_to_layout, ttnn_from_torch_6, layout = ttnn_TILE_LAYOUT);  ttnn_to_layout = ttnn_from_torch_6 = None
    ttnn_add_8 = ttnn_decorators_ttnn_add(ttnn_add_7, ttnn_embedding_2);  ttnn_add_7 = ttnn_embedding_2 = None
    ttnn_from_torch_7 = ttnn_decorators_ttnn_from_torch(arg3_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg3_1 = None
    ttnn_from_torch_8 = ttnn_decorators_ttnn_from_torch(arg4_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg4_1 = None
    ttnn_layer_norm = ttnn_decorators_ttnn_layer_norm(ttnn_add_8, epsilon = 1e-12, weight = ttnn_from_torch_7, bias = ttnn_from_torch_8);  ttnn_add_8 = ttnn_from_torch_7 = ttnn_from_torch_8 = None
    ttnn_experimental_view = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm, [256, 1024])
    
    #LAYER STARTS HERE
    profiler.start(f"Naive_{i}")
    ttnn_from_torch_9 = ttnn_decorators_ttnn_from_torch(arg5_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg5_1 = None
    ttnn_transpose = ttnn_decorators_ttnn_transpose(ttnn_from_torch_9, 0, 1);  ttnn_from_torch_9 = None
    ttnn_matmul = ttnn_decorators_ttnn_matmul(ttnn_experimental_view, ttnn_transpose);  ttnn_transpose = None
    ttnn_from_torch_10 = ttnn_decorators_ttnn_from_torch(arg6_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg6_1 = None
    ttnn_add = ttnn_decorators_ttnn_add(ttnn_from_torch_10, ttnn_matmul);  ttnn_from_torch_10 = ttnn_matmul = None
    ttnn_experimental_view_1 = ttnn_decorators_ttnn_experimental_view(ttnn_add, [1, 256, 1024]);  ttnn_add = None
    ttnn_from_torch_11 = ttnn_decorators_ttnn_from_torch(arg7_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg7_1 = None
    ttnn_transpose_1 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_11, 0, 1);  ttnn_from_torch_11 = None
    ttnn_matmul_1 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view, ttnn_transpose_1);  ttnn_transpose_1 = None
    ttnn_from_torch_12 = ttnn_decorators_ttnn_from_torch(arg8_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg8_1 = None
    ttnn_add_1 = ttnn_decorators_ttnn_add(ttnn_from_torch_12, ttnn_matmul_1);  ttnn_from_torch_12 = ttnn_matmul_1 = None
    ttnn_experimental_view_3 = ttnn_decorators_ttnn_experimental_view(ttnn_add_1, [1, 256, 1024]);  ttnn_add_1 = None
    ttnn_reshape_2 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_3, [1, 256, 16, 64]);  ttnn_experimental_view_3 = None
    ttnn_permute = ttnn_decorators_ttnn_permute(ttnn_reshape_2, [0, 2, 1, 3]);  ttnn_reshape_2 = None
    ttnn_from_torch_13 = ttnn_decorators_ttnn_from_torch(arg9_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg9_1 = None
    ttnn_transpose_2 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_13, 0, 1);  ttnn_from_torch_13 = None
    ttnn_matmul_2 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view, ttnn_transpose_2);  ttnn_experimental_view = ttnn_transpose_2 = None
    ttnn_from_torch_14 = ttnn_decorators_ttnn_from_torch(arg10_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg10_1 = None
    ttnn_add_2 = ttnn_decorators_ttnn_add(ttnn_from_torch_14, ttnn_matmul_2);  ttnn_from_torch_14 = ttnn_matmul_2 = None
    ttnn_experimental_view_5 = ttnn_decorators_ttnn_experimental_view(ttnn_add_2, [1, 256, 1024]);  ttnn_add_2 = None
    ttnn_reshape_3 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_5, [1, 256, 16, 64]);  ttnn_experimental_view_5 = None
    ttnn_permute_1 = ttnn_decorators_ttnn_permute(ttnn_reshape_3, [0, 2, 1, 3]);  ttnn_reshape_3 = None
    ttnn_reshape_4 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_1, [1, 256, 16, 64]);  ttnn_experimental_view_1 = None
    ttnn_permute_2 = ttnn_decorators_ttnn_permute(ttnn_reshape_4, [0, 2, 1, 3]);  ttnn_reshape_4 = None
    ttnn_transpose_3 = ttnn_decorators_ttnn_transpose(ttnn_permute, 3, 2);  ttnn_permute = None
    ttnn_experimental_view_6 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_2, [16, 256, 64]);  ttnn_permute_2 = None
    ttnn_experimental_view_7 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_3, [16, 64, 256]);  ttnn_transpose_3 = None
    ttnn_matmul_3 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_6, ttnn_experimental_view_7);  ttnn_experimental_view_6 = ttnn_experimental_view_7 = None
    ttnn_experimental_view_8 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_3, [1, 16, 256, 256]);  ttnn_matmul_3 = None
    ttnn_multiply_1 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_8, 0.125);  ttnn_experimental_view_8 = None
    ttnn_add_9 = ttnn_decorators_ttnn_add(ttnn_multiply_1, ttnn_multiply);  ttnn_multiply_1 = ttnn_multiply = None
    ttnn_softmax = ttnn_decorators_ttnn_softmax(ttnn_add_9, -1, numeric_stable = True);  ttnn_add_9 = None
    ttnn_experimental_view_9 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax, [16, 256, 256]);  ttnn_softmax = None
    ttnn_experimental_view_10 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_1, [16, 256, 64]);  ttnn_permute_1 = None
    ttnn_matmul_4 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_9, ttnn_experimental_view_10);  ttnn_experimental_view_9 = ttnn_experimental_view_10 = None
    ttnn_experimental_view_11 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_4, [1, 16, 256, 64]);  ttnn_matmul_4 = None
    ttnn_permute_3 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_11, [0, 2, 1, 3]);  ttnn_experimental_view_11 = None
    ttnn_reshape_5 = ttnn_decorators_ttnn_reshape(ttnn_permute_3, [1, 256, 1024]);  ttnn_permute_3 = None
    ttnn_experimental_view_12 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_5, [256, 1024]);  ttnn_reshape_5 = None
    ttnn_from_torch_15 = ttnn_decorators_ttnn_from_torch(arg11_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg11_1 = None
    ttnn_transpose_4 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_15, 0, 1);  ttnn_from_torch_15 = None
    ttnn_matmul_5 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_12, ttnn_transpose_4);  ttnn_transpose_4 = None
    ttnn_from_torch_16 = ttnn_decorators_ttnn_from_torch(arg12_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg12_1 = None
    ttnn_add_3 = ttnn_decorators_ttnn_add(ttnn_from_torch_16, ttnn_matmul_5);  ttnn_from_torch_16 = ttnn_matmul_5 = None
    ttnn_experimental_view_13 = ttnn_decorators_ttnn_experimental_view(ttnn_add_3, [1, 256, 1024]);  ttnn_add_3 = None
    ttnn_add_10 = ttnn_decorators_ttnn_add(ttnn_experimental_view_13, ttnn_layer_norm);  ttnn_experimental_view_13 = ttnn_layer_norm = None
    ttnn_from_torch_17 = ttnn_decorators_ttnn_from_torch(arg13_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg13_1 = None
    ttnn_from_torch_18 = ttnn_decorators_ttnn_from_torch(arg14_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg14_1 = None
    ttnn_layer_norm_1 = ttnn_decorators_ttnn_layer_norm(ttnn_add_10, epsilon = 1e-12, weight = ttnn_from_torch_17, bias = ttnn_from_torch_18);  ttnn_add_10 = ttnn_from_torch_17 = ttnn_from_torch_18 = None
    ttnn_experimental_view_14 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_1, [256, 1024])
    ttnn_from_torch_19 = ttnn_decorators_ttnn_from_torch(arg15_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg15_1 = None
    ttnn_transpose_5 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_19, 0, 1);  ttnn_from_torch_19 = None
    ttnn_matmul_6 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_14, ttnn_transpose_5);  ttnn_experimental_view_14 = ttnn_transpose_5 = None
    ttnn_from_torch_20 = ttnn_decorators_ttnn_from_torch(arg16_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg16_1 = None
    ttnn_add_4 = ttnn_decorators_ttnn_add(ttnn_from_torch_20, ttnn_matmul_6);  ttnn_from_torch_20 = ttnn_matmul_6 = None
    ttnn_experimental_view_15 = ttnn_decorators_ttnn_experimental_view(ttnn_add_4, [1, 256, 4096]);  ttnn_add_4 = None
    ttnn_gelu = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_15);  ttnn_experimental_view_15 = None
    ttnn_experimental_view_16 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu, [256, 4096]);  ttnn_gelu = None
    ttnn_from_torch_21 = ttnn_decorators_ttnn_from_torch(arg17_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg17_1 = None
    ttnn_transpose_6 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_21, 0, 1);  ttnn_from_torch_21 = None
    ttnn_matmul_7 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_16, ttnn_transpose_6);  ttnn_experimental_view_16 = ttnn_transpose_6 = None
    ttnn_from_torch_22 = ttnn_decorators_ttnn_from_torch(arg18_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg18_1 = None
    ttnn_add_5 = ttnn_decorators_ttnn_add(ttnn_from_torch_22, ttnn_matmul_7);  ttnn_from_torch_22 = ttnn_matmul_7 = None
    ttnn_experimental_view_17 = ttnn_decorators_ttnn_experimental_view(ttnn_add_5, [1, 256, 1024]);  ttnn_add_5 = None
    ttnn_add_11 = ttnn_decorators_ttnn_add(ttnn_experimental_view_17, ttnn_layer_norm_1);  ttnn_experimental_view_17 = ttnn_layer_norm_1 = None
    ttnn_from_torch_23 = ttnn_decorators_ttnn_from_torch(arg19_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg19_1 = None
    ttnn_from_torch_24 = ttnn_decorators_ttnn_from_torch(arg20_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg20_1 = None
    ttnn_layer_norm_2 = ttnn_decorators_ttnn_layer_norm(ttnn_add_11, epsilon = 1e-12, weight = ttnn_from_torch_23, bias = ttnn_from_torch_24);  ttnn_add_11 = ttnn_from_torch_23 = ttnn_from_torch_24 = None
    ttnn_experimental_view_18 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_2, [256, 1024]);  ttnn_layer_norm_2 = None
    
    #LAYER ENDS HERE
    profiler.end(f"Naive_{i}")
    
    ttnn_from_torch_25 = ttnn_decorators_ttnn_from_torch(arg21_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg21_1 = None
    ttnn_transpose_7 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_25, 0, 1);  ttnn_from_torch_25 = None
    ttnn_matmul_8 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_18, ttnn_transpose_7); ttnn_transpose_7 = None
    ttnn_from_torch_26 = ttnn_decorators_ttnn_from_torch(arg22_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg22_1 = None
    ttnn_add_6 = ttnn_decorators_ttnn_add(ttnn_from_torch_26, ttnn_matmul_8);  ttnn_from_torch_26 = ttnn_matmul_8 = None
    ttnn_experimental_view_19 = ttnn_decorators_ttnn_experimental_view(ttnn_add_6, [1, 256, 2]);  ttnn_add_6 = None
    ttnn_to_layout_1 = ttnn_decorators_ttnn_to_layout(ttnn_experimental_view_19, ttnn_ROW_MAJOR_LAYOUT);  ttnn_experimental_view_19 = None
    ttnn_split = ttnn_decorators_ttnn_split(ttnn_to_layout_1, 1, 2);  ttnn_to_layout_1 = None
    getitem_3 = ttnn_split[0]
    getitem_4 = ttnn_split[1];  ttnn_split = None
    ttnn_to_layout_2 = ttnn_decorators_ttnn_to_layout(getitem_3, ttnn_TILE_LAYOUT);  getitem_3 = None
    ttnn_squeeze = ttnn_decorators_ttnn_squeeze(ttnn_to_layout_2, -1);  ttnn_to_layout_2 = None
    ttnn_to_layout_3 = ttnn_decorators_ttnn_to_layout(getitem_4, ttnn_TILE_LAYOUT);  getitem_4 = None
    ttnn_squeeze_1 = ttnn_decorators_ttnn_squeeze(ttnn_to_layout_3, -1);  ttnn_to_layout_3 = None
    ttnn_to_torch = ttnn_decorators_ttnn_to_torch(ttnn_squeeze, dtype = torch.bfloat16);  ttnn_squeeze = None
    ttnn_to_torch_1 = ttnn_decorators_ttnn_to_torch(ttnn_squeeze_1, dtype = torch.bfloat16);  ttnn_squeeze_1 = None
    return (ttnn_to_torch, ttnn_to_torch_1, ttnn_experimental_view_12)


# Smart means:
# ttnn.Linear
# KQV fusion and split heads
# more to come!
def smart(ttnn_Specified_Device, i, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1):
    ttnn_from_torch = ttnn_decorators_ttnn_from_torch(arg25_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_uint32);  arg25_1 = None
    ttnn_reshape = ttnn_decorators_ttnn_reshape(ttnn_from_torch, [1, 1, 256]);  ttnn_from_torch = None
    ttnn_reshape_1 = ttnn_decorators_ttnn_reshape(ttnn_reshape, [1, 1, 1, 256]);  ttnn_reshape = None
    ttnn_typecast = ttnn_decorators_ttnn_typecast(ttnn_reshape_1, ttnn_bfloat16);  ttnn_reshape_1 = None
    ttnn_rsub = ttnn_decorators_ttnn_rsub(ttnn_typecast, 1.0);  ttnn_typecast = None
    ttnn_multiply = ttnn_decorators_ttnn_multiply(ttnn_rsub, -3.3895313892515355e+38);  ttnn_rsub = None
    ttnn_from_torch_1 = ttnn_decorators_ttnn_from_torch(arg23_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg23_1 = None
    ttnn_slice = ttnn_decorators_ttnn_slice(ttnn_from_torch_1, [0, 0], [1, 256]);  ttnn_from_torch_1 = None
    ttnn_from_torch_2 = ttnn_decorators_ttnn_from_torch(arg24_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg24_1 = None
    ttnn_from_torch_3 = ttnn_decorators_ttnn_from_torch(arg0_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg0_1 = None
    ttnn_embedding = ttnn_decorators_ttnn_embedding(ttnn_from_torch_2, ttnn_from_torch_3, layout = ttnn_TILE_LAYOUT);  ttnn_from_torch_2 = ttnn_from_torch_3 = None
    ttnn_from_torch_4 = ttnn_decorators_ttnn_from_torch(arg26_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg26_1 = None
    ttnn_from_torch_5 = ttnn_decorators_ttnn_from_torch(arg1_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg1_1 = None
    ttnn_embedding_1 = ttnn_decorators_ttnn_embedding(ttnn_from_torch_4, ttnn_from_torch_5, layout = ttnn_TILE_LAYOUT);  ttnn_from_torch_4 = ttnn_from_torch_5 = None
    ttnn_add_7 = ttnn_decorators_ttnn_add(ttnn_embedding, ttnn_embedding_1);  ttnn_embedding = ttnn_embedding_1 = None
    ttnn_to_layout = ttnn_decorators_ttnn_to_layout(ttnn_slice, ttnn_ROW_MAJOR_LAYOUT);  ttnn_slice = None
    ttnn_from_torch_6 = ttnn_decorators_ttnn_from_torch(arg2_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg2_1 = None
    ttnn_embedding_2 = ttnn_decorators_ttnn_embedding(ttnn_to_layout, ttnn_from_torch_6, layout = ttnn_TILE_LAYOUT);  ttnn_to_layout = ttnn_from_torch_6 = None
    ttnn_add_8 = ttnn_decorators_ttnn_add(ttnn_add_7, ttnn_embedding_2);  ttnn_add_7 = ttnn_embedding_2 = None
    ttnn_from_torch_7 = ttnn_decorators_ttnn_from_torch(arg3_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg3_1 = None
    ttnn_from_torch_8 = ttnn_decorators_ttnn_from_torch(arg4_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg4_1 = None
    ttnn_layer_norm = ttnn_decorators_ttnn_layer_norm(ttnn_add_8, epsilon = 1e-12, weight = ttnn_from_torch_7, bias = ttnn_from_torch_8);  ttnn_add_8 = ttnn_from_torch_7 = ttnn_from_torch_8 = None
    ttnn_experimental_view = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm, [256, 1024])
    ttnn_from_torch_9 = ttnn_decorators_ttnn_from_torch(arg5_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg5_1 = None
    ttnn_transpose_8 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_9, -2, -1);  ttnn_from_torch_9 = None
    ttnn_from_torch_10 = ttnn_decorators_ttnn_from_torch(arg7_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg7_1 = None
    ttnn_transpose_9 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_10, -2, -1);  ttnn_from_torch_10 = None
    ttnn_from_torch_11 = ttnn_decorators_ttnn_from_torch(arg9_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg9_1 = None
    ttnn_transpose_10 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_11, -2, -1);  ttnn_from_torch_11 = None
    ttnn_concat = ttnn_decorators_ttnn_concat([ttnn_transpose_8, ttnn_transpose_9, ttnn_transpose_10], -1);  ttnn_transpose_8 = ttnn_transpose_9 = ttnn_transpose_10 = None
    ttnn_from_torch_12 = ttnn_decorators_ttnn_from_torch(arg6_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg6_1 = None
    ttnn_experimental_view_20 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_12, (1, -1));  ttnn_from_torch_12 = None
    ttnn_from_torch_13 = ttnn_decorators_ttnn_from_torch(arg8_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg8_1 = None
    ttnn_experimental_view_21 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_13, (1, -1));  ttnn_from_torch_13 = None
    ttnn_from_torch_14 = ttnn_decorators_ttnn_from_torch(arg10_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg10_1 = None
    ttnn_experimental_view_22 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_14, (1, -1));  ttnn_from_torch_14 = None
    ttnn_concat_1 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_20, ttnn_experimental_view_21, ttnn_experimental_view_22], -1);  ttnn_experimental_view_20 = ttnn_experimental_view_21 = ttnn_experimental_view_22 = None
    ttnn_to_layout_1 = ttnn_decorators_ttnn_to_layout(ttnn_concat, ttnn_TILE_LAYOUT);  ttnn_concat = None
    ttnn_to_layout_2 = ttnn_decorators_ttnn_to_layout(ttnn_concat_1, ttnn_TILE_LAYOUT);  ttnn_concat_1 = None
    ttnn_linear_7 = ttnn_decorators_ttnn_linear(ttnn_experimental_view, ttnn_to_layout_1, bias = ttnn_to_layout_2);  ttnn_to_layout_1 = ttnn_to_layout_2 = None
    ttnn_experimental_view_23 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_7, [1, 256, 3072]);  ttnn_linear_7 = None
    ttnn_transformer_split_query_key_value_and_split_heads = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_23, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_23 = None
    getitem_5 = ttnn_transformer_split_query_key_value_and_split_heads[0]
    getitem_6 = ttnn_transformer_split_query_key_value_and_split_heads[1]
    getitem_7 = ttnn_transformer_split_query_key_value_and_split_heads[2];  ttnn_transformer_split_query_key_value_and_split_heads = None
    ttnn_matmul_9 = ttnn_decorators_ttnn_matmul(getitem_5, getitem_6);  getitem_5 = getitem_6 = None
    ttnn_transformer_attention_softmax__1 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_9, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_9 = ttnn_multiply = None
    ttnn_matmul_10 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__1, getitem_7);  ttnn_transformer_attention_softmax__1 = getitem_7 = None
    ttnn_transformer_concatenate_heads = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_10);  ttnn_matmul_10 = None
    
    
    
    ttnn_from_torch_15 = ttnn_decorators_ttnn_from_torch(arg11_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg11_1 = None
    ttnn_from_torch_16 = ttnn_decorators_ttnn_from_torch(arg12_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg12_1 = None
    
    ttnn_linear_3 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads, ttnn_from_torch_15, transpose_b = True, bias = ttnn_from_torch_16, activation = None); ttnn_from_torch_15 = ttnn_from_torch_16 = None
    ttnn_experimental_view_13 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_3, [1, 256, 1024]);  ttnn_linear_3 = None
    ttnn_add_10 = ttnn_decorators_ttnn_add(ttnn_experimental_view_13, ttnn_layer_norm);  ttnn_experimental_view_13 = ttnn_layer_norm = None
    ttnn_from_torch_17 = ttnn_decorators_ttnn_from_torch(arg13_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg13_1 = None
    ttnn_from_torch_18 = ttnn_decorators_ttnn_from_torch(arg14_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg14_1 = None
    ttnn_layer_norm_1 = ttnn_decorators_ttnn_layer_norm(ttnn_add_10, epsilon = 1e-12, weight = ttnn_from_torch_17, bias = ttnn_from_torch_18);  ttnn_add_10 = ttnn_from_torch_17 = ttnn_from_torch_18 = None
    ttnn_experimental_view_14 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_1, [256, 1024])

    ttnn_from_torch_19 = ttnn_decorators_ttnn_from_torch(arg15_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg15_1 = None
    ttnn_from_torch_20 = ttnn_decorators_ttnn_from_torch(arg16_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg16_1 = None
    ttnn_linear_4 = ttnn_decorators_ttnn_linear(ttnn_layer_norm_1, ttnn_from_torch_19, transpose_b = True, bias = ttnn_from_torch_20, activation = 'gelu');  ttnn_experimental_view_14 = ttnn_from_torch_19 = ttnn_from_torch_20 = None
    ttnn_experimental_view_16 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_4, [256, 4096]);  ttnn_linear_4 = None
    ttnn_from_torch_21 = ttnn_decorators_ttnn_from_torch(arg17_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg17_1 = None
    ttnn_from_torch_22 = ttnn_decorators_ttnn_from_torch(arg18_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg18_1 = None
    ttnn_linear_5 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_16, ttnn_from_torch_21, transpose_b = True, bias = ttnn_from_torch_22, activation = None);  ttnn_experimental_view_16 = ttnn_from_torch_21 = ttnn_from_torch_22 = None
    ttnn_experimental_view_17 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_5, [1, 256, 1024]);  ttnn_linear_5 = None
    ttnn_add_11 = ttnn_decorators_ttnn_add(ttnn_experimental_view_17, ttnn_layer_norm_1);  ttnn_experimental_view_17 = ttnn_layer_norm_1 = None
    ttnn_from_torch_23 = ttnn_decorators_ttnn_from_torch(arg19_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg19_1 = None
    ttnn_from_torch_24 = ttnn_decorators_ttnn_from_torch(arg20_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg20_1 = None
    ttnn_layer_norm_2 = ttnn_decorators_ttnn_layer_norm(ttnn_add_11, epsilon = 1e-12, weight = ttnn_from_torch_23, bias = ttnn_from_torch_24);  ttnn_add_11 = ttnn_from_torch_23 = ttnn_from_torch_24 = None
    ttnn_experimental_view_18 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_2, [256, 1024]);  ttnn_layer_norm_2 = None
    ttnn_from_torch_25 = ttnn_decorators_ttnn_from_torch(arg21_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg21_1 = None
    ttnn_from_torch_26 = ttnn_decorators_ttnn_from_torch(arg22_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg22_1 = None
    ttnn_linear_6 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_18, ttnn_from_torch_25, transpose_b = True, bias = ttnn_from_torch_26, activation = None);  ttnn_experimental_view_18 = ttnn_from_torch_25 = ttnn_from_torch_26 = None
    ttnn_experimental_view_19 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_6, [1, 256, 2]);  ttnn_linear_6 = None
    ttnn_to_layout_3 = ttnn_decorators_ttnn_to_layout(ttnn_experimental_view_19, ttnn_ROW_MAJOR_LAYOUT);  ttnn_experimental_view_19 = None
    ttnn_split = ttnn_decorators_ttnn_split(ttnn_to_layout_3, 1, 2);  ttnn_to_layout_3 = None
    getitem_3 = ttnn_split[0]
    getitem_4 = ttnn_split[1];  ttnn_split = None
    ttnn_to_layout_4 = ttnn_decorators_ttnn_to_layout(getitem_3, ttnn_TILE_LAYOUT);  getitem_3 = None
    ttnn_squeeze = ttnn_decorators_ttnn_squeeze(ttnn_to_layout_4, -1);  ttnn_to_layout_4 = None
    ttnn_to_layout_5 = ttnn_decorators_ttnn_to_layout(getitem_4, ttnn_TILE_LAYOUT);  getitem_4 = None
    ttnn_squeeze_1 = ttnn_decorators_ttnn_squeeze(ttnn_to_layout_5, -1);  ttnn_to_layout_5 = None
    ttnn_to_torch = ttnn_decorators_ttnn_to_torch(ttnn_squeeze, dtype = torch.bfloat16);  ttnn_squeeze = None
    ttnn_to_torch_1 = ttnn_decorators_ttnn_to_torch(ttnn_squeeze_1, dtype = torch.bfloat16);  ttnn_squeeze_1 = None
    return (ttnn_to_torch, ttnn_to_torch_1, ttnn_transformer_concatenate_heads)


 
def run_bert(ttnn_Specified_Device, iteration):
     # Download model from cloud
    batch_size = 8
    model_name = "phiyodr/bert-large-finetuned-squad2"
    m = AutoModelForQuestionAnswering.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    m.config.num_hidden_layers = 1
    m.bert.encoder.layer = nn.ModuleList([m.bert.encoder.layer[0]])  # Keep only first layer
        
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", torch_dtype=torch.bfloat16)
    questions = [
        "What discipline did Winckelmann create?",
    ]

    contexts = [
        "Winckelmann was a German historian who laid the groundwork for art history and archaeology.",
    ]
    
    # Process each question-context pair
    encoded_inputs = []
    for question, context in zip(questions, contexts):
        encoded = tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=256,
            padding="max_length",
            truncation=True,
        )
        encoded_inputs.append(encoded)
    
    # Stack all encoded inputs
    inputs = type('inputs', (), {})()
    inputs.data = {
        'input_ids': torch.cat([x['input_ids'] for x in encoded_inputs], dim=0),
        'token_type_ids': torch.cat([x['token_type_ids'] for x in encoded_inputs], dim=0),
        'attention_mask': torch.cat([x['attention_mask'] for x in encoded_inputs], dim=0)
    }

    modules = {}
    modules['self'] = m
    
    full_args = []
    params = {
        **dict(m.named_parameters(remove_duplicate=False)),
        **dict(m.named_buffers(remove_duplicate=False)),
    }

    params_items = list(params.items())
    params_items[1], params_items[2] = params_items[2], params_items[1]
    params_items.pop()
    params = dict(params_items)
    params_flat, params_spec = pytree.tree_flatten(params)
    params_flat = list(params_flat)
    full_args.extend(params_flat)

    for i in range(len(params_items)):
        print(f"{i}: {params_items[i][0]}")
    ttnn_Specified_Device.disable_and_clear_program_cache()

    naive_ttnn_to_torch, naive_ttnn_to_torch1, naive_attention_result = naive(ttnn_Specified_Device, 0, *full_args, inputs.data['input_ids'], inputs.data['token_type_ids'], inputs.data['attention_mask'])
    ttnn_Specified_Device.enable_program_cache()

    for i in range(1, 5):
        naive(ttnn_Specified_Device, i, *full_args, inputs.data['input_ids'], inputs.data['token_type_ids'], inputs.data['attention_mask'])
    
    ttnn_Specified_Device.disable_and_clear_program_cache()

    smart_ttnn_to_torch, smart_ttnn_to_torch1, smart_attention_result = smart(ttnn_Specified_Device, 0, *full_args, inputs.data['input_ids'], inputs.data['token_type_ids'], inputs.data['attention_mask'])
    ttnn_Specified_Device.enable_program_cache()

    assert smart_ttnn_to_torch[0] == naive_ttnn_to_torch[0]
    assert smart_ttnn_to_torch1[0] == naive_ttnn_to_torch1[0]
    for i in range(1, 5):
        smart(ttnn_Specified_Device, i, *full_args, inputs.data['input_ids'], inputs.data['token_type_ids'], inputs.data['attention_mask'])
    profiler.disable()
      
if __name__ == "__main__":
    device_id = 0
    ttnn_Specified_Device = ttnn.open_device(device_id=device_id)
    disable_persistent_kernel_cache()
    profiler.enable()
    run_bert(ttnn_Specified_Device, 0)
    ttnn.close_device(ttnn_Specified_Device)
    for i in range(5):
        smart_value = profiler.get(f"Smart_{i}")
        naive_value = profiler.get(f"Naive_{i}")
        print(f"Runtime of one layer smart Iteration {i}: {smart_value}")
        print(f"Runtime of one layer naive Iteration {i}: {naive_value}")