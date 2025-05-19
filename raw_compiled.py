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

# model after https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1024
def after_attention(ttnn_Specified_Device, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1):
    ttnn_from_torch = ttnn_decorators_ttnn_from_torch(arg393_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_uint32);  arg393_1 = None
    ttnn_reshape = ttnn_decorators_ttnn_reshape(ttnn_from_torch, [1, 1, 256]);  ttnn_from_torch = None
    ttnn_reshape_1 = ttnn_decorators_ttnn_reshape(ttnn_reshape, [1, 1, 1, 256]);  ttnn_reshape = None
    ttnn_typecast = ttnn_decorators_ttnn_typecast(ttnn_reshape_1, ttnn_bfloat16);  ttnn_reshape_1 = None
    ttnn_rsub = ttnn_decorators_ttnn_rsub(ttnn_typecast, 1.0);  ttnn_typecast = None
    ttnn_multiply = ttnn_decorators_ttnn_multiply(ttnn_rsub, -3.3895313892515355e+38);  ttnn_rsub = None
    ttnn_from_torch_1 = ttnn_decorators_ttnn_from_torch(arg391_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg391_1 = None
    ttnn_slice = ttnn_decorators_ttnn_slice(ttnn_from_torch_1, [0, 0], [1, 256]);  ttnn_from_torch_1 = None
    ttnn_from_torch_2 = ttnn_decorators_ttnn_from_torch(arg392_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg392_1 = None
    ttnn_from_torch_3 = ttnn_decorators_ttnn_from_torch(arg0_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg0_1 = None
    ttnn_embedding = ttnn_decorators_ttnn_embedding(ttnn_from_torch_2, ttnn_from_torch_3, layout = ttnn_TILE_LAYOUT);  ttnn_from_torch_2 = ttnn_from_torch_3 = None
    ttnn_from_torch_4 = ttnn_decorators_ttnn_from_torch(arg394_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg394_1 = None
    ttnn_from_torch_5 = ttnn_decorators_ttnn_from_torch(arg1_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg1_1 = None
    ttnn_embedding_1 = ttnn_decorators_ttnn_embedding(ttnn_from_torch_4, ttnn_from_torch_5, layout = ttnn_TILE_LAYOUT);  ttnn_from_torch_4 = ttnn_from_torch_5 = None
    ttnn_add_145 = ttnn_decorators_ttnn_add(ttnn_embedding, ttnn_embedding_1);  ttnn_embedding = ttnn_embedding_1 = None
    ttnn_to_layout = ttnn_decorators_ttnn_to_layout(ttnn_slice, ttnn_ROW_MAJOR_LAYOUT);  ttnn_slice = None
    ttnn_from_torch_6 = ttnn_decorators_ttnn_from_torch(arg2_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg2_1 = None
    ttnn_embedding_2 = ttnn_decorators_ttnn_embedding(ttnn_to_layout, ttnn_from_torch_6, layout = ttnn_TILE_LAYOUT);  ttnn_to_layout = ttnn_from_torch_6 = None
    ttnn_add_146 = ttnn_decorators_ttnn_add(ttnn_add_145, ttnn_embedding_2);  ttnn_add_145 = ttnn_embedding_2 = None
    ttnn_from_torch_7 = ttnn_decorators_ttnn_from_torch(arg3_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg3_1 = None
    ttnn_from_torch_8 = ttnn_decorators_ttnn_from_torch(arg4_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg4_1 = None
    ttnn_layer_norm = ttnn_decorators_ttnn_layer_norm(ttnn_add_146, epsilon = 1e-12, weight = ttnn_from_torch_7, bias = ttnn_from_torch_8);  ttnn_add_146 = ttnn_from_torch_7 = ttnn_from_torch_8 = None
    ttnn_experimental_view = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm, [256, 1024])
    ttnn_from_torch_9 = ttnn_decorators_ttnn_from_torch(arg5_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg5_1 = None
    ttnn_transpose_169 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_9, -2, -1);  ttnn_from_torch_9 = None
    ttnn_from_torch_10 = ttnn_decorators_ttnn_from_torch(arg7_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg7_1 = None
    ttnn_transpose_170 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_10, -2, -1);  ttnn_from_torch_10 = None
    ttnn_from_torch_11 = ttnn_decorators_ttnn_from_torch(arg9_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg9_1 = None
    ttnn_transpose_171 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_11, -2, -1);  ttnn_from_torch_11 = None
    ttnn_concat = ttnn_decorators_ttnn_concat([ttnn_transpose_169, ttnn_transpose_170, ttnn_transpose_171], -1);  ttnn_transpose_169 = ttnn_transpose_170 = ttnn_transpose_171 = None
    ttnn_from_torch_12 = ttnn_decorators_ttnn_from_torch(arg6_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg6_1 = None
    ttnn_experimental_view_434 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_12, (1, -1));  ttnn_from_torch_12 = None
    ttnn_from_torch_13 = ttnn_decorators_ttnn_from_torch(arg8_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg8_1 = None
    ttnn_experimental_view_435 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_13, (1, -1));  ttnn_from_torch_13 = None
    ttnn_from_torch_14 = ttnn_decorators_ttnn_from_torch(arg10_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg10_1 = None
    ttnn_experimental_view_436 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_14, (1, -1));  ttnn_from_torch_14 = None
    ttnn_concat_1 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_434, ttnn_experimental_view_435, ttnn_experimental_view_436], -1);  ttnn_experimental_view_434 = ttnn_experimental_view_435 = ttnn_experimental_view_436 = None
    ttnn_to_layout_1 = ttnn_decorators_ttnn_to_layout(ttnn_concat, ttnn_TILE_LAYOUT);  ttnn_concat = None
    ttnn_to_layout_2 = ttnn_decorators_ttnn_to_layout(ttnn_concat_1, ttnn_TILE_LAYOUT);  ttnn_concat_1 = None
    ttnn_linear_145 = ttnn_decorators_ttnn_linear(ttnn_experimental_view, ttnn_to_layout_1, bias = ttnn_to_layout_2);  ttnn_experimental_view = ttnn_to_layout_1 = ttnn_to_layout_2 = None
    ttnn_experimental_view_437 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_145, [1, 256, 3072]);  ttnn_linear_145 = None
    ttnn_transformer_split_query_key_value_and_split_heads = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_437, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_437 = None
    getitem_51 = ttnn_transformer_split_query_key_value_and_split_heads[0]
    getitem_52 = ttnn_transformer_split_query_key_value_and_split_heads[1]
    getitem_53 = ttnn_transformer_split_query_key_value_and_split_heads[2];  ttnn_transformer_split_query_key_value_and_split_heads = None
    ttnn_matmul_193 = ttnn_decorators_ttnn_matmul(getitem_51, getitem_52);  getitem_51 = getitem_52 = None
    ttnn_transformer_attention_softmax__24 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_193, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_193 = None
    ttnn_matmul_194 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__24, getitem_53);  ttnn_transformer_attention_softmax__24 = getitem_53 = None
    ttnn_transformer_concatenate_heads = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_194);  ttnn_matmul_194 = None
    ttnn_from_torch_15 = ttnn_decorators_ttnn_from_torch(arg11_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg11_1 = None
    ttnn_from_torch_16 = ttnn_decorators_ttnn_from_torch(arg12_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg12_1 = None
    ttnn_linear_3 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads, ttnn_from_torch_15, transpose_b = True, bias = ttnn_from_torch_16, activation = None);  ttnn_transformer_concatenate_heads = ttnn_from_torch_15 = ttnn_from_torch_16 = None
    ttnn_experimental_view_13 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_3, [1, 256, 1024]);  ttnn_linear_3 = None
    ttnn_add_148 = ttnn_decorators_ttnn_add(ttnn_experimental_view_13, ttnn_layer_norm);  ttnn_experimental_view_13 = ttnn_layer_norm = None
    ttnn_from_torch_17 = ttnn_decorators_ttnn_from_torch(arg13_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg13_1 = None
    ttnn_from_torch_18 = ttnn_decorators_ttnn_from_torch(arg14_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg14_1 = None
    ttnn_layer_norm_1 = ttnn_decorators_ttnn_layer_norm(ttnn_add_148, epsilon = 1e-12, weight = ttnn_from_torch_17, bias = ttnn_from_torch_18);  ttnn_add_148 = ttnn_from_torch_17 = ttnn_from_torch_18 = None
    ttnn_experimental_view_14 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_1, [256, 1024])
    ttnn_from_torch_19 = ttnn_decorators_ttnn_from_torch(arg15_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg15_1 = None
    ttnn_from_torch_20 = ttnn_decorators_ttnn_from_torch(arg16_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg16_1 = None
    ttnn_linear_4 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_14, ttnn_from_torch_19, transpose_b = True, bias = ttnn_from_torch_20, activation = 'gelu');  ttnn_experimental_view_14 = ttnn_from_torch_19 = ttnn_from_torch_20 = None
    ttnn_experimental_view_16 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_4, [256, 4096]);  ttnn_linear_4 = None
    ttnn_from_torch_21 = ttnn_decorators_ttnn_from_torch(arg17_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg17_1 = None
    ttnn_from_torch_22 = ttnn_decorators_ttnn_from_torch(arg18_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg18_1 = None
    ttnn_linear_5 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_16, ttnn_from_torch_21, transpose_b = True, bias = ttnn_from_torch_22, activation = None);  ttnn_experimental_view_16 = ttnn_from_torch_21 = ttnn_from_torch_22 = None
    ttnn_experimental_view_17 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_5, [1, 256, 1024]);  ttnn_linear_5 = None
    ttnn_add_149 = ttnn_decorators_ttnn_add(ttnn_experimental_view_17, ttnn_layer_norm_1);  ttnn_experimental_view_17 = ttnn_layer_norm_1 = None
    ttnn_from_torch_23 = ttnn_decorators_ttnn_from_torch(arg19_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg19_1 = None
    ttnn_from_torch_24 = ttnn_decorators_ttnn_from_torch(arg20_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg20_1 = None
    ttnn_layer_norm_2 = ttnn_decorators_ttnn_layer_norm(ttnn_add_149, epsilon = 1e-12, weight = ttnn_from_torch_23, bias = ttnn_from_torch_24);  ttnn_add_149 = ttnn_from_torch_23 = ttnn_from_torch_24 = None
    ttnn_experimental_view_18 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_2, [256, 1024])
    ttnn_from_torch_25 = ttnn_decorators_ttnn_from_torch(arg21_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg21_1 = None
    ttnn_transpose_172 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_25, -2, -1);  ttnn_from_torch_25 = None
    ttnn_from_torch_26 = ttnn_decorators_ttnn_from_torch(arg23_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg23_1 = None
    ttnn_transpose_173 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_26, -2, -1);  ttnn_from_torch_26 = None
    ttnn_from_torch_27 = ttnn_decorators_ttnn_from_torch(arg25_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg25_1 = None
    ttnn_transpose_174 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_27, -2, -1);  ttnn_from_torch_27 = None
    ttnn_concat_2 = ttnn_decorators_ttnn_concat([ttnn_transpose_172, ttnn_transpose_173, ttnn_transpose_174], -1);  ttnn_transpose_172 = ttnn_transpose_173 = ttnn_transpose_174 = None
    ttnn_from_torch_28 = ttnn_decorators_ttnn_from_torch(arg22_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg22_1 = None
    ttnn_experimental_view_438 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_28, (1, -1));  ttnn_from_torch_28 = None
    ttnn_from_torch_29 = ttnn_decorators_ttnn_from_torch(arg24_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg24_1 = None
    ttnn_experimental_view_439 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_29, (1, -1));  ttnn_from_torch_29 = None
    ttnn_from_torch_30 = ttnn_decorators_ttnn_from_torch(arg26_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg26_1 = None
    ttnn_experimental_view_440 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_30, (1, -1));  ttnn_from_torch_30 = None
    ttnn_concat_3 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_438, ttnn_experimental_view_439, ttnn_experimental_view_440], -1);  ttnn_experimental_view_438 = ttnn_experimental_view_439 = ttnn_experimental_view_440 = None
    ttnn_to_layout_3 = ttnn_decorators_ttnn_to_layout(ttnn_concat_2, ttnn_TILE_LAYOUT);  ttnn_concat_2 = None
    ttnn_to_layout_4 = ttnn_decorators_ttnn_to_layout(ttnn_concat_3, ttnn_TILE_LAYOUT);  ttnn_concat_3 = None
    ttnn_linear_146 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_18, ttnn_to_layout_3, bias = ttnn_to_layout_4);  ttnn_experimental_view_18 = ttnn_to_layout_3 = ttnn_to_layout_4 = None
    ttnn_experimental_view_441 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_146, [1, 256, 3072]);  ttnn_linear_146 = None
    ttnn_transformer_split_query_key_value_and_split_heads_1 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_441, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_441 = None
    getitem_54 = ttnn_transformer_split_query_key_value_and_split_heads_1[0]
    getitem_55 = ttnn_transformer_split_query_key_value_and_split_heads_1[1]
    getitem_56 = ttnn_transformer_split_query_key_value_and_split_heads_1[2];  ttnn_transformer_split_query_key_value_and_split_heads_1 = None
    ttnn_matmul_195 = ttnn_decorators_ttnn_matmul(getitem_54, getitem_55);  getitem_54 = getitem_55 = None
    ttnn_transformer_attention_softmax__25 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_195, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_195 = None
    ttnn_matmul_196 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__25, getitem_56);  ttnn_transformer_attention_softmax__25 = getitem_56 = None
    ttnn_transformer_concatenate_heads_1 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_196);  ttnn_matmul_196 = None
    ttnn_from_torch_31 = ttnn_decorators_ttnn_from_torch(arg27_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg27_1 = None
    ttnn_from_torch_32 = ttnn_decorators_ttnn_from_torch(arg28_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg28_1 = None
    ttnn_linear_9 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_1, ttnn_from_torch_31, transpose_b = True, bias = ttnn_from_torch_32, activation = None);  ttnn_transformer_concatenate_heads_1 = ttnn_from_torch_31 = ttnn_from_torch_32 = None
    ttnn_experimental_view_31 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_9, [1, 256, 1024]);  ttnn_linear_9 = None
    ttnn_add_151 = ttnn_decorators_ttnn_add(ttnn_experimental_view_31, ttnn_layer_norm_2);  ttnn_experimental_view_31 = ttnn_layer_norm_2 = None
    ttnn_from_torch_33 = ttnn_decorators_ttnn_from_torch(arg29_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg29_1 = None
    ttnn_from_torch_34 = ttnn_decorators_ttnn_from_torch(arg30_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg30_1 = None
    ttnn_layer_norm_3 = ttnn_decorators_ttnn_layer_norm(ttnn_add_151, epsilon = 1e-12, weight = ttnn_from_torch_33, bias = ttnn_from_torch_34);  ttnn_add_151 = ttnn_from_torch_33 = ttnn_from_torch_34 = None
    ttnn_experimental_view_32 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_3, [256, 1024])
    ttnn_from_torch_35 = ttnn_decorators_ttnn_from_torch(arg31_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg31_1 = None
    ttnn_from_torch_36 = ttnn_decorators_ttnn_from_torch(arg32_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg32_1 = None
    ttnn_linear_10 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_32, ttnn_from_torch_35, transpose_b = True, bias = ttnn_from_torch_36, activation = 'gelu');  ttnn_experimental_view_32 = ttnn_from_torch_35 = ttnn_from_torch_36 = None
    ttnn_experimental_view_34 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_10, [256, 4096]);  ttnn_linear_10 = None
    ttnn_from_torch_37 = ttnn_decorators_ttnn_from_torch(arg33_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg33_1 = None
    ttnn_from_torch_38 = ttnn_decorators_ttnn_from_torch(arg34_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg34_1 = None
    ttnn_linear_11 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_34, ttnn_from_torch_37, transpose_b = True, bias = ttnn_from_torch_38, activation = None);  ttnn_experimental_view_34 = ttnn_from_torch_37 = ttnn_from_torch_38 = None
    ttnn_experimental_view_35 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_11, [1, 256, 1024]);  ttnn_linear_11 = None
    ttnn_add_152 = ttnn_decorators_ttnn_add(ttnn_experimental_view_35, ttnn_layer_norm_3);  ttnn_experimental_view_35 = ttnn_layer_norm_3 = None
    ttnn_from_torch_39 = ttnn_decorators_ttnn_from_torch(arg35_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg35_1 = None
    ttnn_from_torch_40 = ttnn_decorators_ttnn_from_torch(arg36_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg36_1 = None
    ttnn_layer_norm_4 = ttnn_decorators_ttnn_layer_norm(ttnn_add_152, epsilon = 1e-12, weight = ttnn_from_torch_39, bias = ttnn_from_torch_40);  ttnn_add_152 = ttnn_from_torch_39 = ttnn_from_torch_40 = None
    ttnn_experimental_view_36 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_4, [256, 1024])
    ttnn_from_torch_41 = ttnn_decorators_ttnn_from_torch(arg37_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg37_1 = None
    ttnn_transpose_175 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_41, -2, -1);  ttnn_from_torch_41 = None
    ttnn_from_torch_42 = ttnn_decorators_ttnn_from_torch(arg39_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg39_1 = None
    ttnn_transpose_176 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_42, -2, -1);  ttnn_from_torch_42 = None
    ttnn_from_torch_43 = ttnn_decorators_ttnn_from_torch(arg41_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg41_1 = None
    ttnn_transpose_177 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_43, -2, -1);  ttnn_from_torch_43 = None
    ttnn_concat_4 = ttnn_decorators_ttnn_concat([ttnn_transpose_175, ttnn_transpose_176, ttnn_transpose_177], -1);  ttnn_transpose_175 = ttnn_transpose_176 = ttnn_transpose_177 = None
    ttnn_from_torch_44 = ttnn_decorators_ttnn_from_torch(arg38_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg38_1 = None
    ttnn_experimental_view_442 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_44, (1, -1));  ttnn_from_torch_44 = None
    ttnn_from_torch_45 = ttnn_decorators_ttnn_from_torch(arg40_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg40_1 = None
    ttnn_experimental_view_443 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_45, (1, -1));  ttnn_from_torch_45 = None
    ttnn_from_torch_46 = ttnn_decorators_ttnn_from_torch(arg42_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg42_1 = None
    ttnn_experimental_view_444 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_46, (1, -1));  ttnn_from_torch_46 = None
    ttnn_concat_5 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_442, ttnn_experimental_view_443, ttnn_experimental_view_444], -1);  ttnn_experimental_view_442 = ttnn_experimental_view_443 = ttnn_experimental_view_444 = None
    ttnn_to_layout_5 = ttnn_decorators_ttnn_to_layout(ttnn_concat_4, ttnn_TILE_LAYOUT);  ttnn_concat_4 = None
    ttnn_to_layout_6 = ttnn_decorators_ttnn_to_layout(ttnn_concat_5, ttnn_TILE_LAYOUT);  ttnn_concat_5 = None
    ttnn_linear_147 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_36, ttnn_to_layout_5, bias = ttnn_to_layout_6);  ttnn_experimental_view_36 = ttnn_to_layout_5 = ttnn_to_layout_6 = None
    ttnn_experimental_view_445 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_147, [1, 256, 3072]);  ttnn_linear_147 = None
    ttnn_transformer_split_query_key_value_and_split_heads_2 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_445, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_445 = None
    getitem_57 = ttnn_transformer_split_query_key_value_and_split_heads_2[0]
    getitem_58 = ttnn_transformer_split_query_key_value_and_split_heads_2[1]
    getitem_59 = ttnn_transformer_split_query_key_value_and_split_heads_2[2];  ttnn_transformer_split_query_key_value_and_split_heads_2 = None
    ttnn_matmul_197 = ttnn_decorators_ttnn_matmul(getitem_57, getitem_58);  getitem_57 = getitem_58 = None
    ttnn_transformer_attention_softmax__26 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_197, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_197 = None
    ttnn_matmul_198 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__26, getitem_59);  ttnn_transformer_attention_softmax__26 = getitem_59 = None
    ttnn_transformer_concatenate_heads_2 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_198);  ttnn_matmul_198 = None
    ttnn_from_torch_47 = ttnn_decorators_ttnn_from_torch(arg43_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg43_1 = None
    ttnn_from_torch_48 = ttnn_decorators_ttnn_from_torch(arg44_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg44_1 = None
    ttnn_linear_15 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_2, ttnn_from_torch_47, transpose_b = True, bias = ttnn_from_torch_48, activation = None);  ttnn_transformer_concatenate_heads_2 = ttnn_from_torch_47 = ttnn_from_torch_48 = None
    ttnn_experimental_view_49 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_15, [1, 256, 1024]);  ttnn_linear_15 = None
    ttnn_add_154 = ttnn_decorators_ttnn_add(ttnn_experimental_view_49, ttnn_layer_norm_4);  ttnn_experimental_view_49 = ttnn_layer_norm_4 = None
    ttnn_from_torch_49 = ttnn_decorators_ttnn_from_torch(arg45_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg45_1 = None
    ttnn_from_torch_50 = ttnn_decorators_ttnn_from_torch(arg46_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg46_1 = None
    ttnn_layer_norm_5 = ttnn_decorators_ttnn_layer_norm(ttnn_add_154, epsilon = 1e-12, weight = ttnn_from_torch_49, bias = ttnn_from_torch_50);  ttnn_add_154 = ttnn_from_torch_49 = ttnn_from_torch_50 = None
    ttnn_experimental_view_50 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_5, [256, 1024])
    ttnn_from_torch_51 = ttnn_decorators_ttnn_from_torch(arg47_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg47_1 = None
    ttnn_from_torch_52 = ttnn_decorators_ttnn_from_torch(arg48_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg48_1 = None
    ttnn_linear_16 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_50, ttnn_from_torch_51, transpose_b = True, bias = ttnn_from_torch_52, activation = 'gelu');  ttnn_experimental_view_50 = ttnn_from_torch_51 = ttnn_from_torch_52 = None
    ttnn_experimental_view_52 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_16, [256, 4096]);  ttnn_linear_16 = None
    ttnn_from_torch_53 = ttnn_decorators_ttnn_from_torch(arg49_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg49_1 = None
    ttnn_from_torch_54 = ttnn_decorators_ttnn_from_torch(arg50_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg50_1 = None
    ttnn_linear_17 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_52, ttnn_from_torch_53, transpose_b = True, bias = ttnn_from_torch_54, activation = None);  ttnn_experimental_view_52 = ttnn_from_torch_53 = ttnn_from_torch_54 = None
    ttnn_experimental_view_53 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_17, [1, 256, 1024]);  ttnn_linear_17 = None
    ttnn_add_155 = ttnn_decorators_ttnn_add(ttnn_experimental_view_53, ttnn_layer_norm_5);  ttnn_experimental_view_53 = ttnn_layer_norm_5 = None
    ttnn_from_torch_55 = ttnn_decorators_ttnn_from_torch(arg51_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg51_1 = None
    ttnn_from_torch_56 = ttnn_decorators_ttnn_from_torch(arg52_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg52_1 = None
    ttnn_layer_norm_6 = ttnn_decorators_ttnn_layer_norm(ttnn_add_155, epsilon = 1e-12, weight = ttnn_from_torch_55, bias = ttnn_from_torch_56);  ttnn_add_155 = ttnn_from_torch_55 = ttnn_from_torch_56 = None
    ttnn_experimental_view_54 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_6, [256, 1024])
    ttnn_from_torch_57 = ttnn_decorators_ttnn_from_torch(arg53_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg53_1 = None
    ttnn_transpose_178 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_57, -2, -1);  ttnn_from_torch_57 = None
    ttnn_from_torch_58 = ttnn_decorators_ttnn_from_torch(arg55_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg55_1 = None
    ttnn_transpose_179 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_58, -2, -1);  ttnn_from_torch_58 = None
    ttnn_from_torch_59 = ttnn_decorators_ttnn_from_torch(arg57_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg57_1 = None
    ttnn_transpose_180 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_59, -2, -1);  ttnn_from_torch_59 = None
    ttnn_concat_6 = ttnn_decorators_ttnn_concat([ttnn_transpose_178, ttnn_transpose_179, ttnn_transpose_180], -1);  ttnn_transpose_178 = ttnn_transpose_179 = ttnn_transpose_180 = None
    ttnn_from_torch_60 = ttnn_decorators_ttnn_from_torch(arg54_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg54_1 = None
    ttnn_experimental_view_446 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_60, (1, -1));  ttnn_from_torch_60 = None
    ttnn_from_torch_61 = ttnn_decorators_ttnn_from_torch(arg56_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg56_1 = None
    ttnn_experimental_view_447 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_61, (1, -1));  ttnn_from_torch_61 = None
    ttnn_from_torch_62 = ttnn_decorators_ttnn_from_torch(arg58_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg58_1 = None
    ttnn_experimental_view_448 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_62, (1, -1));  ttnn_from_torch_62 = None
    ttnn_concat_7 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_446, ttnn_experimental_view_447, ttnn_experimental_view_448], -1);  ttnn_experimental_view_446 = ttnn_experimental_view_447 = ttnn_experimental_view_448 = None
    ttnn_to_layout_7 = ttnn_decorators_ttnn_to_layout(ttnn_concat_6, ttnn_TILE_LAYOUT);  ttnn_concat_6 = None
    ttnn_to_layout_8 = ttnn_decorators_ttnn_to_layout(ttnn_concat_7, ttnn_TILE_LAYOUT);  ttnn_concat_7 = None
    ttnn_linear_148 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_54, ttnn_to_layout_7, bias = ttnn_to_layout_8);  ttnn_experimental_view_54 = ttnn_to_layout_7 = ttnn_to_layout_8 = None
    ttnn_experimental_view_449 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_148, [1, 256, 3072]);  ttnn_linear_148 = None
    ttnn_transformer_split_query_key_value_and_split_heads_3 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_449, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_449 = None
    getitem_60 = ttnn_transformer_split_query_key_value_and_split_heads_3[0]
    getitem_61 = ttnn_transformer_split_query_key_value_and_split_heads_3[1]
    getitem_62 = ttnn_transformer_split_query_key_value_and_split_heads_3[2];  ttnn_transformer_split_query_key_value_and_split_heads_3 = None
    ttnn_matmul_199 = ttnn_decorators_ttnn_matmul(getitem_60, getitem_61);  getitem_60 = getitem_61 = None
    ttnn_transformer_attention_softmax__27 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_199, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_199 = None
    ttnn_matmul_200 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__27, getitem_62);  ttnn_transformer_attention_softmax__27 = getitem_62 = None
    ttnn_transformer_concatenate_heads_3 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_200);  ttnn_matmul_200 = None
    ttnn_from_torch_63 = ttnn_decorators_ttnn_from_torch(arg59_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg59_1 = None
    ttnn_from_torch_64 = ttnn_decorators_ttnn_from_torch(arg60_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg60_1 = None
    ttnn_linear_21 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_3, ttnn_from_torch_63, transpose_b = True, bias = ttnn_from_torch_64, activation = None);  ttnn_transformer_concatenate_heads_3 = ttnn_from_torch_63 = ttnn_from_torch_64 = None
    ttnn_experimental_view_67 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_21, [1, 256, 1024]);  ttnn_linear_21 = None
    ttnn_add_157 = ttnn_decorators_ttnn_add(ttnn_experimental_view_67, ttnn_layer_norm_6);  ttnn_experimental_view_67 = ttnn_layer_norm_6 = None
    ttnn_from_torch_65 = ttnn_decorators_ttnn_from_torch(arg61_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg61_1 = None
    ttnn_from_torch_66 = ttnn_decorators_ttnn_from_torch(arg62_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg62_1 = None
    ttnn_layer_norm_7 = ttnn_decorators_ttnn_layer_norm(ttnn_add_157, epsilon = 1e-12, weight = ttnn_from_torch_65, bias = ttnn_from_torch_66);  ttnn_add_157 = ttnn_from_torch_65 = ttnn_from_torch_66 = None
    ttnn_experimental_view_68 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_7, [256, 1024])
    ttnn_from_torch_67 = ttnn_decorators_ttnn_from_torch(arg63_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg63_1 = None
    ttnn_from_torch_68 = ttnn_decorators_ttnn_from_torch(arg64_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg64_1 = None
    ttnn_linear_22 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_68, ttnn_from_torch_67, transpose_b = True, bias = ttnn_from_torch_68, activation = 'gelu');  ttnn_experimental_view_68 = ttnn_from_torch_67 = ttnn_from_torch_68 = None
    ttnn_experimental_view_70 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_22, [256, 4096]);  ttnn_linear_22 = None
    ttnn_from_torch_69 = ttnn_decorators_ttnn_from_torch(arg65_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg65_1 = None
    ttnn_from_torch_70 = ttnn_decorators_ttnn_from_torch(arg66_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg66_1 = None
    ttnn_linear_23 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_70, ttnn_from_torch_69, transpose_b = True, bias = ttnn_from_torch_70, activation = None);  ttnn_experimental_view_70 = ttnn_from_torch_69 = ttnn_from_torch_70 = None
    ttnn_experimental_view_71 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_23, [1, 256, 1024]);  ttnn_linear_23 = None
    ttnn_add_158 = ttnn_decorators_ttnn_add(ttnn_experimental_view_71, ttnn_layer_norm_7);  ttnn_experimental_view_71 = ttnn_layer_norm_7 = None
    ttnn_from_torch_71 = ttnn_decorators_ttnn_from_torch(arg67_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg67_1 = None
    ttnn_from_torch_72 = ttnn_decorators_ttnn_from_torch(arg68_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg68_1 = None
    ttnn_layer_norm_8 = ttnn_decorators_ttnn_layer_norm(ttnn_add_158, epsilon = 1e-12, weight = ttnn_from_torch_71, bias = ttnn_from_torch_72);  ttnn_add_158 = ttnn_from_torch_71 = ttnn_from_torch_72 = None
    ttnn_experimental_view_72 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_8, [256, 1024])
    ttnn_from_torch_73 = ttnn_decorators_ttnn_from_torch(arg69_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg69_1 = None
    ttnn_transpose_181 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_73, -2, -1);  ttnn_from_torch_73 = None
    ttnn_from_torch_74 = ttnn_decorators_ttnn_from_torch(arg71_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg71_1 = None
    ttnn_transpose_182 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_74, -2, -1);  ttnn_from_torch_74 = None
    ttnn_from_torch_75 = ttnn_decorators_ttnn_from_torch(arg73_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg73_1 = None
    ttnn_transpose_183 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_75, -2, -1);  ttnn_from_torch_75 = None
    ttnn_concat_8 = ttnn_decorators_ttnn_concat([ttnn_transpose_181, ttnn_transpose_182, ttnn_transpose_183], -1);  ttnn_transpose_181 = ttnn_transpose_182 = ttnn_transpose_183 = None
    ttnn_from_torch_76 = ttnn_decorators_ttnn_from_torch(arg70_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg70_1 = None
    ttnn_experimental_view_450 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_76, (1, -1));  ttnn_from_torch_76 = None
    ttnn_from_torch_77 = ttnn_decorators_ttnn_from_torch(arg72_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg72_1 = None
    ttnn_experimental_view_451 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_77, (1, -1));  ttnn_from_torch_77 = None
    ttnn_from_torch_78 = ttnn_decorators_ttnn_from_torch(arg74_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg74_1 = None
    ttnn_experimental_view_452 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_78, (1, -1));  ttnn_from_torch_78 = None
    ttnn_concat_9 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_450, ttnn_experimental_view_451, ttnn_experimental_view_452], -1);  ttnn_experimental_view_450 = ttnn_experimental_view_451 = ttnn_experimental_view_452 = None
    ttnn_to_layout_9 = ttnn_decorators_ttnn_to_layout(ttnn_concat_8, ttnn_TILE_LAYOUT);  ttnn_concat_8 = None
    ttnn_to_layout_10 = ttnn_decorators_ttnn_to_layout(ttnn_concat_9, ttnn_TILE_LAYOUT);  ttnn_concat_9 = None
    ttnn_linear_149 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_72, ttnn_to_layout_9, bias = ttnn_to_layout_10);  ttnn_experimental_view_72 = ttnn_to_layout_9 = ttnn_to_layout_10 = None
    ttnn_experimental_view_453 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_149, [1, 256, 3072]);  ttnn_linear_149 = None
    ttnn_transformer_split_query_key_value_and_split_heads_4 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_453, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_453 = None
    getitem_63 = ttnn_transformer_split_query_key_value_and_split_heads_4[0]
    getitem_64 = ttnn_transformer_split_query_key_value_and_split_heads_4[1]
    getitem_65 = ttnn_transformer_split_query_key_value_and_split_heads_4[2];  ttnn_transformer_split_query_key_value_and_split_heads_4 = None
    ttnn_matmul_201 = ttnn_decorators_ttnn_matmul(getitem_63, getitem_64);  getitem_63 = getitem_64 = None
    ttnn_transformer_attention_softmax__28 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_201, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_201 = None
    ttnn_matmul_202 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__28, getitem_65);  ttnn_transformer_attention_softmax__28 = getitem_65 = None
    ttnn_transformer_concatenate_heads_4 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_202);  ttnn_matmul_202 = None
    ttnn_from_torch_79 = ttnn_decorators_ttnn_from_torch(arg75_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg75_1 = None
    ttnn_from_torch_80 = ttnn_decorators_ttnn_from_torch(arg76_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg76_1 = None
    ttnn_linear_27 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_4, ttnn_from_torch_79, transpose_b = True, bias = ttnn_from_torch_80, activation = None);  ttnn_transformer_concatenate_heads_4 = ttnn_from_torch_79 = ttnn_from_torch_80 = None
    ttnn_experimental_view_85 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_27, [1, 256, 1024]);  ttnn_linear_27 = None
    ttnn_add_160 = ttnn_decorators_ttnn_add(ttnn_experimental_view_85, ttnn_layer_norm_8);  ttnn_experimental_view_85 = ttnn_layer_norm_8 = None
    ttnn_from_torch_81 = ttnn_decorators_ttnn_from_torch(arg77_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg77_1 = None
    ttnn_from_torch_82 = ttnn_decorators_ttnn_from_torch(arg78_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg78_1 = None
    ttnn_layer_norm_9 = ttnn_decorators_ttnn_layer_norm(ttnn_add_160, epsilon = 1e-12, weight = ttnn_from_torch_81, bias = ttnn_from_torch_82);  ttnn_add_160 = ttnn_from_torch_81 = ttnn_from_torch_82 = None
    ttnn_experimental_view_86 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_9, [256, 1024])
    ttnn_from_torch_83 = ttnn_decorators_ttnn_from_torch(arg79_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg79_1 = None
    ttnn_from_torch_84 = ttnn_decorators_ttnn_from_torch(arg80_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg80_1 = None
    ttnn_linear_28 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_86, ttnn_from_torch_83, transpose_b = True, bias = ttnn_from_torch_84, activation = 'gelu');  ttnn_experimental_view_86 = ttnn_from_torch_83 = ttnn_from_torch_84 = None
    ttnn_experimental_view_88 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_28, [256, 4096]);  ttnn_linear_28 = None
    ttnn_from_torch_85 = ttnn_decorators_ttnn_from_torch(arg81_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg81_1 = None
    ttnn_from_torch_86 = ttnn_decorators_ttnn_from_torch(arg82_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg82_1 = None
    ttnn_linear_29 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_88, ttnn_from_torch_85, transpose_b = True, bias = ttnn_from_torch_86, activation = None);  ttnn_experimental_view_88 = ttnn_from_torch_85 = ttnn_from_torch_86 = None
    ttnn_experimental_view_89 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_29, [1, 256, 1024]);  ttnn_linear_29 = None
    ttnn_add_161 = ttnn_decorators_ttnn_add(ttnn_experimental_view_89, ttnn_layer_norm_9);  ttnn_experimental_view_89 = ttnn_layer_norm_9 = None
    ttnn_from_torch_87 = ttnn_decorators_ttnn_from_torch(arg83_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg83_1 = None
    ttnn_from_torch_88 = ttnn_decorators_ttnn_from_torch(arg84_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg84_1 = None
    ttnn_layer_norm_10 = ttnn_decorators_ttnn_layer_norm(ttnn_add_161, epsilon = 1e-12, weight = ttnn_from_torch_87, bias = ttnn_from_torch_88);  ttnn_add_161 = ttnn_from_torch_87 = ttnn_from_torch_88 = None
    ttnn_experimental_view_90 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_10, [256, 1024])
    ttnn_from_torch_89 = ttnn_decorators_ttnn_from_torch(arg85_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg85_1 = None
    ttnn_transpose_184 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_89, -2, -1);  ttnn_from_torch_89 = None
    ttnn_from_torch_90 = ttnn_decorators_ttnn_from_torch(arg87_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg87_1 = None
    ttnn_transpose_185 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_90, -2, -1);  ttnn_from_torch_90 = None
    ttnn_from_torch_91 = ttnn_decorators_ttnn_from_torch(arg89_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg89_1 = None
    ttnn_transpose_186 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_91, -2, -1);  ttnn_from_torch_91 = None
    ttnn_concat_10 = ttnn_decorators_ttnn_concat([ttnn_transpose_184, ttnn_transpose_185, ttnn_transpose_186], -1);  ttnn_transpose_184 = ttnn_transpose_185 = ttnn_transpose_186 = None
    ttnn_from_torch_92 = ttnn_decorators_ttnn_from_torch(arg86_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg86_1 = None
    ttnn_experimental_view_454 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_92, (1, -1));  ttnn_from_torch_92 = None
    ttnn_from_torch_93 = ttnn_decorators_ttnn_from_torch(arg88_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg88_1 = None
    ttnn_experimental_view_455 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_93, (1, -1));  ttnn_from_torch_93 = None
    ttnn_from_torch_94 = ttnn_decorators_ttnn_from_torch(arg90_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg90_1 = None
    ttnn_experimental_view_456 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_94, (1, -1));  ttnn_from_torch_94 = None
    ttnn_concat_11 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_454, ttnn_experimental_view_455, ttnn_experimental_view_456], -1);  ttnn_experimental_view_454 = ttnn_experimental_view_455 = ttnn_experimental_view_456 = None
    ttnn_to_layout_11 = ttnn_decorators_ttnn_to_layout(ttnn_concat_10, ttnn_TILE_LAYOUT);  ttnn_concat_10 = None
    ttnn_to_layout_12 = ttnn_decorators_ttnn_to_layout(ttnn_concat_11, ttnn_TILE_LAYOUT);  ttnn_concat_11 = None
    ttnn_linear_150 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_90, ttnn_to_layout_11, bias = ttnn_to_layout_12);  ttnn_experimental_view_90 = ttnn_to_layout_11 = ttnn_to_layout_12 = None
    ttnn_experimental_view_457 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_150, [1, 256, 3072]);  ttnn_linear_150 = None
    ttnn_transformer_split_query_key_value_and_split_heads_5 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_457, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_457 = None
    getitem_66 = ttnn_transformer_split_query_key_value_and_split_heads_5[0]
    getitem_67 = ttnn_transformer_split_query_key_value_and_split_heads_5[1]
    getitem_68 = ttnn_transformer_split_query_key_value_and_split_heads_5[2];  ttnn_transformer_split_query_key_value_and_split_heads_5 = None
    ttnn_matmul_203 = ttnn_decorators_ttnn_matmul(getitem_66, getitem_67);  getitem_66 = getitem_67 = None
    ttnn_transformer_attention_softmax__29 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_203, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_203 = None
    ttnn_matmul_204 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__29, getitem_68);  ttnn_transformer_attention_softmax__29 = getitem_68 = None
    ttnn_transformer_concatenate_heads_5 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_204);  ttnn_matmul_204 = None
    ttnn_from_torch_95 = ttnn_decorators_ttnn_from_torch(arg91_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg91_1 = None
    ttnn_from_torch_96 = ttnn_decorators_ttnn_from_torch(arg92_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg92_1 = None
    ttnn_linear_33 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_5, ttnn_from_torch_95, transpose_b = True, bias = ttnn_from_torch_96, activation = None);  ttnn_transformer_concatenate_heads_5 = ttnn_from_torch_95 = ttnn_from_torch_96 = None
    ttnn_experimental_view_103 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_33, [1, 256, 1024]);  ttnn_linear_33 = None
    ttnn_add_163 = ttnn_decorators_ttnn_add(ttnn_experimental_view_103, ttnn_layer_norm_10);  ttnn_experimental_view_103 = ttnn_layer_norm_10 = None
    ttnn_from_torch_97 = ttnn_decorators_ttnn_from_torch(arg93_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg93_1 = None
    ttnn_from_torch_98 = ttnn_decorators_ttnn_from_torch(arg94_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg94_1 = None
    ttnn_layer_norm_11 = ttnn_decorators_ttnn_layer_norm(ttnn_add_163, epsilon = 1e-12, weight = ttnn_from_torch_97, bias = ttnn_from_torch_98);  ttnn_add_163 = ttnn_from_torch_97 = ttnn_from_torch_98 = None
    ttnn_experimental_view_104 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_11, [256, 1024])
    ttnn_from_torch_99 = ttnn_decorators_ttnn_from_torch(arg95_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg95_1 = None
    ttnn_from_torch_100 = ttnn_decorators_ttnn_from_torch(arg96_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg96_1 = None
    ttnn_linear_34 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_104, ttnn_from_torch_99, transpose_b = True, bias = ttnn_from_torch_100, activation = 'gelu');  ttnn_experimental_view_104 = ttnn_from_torch_99 = ttnn_from_torch_100 = None
    ttnn_experimental_view_106 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_34, [256, 4096]);  ttnn_linear_34 = None
    ttnn_from_torch_101 = ttnn_decorators_ttnn_from_torch(arg97_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg97_1 = None
    ttnn_from_torch_102 = ttnn_decorators_ttnn_from_torch(arg98_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg98_1 = None
    ttnn_linear_35 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_106, ttnn_from_torch_101, transpose_b = True, bias = ttnn_from_torch_102, activation = None);  ttnn_experimental_view_106 = ttnn_from_torch_101 = ttnn_from_torch_102 = None
    ttnn_experimental_view_107 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_35, [1, 256, 1024]);  ttnn_linear_35 = None
    ttnn_add_164 = ttnn_decorators_ttnn_add(ttnn_experimental_view_107, ttnn_layer_norm_11);  ttnn_experimental_view_107 = ttnn_layer_norm_11 = None
    ttnn_from_torch_103 = ttnn_decorators_ttnn_from_torch(arg99_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg99_1 = None
    ttnn_from_torch_104 = ttnn_decorators_ttnn_from_torch(arg100_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg100_1 = None
    ttnn_layer_norm_12 = ttnn_decorators_ttnn_layer_norm(ttnn_add_164, epsilon = 1e-12, weight = ttnn_from_torch_103, bias = ttnn_from_torch_104);  ttnn_add_164 = ttnn_from_torch_103 = ttnn_from_torch_104 = None
    ttnn_experimental_view_108 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_12, [256, 1024])
    ttnn_from_torch_105 = ttnn_decorators_ttnn_from_torch(arg101_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg101_1 = None
    ttnn_transpose_187 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_105, -2, -1);  ttnn_from_torch_105 = None
    ttnn_from_torch_106 = ttnn_decorators_ttnn_from_torch(arg103_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg103_1 = None
    ttnn_transpose_188 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_106, -2, -1);  ttnn_from_torch_106 = None
    ttnn_from_torch_107 = ttnn_decorators_ttnn_from_torch(arg105_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg105_1 = None
    ttnn_transpose_189 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_107, -2, -1);  ttnn_from_torch_107 = None
    ttnn_concat_12 = ttnn_decorators_ttnn_concat([ttnn_transpose_187, ttnn_transpose_188, ttnn_transpose_189], -1);  ttnn_transpose_187 = ttnn_transpose_188 = ttnn_transpose_189 = None
    ttnn_from_torch_108 = ttnn_decorators_ttnn_from_torch(arg102_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg102_1 = None
    ttnn_experimental_view_458 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_108, (1, -1));  ttnn_from_torch_108 = None
    ttnn_from_torch_109 = ttnn_decorators_ttnn_from_torch(arg104_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg104_1 = None
    ttnn_experimental_view_459 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_109, (1, -1));  ttnn_from_torch_109 = None
    ttnn_from_torch_110 = ttnn_decorators_ttnn_from_torch(arg106_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg106_1 = None
    ttnn_experimental_view_460 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_110, (1, -1));  ttnn_from_torch_110 = None
    ttnn_concat_13 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_458, ttnn_experimental_view_459, ttnn_experimental_view_460], -1);  ttnn_experimental_view_458 = ttnn_experimental_view_459 = ttnn_experimental_view_460 = None
    ttnn_to_layout_13 = ttnn_decorators_ttnn_to_layout(ttnn_concat_12, ttnn_TILE_LAYOUT);  ttnn_concat_12 = None
    ttnn_to_layout_14 = ttnn_decorators_ttnn_to_layout(ttnn_concat_13, ttnn_TILE_LAYOUT);  ttnn_concat_13 = None
    ttnn_linear_151 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_108, ttnn_to_layout_13, bias = ttnn_to_layout_14);  ttnn_experimental_view_108 = ttnn_to_layout_13 = ttnn_to_layout_14 = None
    ttnn_experimental_view_461 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_151, [1, 256, 3072]);  ttnn_linear_151 = None
    ttnn_transformer_split_query_key_value_and_split_heads_6 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_461, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_461 = None
    getitem_69 = ttnn_transformer_split_query_key_value_and_split_heads_6[0]
    getitem_70 = ttnn_transformer_split_query_key_value_and_split_heads_6[1]
    getitem_71 = ttnn_transformer_split_query_key_value_and_split_heads_6[2];  ttnn_transformer_split_query_key_value_and_split_heads_6 = None
    ttnn_matmul_205 = ttnn_decorators_ttnn_matmul(getitem_69, getitem_70);  getitem_69 = getitem_70 = None
    ttnn_transformer_attention_softmax__30 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_205, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_205 = None
    ttnn_matmul_206 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__30, getitem_71);  ttnn_transformer_attention_softmax__30 = getitem_71 = None
    ttnn_transformer_concatenate_heads_6 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_206);  ttnn_matmul_206 = None
    ttnn_from_torch_111 = ttnn_decorators_ttnn_from_torch(arg107_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg107_1 = None
    ttnn_from_torch_112 = ttnn_decorators_ttnn_from_torch(arg108_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg108_1 = None
    ttnn_linear_39 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_6, ttnn_from_torch_111, transpose_b = True, bias = ttnn_from_torch_112, activation = None);  ttnn_transformer_concatenate_heads_6 = ttnn_from_torch_111 = ttnn_from_torch_112 = None
    ttnn_experimental_view_121 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_39, [1, 256, 1024]);  ttnn_linear_39 = None
    ttnn_add_166 = ttnn_decorators_ttnn_add(ttnn_experimental_view_121, ttnn_layer_norm_12);  ttnn_experimental_view_121 = ttnn_layer_norm_12 = None
    ttnn_from_torch_113 = ttnn_decorators_ttnn_from_torch(arg109_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg109_1 = None
    ttnn_from_torch_114 = ttnn_decorators_ttnn_from_torch(arg110_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg110_1 = None
    ttnn_layer_norm_13 = ttnn_decorators_ttnn_layer_norm(ttnn_add_166, epsilon = 1e-12, weight = ttnn_from_torch_113, bias = ttnn_from_torch_114);  ttnn_add_166 = ttnn_from_torch_113 = ttnn_from_torch_114 = None
    ttnn_experimental_view_122 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_13, [256, 1024])
    ttnn_from_torch_115 = ttnn_decorators_ttnn_from_torch(arg111_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg111_1 = None
    ttnn_from_torch_116 = ttnn_decorators_ttnn_from_torch(arg112_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg112_1 = None
    ttnn_linear_40 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_122, ttnn_from_torch_115, transpose_b = True, bias = ttnn_from_torch_116, activation = 'gelu');  ttnn_experimental_view_122 = ttnn_from_torch_115 = ttnn_from_torch_116 = None
    ttnn_experimental_view_124 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_40, [256, 4096]);  ttnn_linear_40 = None
    ttnn_from_torch_117 = ttnn_decorators_ttnn_from_torch(arg113_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg113_1 = None
    ttnn_from_torch_118 = ttnn_decorators_ttnn_from_torch(arg114_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg114_1 = None
    ttnn_linear_41 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_124, ttnn_from_torch_117, transpose_b = True, bias = ttnn_from_torch_118, activation = None);  ttnn_experimental_view_124 = ttnn_from_torch_117 = ttnn_from_torch_118 = None
    ttnn_experimental_view_125 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_41, [1, 256, 1024]);  ttnn_linear_41 = None
    ttnn_add_167 = ttnn_decorators_ttnn_add(ttnn_experimental_view_125, ttnn_layer_norm_13);  ttnn_experimental_view_125 = ttnn_layer_norm_13 = None
    ttnn_from_torch_119 = ttnn_decorators_ttnn_from_torch(arg115_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg115_1 = None
    ttnn_from_torch_120 = ttnn_decorators_ttnn_from_torch(arg116_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg116_1 = None
    ttnn_layer_norm_14 = ttnn_decorators_ttnn_layer_norm(ttnn_add_167, epsilon = 1e-12, weight = ttnn_from_torch_119, bias = ttnn_from_torch_120);  ttnn_add_167 = ttnn_from_torch_119 = ttnn_from_torch_120 = None
    ttnn_experimental_view_126 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_14, [256, 1024])
    ttnn_from_torch_121 = ttnn_decorators_ttnn_from_torch(arg117_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg117_1 = None
    ttnn_transpose_190 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_121, -2, -1);  ttnn_from_torch_121 = None
    ttnn_from_torch_122 = ttnn_decorators_ttnn_from_torch(arg119_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg119_1 = None
    ttnn_transpose_191 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_122, -2, -1);  ttnn_from_torch_122 = None
    ttnn_from_torch_123 = ttnn_decorators_ttnn_from_torch(arg121_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg121_1 = None
    ttnn_transpose_192 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_123, -2, -1);  ttnn_from_torch_123 = None
    ttnn_concat_14 = ttnn_decorators_ttnn_concat([ttnn_transpose_190, ttnn_transpose_191, ttnn_transpose_192], -1);  ttnn_transpose_190 = ttnn_transpose_191 = ttnn_transpose_192 = None
    ttnn_from_torch_124 = ttnn_decorators_ttnn_from_torch(arg118_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg118_1 = None
    ttnn_experimental_view_462 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_124, (1, -1));  ttnn_from_torch_124 = None
    ttnn_from_torch_125 = ttnn_decorators_ttnn_from_torch(arg120_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg120_1 = None
    ttnn_experimental_view_463 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_125, (1, -1));  ttnn_from_torch_125 = None
    ttnn_from_torch_126 = ttnn_decorators_ttnn_from_torch(arg122_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg122_1 = None
    ttnn_experimental_view_464 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_126, (1, -1));  ttnn_from_torch_126 = None
    ttnn_concat_15 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_462, ttnn_experimental_view_463, ttnn_experimental_view_464], -1);  ttnn_experimental_view_462 = ttnn_experimental_view_463 = ttnn_experimental_view_464 = None
    ttnn_to_layout_15 = ttnn_decorators_ttnn_to_layout(ttnn_concat_14, ttnn_TILE_LAYOUT);  ttnn_concat_14 = None
    ttnn_to_layout_16 = ttnn_decorators_ttnn_to_layout(ttnn_concat_15, ttnn_TILE_LAYOUT);  ttnn_concat_15 = None
    ttnn_linear_152 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_126, ttnn_to_layout_15, bias = ttnn_to_layout_16);  ttnn_experimental_view_126 = ttnn_to_layout_15 = ttnn_to_layout_16 = None
    ttnn_experimental_view_465 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_152, [1, 256, 3072]);  ttnn_linear_152 = None
    ttnn_transformer_split_query_key_value_and_split_heads_7 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_465, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_465 = None
    getitem_72 = ttnn_transformer_split_query_key_value_and_split_heads_7[0]
    getitem_73 = ttnn_transformer_split_query_key_value_and_split_heads_7[1]
    getitem_74 = ttnn_transformer_split_query_key_value_and_split_heads_7[2];  ttnn_transformer_split_query_key_value_and_split_heads_7 = None
    ttnn_matmul_207 = ttnn_decorators_ttnn_matmul(getitem_72, getitem_73);  getitem_72 = getitem_73 = None
    ttnn_transformer_attention_softmax__31 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_207, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_207 = None
    ttnn_matmul_208 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__31, getitem_74);  ttnn_transformer_attention_softmax__31 = getitem_74 = None
    ttnn_transformer_concatenate_heads_7 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_208);  ttnn_matmul_208 = None
    ttnn_from_torch_127 = ttnn_decorators_ttnn_from_torch(arg123_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg123_1 = None
    ttnn_from_torch_128 = ttnn_decorators_ttnn_from_torch(arg124_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg124_1 = None
    ttnn_linear_45 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_7, ttnn_from_torch_127, transpose_b = True, bias = ttnn_from_torch_128, activation = None);  ttnn_transformer_concatenate_heads_7 = ttnn_from_torch_127 = ttnn_from_torch_128 = None
    ttnn_experimental_view_139 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_45, [1, 256, 1024]);  ttnn_linear_45 = None
    ttnn_add_169 = ttnn_decorators_ttnn_add(ttnn_experimental_view_139, ttnn_layer_norm_14);  ttnn_experimental_view_139 = ttnn_layer_norm_14 = None
    ttnn_from_torch_129 = ttnn_decorators_ttnn_from_torch(arg125_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg125_1 = None
    ttnn_from_torch_130 = ttnn_decorators_ttnn_from_torch(arg126_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg126_1 = None
    ttnn_layer_norm_15 = ttnn_decorators_ttnn_layer_norm(ttnn_add_169, epsilon = 1e-12, weight = ttnn_from_torch_129, bias = ttnn_from_torch_130);  ttnn_add_169 = ttnn_from_torch_129 = ttnn_from_torch_130 = None
    ttnn_experimental_view_140 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_15, [256, 1024])
    ttnn_from_torch_131 = ttnn_decorators_ttnn_from_torch(arg127_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg127_1 = None
    ttnn_from_torch_132 = ttnn_decorators_ttnn_from_torch(arg128_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg128_1 = None
    ttnn_linear_46 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_140, ttnn_from_torch_131, transpose_b = True, bias = ttnn_from_torch_132, activation = 'gelu');  ttnn_experimental_view_140 = ttnn_from_torch_131 = ttnn_from_torch_132 = None
    ttnn_experimental_view_142 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_46, [256, 4096]);  ttnn_linear_46 = None
    ttnn_from_torch_133 = ttnn_decorators_ttnn_from_torch(arg129_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg129_1 = None
    ttnn_from_torch_134 = ttnn_decorators_ttnn_from_torch(arg130_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg130_1 = None
    ttnn_linear_47 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_142, ttnn_from_torch_133, transpose_b = True, bias = ttnn_from_torch_134, activation = None);  ttnn_experimental_view_142 = ttnn_from_torch_133 = ttnn_from_torch_134 = None
    ttnn_experimental_view_143 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_47, [1, 256, 1024]);  ttnn_linear_47 = None
    ttnn_add_170 = ttnn_decorators_ttnn_add(ttnn_experimental_view_143, ttnn_layer_norm_15);  ttnn_experimental_view_143 = ttnn_layer_norm_15 = None
    ttnn_from_torch_135 = ttnn_decorators_ttnn_from_torch(arg131_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg131_1 = None
    ttnn_from_torch_136 = ttnn_decorators_ttnn_from_torch(arg132_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg132_1 = None
    ttnn_layer_norm_16 = ttnn_decorators_ttnn_layer_norm(ttnn_add_170, epsilon = 1e-12, weight = ttnn_from_torch_135, bias = ttnn_from_torch_136);  ttnn_add_170 = ttnn_from_torch_135 = ttnn_from_torch_136 = None
    ttnn_experimental_view_144 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_16, [256, 1024])
    ttnn_from_torch_137 = ttnn_decorators_ttnn_from_torch(arg133_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg133_1 = None
    ttnn_transpose_193 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_137, -2, -1);  ttnn_from_torch_137 = None
    ttnn_from_torch_138 = ttnn_decorators_ttnn_from_torch(arg135_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg135_1 = None
    ttnn_transpose_194 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_138, -2, -1);  ttnn_from_torch_138 = None
    ttnn_from_torch_139 = ttnn_decorators_ttnn_from_torch(arg137_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg137_1 = None
    ttnn_transpose_195 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_139, -2, -1);  ttnn_from_torch_139 = None
    ttnn_concat_16 = ttnn_decorators_ttnn_concat([ttnn_transpose_193, ttnn_transpose_194, ttnn_transpose_195], -1);  ttnn_transpose_193 = ttnn_transpose_194 = ttnn_transpose_195 = None
    ttnn_from_torch_140 = ttnn_decorators_ttnn_from_torch(arg134_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg134_1 = None
    ttnn_experimental_view_466 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_140, (1, -1));  ttnn_from_torch_140 = None
    ttnn_from_torch_141 = ttnn_decorators_ttnn_from_torch(arg136_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg136_1 = None
    ttnn_experimental_view_467 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_141, (1, -1));  ttnn_from_torch_141 = None
    ttnn_from_torch_142 = ttnn_decorators_ttnn_from_torch(arg138_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg138_1 = None
    ttnn_experimental_view_468 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_142, (1, -1));  ttnn_from_torch_142 = None
    ttnn_concat_17 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_466, ttnn_experimental_view_467, ttnn_experimental_view_468], -1);  ttnn_experimental_view_466 = ttnn_experimental_view_467 = ttnn_experimental_view_468 = None
    ttnn_to_layout_17 = ttnn_decorators_ttnn_to_layout(ttnn_concat_16, ttnn_TILE_LAYOUT);  ttnn_concat_16 = None
    ttnn_to_layout_18 = ttnn_decorators_ttnn_to_layout(ttnn_concat_17, ttnn_TILE_LAYOUT);  ttnn_concat_17 = None
    ttnn_linear_153 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_144, ttnn_to_layout_17, bias = ttnn_to_layout_18);  ttnn_experimental_view_144 = ttnn_to_layout_17 = ttnn_to_layout_18 = None
    ttnn_experimental_view_469 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_153, [1, 256, 3072]);  ttnn_linear_153 = None
    ttnn_transformer_split_query_key_value_and_split_heads_8 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_469, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_469 = None
    getitem_75 = ttnn_transformer_split_query_key_value_and_split_heads_8[0]
    getitem_76 = ttnn_transformer_split_query_key_value_and_split_heads_8[1]
    getitem_77 = ttnn_transformer_split_query_key_value_and_split_heads_8[2];  ttnn_transformer_split_query_key_value_and_split_heads_8 = None
    ttnn_matmul_209 = ttnn_decorators_ttnn_matmul(getitem_75, getitem_76);  getitem_75 = getitem_76 = None
    ttnn_transformer_attention_softmax__32 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_209, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_209 = None
    ttnn_matmul_210 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__32, getitem_77);  ttnn_transformer_attention_softmax__32 = getitem_77 = None
    ttnn_transformer_concatenate_heads_8 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_210);  ttnn_matmul_210 = None
    ttnn_from_torch_143 = ttnn_decorators_ttnn_from_torch(arg139_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg139_1 = None
    ttnn_from_torch_144 = ttnn_decorators_ttnn_from_torch(arg140_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg140_1 = None
    ttnn_linear_51 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_8, ttnn_from_torch_143, transpose_b = True, bias = ttnn_from_torch_144, activation = None);  ttnn_transformer_concatenate_heads_8 = ttnn_from_torch_143 = ttnn_from_torch_144 = None
    ttnn_experimental_view_157 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_51, [1, 256, 1024]);  ttnn_linear_51 = None
    ttnn_add_172 = ttnn_decorators_ttnn_add(ttnn_experimental_view_157, ttnn_layer_norm_16);  ttnn_experimental_view_157 = ttnn_layer_norm_16 = None
    ttnn_from_torch_145 = ttnn_decorators_ttnn_from_torch(arg141_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg141_1 = None
    ttnn_from_torch_146 = ttnn_decorators_ttnn_from_torch(arg142_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg142_1 = None
    ttnn_layer_norm_17 = ttnn_decorators_ttnn_layer_norm(ttnn_add_172, epsilon = 1e-12, weight = ttnn_from_torch_145, bias = ttnn_from_torch_146);  ttnn_add_172 = ttnn_from_torch_145 = ttnn_from_torch_146 = None
    ttnn_experimental_view_158 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_17, [256, 1024])
    ttnn_from_torch_147 = ttnn_decorators_ttnn_from_torch(arg143_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg143_1 = None
    ttnn_from_torch_148 = ttnn_decorators_ttnn_from_torch(arg144_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg144_1 = None
    ttnn_linear_52 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_158, ttnn_from_torch_147, transpose_b = True, bias = ttnn_from_torch_148, activation = 'gelu');  ttnn_experimental_view_158 = ttnn_from_torch_147 = ttnn_from_torch_148 = None
    ttnn_experimental_view_160 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_52, [256, 4096]);  ttnn_linear_52 = None
    ttnn_from_torch_149 = ttnn_decorators_ttnn_from_torch(arg145_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg145_1 = None
    ttnn_from_torch_150 = ttnn_decorators_ttnn_from_torch(arg146_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg146_1 = None
    ttnn_linear_53 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_160, ttnn_from_torch_149, transpose_b = True, bias = ttnn_from_torch_150, activation = None);  ttnn_experimental_view_160 = ttnn_from_torch_149 = ttnn_from_torch_150 = None
    ttnn_experimental_view_161 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_53, [1, 256, 1024]);  ttnn_linear_53 = None
    ttnn_add_173 = ttnn_decorators_ttnn_add(ttnn_experimental_view_161, ttnn_layer_norm_17);  ttnn_experimental_view_161 = ttnn_layer_norm_17 = None
    ttnn_from_torch_151 = ttnn_decorators_ttnn_from_torch(arg147_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg147_1 = None
    ttnn_from_torch_152 = ttnn_decorators_ttnn_from_torch(arg148_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg148_1 = None
    ttnn_layer_norm_18 = ttnn_decorators_ttnn_layer_norm(ttnn_add_173, epsilon = 1e-12, weight = ttnn_from_torch_151, bias = ttnn_from_torch_152);  ttnn_add_173 = ttnn_from_torch_151 = ttnn_from_torch_152 = None
    ttnn_experimental_view_162 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_18, [256, 1024])
    ttnn_from_torch_153 = ttnn_decorators_ttnn_from_torch(arg149_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg149_1 = None
    ttnn_transpose_196 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_153, -2, -1);  ttnn_from_torch_153 = None
    ttnn_from_torch_154 = ttnn_decorators_ttnn_from_torch(arg151_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg151_1 = None
    ttnn_transpose_197 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_154, -2, -1);  ttnn_from_torch_154 = None
    ttnn_from_torch_155 = ttnn_decorators_ttnn_from_torch(arg153_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg153_1 = None
    ttnn_transpose_198 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_155, -2, -1);  ttnn_from_torch_155 = None
    ttnn_concat_18 = ttnn_decorators_ttnn_concat([ttnn_transpose_196, ttnn_transpose_197, ttnn_transpose_198], -1);  ttnn_transpose_196 = ttnn_transpose_197 = ttnn_transpose_198 = None
    ttnn_from_torch_156 = ttnn_decorators_ttnn_from_torch(arg150_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg150_1 = None
    ttnn_experimental_view_470 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_156, (1, -1));  ttnn_from_torch_156 = None
    ttnn_from_torch_157 = ttnn_decorators_ttnn_from_torch(arg152_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg152_1 = None
    ttnn_experimental_view_471 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_157, (1, -1));  ttnn_from_torch_157 = None
    ttnn_from_torch_158 = ttnn_decorators_ttnn_from_torch(arg154_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg154_1 = None
    ttnn_experimental_view_472 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_158, (1, -1));  ttnn_from_torch_158 = None
    ttnn_concat_19 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_470, ttnn_experimental_view_471, ttnn_experimental_view_472], -1);  ttnn_experimental_view_470 = ttnn_experimental_view_471 = ttnn_experimental_view_472 = None
    ttnn_to_layout_19 = ttnn_decorators_ttnn_to_layout(ttnn_concat_18, ttnn_TILE_LAYOUT);  ttnn_concat_18 = None
    ttnn_to_layout_20 = ttnn_decorators_ttnn_to_layout(ttnn_concat_19, ttnn_TILE_LAYOUT);  ttnn_concat_19 = None
    ttnn_linear_154 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_162, ttnn_to_layout_19, bias = ttnn_to_layout_20);  ttnn_experimental_view_162 = ttnn_to_layout_19 = ttnn_to_layout_20 = None
    ttnn_experimental_view_473 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_154, [1, 256, 3072]);  ttnn_linear_154 = None
    ttnn_transformer_split_query_key_value_and_split_heads_9 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_473, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_473 = None
    getitem_78 = ttnn_transformer_split_query_key_value_and_split_heads_9[0]
    getitem_79 = ttnn_transformer_split_query_key_value_and_split_heads_9[1]
    getitem_80 = ttnn_transformer_split_query_key_value_and_split_heads_9[2];  ttnn_transformer_split_query_key_value_and_split_heads_9 = None
    ttnn_matmul_211 = ttnn_decorators_ttnn_matmul(getitem_78, getitem_79);  getitem_78 = getitem_79 = None
    ttnn_transformer_attention_softmax__33 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_211, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_211 = None
    ttnn_matmul_212 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__33, getitem_80);  ttnn_transformer_attention_softmax__33 = getitem_80 = None
    ttnn_transformer_concatenate_heads_9 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_212);  ttnn_matmul_212 = None
    ttnn_from_torch_159 = ttnn_decorators_ttnn_from_torch(arg155_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg155_1 = None
    ttnn_from_torch_160 = ttnn_decorators_ttnn_from_torch(arg156_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg156_1 = None
    ttnn_linear_57 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_9, ttnn_from_torch_159, transpose_b = True, bias = ttnn_from_torch_160, activation = None);  ttnn_transformer_concatenate_heads_9 = ttnn_from_torch_159 = ttnn_from_torch_160 = None
    ttnn_experimental_view_175 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_57, [1, 256, 1024]);  ttnn_linear_57 = None
    ttnn_add_175 = ttnn_decorators_ttnn_add(ttnn_experimental_view_175, ttnn_layer_norm_18);  ttnn_experimental_view_175 = ttnn_layer_norm_18 = None
    ttnn_from_torch_161 = ttnn_decorators_ttnn_from_torch(arg157_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg157_1 = None
    ttnn_from_torch_162 = ttnn_decorators_ttnn_from_torch(arg158_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg158_1 = None
    ttnn_layer_norm_19 = ttnn_decorators_ttnn_layer_norm(ttnn_add_175, epsilon = 1e-12, weight = ttnn_from_torch_161, bias = ttnn_from_torch_162);  ttnn_add_175 = ttnn_from_torch_161 = ttnn_from_torch_162 = None
    ttnn_experimental_view_176 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_19, [256, 1024])
    ttnn_from_torch_163 = ttnn_decorators_ttnn_from_torch(arg159_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg159_1 = None
    ttnn_from_torch_164 = ttnn_decorators_ttnn_from_torch(arg160_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg160_1 = None
    ttnn_linear_58 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_176, ttnn_from_torch_163, transpose_b = True, bias = ttnn_from_torch_164, activation = 'gelu');  ttnn_experimental_view_176 = ttnn_from_torch_163 = ttnn_from_torch_164 = None
    ttnn_experimental_view_178 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_58, [256, 4096]);  ttnn_linear_58 = None
    ttnn_from_torch_165 = ttnn_decorators_ttnn_from_torch(arg161_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg161_1 = None
    ttnn_from_torch_166 = ttnn_decorators_ttnn_from_torch(arg162_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg162_1 = None
    ttnn_linear_59 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_178, ttnn_from_torch_165, transpose_b = True, bias = ttnn_from_torch_166, activation = None);  ttnn_experimental_view_178 = ttnn_from_torch_165 = ttnn_from_torch_166 = None
    ttnn_experimental_view_179 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_59, [1, 256, 1024]);  ttnn_linear_59 = None
    ttnn_add_176 = ttnn_decorators_ttnn_add(ttnn_experimental_view_179, ttnn_layer_norm_19);  ttnn_experimental_view_179 = ttnn_layer_norm_19 = None
    ttnn_from_torch_167 = ttnn_decorators_ttnn_from_torch(arg163_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg163_1 = None
    ttnn_from_torch_168 = ttnn_decorators_ttnn_from_torch(arg164_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg164_1 = None
    ttnn_layer_norm_20 = ttnn_decorators_ttnn_layer_norm(ttnn_add_176, epsilon = 1e-12, weight = ttnn_from_torch_167, bias = ttnn_from_torch_168);  ttnn_add_176 = ttnn_from_torch_167 = ttnn_from_torch_168 = None
    ttnn_experimental_view_180 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_20, [256, 1024])
    ttnn_from_torch_169 = ttnn_decorators_ttnn_from_torch(arg165_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg165_1 = None
    ttnn_transpose_199 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_169, -2, -1);  ttnn_from_torch_169 = None
    ttnn_from_torch_170 = ttnn_decorators_ttnn_from_torch(arg167_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg167_1 = None
    ttnn_transpose_200 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_170, -2, -1);  ttnn_from_torch_170 = None
    ttnn_from_torch_171 = ttnn_decorators_ttnn_from_torch(arg169_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg169_1 = None
    ttnn_transpose_201 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_171, -2, -1);  ttnn_from_torch_171 = None
    ttnn_concat_20 = ttnn_decorators_ttnn_concat([ttnn_transpose_199, ttnn_transpose_200, ttnn_transpose_201], -1);  ttnn_transpose_199 = ttnn_transpose_200 = ttnn_transpose_201 = None
    ttnn_from_torch_172 = ttnn_decorators_ttnn_from_torch(arg166_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg166_1 = None
    ttnn_experimental_view_474 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_172, (1, -1));  ttnn_from_torch_172 = None
    ttnn_from_torch_173 = ttnn_decorators_ttnn_from_torch(arg168_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg168_1 = None
    ttnn_experimental_view_475 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_173, (1, -1));  ttnn_from_torch_173 = None
    ttnn_from_torch_174 = ttnn_decorators_ttnn_from_torch(arg170_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg170_1 = None
    ttnn_experimental_view_476 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_174, (1, -1));  ttnn_from_torch_174 = None
    ttnn_concat_21 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_474, ttnn_experimental_view_475, ttnn_experimental_view_476], -1);  ttnn_experimental_view_474 = ttnn_experimental_view_475 = ttnn_experimental_view_476 = None
    ttnn_to_layout_21 = ttnn_decorators_ttnn_to_layout(ttnn_concat_20, ttnn_TILE_LAYOUT);  ttnn_concat_20 = None
    ttnn_to_layout_22 = ttnn_decorators_ttnn_to_layout(ttnn_concat_21, ttnn_TILE_LAYOUT);  ttnn_concat_21 = None
    ttnn_linear_155 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_180, ttnn_to_layout_21, bias = ttnn_to_layout_22);  ttnn_experimental_view_180 = ttnn_to_layout_21 = ttnn_to_layout_22 = None
    ttnn_experimental_view_477 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_155, [1, 256, 3072]);  ttnn_linear_155 = None
    ttnn_transformer_split_query_key_value_and_split_heads_10 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_477, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_477 = None
    getitem_81 = ttnn_transformer_split_query_key_value_and_split_heads_10[0]
    getitem_82 = ttnn_transformer_split_query_key_value_and_split_heads_10[1]
    getitem_83 = ttnn_transformer_split_query_key_value_and_split_heads_10[2];  ttnn_transformer_split_query_key_value_and_split_heads_10 = None
    ttnn_matmul_213 = ttnn_decorators_ttnn_matmul(getitem_81, getitem_82);  getitem_81 = getitem_82 = None
    ttnn_transformer_attention_softmax__34 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_213, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_213 = None
    ttnn_matmul_214 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__34, getitem_83);  ttnn_transformer_attention_softmax__34 = getitem_83 = None
    ttnn_transformer_concatenate_heads_10 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_214);  ttnn_matmul_214 = None
    ttnn_from_torch_175 = ttnn_decorators_ttnn_from_torch(arg171_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg171_1 = None
    ttnn_from_torch_176 = ttnn_decorators_ttnn_from_torch(arg172_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg172_1 = None
    ttnn_linear_63 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_10, ttnn_from_torch_175, transpose_b = True, bias = ttnn_from_torch_176, activation = None);  ttnn_transformer_concatenate_heads_10 = ttnn_from_torch_175 = ttnn_from_torch_176 = None
    ttnn_experimental_view_193 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_63, [1, 256, 1024]);  ttnn_linear_63 = None
    ttnn_add_178 = ttnn_decorators_ttnn_add(ttnn_experimental_view_193, ttnn_layer_norm_20);  ttnn_experimental_view_193 = ttnn_layer_norm_20 = None
    ttnn_from_torch_177 = ttnn_decorators_ttnn_from_torch(arg173_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg173_1 = None
    ttnn_from_torch_178 = ttnn_decorators_ttnn_from_torch(arg174_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg174_1 = None
    ttnn_layer_norm_21 = ttnn_decorators_ttnn_layer_norm(ttnn_add_178, epsilon = 1e-12, weight = ttnn_from_torch_177, bias = ttnn_from_torch_178);  ttnn_add_178 = ttnn_from_torch_177 = ttnn_from_torch_178 = None
    ttnn_experimental_view_194 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_21, [256, 1024])
    ttnn_from_torch_179 = ttnn_decorators_ttnn_from_torch(arg175_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg175_1 = None
    ttnn_from_torch_180 = ttnn_decorators_ttnn_from_torch(arg176_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg176_1 = None
    ttnn_linear_64 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_194, ttnn_from_torch_179, transpose_b = True, bias = ttnn_from_torch_180, activation = 'gelu');  ttnn_experimental_view_194 = ttnn_from_torch_179 = ttnn_from_torch_180 = None
    ttnn_experimental_view_196 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_64, [256, 4096]);  ttnn_linear_64 = None
    ttnn_from_torch_181 = ttnn_decorators_ttnn_from_torch(arg177_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg177_1 = None
    ttnn_from_torch_182 = ttnn_decorators_ttnn_from_torch(arg178_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg178_1 = None
    ttnn_linear_65 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_196, ttnn_from_torch_181, transpose_b = True, bias = ttnn_from_torch_182, activation = None);  ttnn_experimental_view_196 = ttnn_from_torch_181 = ttnn_from_torch_182 = None
    ttnn_experimental_view_197 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_65, [1, 256, 1024]);  ttnn_linear_65 = None
    ttnn_add_179 = ttnn_decorators_ttnn_add(ttnn_experimental_view_197, ttnn_layer_norm_21);  ttnn_experimental_view_197 = ttnn_layer_norm_21 = None
    ttnn_from_torch_183 = ttnn_decorators_ttnn_from_torch(arg179_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg179_1 = None
    ttnn_from_torch_184 = ttnn_decorators_ttnn_from_torch(arg180_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg180_1 = None
    ttnn_layer_norm_22 = ttnn_decorators_ttnn_layer_norm(ttnn_add_179, epsilon = 1e-12, weight = ttnn_from_torch_183, bias = ttnn_from_torch_184);  ttnn_add_179 = ttnn_from_torch_183 = ttnn_from_torch_184 = None
    ttnn_experimental_view_198 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_22, [256, 1024])
    ttnn_from_torch_185 = ttnn_decorators_ttnn_from_torch(arg181_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg181_1 = None
    ttnn_transpose_202 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_185, -2, -1);  ttnn_from_torch_185 = None
    ttnn_from_torch_186 = ttnn_decorators_ttnn_from_torch(arg183_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg183_1 = None
    ttnn_transpose_203 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_186, -2, -1);  ttnn_from_torch_186 = None
    ttnn_from_torch_187 = ttnn_decorators_ttnn_from_torch(arg185_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg185_1 = None
    ttnn_transpose_204 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_187, -2, -1);  ttnn_from_torch_187 = None
    ttnn_concat_22 = ttnn_decorators_ttnn_concat([ttnn_transpose_202, ttnn_transpose_203, ttnn_transpose_204], -1);  ttnn_transpose_202 = ttnn_transpose_203 = ttnn_transpose_204 = None
    ttnn_from_torch_188 = ttnn_decorators_ttnn_from_torch(arg182_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg182_1 = None
    ttnn_experimental_view_478 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_188, (1, -1));  ttnn_from_torch_188 = None
    ttnn_from_torch_189 = ttnn_decorators_ttnn_from_torch(arg184_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg184_1 = None
    ttnn_experimental_view_479 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_189, (1, -1));  ttnn_from_torch_189 = None
    ttnn_from_torch_190 = ttnn_decorators_ttnn_from_torch(arg186_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg186_1 = None
    ttnn_experimental_view_480 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_190, (1, -1));  ttnn_from_torch_190 = None
    ttnn_concat_23 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_478, ttnn_experimental_view_479, ttnn_experimental_view_480], -1);  ttnn_experimental_view_478 = ttnn_experimental_view_479 = ttnn_experimental_view_480 = None
    ttnn_to_layout_23 = ttnn_decorators_ttnn_to_layout(ttnn_concat_22, ttnn_TILE_LAYOUT);  ttnn_concat_22 = None
    ttnn_to_layout_24 = ttnn_decorators_ttnn_to_layout(ttnn_concat_23, ttnn_TILE_LAYOUT);  ttnn_concat_23 = None
    ttnn_linear_156 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_198, ttnn_to_layout_23, bias = ttnn_to_layout_24);  ttnn_experimental_view_198 = ttnn_to_layout_23 = ttnn_to_layout_24 = None
    ttnn_experimental_view_481 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_156, [1, 256, 3072]);  ttnn_linear_156 = None
    ttnn_transformer_split_query_key_value_and_split_heads_11 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_481, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_481 = None
    getitem_84 = ttnn_transformer_split_query_key_value_and_split_heads_11[0]
    getitem_85 = ttnn_transformer_split_query_key_value_and_split_heads_11[1]
    getitem_86 = ttnn_transformer_split_query_key_value_and_split_heads_11[2];  ttnn_transformer_split_query_key_value_and_split_heads_11 = None
    ttnn_matmul_215 = ttnn_decorators_ttnn_matmul(getitem_84, getitem_85);  getitem_84 = getitem_85 = None
    ttnn_transformer_attention_softmax__35 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_215, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_215 = None
    ttnn_matmul_216 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__35, getitem_86);  ttnn_transformer_attention_softmax__35 = getitem_86 = None
    ttnn_transformer_concatenate_heads_11 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_216);  ttnn_matmul_216 = None
    ttnn_from_torch_191 = ttnn_decorators_ttnn_from_torch(arg187_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg187_1 = None
    ttnn_from_torch_192 = ttnn_decorators_ttnn_from_torch(arg188_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg188_1 = None
    ttnn_linear_69 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_11, ttnn_from_torch_191, transpose_b = True, bias = ttnn_from_torch_192, activation = None);  ttnn_transformer_concatenate_heads_11 = ttnn_from_torch_191 = ttnn_from_torch_192 = None
    ttnn_experimental_view_211 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_69, [1, 256, 1024]);  ttnn_linear_69 = None
    ttnn_add_181 = ttnn_decorators_ttnn_add(ttnn_experimental_view_211, ttnn_layer_norm_22);  ttnn_experimental_view_211 = ttnn_layer_norm_22 = None
    ttnn_from_torch_193 = ttnn_decorators_ttnn_from_torch(arg189_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg189_1 = None
    ttnn_from_torch_194 = ttnn_decorators_ttnn_from_torch(arg190_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg190_1 = None
    ttnn_layer_norm_23 = ttnn_decorators_ttnn_layer_norm(ttnn_add_181, epsilon = 1e-12, weight = ttnn_from_torch_193, bias = ttnn_from_torch_194);  ttnn_add_181 = ttnn_from_torch_193 = ttnn_from_torch_194 = None
    ttnn_experimental_view_212 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_23, [256, 1024])
    ttnn_from_torch_195 = ttnn_decorators_ttnn_from_torch(arg191_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg191_1 = None
    ttnn_from_torch_196 = ttnn_decorators_ttnn_from_torch(arg192_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg192_1 = None
    ttnn_linear_70 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_212, ttnn_from_torch_195, transpose_b = True, bias = ttnn_from_torch_196, activation = 'gelu');  ttnn_experimental_view_212 = ttnn_from_torch_195 = ttnn_from_torch_196 = None
    ttnn_experimental_view_214 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_70, [256, 4096]);  ttnn_linear_70 = None
    ttnn_from_torch_197 = ttnn_decorators_ttnn_from_torch(arg193_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg193_1 = None
    ttnn_from_torch_198 = ttnn_decorators_ttnn_from_torch(arg194_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg194_1 = None
    ttnn_linear_71 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_214, ttnn_from_torch_197, transpose_b = True, bias = ttnn_from_torch_198, activation = None);  ttnn_experimental_view_214 = ttnn_from_torch_197 = ttnn_from_torch_198 = None
    ttnn_experimental_view_215 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_71, [1, 256, 1024]);  ttnn_linear_71 = None
    ttnn_add_182 = ttnn_decorators_ttnn_add(ttnn_experimental_view_215, ttnn_layer_norm_23);  ttnn_experimental_view_215 = ttnn_layer_norm_23 = None
    ttnn_from_torch_199 = ttnn_decorators_ttnn_from_torch(arg195_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg195_1 = None
    ttnn_from_torch_200 = ttnn_decorators_ttnn_from_torch(arg196_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg196_1 = None
    ttnn_layer_norm_24 = ttnn_decorators_ttnn_layer_norm(ttnn_add_182, epsilon = 1e-12, weight = ttnn_from_torch_199, bias = ttnn_from_torch_200);  ttnn_add_182 = ttnn_from_torch_199 = ttnn_from_torch_200 = None
    ttnn_experimental_view_216 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_24, [256, 1024])
    ttnn_from_torch_201 = ttnn_decorators_ttnn_from_torch(arg197_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg197_1 = None
    ttnn_transpose_205 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_201, -2, -1);  ttnn_from_torch_201 = None
    ttnn_from_torch_202 = ttnn_decorators_ttnn_from_torch(arg199_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg199_1 = None
    ttnn_transpose_206 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_202, -2, -1);  ttnn_from_torch_202 = None
    ttnn_from_torch_203 = ttnn_decorators_ttnn_from_torch(arg201_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg201_1 = None
    ttnn_transpose_207 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_203, -2, -1);  ttnn_from_torch_203 = None
    ttnn_concat_24 = ttnn_decorators_ttnn_concat([ttnn_transpose_205, ttnn_transpose_206, ttnn_transpose_207], -1);  ttnn_transpose_205 = ttnn_transpose_206 = ttnn_transpose_207 = None
    ttnn_from_torch_204 = ttnn_decorators_ttnn_from_torch(arg198_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg198_1 = None
    ttnn_experimental_view_482 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_204, (1, -1));  ttnn_from_torch_204 = None
    ttnn_from_torch_205 = ttnn_decorators_ttnn_from_torch(arg200_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg200_1 = None
    ttnn_experimental_view_483 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_205, (1, -1));  ttnn_from_torch_205 = None
    ttnn_from_torch_206 = ttnn_decorators_ttnn_from_torch(arg202_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg202_1 = None
    ttnn_experimental_view_484 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_206, (1, -1));  ttnn_from_torch_206 = None
    ttnn_concat_25 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_482, ttnn_experimental_view_483, ttnn_experimental_view_484], -1);  ttnn_experimental_view_482 = ttnn_experimental_view_483 = ttnn_experimental_view_484 = None
    ttnn_to_layout_25 = ttnn_decorators_ttnn_to_layout(ttnn_concat_24, ttnn_TILE_LAYOUT);  ttnn_concat_24 = None
    ttnn_to_layout_26 = ttnn_decorators_ttnn_to_layout(ttnn_concat_25, ttnn_TILE_LAYOUT);  ttnn_concat_25 = None
    ttnn_linear_157 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_216, ttnn_to_layout_25, bias = ttnn_to_layout_26);  ttnn_experimental_view_216 = ttnn_to_layout_25 = ttnn_to_layout_26 = None
    ttnn_experimental_view_485 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_157, [1, 256, 3072]);  ttnn_linear_157 = None
    ttnn_transformer_split_query_key_value_and_split_heads_12 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_485, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_485 = None
    getitem_87 = ttnn_transformer_split_query_key_value_and_split_heads_12[0]
    getitem_88 = ttnn_transformer_split_query_key_value_and_split_heads_12[1]
    getitem_89 = ttnn_transformer_split_query_key_value_and_split_heads_12[2];  ttnn_transformer_split_query_key_value_and_split_heads_12 = None
    ttnn_matmul_217 = ttnn_decorators_ttnn_matmul(getitem_87, getitem_88);  getitem_87 = getitem_88 = None
    ttnn_transformer_attention_softmax__36 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_217, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_217 = None
    ttnn_matmul_218 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__36, getitem_89);  ttnn_transformer_attention_softmax__36 = getitem_89 = None
    ttnn_transformer_concatenate_heads_12 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_218);  ttnn_matmul_218 = None
    ttnn_from_torch_207 = ttnn_decorators_ttnn_from_torch(arg203_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg203_1 = None
    ttnn_from_torch_208 = ttnn_decorators_ttnn_from_torch(arg204_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg204_1 = None
    ttnn_linear_75 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_12, ttnn_from_torch_207, transpose_b = True, bias = ttnn_from_torch_208, activation = None);  ttnn_transformer_concatenate_heads_12 = ttnn_from_torch_207 = ttnn_from_torch_208 = None
    ttnn_experimental_view_229 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_75, [1, 256, 1024]);  ttnn_linear_75 = None
    ttnn_add_184 = ttnn_decorators_ttnn_add(ttnn_experimental_view_229, ttnn_layer_norm_24);  ttnn_experimental_view_229 = ttnn_layer_norm_24 = None
    ttnn_from_torch_209 = ttnn_decorators_ttnn_from_torch(arg205_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg205_1 = None
    ttnn_from_torch_210 = ttnn_decorators_ttnn_from_torch(arg206_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg206_1 = None
    ttnn_layer_norm_25 = ttnn_decorators_ttnn_layer_norm(ttnn_add_184, epsilon = 1e-12, weight = ttnn_from_torch_209, bias = ttnn_from_torch_210);  ttnn_add_184 = ttnn_from_torch_209 = ttnn_from_torch_210 = None
    ttnn_experimental_view_230 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_25, [256, 1024])
    ttnn_from_torch_211 = ttnn_decorators_ttnn_from_torch(arg207_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg207_1 = None
    ttnn_from_torch_212 = ttnn_decorators_ttnn_from_torch(arg208_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg208_1 = None
    ttnn_linear_76 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_230, ttnn_from_torch_211, transpose_b = True, bias = ttnn_from_torch_212, activation = 'gelu');  ttnn_experimental_view_230 = ttnn_from_torch_211 = ttnn_from_torch_212 = None
    ttnn_experimental_view_232 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_76, [256, 4096]);  ttnn_linear_76 = None
    ttnn_from_torch_213 = ttnn_decorators_ttnn_from_torch(arg209_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg209_1 = None
    ttnn_from_torch_214 = ttnn_decorators_ttnn_from_torch(arg210_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg210_1 = None
    ttnn_linear_77 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_232, ttnn_from_torch_213, transpose_b = True, bias = ttnn_from_torch_214, activation = None);  ttnn_experimental_view_232 = ttnn_from_torch_213 = ttnn_from_torch_214 = None
    ttnn_experimental_view_233 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_77, [1, 256, 1024]);  ttnn_linear_77 = None
    ttnn_add_185 = ttnn_decorators_ttnn_add(ttnn_experimental_view_233, ttnn_layer_norm_25);  ttnn_experimental_view_233 = ttnn_layer_norm_25 = None
    ttnn_from_torch_215 = ttnn_decorators_ttnn_from_torch(arg211_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg211_1 = None
    ttnn_from_torch_216 = ttnn_decorators_ttnn_from_torch(arg212_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg212_1 = None
    ttnn_layer_norm_26 = ttnn_decorators_ttnn_layer_norm(ttnn_add_185, epsilon = 1e-12, weight = ttnn_from_torch_215, bias = ttnn_from_torch_216);  ttnn_add_185 = ttnn_from_torch_215 = ttnn_from_torch_216 = None
    ttnn_experimental_view_234 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_26, [256, 1024])
    ttnn_from_torch_217 = ttnn_decorators_ttnn_from_torch(arg213_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg213_1 = None
    ttnn_transpose_208 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_217, -2, -1);  ttnn_from_torch_217 = None
    ttnn_from_torch_218 = ttnn_decorators_ttnn_from_torch(arg215_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg215_1 = None
    ttnn_transpose_209 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_218, -2, -1);  ttnn_from_torch_218 = None
    ttnn_from_torch_219 = ttnn_decorators_ttnn_from_torch(arg217_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg217_1 = None
    ttnn_transpose_210 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_219, -2, -1);  ttnn_from_torch_219 = None
    ttnn_concat_26 = ttnn_decorators_ttnn_concat([ttnn_transpose_208, ttnn_transpose_209, ttnn_transpose_210], -1);  ttnn_transpose_208 = ttnn_transpose_209 = ttnn_transpose_210 = None
    ttnn_from_torch_220 = ttnn_decorators_ttnn_from_torch(arg214_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg214_1 = None
    ttnn_experimental_view_486 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_220, (1, -1));  ttnn_from_torch_220 = None
    ttnn_from_torch_221 = ttnn_decorators_ttnn_from_torch(arg216_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg216_1 = None
    ttnn_experimental_view_487 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_221, (1, -1));  ttnn_from_torch_221 = None
    ttnn_from_torch_222 = ttnn_decorators_ttnn_from_torch(arg218_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg218_1 = None
    ttnn_experimental_view_488 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_222, (1, -1));  ttnn_from_torch_222 = None
    ttnn_concat_27 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_486, ttnn_experimental_view_487, ttnn_experimental_view_488], -1);  ttnn_experimental_view_486 = ttnn_experimental_view_487 = ttnn_experimental_view_488 = None
    ttnn_to_layout_27 = ttnn_decorators_ttnn_to_layout(ttnn_concat_26, ttnn_TILE_LAYOUT);  ttnn_concat_26 = None
    ttnn_to_layout_28 = ttnn_decorators_ttnn_to_layout(ttnn_concat_27, ttnn_TILE_LAYOUT);  ttnn_concat_27 = None
    ttnn_linear_158 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_234, ttnn_to_layout_27, bias = ttnn_to_layout_28);  ttnn_experimental_view_234 = ttnn_to_layout_27 = ttnn_to_layout_28 = None
    ttnn_experimental_view_489 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_158, [1, 256, 3072]);  ttnn_linear_158 = None
    ttnn_transformer_split_query_key_value_and_split_heads_13 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_489, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_489 = None
    getitem_90 = ttnn_transformer_split_query_key_value_and_split_heads_13[0]
    getitem_91 = ttnn_transformer_split_query_key_value_and_split_heads_13[1]
    getitem_92 = ttnn_transformer_split_query_key_value_and_split_heads_13[2];  ttnn_transformer_split_query_key_value_and_split_heads_13 = None
    ttnn_matmul_219 = ttnn_decorators_ttnn_matmul(getitem_90, getitem_91);  getitem_90 = getitem_91 = None
    ttnn_transformer_attention_softmax__37 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_219, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_219 = None
    ttnn_matmul_220 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__37, getitem_92);  ttnn_transformer_attention_softmax__37 = getitem_92 = None
    ttnn_transformer_concatenate_heads_13 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_220);  ttnn_matmul_220 = None
    ttnn_from_torch_223 = ttnn_decorators_ttnn_from_torch(arg219_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg219_1 = None
    ttnn_from_torch_224 = ttnn_decorators_ttnn_from_torch(arg220_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg220_1 = None
    ttnn_linear_81 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_13, ttnn_from_torch_223, transpose_b = True, bias = ttnn_from_torch_224, activation = None);  ttnn_transformer_concatenate_heads_13 = ttnn_from_torch_223 = ttnn_from_torch_224 = None
    ttnn_experimental_view_247 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_81, [1, 256, 1024]);  ttnn_linear_81 = None
    ttnn_add_187 = ttnn_decorators_ttnn_add(ttnn_experimental_view_247, ttnn_layer_norm_26);  ttnn_experimental_view_247 = ttnn_layer_norm_26 = None
    ttnn_from_torch_225 = ttnn_decorators_ttnn_from_torch(arg221_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg221_1 = None
    ttnn_from_torch_226 = ttnn_decorators_ttnn_from_torch(arg222_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg222_1 = None
    ttnn_layer_norm_27 = ttnn_decorators_ttnn_layer_norm(ttnn_add_187, epsilon = 1e-12, weight = ttnn_from_torch_225, bias = ttnn_from_torch_226);  ttnn_add_187 = ttnn_from_torch_225 = ttnn_from_torch_226 = None
    ttnn_experimental_view_248 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_27, [256, 1024])
    ttnn_from_torch_227 = ttnn_decorators_ttnn_from_torch(arg223_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg223_1 = None
    ttnn_from_torch_228 = ttnn_decorators_ttnn_from_torch(arg224_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg224_1 = None
    ttnn_linear_82 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_248, ttnn_from_torch_227, transpose_b = True, bias = ttnn_from_torch_228, activation = 'gelu');  ttnn_experimental_view_248 = ttnn_from_torch_227 = ttnn_from_torch_228 = None
    ttnn_experimental_view_250 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_82, [256, 4096]);  ttnn_linear_82 = None
    ttnn_from_torch_229 = ttnn_decorators_ttnn_from_torch(arg225_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg225_1 = None
    ttnn_from_torch_230 = ttnn_decorators_ttnn_from_torch(arg226_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg226_1 = None
    ttnn_linear_83 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_250, ttnn_from_torch_229, transpose_b = True, bias = ttnn_from_torch_230, activation = None);  ttnn_experimental_view_250 = ttnn_from_torch_229 = ttnn_from_torch_230 = None
    ttnn_experimental_view_251 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_83, [1, 256, 1024]);  ttnn_linear_83 = None
    ttnn_add_188 = ttnn_decorators_ttnn_add(ttnn_experimental_view_251, ttnn_layer_norm_27);  ttnn_experimental_view_251 = ttnn_layer_norm_27 = None
    ttnn_from_torch_231 = ttnn_decorators_ttnn_from_torch(arg227_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg227_1 = None
    ttnn_from_torch_232 = ttnn_decorators_ttnn_from_torch(arg228_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg228_1 = None
    ttnn_layer_norm_28 = ttnn_decorators_ttnn_layer_norm(ttnn_add_188, epsilon = 1e-12, weight = ttnn_from_torch_231, bias = ttnn_from_torch_232);  ttnn_add_188 = ttnn_from_torch_231 = ttnn_from_torch_232 = None
    ttnn_experimental_view_252 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_28, [256, 1024])
    ttnn_from_torch_233 = ttnn_decorators_ttnn_from_torch(arg229_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg229_1 = None
    ttnn_transpose_211 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_233, -2, -1);  ttnn_from_torch_233 = None
    ttnn_from_torch_234 = ttnn_decorators_ttnn_from_torch(arg231_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg231_1 = None
    ttnn_transpose_212 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_234, -2, -1);  ttnn_from_torch_234 = None
    ttnn_from_torch_235 = ttnn_decorators_ttnn_from_torch(arg233_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg233_1 = None
    ttnn_transpose_213 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_235, -2, -1);  ttnn_from_torch_235 = None
    ttnn_concat_28 = ttnn_decorators_ttnn_concat([ttnn_transpose_211, ttnn_transpose_212, ttnn_transpose_213], -1);  ttnn_transpose_211 = ttnn_transpose_212 = ttnn_transpose_213 = None
    ttnn_from_torch_236 = ttnn_decorators_ttnn_from_torch(arg230_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg230_1 = None
    ttnn_experimental_view_490 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_236, (1, -1));  ttnn_from_torch_236 = None
    ttnn_from_torch_237 = ttnn_decorators_ttnn_from_torch(arg232_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg232_1 = None
    ttnn_experimental_view_491 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_237, (1, -1));  ttnn_from_torch_237 = None
    ttnn_from_torch_238 = ttnn_decorators_ttnn_from_torch(arg234_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg234_1 = None
    ttnn_experimental_view_492 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_238, (1, -1));  ttnn_from_torch_238 = None
    ttnn_concat_29 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_490, ttnn_experimental_view_491, ttnn_experimental_view_492], -1);  ttnn_experimental_view_490 = ttnn_experimental_view_491 = ttnn_experimental_view_492 = None
    ttnn_to_layout_29 = ttnn_decorators_ttnn_to_layout(ttnn_concat_28, ttnn_TILE_LAYOUT);  ttnn_concat_28 = None
    ttnn_to_layout_30 = ttnn_decorators_ttnn_to_layout(ttnn_concat_29, ttnn_TILE_LAYOUT);  ttnn_concat_29 = None
    ttnn_linear_159 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_252, ttnn_to_layout_29, bias = ttnn_to_layout_30);  ttnn_experimental_view_252 = ttnn_to_layout_29 = ttnn_to_layout_30 = None
    ttnn_experimental_view_493 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_159, [1, 256, 3072]);  ttnn_linear_159 = None
    ttnn_transformer_split_query_key_value_and_split_heads_14 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_493, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_493 = None
    getitem_93 = ttnn_transformer_split_query_key_value_and_split_heads_14[0]
    getitem_94 = ttnn_transformer_split_query_key_value_and_split_heads_14[1]
    getitem_95 = ttnn_transformer_split_query_key_value_and_split_heads_14[2];  ttnn_transformer_split_query_key_value_and_split_heads_14 = None
    ttnn_matmul_221 = ttnn_decorators_ttnn_matmul(getitem_93, getitem_94);  getitem_93 = getitem_94 = None
    ttnn_transformer_attention_softmax__38 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_221, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_221 = None
    ttnn_matmul_222 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__38, getitem_95);  ttnn_transformer_attention_softmax__38 = getitem_95 = None
    ttnn_transformer_concatenate_heads_14 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_222);  ttnn_matmul_222 = None
    ttnn_from_torch_239 = ttnn_decorators_ttnn_from_torch(arg235_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg235_1 = None
    ttnn_from_torch_240 = ttnn_decorators_ttnn_from_torch(arg236_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg236_1 = None
    ttnn_linear_87 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_14, ttnn_from_torch_239, transpose_b = True, bias = ttnn_from_torch_240, activation = None);  ttnn_transformer_concatenate_heads_14 = ttnn_from_torch_239 = ttnn_from_torch_240 = None
    ttnn_experimental_view_265 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_87, [1, 256, 1024]);  ttnn_linear_87 = None
    ttnn_add_190 = ttnn_decorators_ttnn_add(ttnn_experimental_view_265, ttnn_layer_norm_28);  ttnn_experimental_view_265 = ttnn_layer_norm_28 = None
    ttnn_from_torch_241 = ttnn_decorators_ttnn_from_torch(arg237_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg237_1 = None
    ttnn_from_torch_242 = ttnn_decorators_ttnn_from_torch(arg238_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg238_1 = None
    ttnn_layer_norm_29 = ttnn_decorators_ttnn_layer_norm(ttnn_add_190, epsilon = 1e-12, weight = ttnn_from_torch_241, bias = ttnn_from_torch_242);  ttnn_add_190 = ttnn_from_torch_241 = ttnn_from_torch_242 = None
    ttnn_experimental_view_266 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_29, [256, 1024])
    ttnn_from_torch_243 = ttnn_decorators_ttnn_from_torch(arg239_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg239_1 = None
    ttnn_from_torch_244 = ttnn_decorators_ttnn_from_torch(arg240_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg240_1 = None
    ttnn_linear_88 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_266, ttnn_from_torch_243, transpose_b = True, bias = ttnn_from_torch_244, activation = 'gelu');  ttnn_experimental_view_266 = ttnn_from_torch_243 = ttnn_from_torch_244 = None
    ttnn_experimental_view_268 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_88, [256, 4096]);  ttnn_linear_88 = None
    ttnn_from_torch_245 = ttnn_decorators_ttnn_from_torch(arg241_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg241_1 = None
    ttnn_from_torch_246 = ttnn_decorators_ttnn_from_torch(arg242_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg242_1 = None
    ttnn_linear_89 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_268, ttnn_from_torch_245, transpose_b = True, bias = ttnn_from_torch_246, activation = None);  ttnn_experimental_view_268 = ttnn_from_torch_245 = ttnn_from_torch_246 = None
    ttnn_experimental_view_269 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_89, [1, 256, 1024]);  ttnn_linear_89 = None
    ttnn_add_191 = ttnn_decorators_ttnn_add(ttnn_experimental_view_269, ttnn_layer_norm_29);  ttnn_experimental_view_269 = ttnn_layer_norm_29 = None
    ttnn_from_torch_247 = ttnn_decorators_ttnn_from_torch(arg243_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg243_1 = None
    ttnn_from_torch_248 = ttnn_decorators_ttnn_from_torch(arg244_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg244_1 = None
    ttnn_layer_norm_30 = ttnn_decorators_ttnn_layer_norm(ttnn_add_191, epsilon = 1e-12, weight = ttnn_from_torch_247, bias = ttnn_from_torch_248);  ttnn_add_191 = ttnn_from_torch_247 = ttnn_from_torch_248 = None
    ttnn_experimental_view_270 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_30, [256, 1024])
    ttnn_from_torch_249 = ttnn_decorators_ttnn_from_torch(arg245_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg245_1 = None
    ttnn_transpose_214 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_249, -2, -1);  ttnn_from_torch_249 = None
    ttnn_from_torch_250 = ttnn_decorators_ttnn_from_torch(arg247_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg247_1 = None
    ttnn_transpose_215 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_250, -2, -1);  ttnn_from_torch_250 = None
    ttnn_from_torch_251 = ttnn_decorators_ttnn_from_torch(arg249_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg249_1 = None
    ttnn_transpose_216 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_251, -2, -1);  ttnn_from_torch_251 = None
    ttnn_concat_30 = ttnn_decorators_ttnn_concat([ttnn_transpose_214, ttnn_transpose_215, ttnn_transpose_216], -1);  ttnn_transpose_214 = ttnn_transpose_215 = ttnn_transpose_216 = None
    ttnn_from_torch_252 = ttnn_decorators_ttnn_from_torch(arg246_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg246_1 = None
    ttnn_experimental_view_494 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_252, (1, -1));  ttnn_from_torch_252 = None
    ttnn_from_torch_253 = ttnn_decorators_ttnn_from_torch(arg248_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg248_1 = None
    ttnn_experimental_view_495 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_253, (1, -1));  ttnn_from_torch_253 = None
    ttnn_from_torch_254 = ttnn_decorators_ttnn_from_torch(arg250_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg250_1 = None
    ttnn_experimental_view_496 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_254, (1, -1));  ttnn_from_torch_254 = None
    ttnn_concat_31 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_494, ttnn_experimental_view_495, ttnn_experimental_view_496], -1);  ttnn_experimental_view_494 = ttnn_experimental_view_495 = ttnn_experimental_view_496 = None
    ttnn_to_layout_31 = ttnn_decorators_ttnn_to_layout(ttnn_concat_30, ttnn_TILE_LAYOUT);  ttnn_concat_30 = None
    ttnn_to_layout_32 = ttnn_decorators_ttnn_to_layout(ttnn_concat_31, ttnn_TILE_LAYOUT);  ttnn_concat_31 = None
    ttnn_linear_160 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_270, ttnn_to_layout_31, bias = ttnn_to_layout_32);  ttnn_experimental_view_270 = ttnn_to_layout_31 = ttnn_to_layout_32 = None
    ttnn_experimental_view_497 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_160, [1, 256, 3072]);  ttnn_linear_160 = None
    ttnn_transformer_split_query_key_value_and_split_heads_15 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_497, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_497 = None
    getitem_96 = ttnn_transformer_split_query_key_value_and_split_heads_15[0]
    getitem_97 = ttnn_transformer_split_query_key_value_and_split_heads_15[1]
    getitem_98 = ttnn_transformer_split_query_key_value_and_split_heads_15[2];  ttnn_transformer_split_query_key_value_and_split_heads_15 = None
    ttnn_matmul_223 = ttnn_decorators_ttnn_matmul(getitem_96, getitem_97);  getitem_96 = getitem_97 = None
    ttnn_transformer_attention_softmax__39 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_223, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_223 = None
    ttnn_matmul_224 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__39, getitem_98);  ttnn_transformer_attention_softmax__39 = getitem_98 = None
    ttnn_transformer_concatenate_heads_15 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_224);  ttnn_matmul_224 = None
    ttnn_from_torch_255 = ttnn_decorators_ttnn_from_torch(arg251_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg251_1 = None
    ttnn_from_torch_256 = ttnn_decorators_ttnn_from_torch(arg252_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg252_1 = None
    ttnn_linear_93 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_15, ttnn_from_torch_255, transpose_b = True, bias = ttnn_from_torch_256, activation = None);  ttnn_transformer_concatenate_heads_15 = ttnn_from_torch_255 = ttnn_from_torch_256 = None
    ttnn_experimental_view_283 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_93, [1, 256, 1024]);  ttnn_linear_93 = None
    ttnn_add_193 = ttnn_decorators_ttnn_add(ttnn_experimental_view_283, ttnn_layer_norm_30);  ttnn_experimental_view_283 = ttnn_layer_norm_30 = None
    ttnn_from_torch_257 = ttnn_decorators_ttnn_from_torch(arg253_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg253_1 = None
    ttnn_from_torch_258 = ttnn_decorators_ttnn_from_torch(arg254_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg254_1 = None
    ttnn_layer_norm_31 = ttnn_decorators_ttnn_layer_norm(ttnn_add_193, epsilon = 1e-12, weight = ttnn_from_torch_257, bias = ttnn_from_torch_258);  ttnn_add_193 = ttnn_from_torch_257 = ttnn_from_torch_258 = None
    ttnn_experimental_view_284 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_31, [256, 1024])
    ttnn_from_torch_259 = ttnn_decorators_ttnn_from_torch(arg255_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg255_1 = None
    ttnn_from_torch_260 = ttnn_decorators_ttnn_from_torch(arg256_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg256_1 = None
    ttnn_linear_94 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_284, ttnn_from_torch_259, transpose_b = True, bias = ttnn_from_torch_260, activation = 'gelu');  ttnn_experimental_view_284 = ttnn_from_torch_259 = ttnn_from_torch_260 = None
    ttnn_experimental_view_286 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_94, [256, 4096]);  ttnn_linear_94 = None
    ttnn_from_torch_261 = ttnn_decorators_ttnn_from_torch(arg257_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg257_1 = None
    ttnn_from_torch_262 = ttnn_decorators_ttnn_from_torch(arg258_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg258_1 = None
    ttnn_linear_95 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_286, ttnn_from_torch_261, transpose_b = True, bias = ttnn_from_torch_262, activation = None);  ttnn_experimental_view_286 = ttnn_from_torch_261 = ttnn_from_torch_262 = None
    ttnn_experimental_view_287 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_95, [1, 256, 1024]);  ttnn_linear_95 = None
    ttnn_add_194 = ttnn_decorators_ttnn_add(ttnn_experimental_view_287, ttnn_layer_norm_31);  ttnn_experimental_view_287 = ttnn_layer_norm_31 = None
    ttnn_from_torch_263 = ttnn_decorators_ttnn_from_torch(arg259_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg259_1 = None
    ttnn_from_torch_264 = ttnn_decorators_ttnn_from_torch(arg260_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg260_1 = None
    ttnn_layer_norm_32 = ttnn_decorators_ttnn_layer_norm(ttnn_add_194, epsilon = 1e-12, weight = ttnn_from_torch_263, bias = ttnn_from_torch_264);  ttnn_add_194 = ttnn_from_torch_263 = ttnn_from_torch_264 = None
    ttnn_experimental_view_288 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_32, [256, 1024])
    ttnn_from_torch_265 = ttnn_decorators_ttnn_from_torch(arg261_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg261_1 = None
    ttnn_transpose_217 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_265, -2, -1);  ttnn_from_torch_265 = None
    ttnn_from_torch_266 = ttnn_decorators_ttnn_from_torch(arg263_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg263_1 = None
    ttnn_transpose_218 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_266, -2, -1);  ttnn_from_torch_266 = None
    ttnn_from_torch_267 = ttnn_decorators_ttnn_from_torch(arg265_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg265_1 = None
    ttnn_transpose_219 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_267, -2, -1);  ttnn_from_torch_267 = None
    ttnn_concat_32 = ttnn_decorators_ttnn_concat([ttnn_transpose_217, ttnn_transpose_218, ttnn_transpose_219], -1);  ttnn_transpose_217 = ttnn_transpose_218 = ttnn_transpose_219 = None
    ttnn_from_torch_268 = ttnn_decorators_ttnn_from_torch(arg262_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg262_1 = None
    ttnn_experimental_view_498 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_268, (1, -1));  ttnn_from_torch_268 = None
    ttnn_from_torch_269 = ttnn_decorators_ttnn_from_torch(arg264_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg264_1 = None
    ttnn_experimental_view_499 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_269, (1, -1));  ttnn_from_torch_269 = None
    ttnn_from_torch_270 = ttnn_decorators_ttnn_from_torch(arg266_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg266_1 = None
    ttnn_experimental_view_500 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_270, (1, -1));  ttnn_from_torch_270 = None
    ttnn_concat_33 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_498, ttnn_experimental_view_499, ttnn_experimental_view_500], -1);  ttnn_experimental_view_498 = ttnn_experimental_view_499 = ttnn_experimental_view_500 = None
    ttnn_to_layout_33 = ttnn_decorators_ttnn_to_layout(ttnn_concat_32, ttnn_TILE_LAYOUT);  ttnn_concat_32 = None
    ttnn_to_layout_34 = ttnn_decorators_ttnn_to_layout(ttnn_concat_33, ttnn_TILE_LAYOUT);  ttnn_concat_33 = None
    ttnn_linear_161 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_288, ttnn_to_layout_33, bias = ttnn_to_layout_34);  ttnn_experimental_view_288 = ttnn_to_layout_33 = ttnn_to_layout_34 = None
    ttnn_experimental_view_501 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_161, [1, 256, 3072]);  ttnn_linear_161 = None
    ttnn_transformer_split_query_key_value_and_split_heads_16 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_501, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_501 = None
    getitem_99 = ttnn_transformer_split_query_key_value_and_split_heads_16[0]
    getitem_100 = ttnn_transformer_split_query_key_value_and_split_heads_16[1]
    getitem_101 = ttnn_transformer_split_query_key_value_and_split_heads_16[2];  ttnn_transformer_split_query_key_value_and_split_heads_16 = None
    ttnn_matmul_225 = ttnn_decorators_ttnn_matmul(getitem_99, getitem_100);  getitem_99 = getitem_100 = None
    ttnn_transformer_attention_softmax__40 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_225, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_225 = None
    ttnn_matmul_226 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__40, getitem_101);  ttnn_transformer_attention_softmax__40 = getitem_101 = None
    ttnn_transformer_concatenate_heads_16 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_226);  ttnn_matmul_226 = None
    ttnn_from_torch_271 = ttnn_decorators_ttnn_from_torch(arg267_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg267_1 = None
    ttnn_from_torch_272 = ttnn_decorators_ttnn_from_torch(arg268_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg268_1 = None
    ttnn_linear_99 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_16, ttnn_from_torch_271, transpose_b = True, bias = ttnn_from_torch_272, activation = None);  ttnn_transformer_concatenate_heads_16 = ttnn_from_torch_271 = ttnn_from_torch_272 = None
    ttnn_experimental_view_301 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_99, [1, 256, 1024]);  ttnn_linear_99 = None
    ttnn_add_196 = ttnn_decorators_ttnn_add(ttnn_experimental_view_301, ttnn_layer_norm_32);  ttnn_experimental_view_301 = ttnn_layer_norm_32 = None
    ttnn_from_torch_273 = ttnn_decorators_ttnn_from_torch(arg269_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg269_1 = None
    ttnn_from_torch_274 = ttnn_decorators_ttnn_from_torch(arg270_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg270_1 = None
    ttnn_layer_norm_33 = ttnn_decorators_ttnn_layer_norm(ttnn_add_196, epsilon = 1e-12, weight = ttnn_from_torch_273, bias = ttnn_from_torch_274);  ttnn_add_196 = ttnn_from_torch_273 = ttnn_from_torch_274 = None
    ttnn_experimental_view_302 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_33, [256, 1024])
    ttnn_from_torch_275 = ttnn_decorators_ttnn_from_torch(arg271_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg271_1 = None
    ttnn_from_torch_276 = ttnn_decorators_ttnn_from_torch(arg272_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg272_1 = None
    ttnn_linear_100 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_302, ttnn_from_torch_275, transpose_b = True, bias = ttnn_from_torch_276, activation = 'gelu');  ttnn_experimental_view_302 = ttnn_from_torch_275 = ttnn_from_torch_276 = None
    ttnn_experimental_view_304 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_100, [256, 4096]);  ttnn_linear_100 = None
    ttnn_from_torch_277 = ttnn_decorators_ttnn_from_torch(arg273_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg273_1 = None
    ttnn_from_torch_278 = ttnn_decorators_ttnn_from_torch(arg274_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg274_1 = None
    ttnn_linear_101 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_304, ttnn_from_torch_277, transpose_b = True, bias = ttnn_from_torch_278, activation = None);  ttnn_experimental_view_304 = ttnn_from_torch_277 = ttnn_from_torch_278 = None
    ttnn_experimental_view_305 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_101, [1, 256, 1024]);  ttnn_linear_101 = None
    ttnn_add_197 = ttnn_decorators_ttnn_add(ttnn_experimental_view_305, ttnn_layer_norm_33);  ttnn_experimental_view_305 = ttnn_layer_norm_33 = None
    ttnn_from_torch_279 = ttnn_decorators_ttnn_from_torch(arg275_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg275_1 = None
    ttnn_from_torch_280 = ttnn_decorators_ttnn_from_torch(arg276_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg276_1 = None
    ttnn_layer_norm_34 = ttnn_decorators_ttnn_layer_norm(ttnn_add_197, epsilon = 1e-12, weight = ttnn_from_torch_279, bias = ttnn_from_torch_280);  ttnn_add_197 = ttnn_from_torch_279 = ttnn_from_torch_280 = None
    ttnn_experimental_view_306 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_34, [256, 1024])
    ttnn_from_torch_281 = ttnn_decorators_ttnn_from_torch(arg277_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg277_1 = None
    ttnn_transpose_220 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_281, -2, -1);  ttnn_from_torch_281 = None
    ttnn_from_torch_282 = ttnn_decorators_ttnn_from_torch(arg279_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg279_1 = None
    ttnn_transpose_221 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_282, -2, -1);  ttnn_from_torch_282 = None
    ttnn_from_torch_283 = ttnn_decorators_ttnn_from_torch(arg281_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg281_1 = None
    ttnn_transpose_222 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_283, -2, -1);  ttnn_from_torch_283 = None
    ttnn_concat_34 = ttnn_decorators_ttnn_concat([ttnn_transpose_220, ttnn_transpose_221, ttnn_transpose_222], -1);  ttnn_transpose_220 = ttnn_transpose_221 = ttnn_transpose_222 = None
    ttnn_from_torch_284 = ttnn_decorators_ttnn_from_torch(arg278_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg278_1 = None
    ttnn_experimental_view_502 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_284, (1, -1));  ttnn_from_torch_284 = None
    ttnn_from_torch_285 = ttnn_decorators_ttnn_from_torch(arg280_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg280_1 = None
    ttnn_experimental_view_503 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_285, (1, -1));  ttnn_from_torch_285 = None
    ttnn_from_torch_286 = ttnn_decorators_ttnn_from_torch(arg282_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg282_1 = None
    ttnn_experimental_view_504 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_286, (1, -1));  ttnn_from_torch_286 = None
    ttnn_concat_35 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_502, ttnn_experimental_view_503, ttnn_experimental_view_504], -1);  ttnn_experimental_view_502 = ttnn_experimental_view_503 = ttnn_experimental_view_504 = None
    ttnn_to_layout_35 = ttnn_decorators_ttnn_to_layout(ttnn_concat_34, ttnn_TILE_LAYOUT);  ttnn_concat_34 = None
    ttnn_to_layout_36 = ttnn_decorators_ttnn_to_layout(ttnn_concat_35, ttnn_TILE_LAYOUT);  ttnn_concat_35 = None
    ttnn_linear_162 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_306, ttnn_to_layout_35, bias = ttnn_to_layout_36);  ttnn_experimental_view_306 = ttnn_to_layout_35 = ttnn_to_layout_36 = None
    ttnn_experimental_view_505 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_162, [1, 256, 3072]);  ttnn_linear_162 = None
    ttnn_transformer_split_query_key_value_and_split_heads_17 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_505, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_505 = None
    getitem_102 = ttnn_transformer_split_query_key_value_and_split_heads_17[0]
    getitem_103 = ttnn_transformer_split_query_key_value_and_split_heads_17[1]
    getitem_104 = ttnn_transformer_split_query_key_value_and_split_heads_17[2];  ttnn_transformer_split_query_key_value_and_split_heads_17 = None
    ttnn_matmul_227 = ttnn_decorators_ttnn_matmul(getitem_102, getitem_103);  getitem_102 = getitem_103 = None
    ttnn_transformer_attention_softmax__41 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_227, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_227 = None
    ttnn_matmul_228 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__41, getitem_104);  ttnn_transformer_attention_softmax__41 = getitem_104 = None
    ttnn_transformer_concatenate_heads_17 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_228);  ttnn_matmul_228 = None
    ttnn_from_torch_287 = ttnn_decorators_ttnn_from_torch(arg283_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg283_1 = None
    ttnn_from_torch_288 = ttnn_decorators_ttnn_from_torch(arg284_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg284_1 = None
    ttnn_linear_105 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_17, ttnn_from_torch_287, transpose_b = True, bias = ttnn_from_torch_288, activation = None);  ttnn_transformer_concatenate_heads_17 = ttnn_from_torch_287 = ttnn_from_torch_288 = None
    ttnn_experimental_view_319 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_105, [1, 256, 1024]);  ttnn_linear_105 = None
    ttnn_add_199 = ttnn_decorators_ttnn_add(ttnn_experimental_view_319, ttnn_layer_norm_34);  ttnn_experimental_view_319 = ttnn_layer_norm_34 = None
    ttnn_from_torch_289 = ttnn_decorators_ttnn_from_torch(arg285_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg285_1 = None
    ttnn_from_torch_290 = ttnn_decorators_ttnn_from_torch(arg286_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg286_1 = None
    ttnn_layer_norm_35 = ttnn_decorators_ttnn_layer_norm(ttnn_add_199, epsilon = 1e-12, weight = ttnn_from_torch_289, bias = ttnn_from_torch_290);  ttnn_add_199 = ttnn_from_torch_289 = ttnn_from_torch_290 = None
    ttnn_experimental_view_320 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_35, [256, 1024])
    ttnn_from_torch_291 = ttnn_decorators_ttnn_from_torch(arg287_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg287_1 = None
    ttnn_from_torch_292 = ttnn_decorators_ttnn_from_torch(arg288_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg288_1 = None
    ttnn_linear_106 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_320, ttnn_from_torch_291, transpose_b = True, bias = ttnn_from_torch_292, activation = 'gelu');  ttnn_experimental_view_320 = ttnn_from_torch_291 = ttnn_from_torch_292 = None
    ttnn_experimental_view_322 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_106, [256, 4096]);  ttnn_linear_106 = None
    ttnn_from_torch_293 = ttnn_decorators_ttnn_from_torch(arg289_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg289_1 = None
    ttnn_from_torch_294 = ttnn_decorators_ttnn_from_torch(arg290_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg290_1 = None
    ttnn_linear_107 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_322, ttnn_from_torch_293, transpose_b = True, bias = ttnn_from_torch_294, activation = None);  ttnn_experimental_view_322 = ttnn_from_torch_293 = ttnn_from_torch_294 = None
    ttnn_experimental_view_323 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_107, [1, 256, 1024]);  ttnn_linear_107 = None
    ttnn_add_200 = ttnn_decorators_ttnn_add(ttnn_experimental_view_323, ttnn_layer_norm_35);  ttnn_experimental_view_323 = ttnn_layer_norm_35 = None
    ttnn_from_torch_295 = ttnn_decorators_ttnn_from_torch(arg291_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg291_1 = None
    ttnn_from_torch_296 = ttnn_decorators_ttnn_from_torch(arg292_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg292_1 = None
    ttnn_layer_norm_36 = ttnn_decorators_ttnn_layer_norm(ttnn_add_200, epsilon = 1e-12, weight = ttnn_from_torch_295, bias = ttnn_from_torch_296);  ttnn_add_200 = ttnn_from_torch_295 = ttnn_from_torch_296 = None
    ttnn_experimental_view_324 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_36, [256, 1024])
    ttnn_from_torch_297 = ttnn_decorators_ttnn_from_torch(arg293_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg293_1 = None
    ttnn_transpose_223 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_297, -2, -1);  ttnn_from_torch_297 = None
    ttnn_from_torch_298 = ttnn_decorators_ttnn_from_torch(arg295_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg295_1 = None
    ttnn_transpose_224 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_298, -2, -1);  ttnn_from_torch_298 = None
    ttnn_from_torch_299 = ttnn_decorators_ttnn_from_torch(arg297_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg297_1 = None
    ttnn_transpose_225 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_299, -2, -1);  ttnn_from_torch_299 = None
    ttnn_concat_36 = ttnn_decorators_ttnn_concat([ttnn_transpose_223, ttnn_transpose_224, ttnn_transpose_225], -1);  ttnn_transpose_223 = ttnn_transpose_224 = ttnn_transpose_225 = None
    ttnn_from_torch_300 = ttnn_decorators_ttnn_from_torch(arg294_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg294_1 = None
    ttnn_experimental_view_506 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_300, (1, -1));  ttnn_from_torch_300 = None
    ttnn_from_torch_301 = ttnn_decorators_ttnn_from_torch(arg296_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg296_1 = None
    ttnn_experimental_view_507 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_301, (1, -1));  ttnn_from_torch_301 = None
    ttnn_from_torch_302 = ttnn_decorators_ttnn_from_torch(arg298_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg298_1 = None
    ttnn_experimental_view_508 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_302, (1, -1));  ttnn_from_torch_302 = None
    ttnn_concat_37 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_506, ttnn_experimental_view_507, ttnn_experimental_view_508], -1);  ttnn_experimental_view_506 = ttnn_experimental_view_507 = ttnn_experimental_view_508 = None
    ttnn_to_layout_37 = ttnn_decorators_ttnn_to_layout(ttnn_concat_36, ttnn_TILE_LAYOUT);  ttnn_concat_36 = None
    ttnn_to_layout_38 = ttnn_decorators_ttnn_to_layout(ttnn_concat_37, ttnn_TILE_LAYOUT);  ttnn_concat_37 = None
    ttnn_linear_163 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_324, ttnn_to_layout_37, bias = ttnn_to_layout_38);  ttnn_experimental_view_324 = ttnn_to_layout_37 = ttnn_to_layout_38 = None
    ttnn_experimental_view_509 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_163, [1, 256, 3072]);  ttnn_linear_163 = None
    ttnn_transformer_split_query_key_value_and_split_heads_18 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_509, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_509 = None
    getitem_105 = ttnn_transformer_split_query_key_value_and_split_heads_18[0]
    getitem_106 = ttnn_transformer_split_query_key_value_and_split_heads_18[1]
    getitem_107 = ttnn_transformer_split_query_key_value_and_split_heads_18[2];  ttnn_transformer_split_query_key_value_and_split_heads_18 = None
    ttnn_matmul_229 = ttnn_decorators_ttnn_matmul(getitem_105, getitem_106);  getitem_105 = getitem_106 = None
    ttnn_transformer_attention_softmax__42 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_229, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_229 = None
    ttnn_matmul_230 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__42, getitem_107);  ttnn_transformer_attention_softmax__42 = getitem_107 = None
    ttnn_transformer_concatenate_heads_18 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_230);  ttnn_matmul_230 = None
    ttnn_from_torch_303 = ttnn_decorators_ttnn_from_torch(arg299_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg299_1 = None
    ttnn_from_torch_304 = ttnn_decorators_ttnn_from_torch(arg300_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg300_1 = None
    ttnn_linear_111 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_18, ttnn_from_torch_303, transpose_b = True, bias = ttnn_from_torch_304, activation = None);  ttnn_transformer_concatenate_heads_18 = ttnn_from_torch_303 = ttnn_from_torch_304 = None
    ttnn_experimental_view_337 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_111, [1, 256, 1024]);  ttnn_linear_111 = None
    ttnn_add_202 = ttnn_decorators_ttnn_add(ttnn_experimental_view_337, ttnn_layer_norm_36);  ttnn_experimental_view_337 = ttnn_layer_norm_36 = None
    ttnn_from_torch_305 = ttnn_decorators_ttnn_from_torch(arg301_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg301_1 = None
    ttnn_from_torch_306 = ttnn_decorators_ttnn_from_torch(arg302_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg302_1 = None
    ttnn_layer_norm_37 = ttnn_decorators_ttnn_layer_norm(ttnn_add_202, epsilon = 1e-12, weight = ttnn_from_torch_305, bias = ttnn_from_torch_306);  ttnn_add_202 = ttnn_from_torch_305 = ttnn_from_torch_306 = None
    ttnn_experimental_view_338 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_37, [256, 1024])
    ttnn_from_torch_307 = ttnn_decorators_ttnn_from_torch(arg303_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg303_1 = None
    ttnn_from_torch_308 = ttnn_decorators_ttnn_from_torch(arg304_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg304_1 = None
    ttnn_linear_112 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_338, ttnn_from_torch_307, transpose_b = True, bias = ttnn_from_torch_308, activation = 'gelu');  ttnn_experimental_view_338 = ttnn_from_torch_307 = ttnn_from_torch_308 = None
    ttnn_experimental_view_340 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_112, [256, 4096]);  ttnn_linear_112 = None
    ttnn_from_torch_309 = ttnn_decorators_ttnn_from_torch(arg305_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg305_1 = None
    ttnn_from_torch_310 = ttnn_decorators_ttnn_from_torch(arg306_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg306_1 = None
    ttnn_linear_113 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_340, ttnn_from_torch_309, transpose_b = True, bias = ttnn_from_torch_310, activation = None);  ttnn_experimental_view_340 = ttnn_from_torch_309 = ttnn_from_torch_310 = None
    ttnn_experimental_view_341 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_113, [1, 256, 1024]);  ttnn_linear_113 = None
    ttnn_add_203 = ttnn_decorators_ttnn_add(ttnn_experimental_view_341, ttnn_layer_norm_37);  ttnn_experimental_view_341 = ttnn_layer_norm_37 = None
    ttnn_from_torch_311 = ttnn_decorators_ttnn_from_torch(arg307_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg307_1 = None
    ttnn_from_torch_312 = ttnn_decorators_ttnn_from_torch(arg308_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg308_1 = None
    ttnn_layer_norm_38 = ttnn_decorators_ttnn_layer_norm(ttnn_add_203, epsilon = 1e-12, weight = ttnn_from_torch_311, bias = ttnn_from_torch_312);  ttnn_add_203 = ttnn_from_torch_311 = ttnn_from_torch_312 = None
    ttnn_experimental_view_342 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_38, [256, 1024])
    ttnn_from_torch_313 = ttnn_decorators_ttnn_from_torch(arg309_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg309_1 = None
    ttnn_transpose_226 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_313, -2, -1);  ttnn_from_torch_313 = None
    ttnn_from_torch_314 = ttnn_decorators_ttnn_from_torch(arg311_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg311_1 = None
    ttnn_transpose_227 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_314, -2, -1);  ttnn_from_torch_314 = None
    ttnn_from_torch_315 = ttnn_decorators_ttnn_from_torch(arg313_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg313_1 = None
    ttnn_transpose_228 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_315, -2, -1);  ttnn_from_torch_315 = None
    ttnn_concat_38 = ttnn_decorators_ttnn_concat([ttnn_transpose_226, ttnn_transpose_227, ttnn_transpose_228], -1);  ttnn_transpose_226 = ttnn_transpose_227 = ttnn_transpose_228 = None
    ttnn_from_torch_316 = ttnn_decorators_ttnn_from_torch(arg310_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg310_1 = None
    ttnn_experimental_view_510 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_316, (1, -1));  ttnn_from_torch_316 = None
    ttnn_from_torch_317 = ttnn_decorators_ttnn_from_torch(arg312_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg312_1 = None
    ttnn_experimental_view_511 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_317, (1, -1));  ttnn_from_torch_317 = None
    ttnn_from_torch_318 = ttnn_decorators_ttnn_from_torch(arg314_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg314_1 = None
    ttnn_experimental_view_512 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_318, (1, -1));  ttnn_from_torch_318 = None
    ttnn_concat_39 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_510, ttnn_experimental_view_511, ttnn_experimental_view_512], -1);  ttnn_experimental_view_510 = ttnn_experimental_view_511 = ttnn_experimental_view_512 = None
    ttnn_to_layout_39 = ttnn_decorators_ttnn_to_layout(ttnn_concat_38, ttnn_TILE_LAYOUT);  ttnn_concat_38 = None
    ttnn_to_layout_40 = ttnn_decorators_ttnn_to_layout(ttnn_concat_39, ttnn_TILE_LAYOUT);  ttnn_concat_39 = None
    ttnn_linear_164 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_342, ttnn_to_layout_39, bias = ttnn_to_layout_40);  ttnn_experimental_view_342 = ttnn_to_layout_39 = ttnn_to_layout_40 = None
    ttnn_experimental_view_513 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_164, [1, 256, 3072]);  ttnn_linear_164 = None
    ttnn_transformer_split_query_key_value_and_split_heads_19 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_513, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_513 = None
    getitem_108 = ttnn_transformer_split_query_key_value_and_split_heads_19[0]
    getitem_109 = ttnn_transformer_split_query_key_value_and_split_heads_19[1]
    getitem_110 = ttnn_transformer_split_query_key_value_and_split_heads_19[2];  ttnn_transformer_split_query_key_value_and_split_heads_19 = None
    ttnn_matmul_231 = ttnn_decorators_ttnn_matmul(getitem_108, getitem_109);  getitem_108 = getitem_109 = None
    ttnn_transformer_attention_softmax__43 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_231, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_231 = None
    ttnn_matmul_232 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__43, getitem_110);  ttnn_transformer_attention_softmax__43 = getitem_110 = None
    ttnn_transformer_concatenate_heads_19 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_232);  ttnn_matmul_232 = None
    ttnn_from_torch_319 = ttnn_decorators_ttnn_from_torch(arg315_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg315_1 = None
    ttnn_from_torch_320 = ttnn_decorators_ttnn_from_torch(arg316_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg316_1 = None
    ttnn_linear_117 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_19, ttnn_from_torch_319, transpose_b = True, bias = ttnn_from_torch_320, activation = None);  ttnn_transformer_concatenate_heads_19 = ttnn_from_torch_319 = ttnn_from_torch_320 = None
    ttnn_experimental_view_355 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_117, [1, 256, 1024]);  ttnn_linear_117 = None
    ttnn_add_205 = ttnn_decorators_ttnn_add(ttnn_experimental_view_355, ttnn_layer_norm_38);  ttnn_experimental_view_355 = ttnn_layer_norm_38 = None
    ttnn_from_torch_321 = ttnn_decorators_ttnn_from_torch(arg317_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg317_1 = None
    ttnn_from_torch_322 = ttnn_decorators_ttnn_from_torch(arg318_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg318_1 = None
    ttnn_layer_norm_39 = ttnn_decorators_ttnn_layer_norm(ttnn_add_205, epsilon = 1e-12, weight = ttnn_from_torch_321, bias = ttnn_from_torch_322);  ttnn_add_205 = ttnn_from_torch_321 = ttnn_from_torch_322 = None
    ttnn_experimental_view_356 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_39, [256, 1024])
    ttnn_from_torch_323 = ttnn_decorators_ttnn_from_torch(arg319_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg319_1 = None
    ttnn_from_torch_324 = ttnn_decorators_ttnn_from_torch(arg320_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg320_1 = None
    ttnn_linear_118 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_356, ttnn_from_torch_323, transpose_b = True, bias = ttnn_from_torch_324, activation = 'gelu');  ttnn_experimental_view_356 = ttnn_from_torch_323 = ttnn_from_torch_324 = None
    ttnn_experimental_view_358 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_118, [256, 4096]);  ttnn_linear_118 = None
    ttnn_from_torch_325 = ttnn_decorators_ttnn_from_torch(arg321_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg321_1 = None
    ttnn_from_torch_326 = ttnn_decorators_ttnn_from_torch(arg322_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg322_1 = None
    ttnn_linear_119 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_358, ttnn_from_torch_325, transpose_b = True, bias = ttnn_from_torch_326, activation = None);  ttnn_experimental_view_358 = ttnn_from_torch_325 = ttnn_from_torch_326 = None
    ttnn_experimental_view_359 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_119, [1, 256, 1024]);  ttnn_linear_119 = None
    ttnn_add_206 = ttnn_decorators_ttnn_add(ttnn_experimental_view_359, ttnn_layer_norm_39);  ttnn_experimental_view_359 = ttnn_layer_norm_39 = None
    ttnn_from_torch_327 = ttnn_decorators_ttnn_from_torch(arg323_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg323_1 = None
    ttnn_from_torch_328 = ttnn_decorators_ttnn_from_torch(arg324_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg324_1 = None
    ttnn_layer_norm_40 = ttnn_decorators_ttnn_layer_norm(ttnn_add_206, epsilon = 1e-12, weight = ttnn_from_torch_327, bias = ttnn_from_torch_328);  ttnn_add_206 = ttnn_from_torch_327 = ttnn_from_torch_328 = None
    ttnn_experimental_view_360 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_40, [256, 1024])
    ttnn_from_torch_329 = ttnn_decorators_ttnn_from_torch(arg325_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg325_1 = None
    ttnn_transpose_229 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_329, -2, -1);  ttnn_from_torch_329 = None
    ttnn_from_torch_330 = ttnn_decorators_ttnn_from_torch(arg327_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg327_1 = None
    ttnn_transpose_230 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_330, -2, -1);  ttnn_from_torch_330 = None
    ttnn_from_torch_331 = ttnn_decorators_ttnn_from_torch(arg329_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg329_1 = None
    ttnn_transpose_231 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_331, -2, -1);  ttnn_from_torch_331 = None
    ttnn_concat_40 = ttnn_decorators_ttnn_concat([ttnn_transpose_229, ttnn_transpose_230, ttnn_transpose_231], -1);  ttnn_transpose_229 = ttnn_transpose_230 = ttnn_transpose_231 = None
    ttnn_from_torch_332 = ttnn_decorators_ttnn_from_torch(arg326_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg326_1 = None
    ttnn_experimental_view_514 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_332, (1, -1));  ttnn_from_torch_332 = None
    ttnn_from_torch_333 = ttnn_decorators_ttnn_from_torch(arg328_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg328_1 = None
    ttnn_experimental_view_515 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_333, (1, -1));  ttnn_from_torch_333 = None
    ttnn_from_torch_334 = ttnn_decorators_ttnn_from_torch(arg330_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg330_1 = None
    ttnn_experimental_view_516 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_334, (1, -1));  ttnn_from_torch_334 = None
    ttnn_concat_41 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_514, ttnn_experimental_view_515, ttnn_experimental_view_516], -1);  ttnn_experimental_view_514 = ttnn_experimental_view_515 = ttnn_experimental_view_516 = None
    ttnn_to_layout_41 = ttnn_decorators_ttnn_to_layout(ttnn_concat_40, ttnn_TILE_LAYOUT);  ttnn_concat_40 = None
    ttnn_to_layout_42 = ttnn_decorators_ttnn_to_layout(ttnn_concat_41, ttnn_TILE_LAYOUT);  ttnn_concat_41 = None
    ttnn_linear_165 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_360, ttnn_to_layout_41, bias = ttnn_to_layout_42);  ttnn_experimental_view_360 = ttnn_to_layout_41 = ttnn_to_layout_42 = None
    ttnn_experimental_view_517 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_165, [1, 256, 3072]);  ttnn_linear_165 = None
    ttnn_transformer_split_query_key_value_and_split_heads_20 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_517, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_517 = None
    getitem_111 = ttnn_transformer_split_query_key_value_and_split_heads_20[0]
    getitem_112 = ttnn_transformer_split_query_key_value_and_split_heads_20[1]
    getitem_113 = ttnn_transformer_split_query_key_value_and_split_heads_20[2];  ttnn_transformer_split_query_key_value_and_split_heads_20 = None
    ttnn_matmul_233 = ttnn_decorators_ttnn_matmul(getitem_111, getitem_112);  getitem_111 = getitem_112 = None
    ttnn_transformer_attention_softmax__44 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_233, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_233 = None
    ttnn_matmul_234 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__44, getitem_113);  ttnn_transformer_attention_softmax__44 = getitem_113 = None
    ttnn_transformer_concatenate_heads_20 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_234);  ttnn_matmul_234 = None
    ttnn_from_torch_335 = ttnn_decorators_ttnn_from_torch(arg331_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg331_1 = None
    ttnn_from_torch_336 = ttnn_decorators_ttnn_from_torch(arg332_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg332_1 = None
    ttnn_linear_123 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_20, ttnn_from_torch_335, transpose_b = True, bias = ttnn_from_torch_336, activation = None);  ttnn_transformer_concatenate_heads_20 = ttnn_from_torch_335 = ttnn_from_torch_336 = None
    ttnn_experimental_view_373 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_123, [1, 256, 1024]);  ttnn_linear_123 = None
    ttnn_add_208 = ttnn_decorators_ttnn_add(ttnn_experimental_view_373, ttnn_layer_norm_40);  ttnn_experimental_view_373 = ttnn_layer_norm_40 = None
    ttnn_from_torch_337 = ttnn_decorators_ttnn_from_torch(arg333_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg333_1 = None
    ttnn_from_torch_338 = ttnn_decorators_ttnn_from_torch(arg334_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg334_1 = None
    ttnn_layer_norm_41 = ttnn_decorators_ttnn_layer_norm(ttnn_add_208, epsilon = 1e-12, weight = ttnn_from_torch_337, bias = ttnn_from_torch_338);  ttnn_add_208 = ttnn_from_torch_337 = ttnn_from_torch_338 = None
    ttnn_experimental_view_374 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_41, [256, 1024])
    ttnn_from_torch_339 = ttnn_decorators_ttnn_from_torch(arg335_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg335_1 = None
    ttnn_from_torch_340 = ttnn_decorators_ttnn_from_torch(arg336_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg336_1 = None
    ttnn_linear_124 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_374, ttnn_from_torch_339, transpose_b = True, bias = ttnn_from_torch_340, activation = 'gelu');  ttnn_experimental_view_374 = ttnn_from_torch_339 = ttnn_from_torch_340 = None
    ttnn_experimental_view_376 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_124, [256, 4096]);  ttnn_linear_124 = None
    ttnn_from_torch_341 = ttnn_decorators_ttnn_from_torch(arg337_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg337_1 = None
    ttnn_from_torch_342 = ttnn_decorators_ttnn_from_torch(arg338_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg338_1 = None
    ttnn_linear_125 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_376, ttnn_from_torch_341, transpose_b = True, bias = ttnn_from_torch_342, activation = None);  ttnn_experimental_view_376 = ttnn_from_torch_341 = ttnn_from_torch_342 = None
    ttnn_experimental_view_377 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_125, [1, 256, 1024]);  ttnn_linear_125 = None
    ttnn_add_209 = ttnn_decorators_ttnn_add(ttnn_experimental_view_377, ttnn_layer_norm_41);  ttnn_experimental_view_377 = ttnn_layer_norm_41 = None
    ttnn_from_torch_343 = ttnn_decorators_ttnn_from_torch(arg339_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg339_1 = None
    ttnn_from_torch_344 = ttnn_decorators_ttnn_from_torch(arg340_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg340_1 = None
    ttnn_layer_norm_42 = ttnn_decorators_ttnn_layer_norm(ttnn_add_209, epsilon = 1e-12, weight = ttnn_from_torch_343, bias = ttnn_from_torch_344);  ttnn_add_209 = ttnn_from_torch_343 = ttnn_from_torch_344 = None
    ttnn_experimental_view_378 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_42, [256, 1024])
    ttnn_from_torch_345 = ttnn_decorators_ttnn_from_torch(arg341_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg341_1 = None
    ttnn_transpose_232 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_345, -2, -1);  ttnn_from_torch_345 = None
    ttnn_from_torch_346 = ttnn_decorators_ttnn_from_torch(arg343_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg343_1 = None
    ttnn_transpose_233 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_346, -2, -1);  ttnn_from_torch_346 = None
    ttnn_from_torch_347 = ttnn_decorators_ttnn_from_torch(arg345_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg345_1 = None
    ttnn_transpose_234 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_347, -2, -1);  ttnn_from_torch_347 = None
    ttnn_concat_42 = ttnn_decorators_ttnn_concat([ttnn_transpose_232, ttnn_transpose_233, ttnn_transpose_234], -1);  ttnn_transpose_232 = ttnn_transpose_233 = ttnn_transpose_234 = None
    ttnn_from_torch_348 = ttnn_decorators_ttnn_from_torch(arg342_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg342_1 = None
    ttnn_experimental_view_518 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_348, (1, -1));  ttnn_from_torch_348 = None
    ttnn_from_torch_349 = ttnn_decorators_ttnn_from_torch(arg344_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg344_1 = None
    ttnn_experimental_view_519 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_349, (1, -1));  ttnn_from_torch_349 = None
    ttnn_from_torch_350 = ttnn_decorators_ttnn_from_torch(arg346_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg346_1 = None
    ttnn_experimental_view_520 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_350, (1, -1));  ttnn_from_torch_350 = None
    ttnn_concat_43 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_518, ttnn_experimental_view_519, ttnn_experimental_view_520], -1);  ttnn_experimental_view_518 = ttnn_experimental_view_519 = ttnn_experimental_view_520 = None
    ttnn_to_layout_43 = ttnn_decorators_ttnn_to_layout(ttnn_concat_42, ttnn_TILE_LAYOUT);  ttnn_concat_42 = None
    ttnn_to_layout_44 = ttnn_decorators_ttnn_to_layout(ttnn_concat_43, ttnn_TILE_LAYOUT);  ttnn_concat_43 = None
    ttnn_linear_166 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_378, ttnn_to_layout_43, bias = ttnn_to_layout_44);  ttnn_experimental_view_378 = ttnn_to_layout_43 = ttnn_to_layout_44 = None
    ttnn_experimental_view_521 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_166, [1, 256, 3072]);  ttnn_linear_166 = None
    ttnn_transformer_split_query_key_value_and_split_heads_21 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_521, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_521 = None
    getitem_114 = ttnn_transformer_split_query_key_value_and_split_heads_21[0]
    getitem_115 = ttnn_transformer_split_query_key_value_and_split_heads_21[1]
    getitem_116 = ttnn_transformer_split_query_key_value_and_split_heads_21[2];  ttnn_transformer_split_query_key_value_and_split_heads_21 = None
    ttnn_matmul_235 = ttnn_decorators_ttnn_matmul(getitem_114, getitem_115);  getitem_114 = getitem_115 = None
    ttnn_transformer_attention_softmax__45 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_235, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_235 = None
    ttnn_matmul_236 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__45, getitem_116);  ttnn_transformer_attention_softmax__45 = getitem_116 = None
    ttnn_transformer_concatenate_heads_21 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_236);  ttnn_matmul_236 = None
    ttnn_from_torch_351 = ttnn_decorators_ttnn_from_torch(arg347_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg347_1 = None
    ttnn_from_torch_352 = ttnn_decorators_ttnn_from_torch(arg348_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg348_1 = None
    ttnn_linear_129 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_21, ttnn_from_torch_351, transpose_b = True, bias = ttnn_from_torch_352, activation = None);  ttnn_transformer_concatenate_heads_21 = ttnn_from_torch_351 = ttnn_from_torch_352 = None
    ttnn_experimental_view_391 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_129, [1, 256, 1024]);  ttnn_linear_129 = None
    ttnn_add_211 = ttnn_decorators_ttnn_add(ttnn_experimental_view_391, ttnn_layer_norm_42);  ttnn_experimental_view_391 = ttnn_layer_norm_42 = None
    ttnn_from_torch_353 = ttnn_decorators_ttnn_from_torch(arg349_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg349_1 = None
    ttnn_from_torch_354 = ttnn_decorators_ttnn_from_torch(arg350_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg350_1 = None
    ttnn_layer_norm_43 = ttnn_decorators_ttnn_layer_norm(ttnn_add_211, epsilon = 1e-12, weight = ttnn_from_torch_353, bias = ttnn_from_torch_354);  ttnn_add_211 = ttnn_from_torch_353 = ttnn_from_torch_354 = None
    ttnn_experimental_view_392 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_43, [256, 1024])
    ttnn_from_torch_355 = ttnn_decorators_ttnn_from_torch(arg351_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg351_1 = None
    ttnn_from_torch_356 = ttnn_decorators_ttnn_from_torch(arg352_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg352_1 = None
    ttnn_linear_130 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_392, ttnn_from_torch_355, transpose_b = True, bias = ttnn_from_torch_356, activation = 'gelu');  ttnn_experimental_view_392 = ttnn_from_torch_355 = ttnn_from_torch_356 = None
    ttnn_experimental_view_394 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_130, [256, 4096]);  ttnn_linear_130 = None
    ttnn_from_torch_357 = ttnn_decorators_ttnn_from_torch(arg353_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg353_1 = None
    ttnn_from_torch_358 = ttnn_decorators_ttnn_from_torch(arg354_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg354_1 = None
    ttnn_linear_131 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_394, ttnn_from_torch_357, transpose_b = True, bias = ttnn_from_torch_358, activation = None);  ttnn_experimental_view_394 = ttnn_from_torch_357 = ttnn_from_torch_358 = None
    ttnn_experimental_view_395 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_131, [1, 256, 1024]);  ttnn_linear_131 = None
    ttnn_add_212 = ttnn_decorators_ttnn_add(ttnn_experimental_view_395, ttnn_layer_norm_43);  ttnn_experimental_view_395 = ttnn_layer_norm_43 = None
    ttnn_from_torch_359 = ttnn_decorators_ttnn_from_torch(arg355_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg355_1 = None
    ttnn_from_torch_360 = ttnn_decorators_ttnn_from_torch(arg356_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg356_1 = None
    ttnn_layer_norm_44 = ttnn_decorators_ttnn_layer_norm(ttnn_add_212, epsilon = 1e-12, weight = ttnn_from_torch_359, bias = ttnn_from_torch_360);  ttnn_add_212 = ttnn_from_torch_359 = ttnn_from_torch_360 = None
    ttnn_experimental_view_396 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_44, [256, 1024])
    ttnn_from_torch_361 = ttnn_decorators_ttnn_from_torch(arg357_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg357_1 = None
    ttnn_transpose_235 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_361, -2, -1);  ttnn_from_torch_361 = None
    ttnn_from_torch_362 = ttnn_decorators_ttnn_from_torch(arg359_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg359_1 = None
    ttnn_transpose_236 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_362, -2, -1);  ttnn_from_torch_362 = None
    ttnn_from_torch_363 = ttnn_decorators_ttnn_from_torch(arg361_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg361_1 = None
    ttnn_transpose_237 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_363, -2, -1);  ttnn_from_torch_363 = None
    ttnn_concat_44 = ttnn_decorators_ttnn_concat([ttnn_transpose_235, ttnn_transpose_236, ttnn_transpose_237], -1);  ttnn_transpose_235 = ttnn_transpose_236 = ttnn_transpose_237 = None
    ttnn_from_torch_364 = ttnn_decorators_ttnn_from_torch(arg358_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg358_1 = None
    ttnn_experimental_view_522 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_364, (1, -1));  ttnn_from_torch_364 = None
    ttnn_from_torch_365 = ttnn_decorators_ttnn_from_torch(arg360_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg360_1 = None
    ttnn_experimental_view_523 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_365, (1, -1));  ttnn_from_torch_365 = None
    ttnn_from_torch_366 = ttnn_decorators_ttnn_from_torch(arg362_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg362_1 = None
    ttnn_experimental_view_524 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_366, (1, -1));  ttnn_from_torch_366 = None
    ttnn_concat_45 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_522, ttnn_experimental_view_523, ttnn_experimental_view_524], -1);  ttnn_experimental_view_522 = ttnn_experimental_view_523 = ttnn_experimental_view_524 = None
    ttnn_to_layout_45 = ttnn_decorators_ttnn_to_layout(ttnn_concat_44, ttnn_TILE_LAYOUT);  ttnn_concat_44 = None
    ttnn_to_layout_46 = ttnn_decorators_ttnn_to_layout(ttnn_concat_45, ttnn_TILE_LAYOUT);  ttnn_concat_45 = None
    ttnn_linear_167 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_396, ttnn_to_layout_45, bias = ttnn_to_layout_46);  ttnn_experimental_view_396 = ttnn_to_layout_45 = ttnn_to_layout_46 = None
    ttnn_experimental_view_525 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_167, [1, 256, 3072]);  ttnn_linear_167 = None
    ttnn_transformer_split_query_key_value_and_split_heads_22 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_525, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_525 = None
    getitem_117 = ttnn_transformer_split_query_key_value_and_split_heads_22[0]
    getitem_118 = ttnn_transformer_split_query_key_value_and_split_heads_22[1]
    getitem_119 = ttnn_transformer_split_query_key_value_and_split_heads_22[2];  ttnn_transformer_split_query_key_value_and_split_heads_22 = None
    ttnn_matmul_237 = ttnn_decorators_ttnn_matmul(getitem_117, getitem_118);  getitem_117 = getitem_118 = None
    ttnn_transformer_attention_softmax__46 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_237, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_237 = None
    ttnn_matmul_238 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__46, getitem_119);  ttnn_transformer_attention_softmax__46 = getitem_119 = None
    ttnn_transformer_concatenate_heads_22 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_238);  ttnn_matmul_238 = None
    ttnn_from_torch_367 = ttnn_decorators_ttnn_from_torch(arg363_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg363_1 = None
    ttnn_from_torch_368 = ttnn_decorators_ttnn_from_torch(arg364_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg364_1 = None
    ttnn_linear_135 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_22, ttnn_from_torch_367, transpose_b = True, bias = ttnn_from_torch_368, activation = None);  ttnn_transformer_concatenate_heads_22 = ttnn_from_torch_367 = ttnn_from_torch_368 = None
    ttnn_experimental_view_409 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_135, [1, 256, 1024]);  ttnn_linear_135 = None
    ttnn_add_214 = ttnn_decorators_ttnn_add(ttnn_experimental_view_409, ttnn_layer_norm_44);  ttnn_experimental_view_409 = ttnn_layer_norm_44 = None
    ttnn_from_torch_369 = ttnn_decorators_ttnn_from_torch(arg365_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg365_1 = None
    ttnn_from_torch_370 = ttnn_decorators_ttnn_from_torch(arg366_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg366_1 = None
    ttnn_layer_norm_45 = ttnn_decorators_ttnn_layer_norm(ttnn_add_214, epsilon = 1e-12, weight = ttnn_from_torch_369, bias = ttnn_from_torch_370);  ttnn_add_214 = ttnn_from_torch_369 = ttnn_from_torch_370 = None
    ttnn_experimental_view_410 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_45, [256, 1024])
    ttnn_from_torch_371 = ttnn_decorators_ttnn_from_torch(arg367_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg367_1 = None
    ttnn_from_torch_372 = ttnn_decorators_ttnn_from_torch(arg368_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg368_1 = None
    ttnn_linear_136 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_410, ttnn_from_torch_371, transpose_b = True, bias = ttnn_from_torch_372, activation = 'gelu');  ttnn_experimental_view_410 = ttnn_from_torch_371 = ttnn_from_torch_372 = None
    ttnn_experimental_view_412 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_136, [256, 4096]);  ttnn_linear_136 = None
    ttnn_from_torch_373 = ttnn_decorators_ttnn_from_torch(arg369_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg369_1 = None
    ttnn_from_torch_374 = ttnn_decorators_ttnn_from_torch(arg370_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg370_1 = None
    ttnn_linear_137 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_412, ttnn_from_torch_373, transpose_b = True, bias = ttnn_from_torch_374, activation = None);  ttnn_experimental_view_412 = ttnn_from_torch_373 = ttnn_from_torch_374 = None
    ttnn_experimental_view_413 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_137, [1, 256, 1024]);  ttnn_linear_137 = None
    ttnn_add_215 = ttnn_decorators_ttnn_add(ttnn_experimental_view_413, ttnn_layer_norm_45);  ttnn_experimental_view_413 = ttnn_layer_norm_45 = None
    ttnn_from_torch_375 = ttnn_decorators_ttnn_from_torch(arg371_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg371_1 = None
    ttnn_from_torch_376 = ttnn_decorators_ttnn_from_torch(arg372_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg372_1 = None
    ttnn_layer_norm_46 = ttnn_decorators_ttnn_layer_norm(ttnn_add_215, epsilon = 1e-12, weight = ttnn_from_torch_375, bias = ttnn_from_torch_376);  ttnn_add_215 = ttnn_from_torch_375 = ttnn_from_torch_376 = None
    ttnn_experimental_view_414 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_46, [256, 1024])
    ttnn_from_torch_377 = ttnn_decorators_ttnn_from_torch(arg373_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg373_1 = None
    ttnn_transpose_238 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_377, -2, -1);  ttnn_from_torch_377 = None
    ttnn_from_torch_378 = ttnn_decorators_ttnn_from_torch(arg375_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg375_1 = None
    ttnn_transpose_239 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_378, -2, -1);  ttnn_from_torch_378 = None
    ttnn_from_torch_379 = ttnn_decorators_ttnn_from_torch(arg377_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg377_1 = None
    ttnn_transpose_240 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_379, -2, -1);  ttnn_from_torch_379 = None
    ttnn_concat_46 = ttnn_decorators_ttnn_concat([ttnn_transpose_238, ttnn_transpose_239, ttnn_transpose_240], -1);  ttnn_transpose_238 = ttnn_transpose_239 = ttnn_transpose_240 = None
    ttnn_from_torch_380 = ttnn_decorators_ttnn_from_torch(arg374_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg374_1 = None
    ttnn_experimental_view_526 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_380, (1, -1));  ttnn_from_torch_380 = None
    ttnn_from_torch_381 = ttnn_decorators_ttnn_from_torch(arg376_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg376_1 = None
    ttnn_experimental_view_527 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_381, (1, -1));  ttnn_from_torch_381 = None
    ttnn_from_torch_382 = ttnn_decorators_ttnn_from_torch(arg378_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg378_1 = None
    ttnn_experimental_view_528 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_382, (1, -1));  ttnn_from_torch_382 = None
    ttnn_concat_47 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_526, ttnn_experimental_view_527, ttnn_experimental_view_528], -1);  ttnn_experimental_view_526 = ttnn_experimental_view_527 = ttnn_experimental_view_528 = None
    ttnn_to_layout_47 = ttnn_decorators_ttnn_to_layout(ttnn_concat_46, ttnn_TILE_LAYOUT);  ttnn_concat_46 = None
    ttnn_to_layout_48 = ttnn_decorators_ttnn_to_layout(ttnn_concat_47, ttnn_TILE_LAYOUT);  ttnn_concat_47 = None
    ttnn_linear_168 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_414, ttnn_to_layout_47, bias = ttnn_to_layout_48);  ttnn_experimental_view_414 = ttnn_to_layout_47 = ttnn_to_layout_48 = None
    ttnn_experimental_view_529 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_168, [1, 256, 3072]);  ttnn_linear_168 = None
    ttnn_transformer_split_query_key_value_and_split_heads_23 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_529, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_529 = None
    getitem_120 = ttnn_transformer_split_query_key_value_and_split_heads_23[0]
    getitem_121 = ttnn_transformer_split_query_key_value_and_split_heads_23[1]
    getitem_122 = ttnn_transformer_split_query_key_value_and_split_heads_23[2];  ttnn_transformer_split_query_key_value_and_split_heads_23 = None
    ttnn_matmul_239 = ttnn_decorators_ttnn_matmul(getitem_120, getitem_121);  getitem_120 = getitem_121 = None
    ttnn_transformer_attention_softmax__47 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_239, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_239 = ttnn_multiply = None
    ttnn_matmul_240 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__47, getitem_122);  ttnn_transformer_attention_softmax__47 = getitem_122 = None
    ttnn_transformer_concatenate_heads_23 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_240);  ttnn_matmul_240 = None
    ttnn_from_torch_383 = ttnn_decorators_ttnn_from_torch(arg379_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg379_1 = None
    ttnn_from_torch_384 = ttnn_decorators_ttnn_from_torch(arg380_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg380_1 = None
    ttnn_linear_141 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_23, ttnn_from_torch_383, transpose_b = True, bias = ttnn_from_torch_384, activation = None);  ttnn_transformer_concatenate_heads_23 = ttnn_from_torch_383 = ttnn_from_torch_384 = None
    ttnn_experimental_view_427 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_141, [1, 256, 1024]);  ttnn_linear_141 = None
    ttnn_add_217 = ttnn_decorators_ttnn_add(ttnn_experimental_view_427, ttnn_layer_norm_46);  ttnn_experimental_view_427 = ttnn_layer_norm_46 = None
    ttnn_from_torch_385 = ttnn_decorators_ttnn_from_torch(arg381_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg381_1 = None
    ttnn_from_torch_386 = ttnn_decorators_ttnn_from_torch(arg382_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg382_1 = None
    ttnn_layer_norm_47 = ttnn_decorators_ttnn_layer_norm(ttnn_add_217, epsilon = 1e-12, weight = ttnn_from_torch_385, bias = ttnn_from_torch_386);  ttnn_add_217 = ttnn_from_torch_385 = ttnn_from_torch_386 = None
    ttnn_experimental_view_428 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_47, [256, 1024])
    ttnn_from_torch_387 = ttnn_decorators_ttnn_from_torch(arg383_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg383_1 = None
    ttnn_from_torch_388 = ttnn_decorators_ttnn_from_torch(arg384_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg384_1 = None
    ttnn_linear_142 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_428, ttnn_from_torch_387, transpose_b = True, bias = ttnn_from_torch_388, activation = 'gelu');  ttnn_experimental_view_428 = ttnn_from_torch_387 = ttnn_from_torch_388 = None
    ttnn_experimental_view_430 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_142, [256, 4096]);  ttnn_linear_142 = None
    ttnn_from_torch_389 = ttnn_decorators_ttnn_from_torch(arg385_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg385_1 = None
    ttnn_from_torch_390 = ttnn_decorators_ttnn_from_torch(arg386_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg386_1 = None
    ttnn_linear_143 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_430, ttnn_from_torch_389, transpose_b = True, bias = ttnn_from_torch_390, activation = None);  ttnn_experimental_view_430 = ttnn_from_torch_389 = ttnn_from_torch_390 = None
    ttnn_experimental_view_431 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_143, [1, 256, 1024]);  ttnn_linear_143 = None
    ttnn_add_218 = ttnn_decorators_ttnn_add(ttnn_experimental_view_431, ttnn_layer_norm_47);  ttnn_experimental_view_431 = ttnn_layer_norm_47 = None
    ttnn_from_torch_391 = ttnn_decorators_ttnn_from_torch(arg387_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg387_1 = None
    ttnn_from_torch_392 = ttnn_decorators_ttnn_from_torch(arg388_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg388_1 = None
    ttnn_layer_norm_48 = ttnn_decorators_ttnn_layer_norm(ttnn_add_218, epsilon = 1e-12, weight = ttnn_from_torch_391, bias = ttnn_from_torch_392);  ttnn_add_218 = ttnn_from_torch_391 = ttnn_from_torch_392 = None
    ttnn_experimental_view_432 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_48, [256, 1024]);  ttnn_layer_norm_48 = None
    ttnn_from_torch_393 = ttnn_decorators_ttnn_from_torch(arg389_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg389_1 = None
    ttnn_from_torch_394 = ttnn_decorators_ttnn_from_torch(arg390_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg390_1 = None
    ttnn_linear_144 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_432, ttnn_from_torch_393, transpose_b = True, bias = ttnn_from_torch_394, activation = None);  ttnn_experimental_view_432 = ttnn_from_torch_393 = ttnn_from_torch_394 = None
    ttnn_experimental_view_433 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_144, [1, 256, 2]);  ttnn_linear_144 = None
    ttnn_to_layout_49 = ttnn_decorators_ttnn_to_layout(ttnn_experimental_view_433, ttnn_ROW_MAJOR_LAYOUT);  ttnn_experimental_view_433 = None
    ttnn_split = ttnn_decorators_ttnn_split(ttnn_to_layout_49, 1, 2);  ttnn_to_layout_49 = None
    getitem_49 = ttnn_split[0]
    getitem_50 = ttnn_split[1];  ttnn_split = None
    ttnn_to_layout_50 = ttnn_decorators_ttnn_to_layout(getitem_49, ttnn_TILE_LAYOUT);  getitem_49 = None
    ttnn_squeeze = ttnn_decorators_ttnn_squeeze(ttnn_to_layout_50, -1);  ttnn_to_layout_50 = None
    ttnn_to_layout_51 = ttnn_decorators_ttnn_to_layout(getitem_50, ttnn_TILE_LAYOUT);  getitem_50 = None
    ttnn_squeeze_1 = ttnn_decorators_ttnn_squeeze(ttnn_to_layout_51, -1);  ttnn_to_layout_51 = None
    ttnn_to_torch = ttnn_decorators_ttnn_to_torch(ttnn_squeeze, dtype = torch.bfloat16);  ttnn_squeeze = None
    ttnn_to_torch_1 = ttnn_decorators_ttnn_to_torch(ttnn_squeeze_1, dtype = torch.bfloat16);  ttnn_squeeze_1 = None
    return (ttnn_to_torch, ttnn_to_torch_1)



# model after https://github.com/tenstorrent/pytorch2.0_ttnn/pull/991
def after_linear_transformation(ttnn_Specified_Device, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1):
    ttnn_from_torch = ttnn_decorators_ttnn_from_torch(arg393_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_uint32);  arg393_1 = None
    ttnn_reshape = ttnn_decorators_ttnn_reshape(ttnn_from_torch, [1, 1, 256]);  ttnn_from_torch = None
    ttnn_reshape_1 = ttnn_decorators_ttnn_reshape(ttnn_reshape, [1, 1, 1, 256]);  ttnn_reshape = None
    ttnn_typecast = ttnn_decorators_ttnn_typecast(ttnn_reshape_1, ttnn_bfloat16);  ttnn_reshape_1 = None
    ttnn_rsub = ttnn_decorators_ttnn_rsub(ttnn_typecast, 1.0);  ttnn_typecast = None
    ttnn_multiply = ttnn_decorators_ttnn_multiply(ttnn_rsub, -3.3895313892515355e+38);  ttnn_rsub = None
    ttnn_from_torch_1 = ttnn_decorators_ttnn_from_torch(arg391_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg391_1 = None
    ttnn_slice = ttnn_decorators_ttnn_slice(ttnn_from_torch_1, [0, 0], [1, 256]);  ttnn_from_torch_1 = None
    ttnn_from_torch_2 = ttnn_decorators_ttnn_from_torch(arg392_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg392_1 = None
    ttnn_from_torch_3 = ttnn_decorators_ttnn_from_torch(arg0_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg0_1 = None
    ttnn_embedding = ttnn_decorators_ttnn_embedding(ttnn_from_torch_2, ttnn_from_torch_3, layout = ttnn_TILE_LAYOUT);  ttnn_from_torch_2 = ttnn_from_torch_3 = None
    ttnn_from_torch_4 = ttnn_decorators_ttnn_from_torch(arg394_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg394_1 = None
    ttnn_from_torch_5 = ttnn_decorators_ttnn_from_torch(arg1_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg1_1 = None
    ttnn_embedding_1 = ttnn_decorators_ttnn_embedding(ttnn_from_torch_4, ttnn_from_torch_5, layout = ttnn_TILE_LAYOUT);  ttnn_from_torch_4 = ttnn_from_torch_5 = None
    ttnn_add_145 = ttnn_decorators_ttnn_add(ttnn_embedding, ttnn_embedding_1);  ttnn_embedding = ttnn_embedding_1 = None
    ttnn_to_layout = ttnn_decorators_ttnn_to_layout(ttnn_slice, ttnn_ROW_MAJOR_LAYOUT);  ttnn_slice = None
    ttnn_from_torch_6 = ttnn_decorators_ttnn_from_torch(arg2_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg2_1 = None
    ttnn_embedding_2 = ttnn_decorators_ttnn_embedding(ttnn_to_layout, ttnn_from_torch_6, layout = ttnn_TILE_LAYOUT);  ttnn_to_layout = ttnn_from_torch_6 = None
    ttnn_add_146 = ttnn_decorators_ttnn_add(ttnn_add_145, ttnn_embedding_2);  ttnn_add_145 = ttnn_embedding_2 = None
    ttnn_from_torch_7 = ttnn_decorators_ttnn_from_torch(arg3_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg3_1 = None
    ttnn_from_torch_8 = ttnn_decorators_ttnn_from_torch(arg4_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg4_1 = None
    ttnn_layer_norm = ttnn_decorators_ttnn_layer_norm(ttnn_add_146, epsilon = 1e-12, weight = ttnn_from_torch_7, bias = ttnn_from_torch_8);  ttnn_add_146 = ttnn_from_torch_7 = ttnn_from_torch_8 = None
    ttnn_experimental_view = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm, [256, 1024])
    ttnn_from_torch_9 = ttnn_decorators_ttnn_from_torch(arg5_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg5_1 = None
    ttnn_transpose_169 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_9, -2, -1);  ttnn_from_torch_9 = None
    ttnn_from_torch_10 = ttnn_decorators_ttnn_from_torch(arg7_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg7_1 = None
    ttnn_transpose_170 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_10, -2, -1);  ttnn_from_torch_10 = None
    ttnn_from_torch_11 = ttnn_decorators_ttnn_from_torch(arg9_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg9_1 = None
    ttnn_transpose_171 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_11, -2, -1);  ttnn_from_torch_11 = None
    ttnn_concat = ttnn_decorators_ttnn_concat([ttnn_transpose_169, ttnn_transpose_170, ttnn_transpose_171], -1);  ttnn_transpose_169 = ttnn_transpose_170 = ttnn_transpose_171 = None
    ttnn_from_torch_12 = ttnn_decorators_ttnn_from_torch(arg6_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg6_1 = None
    ttnn_experimental_view_434 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_12, (1, -1));  ttnn_from_torch_12 = None
    ttnn_from_torch_13 = ttnn_decorators_ttnn_from_torch(arg8_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg8_1 = None
    ttnn_experimental_view_435 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_13, (1, -1));  ttnn_from_torch_13 = None
    ttnn_from_torch_14 = ttnn_decorators_ttnn_from_torch(arg10_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg10_1 = None
    ttnn_experimental_view_436 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_14, (1, -1));  ttnn_from_torch_14 = None
    ttnn_concat_1 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_434, ttnn_experimental_view_435, ttnn_experimental_view_436], -1);  ttnn_experimental_view_434 = ttnn_experimental_view_435 = ttnn_experimental_view_436 = None
    ttnn_to_layout_1 = ttnn_decorators_ttnn_to_layout(ttnn_concat, ttnn_TILE_LAYOUT);  ttnn_concat = None
    ttnn_to_layout_2 = ttnn_decorators_ttnn_to_layout(ttnn_concat_1, ttnn_TILE_LAYOUT);  ttnn_concat_1 = None
    ttnn_linear_145 = ttnn_decorators_ttnn_linear(ttnn_experimental_view, ttnn_to_layout_1, bias = ttnn_to_layout_2);  ttnn_experimental_view = ttnn_to_layout_1 = ttnn_to_layout_2 = None
    ttnn_experimental_view_437 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_145, [1, 256, 3072]);  ttnn_linear_145 = None
    ttnn_transformer_split_query_key_value_and_split_heads = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_437, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_437 = None
    getitem_51 = ttnn_transformer_split_query_key_value_and_split_heads[0]
    getitem_52 = ttnn_transformer_split_query_key_value_and_split_heads[1]
    getitem_53 = ttnn_transformer_split_query_key_value_and_split_heads[2];  ttnn_transformer_split_query_key_value_and_split_heads = None
    ttnn_matmul_193 = ttnn_decorators_ttnn_matmul(getitem_51, getitem_52);  getitem_51 = getitem_52 = None
    ttnn_transformer_attention_softmax__24 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_193, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_193 = None
    ttnn_matmul_194 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__24, getitem_53);  ttnn_transformer_attention_softmax__24 = getitem_53 = None
    ttnn_transformer_concatenate_heads = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_194);  ttnn_matmul_194 = None
    ttnn_from_torch_15 = ttnn_decorators_ttnn_from_torch(arg11_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg11_1 = None
    ttnn_from_torch_16 = ttnn_decorators_ttnn_from_torch(arg12_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg12_1 = None
    ttnn_linear_3 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads, ttnn_from_torch_15, transpose_b = True, bias = ttnn_from_torch_16, activation = None);  ttnn_transformer_concatenate_heads = ttnn_from_torch_15 = ttnn_from_torch_16 = None
    ttnn_experimental_view_13 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_3, [1, 256, 1024]);  ttnn_linear_3 = None
    ttnn_add_148 = ttnn_decorators_ttnn_add(ttnn_experimental_view_13, ttnn_layer_norm);  ttnn_experimental_view_13 = ttnn_layer_norm = None
    ttnn_from_torch_17 = ttnn_decorators_ttnn_from_torch(arg13_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg13_1 = None
    ttnn_from_torch_18 = ttnn_decorators_ttnn_from_torch(arg14_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg14_1 = None
    ttnn_layer_norm_1 = ttnn_decorators_ttnn_layer_norm(ttnn_add_148, epsilon = 1e-12, weight = ttnn_from_torch_17, bias = ttnn_from_torch_18);  ttnn_add_148 = ttnn_from_torch_17 = ttnn_from_torch_18 = None
    ttnn_experimental_view_14 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_1, [256, 1024])
    ttnn_from_torch_19 = ttnn_decorators_ttnn_from_torch(arg15_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg15_1 = None
    ttnn_from_torch_20 = ttnn_decorators_ttnn_from_torch(arg16_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg16_1 = None
    ttnn_linear_4 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_14, ttnn_from_torch_19, transpose_b = True, bias = ttnn_from_torch_20, activation = 'gelu');  ttnn_experimental_view_14 = ttnn_from_torch_19 = ttnn_from_torch_20 = None
    ttnn_experimental_view_16 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_4, [256, 4096]);  ttnn_linear_4 = None
    ttnn_from_torch_21 = ttnn_decorators_ttnn_from_torch(arg17_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg17_1 = None
    ttnn_from_torch_22 = ttnn_decorators_ttnn_from_torch(arg18_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg18_1 = None
    ttnn_linear_5 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_16, ttnn_from_torch_21, transpose_b = True, bias = ttnn_from_torch_22, activation = None);  ttnn_experimental_view_16 = ttnn_from_torch_21 = ttnn_from_torch_22 = None
    ttnn_experimental_view_17 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_5, [1, 256, 1024]);  ttnn_linear_5 = None
    ttnn_add_149 = ttnn_decorators_ttnn_add(ttnn_experimental_view_17, ttnn_layer_norm_1);  ttnn_experimental_view_17 = ttnn_layer_norm_1 = None
    ttnn_from_torch_23 = ttnn_decorators_ttnn_from_torch(arg19_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg19_1 = None
    ttnn_from_torch_24 = ttnn_decorators_ttnn_from_torch(arg20_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg20_1 = None
    ttnn_layer_norm_2 = ttnn_decorators_ttnn_layer_norm(ttnn_add_149, epsilon = 1e-12, weight = ttnn_from_torch_23, bias = ttnn_from_torch_24);  ttnn_add_149 = ttnn_from_torch_23 = ttnn_from_torch_24 = None
    ttnn_experimental_view_18 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_2, [256, 1024])
    ttnn_from_torch_25 = ttnn_decorators_ttnn_from_torch(arg21_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg21_1 = None
    ttnn_transpose_172 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_25, -2, -1);  ttnn_from_torch_25 = None
    ttnn_from_torch_26 = ttnn_decorators_ttnn_from_torch(arg23_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg23_1 = None
    ttnn_transpose_173 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_26, -2, -1);  ttnn_from_torch_26 = None
    ttnn_from_torch_27 = ttnn_decorators_ttnn_from_torch(arg25_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg25_1 = None
    ttnn_transpose_174 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_27, -2, -1);  ttnn_from_torch_27 = None
    ttnn_concat_2 = ttnn_decorators_ttnn_concat([ttnn_transpose_172, ttnn_transpose_173, ttnn_transpose_174], -1);  ttnn_transpose_172 = ttnn_transpose_173 = ttnn_transpose_174 = None
    ttnn_from_torch_28 = ttnn_decorators_ttnn_from_torch(arg22_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg22_1 = None
    ttnn_experimental_view_438 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_28, (1, -1));  ttnn_from_torch_28 = None
    ttnn_from_torch_29 = ttnn_decorators_ttnn_from_torch(arg24_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg24_1 = None
    ttnn_experimental_view_439 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_29, (1, -1));  ttnn_from_torch_29 = None
    ttnn_from_torch_30 = ttnn_decorators_ttnn_from_torch(arg26_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg26_1 = None
    ttnn_experimental_view_440 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_30, (1, -1));  ttnn_from_torch_30 = None
    ttnn_concat_3 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_438, ttnn_experimental_view_439, ttnn_experimental_view_440], -1);  ttnn_experimental_view_438 = ttnn_experimental_view_439 = ttnn_experimental_view_440 = None
    ttnn_to_layout_3 = ttnn_decorators_ttnn_to_layout(ttnn_concat_2, ttnn_TILE_LAYOUT);  ttnn_concat_2 = None
    ttnn_to_layout_4 = ttnn_decorators_ttnn_to_layout(ttnn_concat_3, ttnn_TILE_LAYOUT);  ttnn_concat_3 = None
    ttnn_linear_146 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_18, ttnn_to_layout_3, bias = ttnn_to_layout_4);  ttnn_experimental_view_18 = ttnn_to_layout_3 = ttnn_to_layout_4 = None
    ttnn_experimental_view_441 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_146, [1, 256, 3072]);  ttnn_linear_146 = None
    ttnn_transformer_split_query_key_value_and_split_heads_1 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_441, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_441 = None
    getitem_54 = ttnn_transformer_split_query_key_value_and_split_heads_1[0]
    getitem_55 = ttnn_transformer_split_query_key_value_and_split_heads_1[1]
    getitem_56 = ttnn_transformer_split_query_key_value_and_split_heads_1[2];  ttnn_transformer_split_query_key_value_and_split_heads_1 = None
    ttnn_matmul_195 = ttnn_decorators_ttnn_matmul(getitem_54, getitem_55);  getitem_54 = getitem_55 = None
    ttnn_transformer_attention_softmax__25 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_195, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_195 = None
    ttnn_matmul_196 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__25, getitem_56);  ttnn_transformer_attention_softmax__25 = getitem_56 = None
    ttnn_transformer_concatenate_heads_1 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_196);  ttnn_matmul_196 = None
    ttnn_from_torch_31 = ttnn_decorators_ttnn_from_torch(arg27_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg27_1 = None
    ttnn_from_torch_32 = ttnn_decorators_ttnn_from_torch(arg28_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg28_1 = None
    ttnn_linear_9 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_1, ttnn_from_torch_31, transpose_b = True, bias = ttnn_from_torch_32, activation = None);  ttnn_transformer_concatenate_heads_1 = ttnn_from_torch_31 = ttnn_from_torch_32 = None
    ttnn_experimental_view_31 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_9, [1, 256, 1024]);  ttnn_linear_9 = None
    ttnn_add_151 = ttnn_decorators_ttnn_add(ttnn_experimental_view_31, ttnn_layer_norm_2);  ttnn_experimental_view_31 = ttnn_layer_norm_2 = None
    ttnn_from_torch_33 = ttnn_decorators_ttnn_from_torch(arg29_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg29_1 = None
    ttnn_from_torch_34 = ttnn_decorators_ttnn_from_torch(arg30_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg30_1 = None
    ttnn_layer_norm_3 = ttnn_decorators_ttnn_layer_norm(ttnn_add_151, epsilon = 1e-12, weight = ttnn_from_torch_33, bias = ttnn_from_torch_34);  ttnn_add_151 = ttnn_from_torch_33 = ttnn_from_torch_34 = None
    ttnn_experimental_view_32 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_3, [256, 1024])
    ttnn_from_torch_35 = ttnn_decorators_ttnn_from_torch(arg31_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg31_1 = None
    ttnn_from_torch_36 = ttnn_decorators_ttnn_from_torch(arg32_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg32_1 = None
    ttnn_linear_10 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_32, ttnn_from_torch_35, transpose_b = True, bias = ttnn_from_torch_36, activation = 'gelu');  ttnn_experimental_view_32 = ttnn_from_torch_35 = ttnn_from_torch_36 = None
    ttnn_experimental_view_34 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_10, [256, 4096]);  ttnn_linear_10 = None
    ttnn_from_torch_37 = ttnn_decorators_ttnn_from_torch(arg33_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg33_1 = None
    ttnn_from_torch_38 = ttnn_decorators_ttnn_from_torch(arg34_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg34_1 = None
    ttnn_linear_11 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_34, ttnn_from_torch_37, transpose_b = True, bias = ttnn_from_torch_38, activation = None);  ttnn_experimental_view_34 = ttnn_from_torch_37 = ttnn_from_torch_38 = None
    ttnn_experimental_view_35 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_11, [1, 256, 1024]);  ttnn_linear_11 = None
    ttnn_add_152 = ttnn_decorators_ttnn_add(ttnn_experimental_view_35, ttnn_layer_norm_3);  ttnn_experimental_view_35 = ttnn_layer_norm_3 = None
    ttnn_from_torch_39 = ttnn_decorators_ttnn_from_torch(arg35_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg35_1 = None
    ttnn_from_torch_40 = ttnn_decorators_ttnn_from_torch(arg36_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg36_1 = None
    ttnn_layer_norm_4 = ttnn_decorators_ttnn_layer_norm(ttnn_add_152, epsilon = 1e-12, weight = ttnn_from_torch_39, bias = ttnn_from_torch_40);  ttnn_add_152 = ttnn_from_torch_39 = ttnn_from_torch_40 = None
    ttnn_experimental_view_36 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_4, [256, 1024])
    ttnn_from_torch_41 = ttnn_decorators_ttnn_from_torch(arg37_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg37_1 = None
    ttnn_transpose_175 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_41, -2, -1);  ttnn_from_torch_41 = None
    ttnn_from_torch_42 = ttnn_decorators_ttnn_from_torch(arg39_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg39_1 = None
    ttnn_transpose_176 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_42, -2, -1);  ttnn_from_torch_42 = None
    ttnn_from_torch_43 = ttnn_decorators_ttnn_from_torch(arg41_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg41_1 = None
    ttnn_transpose_177 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_43, -2, -1);  ttnn_from_torch_43 = None
    ttnn_concat_4 = ttnn_decorators_ttnn_concat([ttnn_transpose_175, ttnn_transpose_176, ttnn_transpose_177], -1);  ttnn_transpose_175 = ttnn_transpose_176 = ttnn_transpose_177 = None
    ttnn_from_torch_44 = ttnn_decorators_ttnn_from_torch(arg38_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg38_1 = None
    ttnn_experimental_view_442 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_44, (1, -1));  ttnn_from_torch_44 = None
    ttnn_from_torch_45 = ttnn_decorators_ttnn_from_torch(arg40_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg40_1 = None
    ttnn_experimental_view_443 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_45, (1, -1));  ttnn_from_torch_45 = None
    ttnn_from_torch_46 = ttnn_decorators_ttnn_from_torch(arg42_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg42_1 = None
    ttnn_experimental_view_444 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_46, (1, -1));  ttnn_from_torch_46 = None
    ttnn_concat_5 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_442, ttnn_experimental_view_443, ttnn_experimental_view_444], -1);  ttnn_experimental_view_442 = ttnn_experimental_view_443 = ttnn_experimental_view_444 = None
    ttnn_to_layout_5 = ttnn_decorators_ttnn_to_layout(ttnn_concat_4, ttnn_TILE_LAYOUT);  ttnn_concat_4 = None
    ttnn_to_layout_6 = ttnn_decorators_ttnn_to_layout(ttnn_concat_5, ttnn_TILE_LAYOUT);  ttnn_concat_5 = None
    ttnn_linear_147 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_36, ttnn_to_layout_5, bias = ttnn_to_layout_6);  ttnn_experimental_view_36 = ttnn_to_layout_5 = ttnn_to_layout_6 = None
    ttnn_experimental_view_445 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_147, [1, 256, 3072]);  ttnn_linear_147 = None
    ttnn_transformer_split_query_key_value_and_split_heads_2 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_445, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_445 = None
    getitem_57 = ttnn_transformer_split_query_key_value_and_split_heads_2[0]
    getitem_58 = ttnn_transformer_split_query_key_value_and_split_heads_2[1]
    getitem_59 = ttnn_transformer_split_query_key_value_and_split_heads_2[2];  ttnn_transformer_split_query_key_value_and_split_heads_2 = None
    ttnn_matmul_197 = ttnn_decorators_ttnn_matmul(getitem_57, getitem_58);  getitem_57 = getitem_58 = None
    ttnn_transformer_attention_softmax__26 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_197, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_197 = None
    ttnn_matmul_198 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__26, getitem_59);  ttnn_transformer_attention_softmax__26 = getitem_59 = None
    ttnn_transformer_concatenate_heads_2 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_198);  ttnn_matmul_198 = None
    ttnn_from_torch_47 = ttnn_decorators_ttnn_from_torch(arg43_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg43_1 = None
    ttnn_from_torch_48 = ttnn_decorators_ttnn_from_torch(arg44_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg44_1 = None
    ttnn_linear_15 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_2, ttnn_from_torch_47, transpose_b = True, bias = ttnn_from_torch_48, activation = None);  ttnn_transformer_concatenate_heads_2 = ttnn_from_torch_47 = ttnn_from_torch_48 = None
    ttnn_experimental_view_49 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_15, [1, 256, 1024]);  ttnn_linear_15 = None
    ttnn_add_154 = ttnn_decorators_ttnn_add(ttnn_experimental_view_49, ttnn_layer_norm_4);  ttnn_experimental_view_49 = ttnn_layer_norm_4 = None
    ttnn_from_torch_49 = ttnn_decorators_ttnn_from_torch(arg45_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg45_1 = None
    ttnn_from_torch_50 = ttnn_decorators_ttnn_from_torch(arg46_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg46_1 = None
    ttnn_layer_norm_5 = ttnn_decorators_ttnn_layer_norm(ttnn_add_154, epsilon = 1e-12, weight = ttnn_from_torch_49, bias = ttnn_from_torch_50);  ttnn_add_154 = ttnn_from_torch_49 = ttnn_from_torch_50 = None
    ttnn_experimental_view_50 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_5, [256, 1024])
    ttnn_from_torch_51 = ttnn_decorators_ttnn_from_torch(arg47_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg47_1 = None
    ttnn_from_torch_52 = ttnn_decorators_ttnn_from_torch(arg48_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg48_1 = None
    ttnn_linear_16 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_50, ttnn_from_torch_51, transpose_b = True, bias = ttnn_from_torch_52, activation = 'gelu');  ttnn_experimental_view_50 = ttnn_from_torch_51 = ttnn_from_torch_52 = None
    ttnn_experimental_view_52 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_16, [256, 4096]);  ttnn_linear_16 = None
    ttnn_from_torch_53 = ttnn_decorators_ttnn_from_torch(arg49_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg49_1 = None
    ttnn_from_torch_54 = ttnn_decorators_ttnn_from_torch(arg50_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg50_1 = None
    ttnn_linear_17 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_52, ttnn_from_torch_53, transpose_b = True, bias = ttnn_from_torch_54, activation = None);  ttnn_experimental_view_52 = ttnn_from_torch_53 = ttnn_from_torch_54 = None
    ttnn_experimental_view_53 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_17, [1, 256, 1024]);  ttnn_linear_17 = None
    ttnn_add_155 = ttnn_decorators_ttnn_add(ttnn_experimental_view_53, ttnn_layer_norm_5);  ttnn_experimental_view_53 = ttnn_layer_norm_5 = None
    ttnn_from_torch_55 = ttnn_decorators_ttnn_from_torch(arg51_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg51_1 = None
    ttnn_from_torch_56 = ttnn_decorators_ttnn_from_torch(arg52_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg52_1 = None
    ttnn_layer_norm_6 = ttnn_decorators_ttnn_layer_norm(ttnn_add_155, epsilon = 1e-12, weight = ttnn_from_torch_55, bias = ttnn_from_torch_56);  ttnn_add_155 = ttnn_from_torch_55 = ttnn_from_torch_56 = None
    ttnn_experimental_view_54 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_6, [256, 1024])
    ttnn_from_torch_57 = ttnn_decorators_ttnn_from_torch(arg53_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg53_1 = None
    ttnn_transpose_178 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_57, -2, -1);  ttnn_from_torch_57 = None
    ttnn_from_torch_58 = ttnn_decorators_ttnn_from_torch(arg55_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg55_1 = None
    ttnn_transpose_179 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_58, -2, -1);  ttnn_from_torch_58 = None
    ttnn_from_torch_59 = ttnn_decorators_ttnn_from_torch(arg57_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg57_1 = None
    ttnn_transpose_180 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_59, -2, -1);  ttnn_from_torch_59 = None
    ttnn_concat_6 = ttnn_decorators_ttnn_concat([ttnn_transpose_178, ttnn_transpose_179, ttnn_transpose_180], -1);  ttnn_transpose_178 = ttnn_transpose_179 = ttnn_transpose_180 = None
    ttnn_from_torch_60 = ttnn_decorators_ttnn_from_torch(arg54_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg54_1 = None
    ttnn_experimental_view_446 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_60, (1, -1));  ttnn_from_torch_60 = None
    ttnn_from_torch_61 = ttnn_decorators_ttnn_from_torch(arg56_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg56_1 = None
    ttnn_experimental_view_447 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_61, (1, -1));  ttnn_from_torch_61 = None
    ttnn_from_torch_62 = ttnn_decorators_ttnn_from_torch(arg58_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg58_1 = None
    ttnn_experimental_view_448 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_62, (1, -1));  ttnn_from_torch_62 = None
    ttnn_concat_7 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_446, ttnn_experimental_view_447, ttnn_experimental_view_448], -1);  ttnn_experimental_view_446 = ttnn_experimental_view_447 = ttnn_experimental_view_448 = None
    ttnn_to_layout_7 = ttnn_decorators_ttnn_to_layout(ttnn_concat_6, ttnn_TILE_LAYOUT);  ttnn_concat_6 = None
    ttnn_to_layout_8 = ttnn_decorators_ttnn_to_layout(ttnn_concat_7, ttnn_TILE_LAYOUT);  ttnn_concat_7 = None
    ttnn_linear_148 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_54, ttnn_to_layout_7, bias = ttnn_to_layout_8);  ttnn_experimental_view_54 = ttnn_to_layout_7 = ttnn_to_layout_8 = None
    ttnn_experimental_view_449 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_148, [1, 256, 3072]);  ttnn_linear_148 = None
    ttnn_transformer_split_query_key_value_and_split_heads_3 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_449, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_449 = None
    getitem_60 = ttnn_transformer_split_query_key_value_and_split_heads_3[0]
    getitem_61 = ttnn_transformer_split_query_key_value_and_split_heads_3[1]
    getitem_62 = ttnn_transformer_split_query_key_value_and_split_heads_3[2];  ttnn_transformer_split_query_key_value_and_split_heads_3 = None
    ttnn_matmul_199 = ttnn_decorators_ttnn_matmul(getitem_60, getitem_61);  getitem_60 = getitem_61 = None
    ttnn_transformer_attention_softmax__27 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_199, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_199 = None
    ttnn_matmul_200 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__27, getitem_62);  ttnn_transformer_attention_softmax__27 = getitem_62 = None
    ttnn_transformer_concatenate_heads_3 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_200);  ttnn_matmul_200 = None
    ttnn_from_torch_63 = ttnn_decorators_ttnn_from_torch(arg59_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg59_1 = None
    ttnn_from_torch_64 = ttnn_decorators_ttnn_from_torch(arg60_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg60_1 = None
    ttnn_linear_21 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_3, ttnn_from_torch_63, transpose_b = True, bias = ttnn_from_torch_64, activation = None);  ttnn_transformer_concatenate_heads_3 = ttnn_from_torch_63 = ttnn_from_torch_64 = None
    ttnn_experimental_view_67 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_21, [1, 256, 1024]);  ttnn_linear_21 = None
    ttnn_add_157 = ttnn_decorators_ttnn_add(ttnn_experimental_view_67, ttnn_layer_norm_6);  ttnn_experimental_view_67 = ttnn_layer_norm_6 = None
    ttnn_from_torch_65 = ttnn_decorators_ttnn_from_torch(arg61_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg61_1 = None
    ttnn_from_torch_66 = ttnn_decorators_ttnn_from_torch(arg62_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg62_1 = None
    ttnn_layer_norm_7 = ttnn_decorators_ttnn_layer_norm(ttnn_add_157, epsilon = 1e-12, weight = ttnn_from_torch_65, bias = ttnn_from_torch_66);  ttnn_add_157 = ttnn_from_torch_65 = ttnn_from_torch_66 = None
    ttnn_experimental_view_68 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_7, [256, 1024])
    ttnn_from_torch_67 = ttnn_decorators_ttnn_from_torch(arg63_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg63_1 = None
    ttnn_from_torch_68 = ttnn_decorators_ttnn_from_torch(arg64_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg64_1 = None
    ttnn_linear_22 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_68, ttnn_from_torch_67, transpose_b = True, bias = ttnn_from_torch_68, activation = 'gelu');  ttnn_experimental_view_68 = ttnn_from_torch_67 = ttnn_from_torch_68 = None
    ttnn_experimental_view_70 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_22, [256, 4096]);  ttnn_linear_22 = None
    ttnn_from_torch_69 = ttnn_decorators_ttnn_from_torch(arg65_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg65_1 = None
    ttnn_from_torch_70 = ttnn_decorators_ttnn_from_torch(arg66_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg66_1 = None
    ttnn_linear_23 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_70, ttnn_from_torch_69, transpose_b = True, bias = ttnn_from_torch_70, activation = None);  ttnn_experimental_view_70 = ttnn_from_torch_69 = ttnn_from_torch_70 = None
    ttnn_experimental_view_71 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_23, [1, 256, 1024]);  ttnn_linear_23 = None
    ttnn_add_158 = ttnn_decorators_ttnn_add(ttnn_experimental_view_71, ttnn_layer_norm_7);  ttnn_experimental_view_71 = ttnn_layer_norm_7 = None
    ttnn_from_torch_71 = ttnn_decorators_ttnn_from_torch(arg67_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg67_1 = None
    ttnn_from_torch_72 = ttnn_decorators_ttnn_from_torch(arg68_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg68_1 = None
    ttnn_layer_norm_8 = ttnn_decorators_ttnn_layer_norm(ttnn_add_158, epsilon = 1e-12, weight = ttnn_from_torch_71, bias = ttnn_from_torch_72);  ttnn_add_158 = ttnn_from_torch_71 = ttnn_from_torch_72 = None
    ttnn_experimental_view_72 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_8, [256, 1024])
    ttnn_from_torch_73 = ttnn_decorators_ttnn_from_torch(arg69_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg69_1 = None
    ttnn_transpose_181 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_73, -2, -1);  ttnn_from_torch_73 = None
    ttnn_from_torch_74 = ttnn_decorators_ttnn_from_torch(arg71_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg71_1 = None
    ttnn_transpose_182 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_74, -2, -1);  ttnn_from_torch_74 = None
    ttnn_from_torch_75 = ttnn_decorators_ttnn_from_torch(arg73_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg73_1 = None
    ttnn_transpose_183 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_75, -2, -1);  ttnn_from_torch_75 = None
    ttnn_concat_8 = ttnn_decorators_ttnn_concat([ttnn_transpose_181, ttnn_transpose_182, ttnn_transpose_183], -1);  ttnn_transpose_181 = ttnn_transpose_182 = ttnn_transpose_183 = None
    ttnn_from_torch_76 = ttnn_decorators_ttnn_from_torch(arg70_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg70_1 = None
    ttnn_experimental_view_450 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_76, (1, -1));  ttnn_from_torch_76 = None
    ttnn_from_torch_77 = ttnn_decorators_ttnn_from_torch(arg72_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg72_1 = None
    ttnn_experimental_view_451 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_77, (1, -1));  ttnn_from_torch_77 = None
    ttnn_from_torch_78 = ttnn_decorators_ttnn_from_torch(arg74_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg74_1 = None
    ttnn_experimental_view_452 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_78, (1, -1));  ttnn_from_torch_78 = None
    ttnn_concat_9 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_450, ttnn_experimental_view_451, ttnn_experimental_view_452], -1);  ttnn_experimental_view_450 = ttnn_experimental_view_451 = ttnn_experimental_view_452 = None
    ttnn_to_layout_9 = ttnn_decorators_ttnn_to_layout(ttnn_concat_8, ttnn_TILE_LAYOUT);  ttnn_concat_8 = None
    ttnn_to_layout_10 = ttnn_decorators_ttnn_to_layout(ttnn_concat_9, ttnn_TILE_LAYOUT);  ttnn_concat_9 = None
    ttnn_linear_149 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_72, ttnn_to_layout_9, bias = ttnn_to_layout_10);  ttnn_experimental_view_72 = ttnn_to_layout_9 = ttnn_to_layout_10 = None
    ttnn_experimental_view_453 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_149, [1, 256, 3072]);  ttnn_linear_149 = None
    ttnn_transformer_split_query_key_value_and_split_heads_4 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_453, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_453 = None
    getitem_63 = ttnn_transformer_split_query_key_value_and_split_heads_4[0]
    getitem_64 = ttnn_transformer_split_query_key_value_and_split_heads_4[1]
    getitem_65 = ttnn_transformer_split_query_key_value_and_split_heads_4[2];  ttnn_transformer_split_query_key_value_and_split_heads_4 = None
    ttnn_matmul_201 = ttnn_decorators_ttnn_matmul(getitem_63, getitem_64);  getitem_63 = getitem_64 = None
    ttnn_transformer_attention_softmax__28 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_201, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_201 = None
    ttnn_matmul_202 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__28, getitem_65);  ttnn_transformer_attention_softmax__28 = getitem_65 = None
    ttnn_transformer_concatenate_heads_4 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_202);  ttnn_matmul_202 = None
    ttnn_from_torch_79 = ttnn_decorators_ttnn_from_torch(arg75_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg75_1 = None
    ttnn_from_torch_80 = ttnn_decorators_ttnn_from_torch(arg76_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg76_1 = None
    ttnn_linear_27 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_4, ttnn_from_torch_79, transpose_b = True, bias = ttnn_from_torch_80, activation = None);  ttnn_transformer_concatenate_heads_4 = ttnn_from_torch_79 = ttnn_from_torch_80 = None
    ttnn_experimental_view_85 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_27, [1, 256, 1024]);  ttnn_linear_27 = None
    ttnn_add_160 = ttnn_decorators_ttnn_add(ttnn_experimental_view_85, ttnn_layer_norm_8);  ttnn_experimental_view_85 = ttnn_layer_norm_8 = None
    ttnn_from_torch_81 = ttnn_decorators_ttnn_from_torch(arg77_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg77_1 = None
    ttnn_from_torch_82 = ttnn_decorators_ttnn_from_torch(arg78_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg78_1 = None
    ttnn_layer_norm_9 = ttnn_decorators_ttnn_layer_norm(ttnn_add_160, epsilon = 1e-12, weight = ttnn_from_torch_81, bias = ttnn_from_torch_82);  ttnn_add_160 = ttnn_from_torch_81 = ttnn_from_torch_82 = None
    ttnn_experimental_view_86 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_9, [256, 1024])
    ttnn_from_torch_83 = ttnn_decorators_ttnn_from_torch(arg79_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg79_1 = None
    ttnn_from_torch_84 = ttnn_decorators_ttnn_from_torch(arg80_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg80_1 = None
    ttnn_linear_28 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_86, ttnn_from_torch_83, transpose_b = True, bias = ttnn_from_torch_84, activation = 'gelu');  ttnn_experimental_view_86 = ttnn_from_torch_83 = ttnn_from_torch_84 = None
    ttnn_experimental_view_88 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_28, [256, 4096]);  ttnn_linear_28 = None
    ttnn_from_torch_85 = ttnn_decorators_ttnn_from_torch(arg81_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg81_1 = None
    ttnn_from_torch_86 = ttnn_decorators_ttnn_from_torch(arg82_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg82_1 = None
    ttnn_linear_29 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_88, ttnn_from_torch_85, transpose_b = True, bias = ttnn_from_torch_86, activation = None);  ttnn_experimental_view_88 = ttnn_from_torch_85 = ttnn_from_torch_86 = None
    ttnn_experimental_view_89 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_29, [1, 256, 1024]);  ttnn_linear_29 = None
    ttnn_add_161 = ttnn_decorators_ttnn_add(ttnn_experimental_view_89, ttnn_layer_norm_9);  ttnn_experimental_view_89 = ttnn_layer_norm_9 = None
    ttnn_from_torch_87 = ttnn_decorators_ttnn_from_torch(arg83_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg83_1 = None
    ttnn_from_torch_88 = ttnn_decorators_ttnn_from_torch(arg84_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg84_1 = None
    ttnn_layer_norm_10 = ttnn_decorators_ttnn_layer_norm(ttnn_add_161, epsilon = 1e-12, weight = ttnn_from_torch_87, bias = ttnn_from_torch_88);  ttnn_add_161 = ttnn_from_torch_87 = ttnn_from_torch_88 = None
    ttnn_experimental_view_90 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_10, [256, 1024])
    ttnn_from_torch_89 = ttnn_decorators_ttnn_from_torch(arg85_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg85_1 = None
    ttnn_transpose_184 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_89, -2, -1);  ttnn_from_torch_89 = None
    ttnn_from_torch_90 = ttnn_decorators_ttnn_from_torch(arg87_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg87_1 = None
    ttnn_transpose_185 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_90, -2, -1);  ttnn_from_torch_90 = None
    ttnn_from_torch_91 = ttnn_decorators_ttnn_from_torch(arg89_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg89_1 = None
    ttnn_transpose_186 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_91, -2, -1);  ttnn_from_torch_91 = None
    ttnn_concat_10 = ttnn_decorators_ttnn_concat([ttnn_transpose_184, ttnn_transpose_185, ttnn_transpose_186], -1);  ttnn_transpose_184 = ttnn_transpose_185 = ttnn_transpose_186 = None
    ttnn_from_torch_92 = ttnn_decorators_ttnn_from_torch(arg86_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg86_1 = None
    ttnn_experimental_view_454 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_92, (1, -1));  ttnn_from_torch_92 = None
    ttnn_from_torch_93 = ttnn_decorators_ttnn_from_torch(arg88_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg88_1 = None
    ttnn_experimental_view_455 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_93, (1, -1));  ttnn_from_torch_93 = None
    ttnn_from_torch_94 = ttnn_decorators_ttnn_from_torch(arg90_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg90_1 = None
    ttnn_experimental_view_456 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_94, (1, -1));  ttnn_from_torch_94 = None
    ttnn_concat_11 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_454, ttnn_experimental_view_455, ttnn_experimental_view_456], -1);  ttnn_experimental_view_454 = ttnn_experimental_view_455 = ttnn_experimental_view_456 = None
    ttnn_to_layout_11 = ttnn_decorators_ttnn_to_layout(ttnn_concat_10, ttnn_TILE_LAYOUT);  ttnn_concat_10 = None
    ttnn_to_layout_12 = ttnn_decorators_ttnn_to_layout(ttnn_concat_11, ttnn_TILE_LAYOUT);  ttnn_concat_11 = None
    ttnn_linear_150 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_90, ttnn_to_layout_11, bias = ttnn_to_layout_12);  ttnn_experimental_view_90 = ttnn_to_layout_11 = ttnn_to_layout_12 = None
    ttnn_experimental_view_457 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_150, [1, 256, 3072]);  ttnn_linear_150 = None
    ttnn_transformer_split_query_key_value_and_split_heads_5 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_457, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_457 = None
    getitem_66 = ttnn_transformer_split_query_key_value_and_split_heads_5[0]
    getitem_67 = ttnn_transformer_split_query_key_value_and_split_heads_5[1]
    getitem_68 = ttnn_transformer_split_query_key_value_and_split_heads_5[2];  ttnn_transformer_split_query_key_value_and_split_heads_5 = None
    ttnn_matmul_203 = ttnn_decorators_ttnn_matmul(getitem_66, getitem_67);  getitem_66 = getitem_67 = None
    ttnn_transformer_attention_softmax__29 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_203, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_203 = None
    ttnn_matmul_204 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__29, getitem_68);  ttnn_transformer_attention_softmax__29 = getitem_68 = None
    ttnn_transformer_concatenate_heads_5 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_204);  ttnn_matmul_204 = None
    ttnn_from_torch_95 = ttnn_decorators_ttnn_from_torch(arg91_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg91_1 = None
    ttnn_from_torch_96 = ttnn_decorators_ttnn_from_torch(arg92_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg92_1 = None
    ttnn_linear_33 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_5, ttnn_from_torch_95, transpose_b = True, bias = ttnn_from_torch_96, activation = None);  ttnn_transformer_concatenate_heads_5 = ttnn_from_torch_95 = ttnn_from_torch_96 = None
    ttnn_experimental_view_103 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_33, [1, 256, 1024]);  ttnn_linear_33 = None
    ttnn_add_163 = ttnn_decorators_ttnn_add(ttnn_experimental_view_103, ttnn_layer_norm_10);  ttnn_experimental_view_103 = ttnn_layer_norm_10 = None
    ttnn_from_torch_97 = ttnn_decorators_ttnn_from_torch(arg93_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg93_1 = None
    ttnn_from_torch_98 = ttnn_decorators_ttnn_from_torch(arg94_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg94_1 = None
    ttnn_layer_norm_11 = ttnn_decorators_ttnn_layer_norm(ttnn_add_163, epsilon = 1e-12, weight = ttnn_from_torch_97, bias = ttnn_from_torch_98);  ttnn_add_163 = ttnn_from_torch_97 = ttnn_from_torch_98 = None
    ttnn_experimental_view_104 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_11, [256, 1024])
    ttnn_from_torch_99 = ttnn_decorators_ttnn_from_torch(arg95_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg95_1 = None
    ttnn_from_torch_100 = ttnn_decorators_ttnn_from_torch(arg96_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg96_1 = None
    ttnn_linear_34 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_104, ttnn_from_torch_99, transpose_b = True, bias = ttnn_from_torch_100, activation = 'gelu');  ttnn_experimental_view_104 = ttnn_from_torch_99 = ttnn_from_torch_100 = None
    ttnn_experimental_view_106 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_34, [256, 4096]);  ttnn_linear_34 = None
    ttnn_from_torch_101 = ttnn_decorators_ttnn_from_torch(arg97_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg97_1 = None
    ttnn_from_torch_102 = ttnn_decorators_ttnn_from_torch(arg98_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg98_1 = None
    ttnn_linear_35 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_106, ttnn_from_torch_101, transpose_b = True, bias = ttnn_from_torch_102, activation = None);  ttnn_experimental_view_106 = ttnn_from_torch_101 = ttnn_from_torch_102 = None
    ttnn_experimental_view_107 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_35, [1, 256, 1024]);  ttnn_linear_35 = None
    ttnn_add_164 = ttnn_decorators_ttnn_add(ttnn_experimental_view_107, ttnn_layer_norm_11);  ttnn_experimental_view_107 = ttnn_layer_norm_11 = None
    ttnn_from_torch_103 = ttnn_decorators_ttnn_from_torch(arg99_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg99_1 = None
    ttnn_from_torch_104 = ttnn_decorators_ttnn_from_torch(arg100_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg100_1 = None
    ttnn_layer_norm_12 = ttnn_decorators_ttnn_layer_norm(ttnn_add_164, epsilon = 1e-12, weight = ttnn_from_torch_103, bias = ttnn_from_torch_104);  ttnn_add_164 = ttnn_from_torch_103 = ttnn_from_torch_104 = None
    ttnn_experimental_view_108 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_12, [256, 1024])
    ttnn_from_torch_105 = ttnn_decorators_ttnn_from_torch(arg101_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg101_1 = None
    ttnn_transpose_187 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_105, -2, -1);  ttnn_from_torch_105 = None
    ttnn_from_torch_106 = ttnn_decorators_ttnn_from_torch(arg103_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg103_1 = None
    ttnn_transpose_188 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_106, -2, -1);  ttnn_from_torch_106 = None
    ttnn_from_torch_107 = ttnn_decorators_ttnn_from_torch(arg105_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg105_1 = None
    ttnn_transpose_189 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_107, -2, -1);  ttnn_from_torch_107 = None
    ttnn_concat_12 = ttnn_decorators_ttnn_concat([ttnn_transpose_187, ttnn_transpose_188, ttnn_transpose_189], -1);  ttnn_transpose_187 = ttnn_transpose_188 = ttnn_transpose_189 = None
    ttnn_from_torch_108 = ttnn_decorators_ttnn_from_torch(arg102_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg102_1 = None
    ttnn_experimental_view_458 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_108, (1, -1));  ttnn_from_torch_108 = None
    ttnn_from_torch_109 = ttnn_decorators_ttnn_from_torch(arg104_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg104_1 = None
    ttnn_experimental_view_459 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_109, (1, -1));  ttnn_from_torch_109 = None
    ttnn_from_torch_110 = ttnn_decorators_ttnn_from_torch(arg106_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg106_1 = None
    ttnn_experimental_view_460 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_110, (1, -1));  ttnn_from_torch_110 = None
    ttnn_concat_13 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_458, ttnn_experimental_view_459, ttnn_experimental_view_460], -1);  ttnn_experimental_view_458 = ttnn_experimental_view_459 = ttnn_experimental_view_460 = None
    ttnn_to_layout_13 = ttnn_decorators_ttnn_to_layout(ttnn_concat_12, ttnn_TILE_LAYOUT);  ttnn_concat_12 = None
    ttnn_to_layout_14 = ttnn_decorators_ttnn_to_layout(ttnn_concat_13, ttnn_TILE_LAYOUT);  ttnn_concat_13 = None
    ttnn_linear_151 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_108, ttnn_to_layout_13, bias = ttnn_to_layout_14);  ttnn_experimental_view_108 = ttnn_to_layout_13 = ttnn_to_layout_14 = None
    ttnn_experimental_view_461 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_151, [1, 256, 3072]);  ttnn_linear_151 = None
    ttnn_transformer_split_query_key_value_and_split_heads_6 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_461, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_461 = None
    getitem_69 = ttnn_transformer_split_query_key_value_and_split_heads_6[0]
    getitem_70 = ttnn_transformer_split_query_key_value_and_split_heads_6[1]
    getitem_71 = ttnn_transformer_split_query_key_value_and_split_heads_6[2];  ttnn_transformer_split_query_key_value_and_split_heads_6 = None
    ttnn_matmul_205 = ttnn_decorators_ttnn_matmul(getitem_69, getitem_70);  getitem_69 = getitem_70 = None
    ttnn_transformer_attention_softmax__30 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_205, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_205 = None
    ttnn_matmul_206 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__30, getitem_71);  ttnn_transformer_attention_softmax__30 = getitem_71 = None
    ttnn_transformer_concatenate_heads_6 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_206);  ttnn_matmul_206 = None
    ttnn_from_torch_111 = ttnn_decorators_ttnn_from_torch(arg107_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg107_1 = None
    ttnn_from_torch_112 = ttnn_decorators_ttnn_from_torch(arg108_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg108_1 = None
    ttnn_linear_39 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_6, ttnn_from_torch_111, transpose_b = True, bias = ttnn_from_torch_112, activation = None);  ttnn_transformer_concatenate_heads_6 = ttnn_from_torch_111 = ttnn_from_torch_112 = None
    ttnn_experimental_view_121 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_39, [1, 256, 1024]);  ttnn_linear_39 = None
    ttnn_add_166 = ttnn_decorators_ttnn_add(ttnn_experimental_view_121, ttnn_layer_norm_12);  ttnn_experimental_view_121 = ttnn_layer_norm_12 = None
    ttnn_from_torch_113 = ttnn_decorators_ttnn_from_torch(arg109_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg109_1 = None
    ttnn_from_torch_114 = ttnn_decorators_ttnn_from_torch(arg110_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg110_1 = None
    ttnn_layer_norm_13 = ttnn_decorators_ttnn_layer_norm(ttnn_add_166, epsilon = 1e-12, weight = ttnn_from_torch_113, bias = ttnn_from_torch_114);  ttnn_add_166 = ttnn_from_torch_113 = ttnn_from_torch_114 = None
    ttnn_experimental_view_122 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_13, [256, 1024])
    ttnn_from_torch_115 = ttnn_decorators_ttnn_from_torch(arg111_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg111_1 = None
    ttnn_from_torch_116 = ttnn_decorators_ttnn_from_torch(arg112_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg112_1 = None
    ttnn_linear_40 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_122, ttnn_from_torch_115, transpose_b = True, bias = ttnn_from_torch_116, activation = 'gelu');  ttnn_experimental_view_122 = ttnn_from_torch_115 = ttnn_from_torch_116 = None
    ttnn_experimental_view_124 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_40, [256, 4096]);  ttnn_linear_40 = None
    ttnn_from_torch_117 = ttnn_decorators_ttnn_from_torch(arg113_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg113_1 = None
    ttnn_from_torch_118 = ttnn_decorators_ttnn_from_torch(arg114_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg114_1 = None
    ttnn_linear_41 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_124, ttnn_from_torch_117, transpose_b = True, bias = ttnn_from_torch_118, activation = None);  ttnn_experimental_view_124 = ttnn_from_torch_117 = ttnn_from_torch_118 = None
    ttnn_experimental_view_125 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_41, [1, 256, 1024]);  ttnn_linear_41 = None
    ttnn_add_167 = ttnn_decorators_ttnn_add(ttnn_experimental_view_125, ttnn_layer_norm_13);  ttnn_experimental_view_125 = ttnn_layer_norm_13 = None
    ttnn_from_torch_119 = ttnn_decorators_ttnn_from_torch(arg115_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg115_1 = None
    ttnn_from_torch_120 = ttnn_decorators_ttnn_from_torch(arg116_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg116_1 = None
    ttnn_layer_norm_14 = ttnn_decorators_ttnn_layer_norm(ttnn_add_167, epsilon = 1e-12, weight = ttnn_from_torch_119, bias = ttnn_from_torch_120);  ttnn_add_167 = ttnn_from_torch_119 = ttnn_from_torch_120 = None
    ttnn_experimental_view_126 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_14, [256, 1024])
    ttnn_from_torch_121 = ttnn_decorators_ttnn_from_torch(arg117_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg117_1 = None
    ttnn_transpose_190 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_121, -2, -1);  ttnn_from_torch_121 = None
    ttnn_from_torch_122 = ttnn_decorators_ttnn_from_torch(arg119_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg119_1 = None
    ttnn_transpose_191 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_122, -2, -1);  ttnn_from_torch_122 = None
    ttnn_from_torch_123 = ttnn_decorators_ttnn_from_torch(arg121_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg121_1 = None
    ttnn_transpose_192 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_123, -2, -1);  ttnn_from_torch_123 = None
    ttnn_concat_14 = ttnn_decorators_ttnn_concat([ttnn_transpose_190, ttnn_transpose_191, ttnn_transpose_192], -1);  ttnn_transpose_190 = ttnn_transpose_191 = ttnn_transpose_192 = None
    ttnn_from_torch_124 = ttnn_decorators_ttnn_from_torch(arg118_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg118_1 = None
    ttnn_experimental_view_462 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_124, (1, -1));  ttnn_from_torch_124 = None
    ttnn_from_torch_125 = ttnn_decorators_ttnn_from_torch(arg120_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg120_1 = None
    ttnn_experimental_view_463 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_125, (1, -1));  ttnn_from_torch_125 = None
    ttnn_from_torch_126 = ttnn_decorators_ttnn_from_torch(arg122_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg122_1 = None
    ttnn_experimental_view_464 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_126, (1, -1));  ttnn_from_torch_126 = None
    ttnn_concat_15 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_462, ttnn_experimental_view_463, ttnn_experimental_view_464], -1);  ttnn_experimental_view_462 = ttnn_experimental_view_463 = ttnn_experimental_view_464 = None
    ttnn_to_layout_15 = ttnn_decorators_ttnn_to_layout(ttnn_concat_14, ttnn_TILE_LAYOUT);  ttnn_concat_14 = None
    ttnn_to_layout_16 = ttnn_decorators_ttnn_to_layout(ttnn_concat_15, ttnn_TILE_LAYOUT);  ttnn_concat_15 = None
    ttnn_linear_152 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_126, ttnn_to_layout_15, bias = ttnn_to_layout_16);  ttnn_experimental_view_126 = ttnn_to_layout_15 = ttnn_to_layout_16 = None
    ttnn_experimental_view_465 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_152, [1, 256, 3072]);  ttnn_linear_152 = None
    ttnn_transformer_split_query_key_value_and_split_heads_7 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_465, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_465 = None
    getitem_72 = ttnn_transformer_split_query_key_value_and_split_heads_7[0]
    getitem_73 = ttnn_transformer_split_query_key_value_and_split_heads_7[1]
    getitem_74 = ttnn_transformer_split_query_key_value_and_split_heads_7[2];  ttnn_transformer_split_query_key_value_and_split_heads_7 = None
    ttnn_matmul_207 = ttnn_decorators_ttnn_matmul(getitem_72, getitem_73);  getitem_72 = getitem_73 = None
    ttnn_transformer_attention_softmax__31 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_207, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_207 = None
    ttnn_matmul_208 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__31, getitem_74);  ttnn_transformer_attention_softmax__31 = getitem_74 = None
    ttnn_transformer_concatenate_heads_7 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_208);  ttnn_matmul_208 = None
    ttnn_from_torch_127 = ttnn_decorators_ttnn_from_torch(arg123_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg123_1 = None
    ttnn_from_torch_128 = ttnn_decorators_ttnn_from_torch(arg124_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg124_1 = None
    ttnn_linear_45 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_7, ttnn_from_torch_127, transpose_b = True, bias = ttnn_from_torch_128, activation = None);  ttnn_transformer_concatenate_heads_7 = ttnn_from_torch_127 = ttnn_from_torch_128 = None
    ttnn_experimental_view_139 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_45, [1, 256, 1024]);  ttnn_linear_45 = None
    ttnn_add_169 = ttnn_decorators_ttnn_add(ttnn_experimental_view_139, ttnn_layer_norm_14);  ttnn_experimental_view_139 = ttnn_layer_norm_14 = None
    ttnn_from_torch_129 = ttnn_decorators_ttnn_from_torch(arg125_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg125_1 = None
    ttnn_from_torch_130 = ttnn_decorators_ttnn_from_torch(arg126_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg126_1 = None
    ttnn_layer_norm_15 = ttnn_decorators_ttnn_layer_norm(ttnn_add_169, epsilon = 1e-12, weight = ttnn_from_torch_129, bias = ttnn_from_torch_130);  ttnn_add_169 = ttnn_from_torch_129 = ttnn_from_torch_130 = None
    ttnn_experimental_view_140 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_15, [256, 1024])
    ttnn_from_torch_131 = ttnn_decorators_ttnn_from_torch(arg127_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg127_1 = None
    ttnn_from_torch_132 = ttnn_decorators_ttnn_from_torch(arg128_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg128_1 = None
    ttnn_linear_46 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_140, ttnn_from_torch_131, transpose_b = True, bias = ttnn_from_torch_132, activation = 'gelu');  ttnn_experimental_view_140 = ttnn_from_torch_131 = ttnn_from_torch_132 = None
    ttnn_experimental_view_142 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_46, [256, 4096]);  ttnn_linear_46 = None
    ttnn_from_torch_133 = ttnn_decorators_ttnn_from_torch(arg129_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg129_1 = None
    ttnn_from_torch_134 = ttnn_decorators_ttnn_from_torch(arg130_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg130_1 = None
    ttnn_linear_47 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_142, ttnn_from_torch_133, transpose_b = True, bias = ttnn_from_torch_134, activation = None);  ttnn_experimental_view_142 = ttnn_from_torch_133 = ttnn_from_torch_134 = None
    ttnn_experimental_view_143 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_47, [1, 256, 1024]);  ttnn_linear_47 = None
    ttnn_add_170 = ttnn_decorators_ttnn_add(ttnn_experimental_view_143, ttnn_layer_norm_15);  ttnn_experimental_view_143 = ttnn_layer_norm_15 = None
    ttnn_from_torch_135 = ttnn_decorators_ttnn_from_torch(arg131_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg131_1 = None
    ttnn_from_torch_136 = ttnn_decorators_ttnn_from_torch(arg132_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg132_1 = None
    ttnn_layer_norm_16 = ttnn_decorators_ttnn_layer_norm(ttnn_add_170, epsilon = 1e-12, weight = ttnn_from_torch_135, bias = ttnn_from_torch_136);  ttnn_add_170 = ttnn_from_torch_135 = ttnn_from_torch_136 = None
    ttnn_experimental_view_144 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_16, [256, 1024])
    ttnn_from_torch_137 = ttnn_decorators_ttnn_from_torch(arg133_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg133_1 = None
    ttnn_transpose_193 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_137, -2, -1);  ttnn_from_torch_137 = None
    ttnn_from_torch_138 = ttnn_decorators_ttnn_from_torch(arg135_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg135_1 = None
    ttnn_transpose_194 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_138, -2, -1);  ttnn_from_torch_138 = None
    ttnn_from_torch_139 = ttnn_decorators_ttnn_from_torch(arg137_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg137_1 = None
    ttnn_transpose_195 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_139, -2, -1);  ttnn_from_torch_139 = None
    ttnn_concat_16 = ttnn_decorators_ttnn_concat([ttnn_transpose_193, ttnn_transpose_194, ttnn_transpose_195], -1);  ttnn_transpose_193 = ttnn_transpose_194 = ttnn_transpose_195 = None
    ttnn_from_torch_140 = ttnn_decorators_ttnn_from_torch(arg134_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg134_1 = None
    ttnn_experimental_view_466 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_140, (1, -1));  ttnn_from_torch_140 = None
    ttnn_from_torch_141 = ttnn_decorators_ttnn_from_torch(arg136_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg136_1 = None
    ttnn_experimental_view_467 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_141, (1, -1));  ttnn_from_torch_141 = None
    ttnn_from_torch_142 = ttnn_decorators_ttnn_from_torch(arg138_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg138_1 = None
    ttnn_experimental_view_468 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_142, (1, -1));  ttnn_from_torch_142 = None
    ttnn_concat_17 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_466, ttnn_experimental_view_467, ttnn_experimental_view_468], -1);  ttnn_experimental_view_466 = ttnn_experimental_view_467 = ttnn_experimental_view_468 = None
    ttnn_to_layout_17 = ttnn_decorators_ttnn_to_layout(ttnn_concat_16, ttnn_TILE_LAYOUT);  ttnn_concat_16 = None
    ttnn_to_layout_18 = ttnn_decorators_ttnn_to_layout(ttnn_concat_17, ttnn_TILE_LAYOUT);  ttnn_concat_17 = None
    ttnn_linear_153 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_144, ttnn_to_layout_17, bias = ttnn_to_layout_18);  ttnn_experimental_view_144 = ttnn_to_layout_17 = ttnn_to_layout_18 = None
    ttnn_experimental_view_469 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_153, [1, 256, 3072]);  ttnn_linear_153 = None
    ttnn_transformer_split_query_key_value_and_split_heads_8 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_469, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_469 = None
    getitem_75 = ttnn_transformer_split_query_key_value_and_split_heads_8[0]
    getitem_76 = ttnn_transformer_split_query_key_value_and_split_heads_8[1]
    getitem_77 = ttnn_transformer_split_query_key_value_and_split_heads_8[2];  ttnn_transformer_split_query_key_value_and_split_heads_8 = None
    ttnn_matmul_209 = ttnn_decorators_ttnn_matmul(getitem_75, getitem_76);  getitem_75 = getitem_76 = None
    ttnn_transformer_attention_softmax__32 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_209, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_209 = None
    ttnn_matmul_210 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__32, getitem_77);  ttnn_transformer_attention_softmax__32 = getitem_77 = None
    ttnn_transformer_concatenate_heads_8 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_210);  ttnn_matmul_210 = None
    ttnn_from_torch_143 = ttnn_decorators_ttnn_from_torch(arg139_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg139_1 = None
    ttnn_from_torch_144 = ttnn_decorators_ttnn_from_torch(arg140_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg140_1 = None
    ttnn_linear_51 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_8, ttnn_from_torch_143, transpose_b = True, bias = ttnn_from_torch_144, activation = None);  ttnn_transformer_concatenate_heads_8 = ttnn_from_torch_143 = ttnn_from_torch_144 = None
    ttnn_experimental_view_157 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_51, [1, 256, 1024]);  ttnn_linear_51 = None
    ttnn_add_172 = ttnn_decorators_ttnn_add(ttnn_experimental_view_157, ttnn_layer_norm_16);  ttnn_experimental_view_157 = ttnn_layer_norm_16 = None
    ttnn_from_torch_145 = ttnn_decorators_ttnn_from_torch(arg141_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg141_1 = None
    ttnn_from_torch_146 = ttnn_decorators_ttnn_from_torch(arg142_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg142_1 = None
    ttnn_layer_norm_17 = ttnn_decorators_ttnn_layer_norm(ttnn_add_172, epsilon = 1e-12, weight = ttnn_from_torch_145, bias = ttnn_from_torch_146);  ttnn_add_172 = ttnn_from_torch_145 = ttnn_from_torch_146 = None
    ttnn_experimental_view_158 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_17, [256, 1024])
    ttnn_from_torch_147 = ttnn_decorators_ttnn_from_torch(arg143_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg143_1 = None
    ttnn_from_torch_148 = ttnn_decorators_ttnn_from_torch(arg144_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg144_1 = None
    ttnn_linear_52 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_158, ttnn_from_torch_147, transpose_b = True, bias = ttnn_from_torch_148, activation = 'gelu');  ttnn_experimental_view_158 = ttnn_from_torch_147 = ttnn_from_torch_148 = None
    ttnn_experimental_view_160 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_52, [256, 4096]);  ttnn_linear_52 = None
    ttnn_from_torch_149 = ttnn_decorators_ttnn_from_torch(arg145_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg145_1 = None
    ttnn_from_torch_150 = ttnn_decorators_ttnn_from_torch(arg146_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg146_1 = None
    ttnn_linear_53 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_160, ttnn_from_torch_149, transpose_b = True, bias = ttnn_from_torch_150, activation = None);  ttnn_experimental_view_160 = ttnn_from_torch_149 = ttnn_from_torch_150 = None
    ttnn_experimental_view_161 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_53, [1, 256, 1024]);  ttnn_linear_53 = None
    ttnn_add_173 = ttnn_decorators_ttnn_add(ttnn_experimental_view_161, ttnn_layer_norm_17);  ttnn_experimental_view_161 = ttnn_layer_norm_17 = None
    ttnn_from_torch_151 = ttnn_decorators_ttnn_from_torch(arg147_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg147_1 = None
    ttnn_from_torch_152 = ttnn_decorators_ttnn_from_torch(arg148_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg148_1 = None
    ttnn_layer_norm_18 = ttnn_decorators_ttnn_layer_norm(ttnn_add_173, epsilon = 1e-12, weight = ttnn_from_torch_151, bias = ttnn_from_torch_152);  ttnn_add_173 = ttnn_from_torch_151 = ttnn_from_torch_152 = None
    ttnn_experimental_view_162 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_18, [256, 1024])
    ttnn_from_torch_153 = ttnn_decorators_ttnn_from_torch(arg149_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg149_1 = None
    ttnn_transpose_196 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_153, -2, -1);  ttnn_from_torch_153 = None
    ttnn_from_torch_154 = ttnn_decorators_ttnn_from_torch(arg151_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg151_1 = None
    ttnn_transpose_197 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_154, -2, -1);  ttnn_from_torch_154 = None
    ttnn_from_torch_155 = ttnn_decorators_ttnn_from_torch(arg153_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg153_1 = None
    ttnn_transpose_198 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_155, -2, -1);  ttnn_from_torch_155 = None
    ttnn_concat_18 = ttnn_decorators_ttnn_concat([ttnn_transpose_196, ttnn_transpose_197, ttnn_transpose_198], -1);  ttnn_transpose_196 = ttnn_transpose_197 = ttnn_transpose_198 = None
    ttnn_from_torch_156 = ttnn_decorators_ttnn_from_torch(arg150_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg150_1 = None
    ttnn_experimental_view_470 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_156, (1, -1));  ttnn_from_torch_156 = None
    ttnn_from_torch_157 = ttnn_decorators_ttnn_from_torch(arg152_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg152_1 = None
    ttnn_experimental_view_471 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_157, (1, -1));  ttnn_from_torch_157 = None
    ttnn_from_torch_158 = ttnn_decorators_ttnn_from_torch(arg154_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg154_1 = None
    ttnn_experimental_view_472 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_158, (1, -1));  ttnn_from_torch_158 = None
    ttnn_concat_19 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_470, ttnn_experimental_view_471, ttnn_experimental_view_472], -1);  ttnn_experimental_view_470 = ttnn_experimental_view_471 = ttnn_experimental_view_472 = None
    ttnn_to_layout_19 = ttnn_decorators_ttnn_to_layout(ttnn_concat_18, ttnn_TILE_LAYOUT);  ttnn_concat_18 = None
    ttnn_to_layout_20 = ttnn_decorators_ttnn_to_layout(ttnn_concat_19, ttnn_TILE_LAYOUT);  ttnn_concat_19 = None
    ttnn_linear_154 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_162, ttnn_to_layout_19, bias = ttnn_to_layout_20);  ttnn_experimental_view_162 = ttnn_to_layout_19 = ttnn_to_layout_20 = None
    ttnn_experimental_view_473 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_154, [1, 256, 3072]);  ttnn_linear_154 = None
    ttnn_transformer_split_query_key_value_and_split_heads_9 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_473, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_473 = None
    getitem_78 = ttnn_transformer_split_query_key_value_and_split_heads_9[0]
    getitem_79 = ttnn_transformer_split_query_key_value_and_split_heads_9[1]
    getitem_80 = ttnn_transformer_split_query_key_value_and_split_heads_9[2];  ttnn_transformer_split_query_key_value_and_split_heads_9 = None
    ttnn_matmul_211 = ttnn_decorators_ttnn_matmul(getitem_78, getitem_79);  getitem_78 = getitem_79 = None
    ttnn_transformer_attention_softmax__33 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_211, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_211 = None
    ttnn_matmul_212 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__33, getitem_80);  ttnn_transformer_attention_softmax__33 = getitem_80 = None
    ttnn_transformer_concatenate_heads_9 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_212);  ttnn_matmul_212 = None
    ttnn_from_torch_159 = ttnn_decorators_ttnn_from_torch(arg155_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg155_1 = None
    ttnn_from_torch_160 = ttnn_decorators_ttnn_from_torch(arg156_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg156_1 = None
    ttnn_linear_57 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_9, ttnn_from_torch_159, transpose_b = True, bias = ttnn_from_torch_160, activation = None);  ttnn_transformer_concatenate_heads_9 = ttnn_from_torch_159 = ttnn_from_torch_160 = None
    ttnn_experimental_view_175 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_57, [1, 256, 1024]);  ttnn_linear_57 = None
    ttnn_add_175 = ttnn_decorators_ttnn_add(ttnn_experimental_view_175, ttnn_layer_norm_18);  ttnn_experimental_view_175 = ttnn_layer_norm_18 = None
    ttnn_from_torch_161 = ttnn_decorators_ttnn_from_torch(arg157_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg157_1 = None
    ttnn_from_torch_162 = ttnn_decorators_ttnn_from_torch(arg158_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg158_1 = None
    ttnn_layer_norm_19 = ttnn_decorators_ttnn_layer_norm(ttnn_add_175, epsilon = 1e-12, weight = ttnn_from_torch_161, bias = ttnn_from_torch_162);  ttnn_add_175 = ttnn_from_torch_161 = ttnn_from_torch_162 = None
    ttnn_experimental_view_176 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_19, [256, 1024])
    ttnn_from_torch_163 = ttnn_decorators_ttnn_from_torch(arg159_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg159_1 = None
    ttnn_from_torch_164 = ttnn_decorators_ttnn_from_torch(arg160_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg160_1 = None
    ttnn_linear_58 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_176, ttnn_from_torch_163, transpose_b = True, bias = ttnn_from_torch_164, activation = 'gelu');  ttnn_experimental_view_176 = ttnn_from_torch_163 = ttnn_from_torch_164 = None
    ttnn_experimental_view_178 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_58, [256, 4096]);  ttnn_linear_58 = None
    ttnn_from_torch_165 = ttnn_decorators_ttnn_from_torch(arg161_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg161_1 = None
    ttnn_from_torch_166 = ttnn_decorators_ttnn_from_torch(arg162_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg162_1 = None
    ttnn_linear_59 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_178, ttnn_from_torch_165, transpose_b = True, bias = ttnn_from_torch_166, activation = None);  ttnn_experimental_view_178 = ttnn_from_torch_165 = ttnn_from_torch_166 = None
    ttnn_experimental_view_179 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_59, [1, 256, 1024]);  ttnn_linear_59 = None
    ttnn_add_176 = ttnn_decorators_ttnn_add(ttnn_experimental_view_179, ttnn_layer_norm_19);  ttnn_experimental_view_179 = ttnn_layer_norm_19 = None
    ttnn_from_torch_167 = ttnn_decorators_ttnn_from_torch(arg163_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg163_1 = None
    ttnn_from_torch_168 = ttnn_decorators_ttnn_from_torch(arg164_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg164_1 = None
    ttnn_layer_norm_20 = ttnn_decorators_ttnn_layer_norm(ttnn_add_176, epsilon = 1e-12, weight = ttnn_from_torch_167, bias = ttnn_from_torch_168);  ttnn_add_176 = ttnn_from_torch_167 = ttnn_from_torch_168 = None
    ttnn_experimental_view_180 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_20, [256, 1024])
    ttnn_from_torch_169 = ttnn_decorators_ttnn_from_torch(arg165_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg165_1 = None
    ttnn_transpose_199 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_169, -2, -1);  ttnn_from_torch_169 = None
    ttnn_from_torch_170 = ttnn_decorators_ttnn_from_torch(arg167_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg167_1 = None
    ttnn_transpose_200 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_170, -2, -1);  ttnn_from_torch_170 = None
    ttnn_from_torch_171 = ttnn_decorators_ttnn_from_torch(arg169_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg169_1 = None
    ttnn_transpose_201 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_171, -2, -1);  ttnn_from_torch_171 = None
    ttnn_concat_20 = ttnn_decorators_ttnn_concat([ttnn_transpose_199, ttnn_transpose_200, ttnn_transpose_201], -1);  ttnn_transpose_199 = ttnn_transpose_200 = ttnn_transpose_201 = None
    ttnn_from_torch_172 = ttnn_decorators_ttnn_from_torch(arg166_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg166_1 = None
    ttnn_experimental_view_474 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_172, (1, -1));  ttnn_from_torch_172 = None
    ttnn_from_torch_173 = ttnn_decorators_ttnn_from_torch(arg168_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg168_1 = None
    ttnn_experimental_view_475 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_173, (1, -1));  ttnn_from_torch_173 = None
    ttnn_from_torch_174 = ttnn_decorators_ttnn_from_torch(arg170_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg170_1 = None
    ttnn_experimental_view_476 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_174, (1, -1));  ttnn_from_torch_174 = None
    ttnn_concat_21 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_474, ttnn_experimental_view_475, ttnn_experimental_view_476], -1);  ttnn_experimental_view_474 = ttnn_experimental_view_475 = ttnn_experimental_view_476 = None
    ttnn_to_layout_21 = ttnn_decorators_ttnn_to_layout(ttnn_concat_20, ttnn_TILE_LAYOUT);  ttnn_concat_20 = None
    ttnn_to_layout_22 = ttnn_decorators_ttnn_to_layout(ttnn_concat_21, ttnn_TILE_LAYOUT);  ttnn_concat_21 = None
    ttnn_linear_155 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_180, ttnn_to_layout_21, bias = ttnn_to_layout_22);  ttnn_experimental_view_180 = ttnn_to_layout_21 = ttnn_to_layout_22 = None
    ttnn_experimental_view_477 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_155, [1, 256, 3072]);  ttnn_linear_155 = None
    ttnn_transformer_split_query_key_value_and_split_heads_10 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_477, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_477 = None
    getitem_81 = ttnn_transformer_split_query_key_value_and_split_heads_10[0]
    getitem_82 = ttnn_transformer_split_query_key_value_and_split_heads_10[1]
    getitem_83 = ttnn_transformer_split_query_key_value_and_split_heads_10[2];  ttnn_transformer_split_query_key_value_and_split_heads_10 = None
    ttnn_matmul_213 = ttnn_decorators_ttnn_matmul(getitem_81, getitem_82);  getitem_81 = getitem_82 = None
    ttnn_transformer_attention_softmax__34 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_213, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_213 = None
    ttnn_matmul_214 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__34, getitem_83);  ttnn_transformer_attention_softmax__34 = getitem_83 = None
    ttnn_transformer_concatenate_heads_10 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_214);  ttnn_matmul_214 = None
    ttnn_from_torch_175 = ttnn_decorators_ttnn_from_torch(arg171_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg171_1 = None
    ttnn_from_torch_176 = ttnn_decorators_ttnn_from_torch(arg172_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg172_1 = None
    ttnn_linear_63 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_10, ttnn_from_torch_175, transpose_b = True, bias = ttnn_from_torch_176, activation = None);  ttnn_transformer_concatenate_heads_10 = ttnn_from_torch_175 = ttnn_from_torch_176 = None
    ttnn_experimental_view_193 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_63, [1, 256, 1024]);  ttnn_linear_63 = None
    ttnn_add_178 = ttnn_decorators_ttnn_add(ttnn_experimental_view_193, ttnn_layer_norm_20);  ttnn_experimental_view_193 = ttnn_layer_norm_20 = None
    ttnn_from_torch_177 = ttnn_decorators_ttnn_from_torch(arg173_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg173_1 = None
    ttnn_from_torch_178 = ttnn_decorators_ttnn_from_torch(arg174_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg174_1 = None
    ttnn_layer_norm_21 = ttnn_decorators_ttnn_layer_norm(ttnn_add_178, epsilon = 1e-12, weight = ttnn_from_torch_177, bias = ttnn_from_torch_178);  ttnn_add_178 = ttnn_from_torch_177 = ttnn_from_torch_178 = None
    ttnn_experimental_view_194 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_21, [256, 1024])
    ttnn_from_torch_179 = ttnn_decorators_ttnn_from_torch(arg175_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg175_1 = None
    ttnn_from_torch_180 = ttnn_decorators_ttnn_from_torch(arg176_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg176_1 = None
    ttnn_linear_64 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_194, ttnn_from_torch_179, transpose_b = True, bias = ttnn_from_torch_180, activation = 'gelu');  ttnn_experimental_view_194 = ttnn_from_torch_179 = ttnn_from_torch_180 = None
    ttnn_experimental_view_196 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_64, [256, 4096]);  ttnn_linear_64 = None
    ttnn_from_torch_181 = ttnn_decorators_ttnn_from_torch(arg177_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg177_1 = None
    ttnn_from_torch_182 = ttnn_decorators_ttnn_from_torch(arg178_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg178_1 = None
    ttnn_linear_65 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_196, ttnn_from_torch_181, transpose_b = True, bias = ttnn_from_torch_182, activation = None);  ttnn_experimental_view_196 = ttnn_from_torch_181 = ttnn_from_torch_182 = None
    ttnn_experimental_view_197 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_65, [1, 256, 1024]);  ttnn_linear_65 = None
    ttnn_add_179 = ttnn_decorators_ttnn_add(ttnn_experimental_view_197, ttnn_layer_norm_21);  ttnn_experimental_view_197 = ttnn_layer_norm_21 = None
    ttnn_from_torch_183 = ttnn_decorators_ttnn_from_torch(arg179_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg179_1 = None
    ttnn_from_torch_184 = ttnn_decorators_ttnn_from_torch(arg180_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg180_1 = None
    ttnn_layer_norm_22 = ttnn_decorators_ttnn_layer_norm(ttnn_add_179, epsilon = 1e-12, weight = ttnn_from_torch_183, bias = ttnn_from_torch_184);  ttnn_add_179 = ttnn_from_torch_183 = ttnn_from_torch_184 = None
    ttnn_experimental_view_198 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_22, [256, 1024])
    ttnn_from_torch_185 = ttnn_decorators_ttnn_from_torch(arg181_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg181_1 = None
    ttnn_transpose_202 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_185, -2, -1);  ttnn_from_torch_185 = None
    ttnn_from_torch_186 = ttnn_decorators_ttnn_from_torch(arg183_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg183_1 = None
    ttnn_transpose_203 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_186, -2, -1);  ttnn_from_torch_186 = None
    ttnn_from_torch_187 = ttnn_decorators_ttnn_from_torch(arg185_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg185_1 = None
    ttnn_transpose_204 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_187, -2, -1);  ttnn_from_torch_187 = None
    ttnn_concat_22 = ttnn_decorators_ttnn_concat([ttnn_transpose_202, ttnn_transpose_203, ttnn_transpose_204], -1);  ttnn_transpose_202 = ttnn_transpose_203 = ttnn_transpose_204 = None
    ttnn_from_torch_188 = ttnn_decorators_ttnn_from_torch(arg182_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg182_1 = None
    ttnn_experimental_view_478 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_188, (1, -1));  ttnn_from_torch_188 = None
    ttnn_from_torch_189 = ttnn_decorators_ttnn_from_torch(arg184_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg184_1 = None
    ttnn_experimental_view_479 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_189, (1, -1));  ttnn_from_torch_189 = None
    ttnn_from_torch_190 = ttnn_decorators_ttnn_from_torch(arg186_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg186_1 = None
    ttnn_experimental_view_480 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_190, (1, -1));  ttnn_from_torch_190 = None
    ttnn_concat_23 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_478, ttnn_experimental_view_479, ttnn_experimental_view_480], -1);  ttnn_experimental_view_478 = ttnn_experimental_view_479 = ttnn_experimental_view_480 = None
    ttnn_to_layout_23 = ttnn_decorators_ttnn_to_layout(ttnn_concat_22, ttnn_TILE_LAYOUT);  ttnn_concat_22 = None
    ttnn_to_layout_24 = ttnn_decorators_ttnn_to_layout(ttnn_concat_23, ttnn_TILE_LAYOUT);  ttnn_concat_23 = None
    ttnn_linear_156 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_198, ttnn_to_layout_23, bias = ttnn_to_layout_24);  ttnn_experimental_view_198 = ttnn_to_layout_23 = ttnn_to_layout_24 = None
    ttnn_experimental_view_481 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_156, [1, 256, 3072]);  ttnn_linear_156 = None
    ttnn_transformer_split_query_key_value_and_split_heads_11 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_481, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_481 = None
    getitem_84 = ttnn_transformer_split_query_key_value_and_split_heads_11[0]
    getitem_85 = ttnn_transformer_split_query_key_value_and_split_heads_11[1]
    getitem_86 = ttnn_transformer_split_query_key_value_and_split_heads_11[2];  ttnn_transformer_split_query_key_value_and_split_heads_11 = None
    ttnn_matmul_215 = ttnn_decorators_ttnn_matmul(getitem_84, getitem_85);  getitem_84 = getitem_85 = None
    ttnn_transformer_attention_softmax__35 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_215, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_215 = None
    ttnn_matmul_216 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__35, getitem_86);  ttnn_transformer_attention_softmax__35 = getitem_86 = None
    ttnn_transformer_concatenate_heads_11 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_216);  ttnn_matmul_216 = None
    ttnn_from_torch_191 = ttnn_decorators_ttnn_from_torch(arg187_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg187_1 = None
    ttnn_from_torch_192 = ttnn_decorators_ttnn_from_torch(arg188_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg188_1 = None
    ttnn_linear_69 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_11, ttnn_from_torch_191, transpose_b = True, bias = ttnn_from_torch_192, activation = None);  ttnn_transformer_concatenate_heads_11 = ttnn_from_torch_191 = ttnn_from_torch_192 = None
    ttnn_experimental_view_211 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_69, [1, 256, 1024]);  ttnn_linear_69 = None
    ttnn_add_181 = ttnn_decorators_ttnn_add(ttnn_experimental_view_211, ttnn_layer_norm_22);  ttnn_experimental_view_211 = ttnn_layer_norm_22 = None
    ttnn_from_torch_193 = ttnn_decorators_ttnn_from_torch(arg189_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg189_1 = None
    ttnn_from_torch_194 = ttnn_decorators_ttnn_from_torch(arg190_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg190_1 = None
    ttnn_layer_norm_23 = ttnn_decorators_ttnn_layer_norm(ttnn_add_181, epsilon = 1e-12, weight = ttnn_from_torch_193, bias = ttnn_from_torch_194);  ttnn_add_181 = ttnn_from_torch_193 = ttnn_from_torch_194 = None
    ttnn_experimental_view_212 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_23, [256, 1024])
    ttnn_from_torch_195 = ttnn_decorators_ttnn_from_torch(arg191_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg191_1 = None
    ttnn_from_torch_196 = ttnn_decorators_ttnn_from_torch(arg192_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg192_1 = None
    ttnn_linear_70 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_212, ttnn_from_torch_195, transpose_b = True, bias = ttnn_from_torch_196, activation = 'gelu');  ttnn_experimental_view_212 = ttnn_from_torch_195 = ttnn_from_torch_196 = None
    ttnn_experimental_view_214 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_70, [256, 4096]);  ttnn_linear_70 = None
    ttnn_from_torch_197 = ttnn_decorators_ttnn_from_torch(arg193_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg193_1 = None
    ttnn_from_torch_198 = ttnn_decorators_ttnn_from_torch(arg194_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg194_1 = None
    ttnn_linear_71 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_214, ttnn_from_torch_197, transpose_b = True, bias = ttnn_from_torch_198, activation = None);  ttnn_experimental_view_214 = ttnn_from_torch_197 = ttnn_from_torch_198 = None
    ttnn_experimental_view_215 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_71, [1, 256, 1024]);  ttnn_linear_71 = None
    ttnn_add_182 = ttnn_decorators_ttnn_add(ttnn_experimental_view_215, ttnn_layer_norm_23);  ttnn_experimental_view_215 = ttnn_layer_norm_23 = None
    ttnn_from_torch_199 = ttnn_decorators_ttnn_from_torch(arg195_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg195_1 = None
    ttnn_from_torch_200 = ttnn_decorators_ttnn_from_torch(arg196_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg196_1 = None
    ttnn_layer_norm_24 = ttnn_decorators_ttnn_layer_norm(ttnn_add_182, epsilon = 1e-12, weight = ttnn_from_torch_199, bias = ttnn_from_torch_200);  ttnn_add_182 = ttnn_from_torch_199 = ttnn_from_torch_200 = None
    ttnn_experimental_view_216 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_24, [256, 1024])
    ttnn_from_torch_201 = ttnn_decorators_ttnn_from_torch(arg197_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg197_1 = None
    ttnn_transpose_205 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_201, -2, -1);  ttnn_from_torch_201 = None
    ttnn_from_torch_202 = ttnn_decorators_ttnn_from_torch(arg199_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg199_1 = None
    ttnn_transpose_206 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_202, -2, -1);  ttnn_from_torch_202 = None
    ttnn_from_torch_203 = ttnn_decorators_ttnn_from_torch(arg201_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg201_1 = None
    ttnn_transpose_207 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_203, -2, -1);  ttnn_from_torch_203 = None
    ttnn_concat_24 = ttnn_decorators_ttnn_concat([ttnn_transpose_205, ttnn_transpose_206, ttnn_transpose_207], -1);  ttnn_transpose_205 = ttnn_transpose_206 = ttnn_transpose_207 = None
    ttnn_from_torch_204 = ttnn_decorators_ttnn_from_torch(arg198_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg198_1 = None
    ttnn_experimental_view_482 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_204, (1, -1));  ttnn_from_torch_204 = None
    ttnn_from_torch_205 = ttnn_decorators_ttnn_from_torch(arg200_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg200_1 = None
    ttnn_experimental_view_483 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_205, (1, -1));  ttnn_from_torch_205 = None
    ttnn_from_torch_206 = ttnn_decorators_ttnn_from_torch(arg202_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg202_1 = None
    ttnn_experimental_view_484 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_206, (1, -1));  ttnn_from_torch_206 = None
    ttnn_concat_25 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_482, ttnn_experimental_view_483, ttnn_experimental_view_484], -1);  ttnn_experimental_view_482 = ttnn_experimental_view_483 = ttnn_experimental_view_484 = None
    ttnn_to_layout_25 = ttnn_decorators_ttnn_to_layout(ttnn_concat_24, ttnn_TILE_LAYOUT);  ttnn_concat_24 = None
    ttnn_to_layout_26 = ttnn_decorators_ttnn_to_layout(ttnn_concat_25, ttnn_TILE_LAYOUT);  ttnn_concat_25 = None
    ttnn_linear_157 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_216, ttnn_to_layout_25, bias = ttnn_to_layout_26);  ttnn_experimental_view_216 = ttnn_to_layout_25 = ttnn_to_layout_26 = None
    ttnn_experimental_view_485 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_157, [1, 256, 3072]);  ttnn_linear_157 = None
    ttnn_transformer_split_query_key_value_and_split_heads_12 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_485, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_485 = None
    getitem_87 = ttnn_transformer_split_query_key_value_and_split_heads_12[0]
    getitem_88 = ttnn_transformer_split_query_key_value_and_split_heads_12[1]
    getitem_89 = ttnn_transformer_split_query_key_value_and_split_heads_12[2];  ttnn_transformer_split_query_key_value_and_split_heads_12 = None
    ttnn_matmul_217 = ttnn_decorators_ttnn_matmul(getitem_87, getitem_88);  getitem_87 = getitem_88 = None
    ttnn_transformer_attention_softmax__36 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_217, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_217 = None
    ttnn_matmul_218 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__36, getitem_89);  ttnn_transformer_attention_softmax__36 = getitem_89 = None
    ttnn_transformer_concatenate_heads_12 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_218);  ttnn_matmul_218 = None
    ttnn_from_torch_207 = ttnn_decorators_ttnn_from_torch(arg203_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg203_1 = None
    ttnn_from_torch_208 = ttnn_decorators_ttnn_from_torch(arg204_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg204_1 = None
    ttnn_linear_75 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_12, ttnn_from_torch_207, transpose_b = True, bias = ttnn_from_torch_208, activation = None);  ttnn_transformer_concatenate_heads_12 = ttnn_from_torch_207 = ttnn_from_torch_208 = None
    ttnn_experimental_view_229 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_75, [1, 256, 1024]);  ttnn_linear_75 = None
    ttnn_add_184 = ttnn_decorators_ttnn_add(ttnn_experimental_view_229, ttnn_layer_norm_24);  ttnn_experimental_view_229 = ttnn_layer_norm_24 = None
    ttnn_from_torch_209 = ttnn_decorators_ttnn_from_torch(arg205_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg205_1 = None
    ttnn_from_torch_210 = ttnn_decorators_ttnn_from_torch(arg206_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg206_1 = None
    ttnn_layer_norm_25 = ttnn_decorators_ttnn_layer_norm(ttnn_add_184, epsilon = 1e-12, weight = ttnn_from_torch_209, bias = ttnn_from_torch_210);  ttnn_add_184 = ttnn_from_torch_209 = ttnn_from_torch_210 = None
    ttnn_experimental_view_230 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_25, [256, 1024])
    ttnn_from_torch_211 = ttnn_decorators_ttnn_from_torch(arg207_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg207_1 = None
    ttnn_from_torch_212 = ttnn_decorators_ttnn_from_torch(arg208_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg208_1 = None
    ttnn_linear_76 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_230, ttnn_from_torch_211, transpose_b = True, bias = ttnn_from_torch_212, activation = 'gelu');  ttnn_experimental_view_230 = ttnn_from_torch_211 = ttnn_from_torch_212 = None
    ttnn_experimental_view_232 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_76, [256, 4096]);  ttnn_linear_76 = None
    ttnn_from_torch_213 = ttnn_decorators_ttnn_from_torch(arg209_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg209_1 = None
    ttnn_from_torch_214 = ttnn_decorators_ttnn_from_torch(arg210_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg210_1 = None
    ttnn_linear_77 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_232, ttnn_from_torch_213, transpose_b = True, bias = ttnn_from_torch_214, activation = None);  ttnn_experimental_view_232 = ttnn_from_torch_213 = ttnn_from_torch_214 = None
    ttnn_experimental_view_233 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_77, [1, 256, 1024]);  ttnn_linear_77 = None
    ttnn_add_185 = ttnn_decorators_ttnn_add(ttnn_experimental_view_233, ttnn_layer_norm_25);  ttnn_experimental_view_233 = ttnn_layer_norm_25 = None
    ttnn_from_torch_215 = ttnn_decorators_ttnn_from_torch(arg211_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg211_1 = None
    ttnn_from_torch_216 = ttnn_decorators_ttnn_from_torch(arg212_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg212_1 = None
    ttnn_layer_norm_26 = ttnn_decorators_ttnn_layer_norm(ttnn_add_185, epsilon = 1e-12, weight = ttnn_from_torch_215, bias = ttnn_from_torch_216);  ttnn_add_185 = ttnn_from_torch_215 = ttnn_from_torch_216 = None
    ttnn_experimental_view_234 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_26, [256, 1024])
    ttnn_from_torch_217 = ttnn_decorators_ttnn_from_torch(arg213_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg213_1 = None
    ttnn_transpose_208 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_217, -2, -1);  ttnn_from_torch_217 = None
    ttnn_from_torch_218 = ttnn_decorators_ttnn_from_torch(arg215_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg215_1 = None
    ttnn_transpose_209 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_218, -2, -1);  ttnn_from_torch_218 = None
    ttnn_from_torch_219 = ttnn_decorators_ttnn_from_torch(arg217_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg217_1 = None
    ttnn_transpose_210 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_219, -2, -1);  ttnn_from_torch_219 = None
    ttnn_concat_26 = ttnn_decorators_ttnn_concat([ttnn_transpose_208, ttnn_transpose_209, ttnn_transpose_210], -1);  ttnn_transpose_208 = ttnn_transpose_209 = ttnn_transpose_210 = None
    ttnn_from_torch_220 = ttnn_decorators_ttnn_from_torch(arg214_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg214_1 = None
    ttnn_experimental_view_486 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_220, (1, -1));  ttnn_from_torch_220 = None
    ttnn_from_torch_221 = ttnn_decorators_ttnn_from_torch(arg216_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg216_1 = None
    ttnn_experimental_view_487 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_221, (1, -1));  ttnn_from_torch_221 = None
    ttnn_from_torch_222 = ttnn_decorators_ttnn_from_torch(arg218_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg218_1 = None
    ttnn_experimental_view_488 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_222, (1, -1));  ttnn_from_torch_222 = None
    ttnn_concat_27 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_486, ttnn_experimental_view_487, ttnn_experimental_view_488], -1);  ttnn_experimental_view_486 = ttnn_experimental_view_487 = ttnn_experimental_view_488 = None
    ttnn_to_layout_27 = ttnn_decorators_ttnn_to_layout(ttnn_concat_26, ttnn_TILE_LAYOUT);  ttnn_concat_26 = None
    ttnn_to_layout_28 = ttnn_decorators_ttnn_to_layout(ttnn_concat_27, ttnn_TILE_LAYOUT);  ttnn_concat_27 = None
    ttnn_linear_158 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_234, ttnn_to_layout_27, bias = ttnn_to_layout_28);  ttnn_experimental_view_234 = ttnn_to_layout_27 = ttnn_to_layout_28 = None
    ttnn_experimental_view_489 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_158, [1, 256, 3072]);  ttnn_linear_158 = None
    ttnn_transformer_split_query_key_value_and_split_heads_13 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_489, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_489 = None
    getitem_90 = ttnn_transformer_split_query_key_value_and_split_heads_13[0]
    getitem_91 = ttnn_transformer_split_query_key_value_and_split_heads_13[1]
    getitem_92 = ttnn_transformer_split_query_key_value_and_split_heads_13[2];  ttnn_transformer_split_query_key_value_and_split_heads_13 = None
    ttnn_matmul_219 = ttnn_decorators_ttnn_matmul(getitem_90, getitem_91);  getitem_90 = getitem_91 = None
    ttnn_transformer_attention_softmax__37 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_219, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_219 = None
    ttnn_matmul_220 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__37, getitem_92);  ttnn_transformer_attention_softmax__37 = getitem_92 = None
    ttnn_transformer_concatenate_heads_13 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_220);  ttnn_matmul_220 = None
    ttnn_from_torch_223 = ttnn_decorators_ttnn_from_torch(arg219_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg219_1 = None
    ttnn_from_torch_224 = ttnn_decorators_ttnn_from_torch(arg220_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg220_1 = None
    ttnn_linear_81 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_13, ttnn_from_torch_223, transpose_b = True, bias = ttnn_from_torch_224, activation = None);  ttnn_transformer_concatenate_heads_13 = ttnn_from_torch_223 = ttnn_from_torch_224 = None
    ttnn_experimental_view_247 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_81, [1, 256, 1024]);  ttnn_linear_81 = None
    ttnn_add_187 = ttnn_decorators_ttnn_add(ttnn_experimental_view_247, ttnn_layer_norm_26);  ttnn_experimental_view_247 = ttnn_layer_norm_26 = None
    ttnn_from_torch_225 = ttnn_decorators_ttnn_from_torch(arg221_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg221_1 = None
    ttnn_from_torch_226 = ttnn_decorators_ttnn_from_torch(arg222_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg222_1 = None
    ttnn_layer_norm_27 = ttnn_decorators_ttnn_layer_norm(ttnn_add_187, epsilon = 1e-12, weight = ttnn_from_torch_225, bias = ttnn_from_torch_226);  ttnn_add_187 = ttnn_from_torch_225 = ttnn_from_torch_226 = None
    ttnn_experimental_view_248 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_27, [256, 1024])
    ttnn_from_torch_227 = ttnn_decorators_ttnn_from_torch(arg223_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg223_1 = None
    ttnn_from_torch_228 = ttnn_decorators_ttnn_from_torch(arg224_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg224_1 = None
    ttnn_linear_82 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_248, ttnn_from_torch_227, transpose_b = True, bias = ttnn_from_torch_228, activation = 'gelu');  ttnn_experimental_view_248 = ttnn_from_torch_227 = ttnn_from_torch_228 = None
    ttnn_experimental_view_250 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_82, [256, 4096]);  ttnn_linear_82 = None
    ttnn_from_torch_229 = ttnn_decorators_ttnn_from_torch(arg225_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg225_1 = None
    ttnn_from_torch_230 = ttnn_decorators_ttnn_from_torch(arg226_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg226_1 = None
    ttnn_linear_83 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_250, ttnn_from_torch_229, transpose_b = True, bias = ttnn_from_torch_230, activation = None);  ttnn_experimental_view_250 = ttnn_from_torch_229 = ttnn_from_torch_230 = None
    ttnn_experimental_view_251 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_83, [1, 256, 1024]);  ttnn_linear_83 = None
    ttnn_add_188 = ttnn_decorators_ttnn_add(ttnn_experimental_view_251, ttnn_layer_norm_27);  ttnn_experimental_view_251 = ttnn_layer_norm_27 = None
    ttnn_from_torch_231 = ttnn_decorators_ttnn_from_torch(arg227_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg227_1 = None
    ttnn_from_torch_232 = ttnn_decorators_ttnn_from_torch(arg228_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg228_1 = None
    ttnn_layer_norm_28 = ttnn_decorators_ttnn_layer_norm(ttnn_add_188, epsilon = 1e-12, weight = ttnn_from_torch_231, bias = ttnn_from_torch_232);  ttnn_add_188 = ttnn_from_torch_231 = ttnn_from_torch_232 = None
    ttnn_experimental_view_252 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_28, [256, 1024])
    ttnn_from_torch_233 = ttnn_decorators_ttnn_from_torch(arg229_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg229_1 = None
    ttnn_transpose_211 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_233, -2, -1);  ttnn_from_torch_233 = None
    ttnn_from_torch_234 = ttnn_decorators_ttnn_from_torch(arg231_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg231_1 = None
    ttnn_transpose_212 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_234, -2, -1);  ttnn_from_torch_234 = None
    ttnn_from_torch_235 = ttnn_decorators_ttnn_from_torch(arg233_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg233_1 = None
    ttnn_transpose_213 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_235, -2, -1);  ttnn_from_torch_235 = None
    ttnn_concat_28 = ttnn_decorators_ttnn_concat([ttnn_transpose_211, ttnn_transpose_212, ttnn_transpose_213], -1);  ttnn_transpose_211 = ttnn_transpose_212 = ttnn_transpose_213 = None
    ttnn_from_torch_236 = ttnn_decorators_ttnn_from_torch(arg230_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg230_1 = None
    ttnn_experimental_view_490 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_236, (1, -1));  ttnn_from_torch_236 = None
    ttnn_from_torch_237 = ttnn_decorators_ttnn_from_torch(arg232_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg232_1 = None
    ttnn_experimental_view_491 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_237, (1, -1));  ttnn_from_torch_237 = None
    ttnn_from_torch_238 = ttnn_decorators_ttnn_from_torch(arg234_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg234_1 = None
    ttnn_experimental_view_492 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_238, (1, -1));  ttnn_from_torch_238 = None
    ttnn_concat_29 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_490, ttnn_experimental_view_491, ttnn_experimental_view_492], -1);  ttnn_experimental_view_490 = ttnn_experimental_view_491 = ttnn_experimental_view_492 = None
    ttnn_to_layout_29 = ttnn_decorators_ttnn_to_layout(ttnn_concat_28, ttnn_TILE_LAYOUT);  ttnn_concat_28 = None
    ttnn_to_layout_30 = ttnn_decorators_ttnn_to_layout(ttnn_concat_29, ttnn_TILE_LAYOUT);  ttnn_concat_29 = None
    ttnn_linear_159 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_252, ttnn_to_layout_29, bias = ttnn_to_layout_30);  ttnn_experimental_view_252 = ttnn_to_layout_29 = ttnn_to_layout_30 = None
    ttnn_experimental_view_493 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_159, [1, 256, 3072]);  ttnn_linear_159 = None
    ttnn_transformer_split_query_key_value_and_split_heads_14 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_493, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_493 = None
    getitem_93 = ttnn_transformer_split_query_key_value_and_split_heads_14[0]
    getitem_94 = ttnn_transformer_split_query_key_value_and_split_heads_14[1]
    getitem_95 = ttnn_transformer_split_query_key_value_and_split_heads_14[2];  ttnn_transformer_split_query_key_value_and_split_heads_14 = None
    ttnn_matmul_221 = ttnn_decorators_ttnn_matmul(getitem_93, getitem_94);  getitem_93 = getitem_94 = None
    ttnn_transformer_attention_softmax__38 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_221, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_221 = None
    ttnn_matmul_222 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__38, getitem_95);  ttnn_transformer_attention_softmax__38 = getitem_95 = None
    ttnn_transformer_concatenate_heads_14 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_222);  ttnn_matmul_222 = None
    ttnn_from_torch_239 = ttnn_decorators_ttnn_from_torch(arg235_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg235_1 = None
    ttnn_from_torch_240 = ttnn_decorators_ttnn_from_torch(arg236_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg236_1 = None
    ttnn_linear_87 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_14, ttnn_from_torch_239, transpose_b = True, bias = ttnn_from_torch_240, activation = None);  ttnn_transformer_concatenate_heads_14 = ttnn_from_torch_239 = ttnn_from_torch_240 = None
    ttnn_experimental_view_265 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_87, [1, 256, 1024]);  ttnn_linear_87 = None
    ttnn_add_190 = ttnn_decorators_ttnn_add(ttnn_experimental_view_265, ttnn_layer_norm_28);  ttnn_experimental_view_265 = ttnn_layer_norm_28 = None
    ttnn_from_torch_241 = ttnn_decorators_ttnn_from_torch(arg237_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg237_1 = None
    ttnn_from_torch_242 = ttnn_decorators_ttnn_from_torch(arg238_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg238_1 = None
    ttnn_layer_norm_29 = ttnn_decorators_ttnn_layer_norm(ttnn_add_190, epsilon = 1e-12, weight = ttnn_from_torch_241, bias = ttnn_from_torch_242);  ttnn_add_190 = ttnn_from_torch_241 = ttnn_from_torch_242 = None
    ttnn_experimental_view_266 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_29, [256, 1024])
    ttnn_from_torch_243 = ttnn_decorators_ttnn_from_torch(arg239_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg239_1 = None
    ttnn_from_torch_244 = ttnn_decorators_ttnn_from_torch(arg240_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg240_1 = None
    ttnn_linear_88 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_266, ttnn_from_torch_243, transpose_b = True, bias = ttnn_from_torch_244, activation = 'gelu');  ttnn_experimental_view_266 = ttnn_from_torch_243 = ttnn_from_torch_244 = None
    ttnn_experimental_view_268 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_88, [256, 4096]);  ttnn_linear_88 = None
    ttnn_from_torch_245 = ttnn_decorators_ttnn_from_torch(arg241_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg241_1 = None
    ttnn_from_torch_246 = ttnn_decorators_ttnn_from_torch(arg242_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg242_1 = None
    ttnn_linear_89 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_268, ttnn_from_torch_245, transpose_b = True, bias = ttnn_from_torch_246, activation = None);  ttnn_experimental_view_268 = ttnn_from_torch_245 = ttnn_from_torch_246 = None
    ttnn_experimental_view_269 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_89, [1, 256, 1024]);  ttnn_linear_89 = None
    ttnn_add_191 = ttnn_decorators_ttnn_add(ttnn_experimental_view_269, ttnn_layer_norm_29);  ttnn_experimental_view_269 = ttnn_layer_norm_29 = None
    ttnn_from_torch_247 = ttnn_decorators_ttnn_from_torch(arg243_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg243_1 = None
    ttnn_from_torch_248 = ttnn_decorators_ttnn_from_torch(arg244_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg244_1 = None
    ttnn_layer_norm_30 = ttnn_decorators_ttnn_layer_norm(ttnn_add_191, epsilon = 1e-12, weight = ttnn_from_torch_247, bias = ttnn_from_torch_248);  ttnn_add_191 = ttnn_from_torch_247 = ttnn_from_torch_248 = None
    ttnn_experimental_view_270 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_30, [256, 1024])
    ttnn_from_torch_249 = ttnn_decorators_ttnn_from_torch(arg245_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg245_1 = None
    ttnn_transpose_214 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_249, -2, -1);  ttnn_from_torch_249 = None
    ttnn_from_torch_250 = ttnn_decorators_ttnn_from_torch(arg247_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg247_1 = None
    ttnn_transpose_215 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_250, -2, -1);  ttnn_from_torch_250 = None
    ttnn_from_torch_251 = ttnn_decorators_ttnn_from_torch(arg249_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg249_1 = None
    ttnn_transpose_216 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_251, -2, -1);  ttnn_from_torch_251 = None
    ttnn_concat_30 = ttnn_decorators_ttnn_concat([ttnn_transpose_214, ttnn_transpose_215, ttnn_transpose_216], -1);  ttnn_transpose_214 = ttnn_transpose_215 = ttnn_transpose_216 = None
    ttnn_from_torch_252 = ttnn_decorators_ttnn_from_torch(arg246_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg246_1 = None
    ttnn_experimental_view_494 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_252, (1, -1));  ttnn_from_torch_252 = None
    ttnn_from_torch_253 = ttnn_decorators_ttnn_from_torch(arg248_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg248_1 = None
    ttnn_experimental_view_495 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_253, (1, -1));  ttnn_from_torch_253 = None
    ttnn_from_torch_254 = ttnn_decorators_ttnn_from_torch(arg250_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg250_1 = None
    ttnn_experimental_view_496 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_254, (1, -1));  ttnn_from_torch_254 = None
    ttnn_concat_31 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_494, ttnn_experimental_view_495, ttnn_experimental_view_496], -1);  ttnn_experimental_view_494 = ttnn_experimental_view_495 = ttnn_experimental_view_496 = None
    ttnn_to_layout_31 = ttnn_decorators_ttnn_to_layout(ttnn_concat_30, ttnn_TILE_LAYOUT);  ttnn_concat_30 = None
    ttnn_to_layout_32 = ttnn_decorators_ttnn_to_layout(ttnn_concat_31, ttnn_TILE_LAYOUT);  ttnn_concat_31 = None
    ttnn_linear_160 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_270, ttnn_to_layout_31, bias = ttnn_to_layout_32);  ttnn_experimental_view_270 = ttnn_to_layout_31 = ttnn_to_layout_32 = None
    ttnn_experimental_view_497 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_160, [1, 256, 3072]);  ttnn_linear_160 = None
    ttnn_transformer_split_query_key_value_and_split_heads_15 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_497, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_497 = None
    getitem_96 = ttnn_transformer_split_query_key_value_and_split_heads_15[0]
    getitem_97 = ttnn_transformer_split_query_key_value_and_split_heads_15[1]
    getitem_98 = ttnn_transformer_split_query_key_value_and_split_heads_15[2];  ttnn_transformer_split_query_key_value_and_split_heads_15 = None
    ttnn_matmul_223 = ttnn_decorators_ttnn_matmul(getitem_96, getitem_97);  getitem_96 = getitem_97 = None
    ttnn_transformer_attention_softmax__39 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_223, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_223 = None
    ttnn_matmul_224 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__39, getitem_98);  ttnn_transformer_attention_softmax__39 = getitem_98 = None
    ttnn_transformer_concatenate_heads_15 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_224);  ttnn_matmul_224 = None
    ttnn_from_torch_255 = ttnn_decorators_ttnn_from_torch(arg251_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg251_1 = None
    ttnn_from_torch_256 = ttnn_decorators_ttnn_from_torch(arg252_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg252_1 = None
    ttnn_linear_93 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_15, ttnn_from_torch_255, transpose_b = True, bias = ttnn_from_torch_256, activation = None);  ttnn_transformer_concatenate_heads_15 = ttnn_from_torch_255 = ttnn_from_torch_256 = None
    ttnn_experimental_view_283 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_93, [1, 256, 1024]);  ttnn_linear_93 = None
    ttnn_add_193 = ttnn_decorators_ttnn_add(ttnn_experimental_view_283, ttnn_layer_norm_30);  ttnn_experimental_view_283 = ttnn_layer_norm_30 = None
    ttnn_from_torch_257 = ttnn_decorators_ttnn_from_torch(arg253_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg253_1 = None
    ttnn_from_torch_258 = ttnn_decorators_ttnn_from_torch(arg254_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg254_1 = None
    ttnn_layer_norm_31 = ttnn_decorators_ttnn_layer_norm(ttnn_add_193, epsilon = 1e-12, weight = ttnn_from_torch_257, bias = ttnn_from_torch_258);  ttnn_add_193 = ttnn_from_torch_257 = ttnn_from_torch_258 = None
    ttnn_experimental_view_284 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_31, [256, 1024])
    ttnn_from_torch_259 = ttnn_decorators_ttnn_from_torch(arg255_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg255_1 = None
    ttnn_from_torch_260 = ttnn_decorators_ttnn_from_torch(arg256_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg256_1 = None
    ttnn_linear_94 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_284, ttnn_from_torch_259, transpose_b = True, bias = ttnn_from_torch_260, activation = 'gelu');  ttnn_experimental_view_284 = ttnn_from_torch_259 = ttnn_from_torch_260 = None
    ttnn_experimental_view_286 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_94, [256, 4096]);  ttnn_linear_94 = None
    ttnn_from_torch_261 = ttnn_decorators_ttnn_from_torch(arg257_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg257_1 = None
    ttnn_from_torch_262 = ttnn_decorators_ttnn_from_torch(arg258_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg258_1 = None
    ttnn_linear_95 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_286, ttnn_from_torch_261, transpose_b = True, bias = ttnn_from_torch_262, activation = None);  ttnn_experimental_view_286 = ttnn_from_torch_261 = ttnn_from_torch_262 = None
    ttnn_experimental_view_287 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_95, [1, 256, 1024]);  ttnn_linear_95 = None
    ttnn_add_194 = ttnn_decorators_ttnn_add(ttnn_experimental_view_287, ttnn_layer_norm_31);  ttnn_experimental_view_287 = ttnn_layer_norm_31 = None
    ttnn_from_torch_263 = ttnn_decorators_ttnn_from_torch(arg259_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg259_1 = None
    ttnn_from_torch_264 = ttnn_decorators_ttnn_from_torch(arg260_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg260_1 = None
    ttnn_layer_norm_32 = ttnn_decorators_ttnn_layer_norm(ttnn_add_194, epsilon = 1e-12, weight = ttnn_from_torch_263, bias = ttnn_from_torch_264);  ttnn_add_194 = ttnn_from_torch_263 = ttnn_from_torch_264 = None
    ttnn_experimental_view_288 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_32, [256, 1024])
    ttnn_from_torch_265 = ttnn_decorators_ttnn_from_torch(arg261_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg261_1 = None
    ttnn_transpose_217 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_265, -2, -1);  ttnn_from_torch_265 = None
    ttnn_from_torch_266 = ttnn_decorators_ttnn_from_torch(arg263_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg263_1 = None
    ttnn_transpose_218 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_266, -2, -1);  ttnn_from_torch_266 = None
    ttnn_from_torch_267 = ttnn_decorators_ttnn_from_torch(arg265_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg265_1 = None
    ttnn_transpose_219 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_267, -2, -1);  ttnn_from_torch_267 = None
    ttnn_concat_32 = ttnn_decorators_ttnn_concat([ttnn_transpose_217, ttnn_transpose_218, ttnn_transpose_219], -1);  ttnn_transpose_217 = ttnn_transpose_218 = ttnn_transpose_219 = None
    ttnn_from_torch_268 = ttnn_decorators_ttnn_from_torch(arg262_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg262_1 = None
    ttnn_experimental_view_498 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_268, (1, -1));  ttnn_from_torch_268 = None
    ttnn_from_torch_269 = ttnn_decorators_ttnn_from_torch(arg264_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg264_1 = None
    ttnn_experimental_view_499 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_269, (1, -1));  ttnn_from_torch_269 = None
    ttnn_from_torch_270 = ttnn_decorators_ttnn_from_torch(arg266_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg266_1 = None
    ttnn_experimental_view_500 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_270, (1, -1));  ttnn_from_torch_270 = None
    ttnn_concat_33 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_498, ttnn_experimental_view_499, ttnn_experimental_view_500], -1);  ttnn_experimental_view_498 = ttnn_experimental_view_499 = ttnn_experimental_view_500 = None
    ttnn_to_layout_33 = ttnn_decorators_ttnn_to_layout(ttnn_concat_32, ttnn_TILE_LAYOUT);  ttnn_concat_32 = None
    ttnn_to_layout_34 = ttnn_decorators_ttnn_to_layout(ttnn_concat_33, ttnn_TILE_LAYOUT);  ttnn_concat_33 = None
    ttnn_linear_161 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_288, ttnn_to_layout_33, bias = ttnn_to_layout_34);  ttnn_experimental_view_288 = ttnn_to_layout_33 = ttnn_to_layout_34 = None
    ttnn_experimental_view_501 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_161, [1, 256, 3072]);  ttnn_linear_161 = None
    ttnn_transformer_split_query_key_value_and_split_heads_16 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_501, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_501 = None
    getitem_99 = ttnn_transformer_split_query_key_value_and_split_heads_16[0]
    getitem_100 = ttnn_transformer_split_query_key_value_and_split_heads_16[1]
    getitem_101 = ttnn_transformer_split_query_key_value_and_split_heads_16[2];  ttnn_transformer_split_query_key_value_and_split_heads_16 = None
    ttnn_matmul_225 = ttnn_decorators_ttnn_matmul(getitem_99, getitem_100);  getitem_99 = getitem_100 = None
    ttnn_transformer_attention_softmax__40 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_225, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_225 = None
    ttnn_matmul_226 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__40, getitem_101);  ttnn_transformer_attention_softmax__40 = getitem_101 = None
    ttnn_transformer_concatenate_heads_16 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_226);  ttnn_matmul_226 = None
    ttnn_from_torch_271 = ttnn_decorators_ttnn_from_torch(arg267_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg267_1 = None
    ttnn_from_torch_272 = ttnn_decorators_ttnn_from_torch(arg268_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg268_1 = None
    ttnn_linear_99 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_16, ttnn_from_torch_271, transpose_b = True, bias = ttnn_from_torch_272, activation = None);  ttnn_transformer_concatenate_heads_16 = ttnn_from_torch_271 = ttnn_from_torch_272 = None
    ttnn_experimental_view_301 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_99, [1, 256, 1024]);  ttnn_linear_99 = None
    ttnn_add_196 = ttnn_decorators_ttnn_add(ttnn_experimental_view_301, ttnn_layer_norm_32);  ttnn_experimental_view_301 = ttnn_layer_norm_32 = None
    ttnn_from_torch_273 = ttnn_decorators_ttnn_from_torch(arg269_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg269_1 = None
    ttnn_from_torch_274 = ttnn_decorators_ttnn_from_torch(arg270_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg270_1 = None
    ttnn_layer_norm_33 = ttnn_decorators_ttnn_layer_norm(ttnn_add_196, epsilon = 1e-12, weight = ttnn_from_torch_273, bias = ttnn_from_torch_274);  ttnn_add_196 = ttnn_from_torch_273 = ttnn_from_torch_274 = None
    ttnn_experimental_view_302 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_33, [256, 1024])
    ttnn_from_torch_275 = ttnn_decorators_ttnn_from_torch(arg271_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg271_1 = None
    ttnn_from_torch_276 = ttnn_decorators_ttnn_from_torch(arg272_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg272_1 = None
    ttnn_linear_100 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_302, ttnn_from_torch_275, transpose_b = True, bias = ttnn_from_torch_276, activation = 'gelu');  ttnn_experimental_view_302 = ttnn_from_torch_275 = ttnn_from_torch_276 = None
    ttnn_experimental_view_304 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_100, [256, 4096]);  ttnn_linear_100 = None
    ttnn_from_torch_277 = ttnn_decorators_ttnn_from_torch(arg273_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg273_1 = None
    ttnn_from_torch_278 = ttnn_decorators_ttnn_from_torch(arg274_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg274_1 = None
    ttnn_linear_101 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_304, ttnn_from_torch_277, transpose_b = True, bias = ttnn_from_torch_278, activation = None);  ttnn_experimental_view_304 = ttnn_from_torch_277 = ttnn_from_torch_278 = None
    ttnn_experimental_view_305 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_101, [1, 256, 1024]);  ttnn_linear_101 = None
    ttnn_add_197 = ttnn_decorators_ttnn_add(ttnn_experimental_view_305, ttnn_layer_norm_33);  ttnn_experimental_view_305 = ttnn_layer_norm_33 = None
    ttnn_from_torch_279 = ttnn_decorators_ttnn_from_torch(arg275_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg275_1 = None
    ttnn_from_torch_280 = ttnn_decorators_ttnn_from_torch(arg276_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg276_1 = None
    ttnn_layer_norm_34 = ttnn_decorators_ttnn_layer_norm(ttnn_add_197, epsilon = 1e-12, weight = ttnn_from_torch_279, bias = ttnn_from_torch_280);  ttnn_add_197 = ttnn_from_torch_279 = ttnn_from_torch_280 = None
    ttnn_experimental_view_306 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_34, [256, 1024])
    ttnn_from_torch_281 = ttnn_decorators_ttnn_from_torch(arg277_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg277_1 = None
    ttnn_transpose_220 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_281, -2, -1);  ttnn_from_torch_281 = None
    ttnn_from_torch_282 = ttnn_decorators_ttnn_from_torch(arg279_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg279_1 = None
    ttnn_transpose_221 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_282, -2, -1);  ttnn_from_torch_282 = None
    ttnn_from_torch_283 = ttnn_decorators_ttnn_from_torch(arg281_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg281_1 = None
    ttnn_transpose_222 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_283, -2, -1);  ttnn_from_torch_283 = None
    ttnn_concat_34 = ttnn_decorators_ttnn_concat([ttnn_transpose_220, ttnn_transpose_221, ttnn_transpose_222], -1);  ttnn_transpose_220 = ttnn_transpose_221 = ttnn_transpose_222 = None
    ttnn_from_torch_284 = ttnn_decorators_ttnn_from_torch(arg278_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg278_1 = None
    ttnn_experimental_view_502 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_284, (1, -1));  ttnn_from_torch_284 = None
    ttnn_from_torch_285 = ttnn_decorators_ttnn_from_torch(arg280_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg280_1 = None
    ttnn_experimental_view_503 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_285, (1, -1));  ttnn_from_torch_285 = None
    ttnn_from_torch_286 = ttnn_decorators_ttnn_from_torch(arg282_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg282_1 = None
    ttnn_experimental_view_504 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_286, (1, -1));  ttnn_from_torch_286 = None
    ttnn_concat_35 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_502, ttnn_experimental_view_503, ttnn_experimental_view_504], -1);  ttnn_experimental_view_502 = ttnn_experimental_view_503 = ttnn_experimental_view_504 = None
    ttnn_to_layout_35 = ttnn_decorators_ttnn_to_layout(ttnn_concat_34, ttnn_TILE_LAYOUT);  ttnn_concat_34 = None
    ttnn_to_layout_36 = ttnn_decorators_ttnn_to_layout(ttnn_concat_35, ttnn_TILE_LAYOUT);  ttnn_concat_35 = None
    ttnn_linear_162 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_306, ttnn_to_layout_35, bias = ttnn_to_layout_36);  ttnn_experimental_view_306 = ttnn_to_layout_35 = ttnn_to_layout_36 = None
    ttnn_experimental_view_505 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_162, [1, 256, 3072]);  ttnn_linear_162 = None
    ttnn_transformer_split_query_key_value_and_split_heads_17 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_505, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_505 = None
    getitem_102 = ttnn_transformer_split_query_key_value_and_split_heads_17[0]
    getitem_103 = ttnn_transformer_split_query_key_value_and_split_heads_17[1]
    getitem_104 = ttnn_transformer_split_query_key_value_and_split_heads_17[2];  ttnn_transformer_split_query_key_value_and_split_heads_17 = None
    ttnn_matmul_227 = ttnn_decorators_ttnn_matmul(getitem_102, getitem_103);  getitem_102 = getitem_103 = None
    ttnn_transformer_attention_softmax__41 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_227, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_227 = None
    ttnn_matmul_228 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__41, getitem_104);  ttnn_transformer_attention_softmax__41 = getitem_104 = None
    ttnn_transformer_concatenate_heads_17 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_228);  ttnn_matmul_228 = None
    ttnn_from_torch_287 = ttnn_decorators_ttnn_from_torch(arg283_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg283_1 = None
    ttnn_from_torch_288 = ttnn_decorators_ttnn_from_torch(arg284_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg284_1 = None
    ttnn_linear_105 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_17, ttnn_from_torch_287, transpose_b = True, bias = ttnn_from_torch_288, activation = None);  ttnn_transformer_concatenate_heads_17 = ttnn_from_torch_287 = ttnn_from_torch_288 = None
    ttnn_experimental_view_319 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_105, [1, 256, 1024]);  ttnn_linear_105 = None
    ttnn_add_199 = ttnn_decorators_ttnn_add(ttnn_experimental_view_319, ttnn_layer_norm_34);  ttnn_experimental_view_319 = ttnn_layer_norm_34 = None
    ttnn_from_torch_289 = ttnn_decorators_ttnn_from_torch(arg285_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg285_1 = None
    ttnn_from_torch_290 = ttnn_decorators_ttnn_from_torch(arg286_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg286_1 = None
    ttnn_layer_norm_35 = ttnn_decorators_ttnn_layer_norm(ttnn_add_199, epsilon = 1e-12, weight = ttnn_from_torch_289, bias = ttnn_from_torch_290);  ttnn_add_199 = ttnn_from_torch_289 = ttnn_from_torch_290 = None
    ttnn_experimental_view_320 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_35, [256, 1024])
    ttnn_from_torch_291 = ttnn_decorators_ttnn_from_torch(arg287_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg287_1 = None
    ttnn_from_torch_292 = ttnn_decorators_ttnn_from_torch(arg288_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg288_1 = None
    ttnn_linear_106 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_320, ttnn_from_torch_291, transpose_b = True, bias = ttnn_from_torch_292, activation = 'gelu');  ttnn_experimental_view_320 = ttnn_from_torch_291 = ttnn_from_torch_292 = None
    ttnn_experimental_view_322 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_106, [256, 4096]);  ttnn_linear_106 = None
    ttnn_from_torch_293 = ttnn_decorators_ttnn_from_torch(arg289_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg289_1 = None
    ttnn_from_torch_294 = ttnn_decorators_ttnn_from_torch(arg290_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg290_1 = None
    ttnn_linear_107 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_322, ttnn_from_torch_293, transpose_b = True, bias = ttnn_from_torch_294, activation = None);  ttnn_experimental_view_322 = ttnn_from_torch_293 = ttnn_from_torch_294 = None
    ttnn_experimental_view_323 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_107, [1, 256, 1024]);  ttnn_linear_107 = None
    ttnn_add_200 = ttnn_decorators_ttnn_add(ttnn_experimental_view_323, ttnn_layer_norm_35);  ttnn_experimental_view_323 = ttnn_layer_norm_35 = None
    ttnn_from_torch_295 = ttnn_decorators_ttnn_from_torch(arg291_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg291_1 = None
    ttnn_from_torch_296 = ttnn_decorators_ttnn_from_torch(arg292_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg292_1 = None
    ttnn_layer_norm_36 = ttnn_decorators_ttnn_layer_norm(ttnn_add_200, epsilon = 1e-12, weight = ttnn_from_torch_295, bias = ttnn_from_torch_296);  ttnn_add_200 = ttnn_from_torch_295 = ttnn_from_torch_296 = None
    ttnn_experimental_view_324 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_36, [256, 1024])
    ttnn_from_torch_297 = ttnn_decorators_ttnn_from_torch(arg293_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg293_1 = None
    ttnn_transpose_223 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_297, -2, -1);  ttnn_from_torch_297 = None
    ttnn_from_torch_298 = ttnn_decorators_ttnn_from_torch(arg295_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg295_1 = None
    ttnn_transpose_224 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_298, -2, -1);  ttnn_from_torch_298 = None
    ttnn_from_torch_299 = ttnn_decorators_ttnn_from_torch(arg297_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg297_1 = None
    ttnn_transpose_225 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_299, -2, -1);  ttnn_from_torch_299 = None
    ttnn_concat_36 = ttnn_decorators_ttnn_concat([ttnn_transpose_223, ttnn_transpose_224, ttnn_transpose_225], -1);  ttnn_transpose_223 = ttnn_transpose_224 = ttnn_transpose_225 = None
    ttnn_from_torch_300 = ttnn_decorators_ttnn_from_torch(arg294_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg294_1 = None
    ttnn_experimental_view_506 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_300, (1, -1));  ttnn_from_torch_300 = None
    ttnn_from_torch_301 = ttnn_decorators_ttnn_from_torch(arg296_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg296_1 = None
    ttnn_experimental_view_507 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_301, (1, -1));  ttnn_from_torch_301 = None
    ttnn_from_torch_302 = ttnn_decorators_ttnn_from_torch(arg298_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg298_1 = None
    ttnn_experimental_view_508 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_302, (1, -1));  ttnn_from_torch_302 = None
    ttnn_concat_37 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_506, ttnn_experimental_view_507, ttnn_experimental_view_508], -1);  ttnn_experimental_view_506 = ttnn_experimental_view_507 = ttnn_experimental_view_508 = None
    ttnn_to_layout_37 = ttnn_decorators_ttnn_to_layout(ttnn_concat_36, ttnn_TILE_LAYOUT);  ttnn_concat_36 = None
    ttnn_to_layout_38 = ttnn_decorators_ttnn_to_layout(ttnn_concat_37, ttnn_TILE_LAYOUT);  ttnn_concat_37 = None
    ttnn_linear_163 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_324, ttnn_to_layout_37, bias = ttnn_to_layout_38);  ttnn_experimental_view_324 = ttnn_to_layout_37 = ttnn_to_layout_38 = None
    ttnn_experimental_view_509 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_163, [1, 256, 3072]);  ttnn_linear_163 = None
    ttnn_transformer_split_query_key_value_and_split_heads_18 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_509, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_509 = None
    getitem_105 = ttnn_transformer_split_query_key_value_and_split_heads_18[0]
    getitem_106 = ttnn_transformer_split_query_key_value_and_split_heads_18[1]
    getitem_107 = ttnn_transformer_split_query_key_value_and_split_heads_18[2];  ttnn_transformer_split_query_key_value_and_split_heads_18 = None
    ttnn_matmul_229 = ttnn_decorators_ttnn_matmul(getitem_105, getitem_106);  getitem_105 = getitem_106 = None
    ttnn_transformer_attention_softmax__42 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_229, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_229 = None
    ttnn_matmul_230 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__42, getitem_107);  ttnn_transformer_attention_softmax__42 = getitem_107 = None
    ttnn_transformer_concatenate_heads_18 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_230);  ttnn_matmul_230 = None
    ttnn_from_torch_303 = ttnn_decorators_ttnn_from_torch(arg299_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg299_1 = None
    ttnn_from_torch_304 = ttnn_decorators_ttnn_from_torch(arg300_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg300_1 = None
    ttnn_linear_111 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_18, ttnn_from_torch_303, transpose_b = True, bias = ttnn_from_torch_304, activation = None);  ttnn_transformer_concatenate_heads_18 = ttnn_from_torch_303 = ttnn_from_torch_304 = None
    ttnn_experimental_view_337 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_111, [1, 256, 1024]);  ttnn_linear_111 = None
    ttnn_add_202 = ttnn_decorators_ttnn_add(ttnn_experimental_view_337, ttnn_layer_norm_36);  ttnn_experimental_view_337 = ttnn_layer_norm_36 = None
    ttnn_from_torch_305 = ttnn_decorators_ttnn_from_torch(arg301_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg301_1 = None
    ttnn_from_torch_306 = ttnn_decorators_ttnn_from_torch(arg302_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg302_1 = None
    ttnn_layer_norm_37 = ttnn_decorators_ttnn_layer_norm(ttnn_add_202, epsilon = 1e-12, weight = ttnn_from_torch_305, bias = ttnn_from_torch_306);  ttnn_add_202 = ttnn_from_torch_305 = ttnn_from_torch_306 = None
    ttnn_experimental_view_338 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_37, [256, 1024])
    ttnn_from_torch_307 = ttnn_decorators_ttnn_from_torch(arg303_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg303_1 = None
    ttnn_from_torch_308 = ttnn_decorators_ttnn_from_torch(arg304_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg304_1 = None
    ttnn_linear_112 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_338, ttnn_from_torch_307, transpose_b = True, bias = ttnn_from_torch_308, activation = 'gelu');  ttnn_experimental_view_338 = ttnn_from_torch_307 = ttnn_from_torch_308 = None
    ttnn_experimental_view_340 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_112, [256, 4096]);  ttnn_linear_112 = None
    ttnn_from_torch_309 = ttnn_decorators_ttnn_from_torch(arg305_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg305_1 = None
    ttnn_from_torch_310 = ttnn_decorators_ttnn_from_torch(arg306_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg306_1 = None
    ttnn_linear_113 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_340, ttnn_from_torch_309, transpose_b = True, bias = ttnn_from_torch_310, activation = None);  ttnn_experimental_view_340 = ttnn_from_torch_309 = ttnn_from_torch_310 = None
    ttnn_experimental_view_341 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_113, [1, 256, 1024]);  ttnn_linear_113 = None
    ttnn_add_203 = ttnn_decorators_ttnn_add(ttnn_experimental_view_341, ttnn_layer_norm_37);  ttnn_experimental_view_341 = ttnn_layer_norm_37 = None
    ttnn_from_torch_311 = ttnn_decorators_ttnn_from_torch(arg307_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg307_1 = None
    ttnn_from_torch_312 = ttnn_decorators_ttnn_from_torch(arg308_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg308_1 = None
    ttnn_layer_norm_38 = ttnn_decorators_ttnn_layer_norm(ttnn_add_203, epsilon = 1e-12, weight = ttnn_from_torch_311, bias = ttnn_from_torch_312);  ttnn_add_203 = ttnn_from_torch_311 = ttnn_from_torch_312 = None
    ttnn_experimental_view_342 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_38, [256, 1024])
    ttnn_from_torch_313 = ttnn_decorators_ttnn_from_torch(arg309_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg309_1 = None
    ttnn_transpose_226 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_313, -2, -1);  ttnn_from_torch_313 = None
    ttnn_from_torch_314 = ttnn_decorators_ttnn_from_torch(arg311_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg311_1 = None
    ttnn_transpose_227 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_314, -2, -1);  ttnn_from_torch_314 = None
    ttnn_from_torch_315 = ttnn_decorators_ttnn_from_torch(arg313_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg313_1 = None
    ttnn_transpose_228 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_315, -2, -1);  ttnn_from_torch_315 = None
    ttnn_concat_38 = ttnn_decorators_ttnn_concat([ttnn_transpose_226, ttnn_transpose_227, ttnn_transpose_228], -1);  ttnn_transpose_226 = ttnn_transpose_227 = ttnn_transpose_228 = None
    ttnn_from_torch_316 = ttnn_decorators_ttnn_from_torch(arg310_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg310_1 = None
    ttnn_experimental_view_510 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_316, (1, -1));  ttnn_from_torch_316 = None
    ttnn_from_torch_317 = ttnn_decorators_ttnn_from_torch(arg312_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg312_1 = None
    ttnn_experimental_view_511 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_317, (1, -1));  ttnn_from_torch_317 = None
    ttnn_from_torch_318 = ttnn_decorators_ttnn_from_torch(arg314_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg314_1 = None
    ttnn_experimental_view_512 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_318, (1, -1));  ttnn_from_torch_318 = None
    ttnn_concat_39 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_510, ttnn_experimental_view_511, ttnn_experimental_view_512], -1);  ttnn_experimental_view_510 = ttnn_experimental_view_511 = ttnn_experimental_view_512 = None
    ttnn_to_layout_39 = ttnn_decorators_ttnn_to_layout(ttnn_concat_38, ttnn_TILE_LAYOUT);  ttnn_concat_38 = None
    ttnn_to_layout_40 = ttnn_decorators_ttnn_to_layout(ttnn_concat_39, ttnn_TILE_LAYOUT);  ttnn_concat_39 = None
    ttnn_linear_164 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_342, ttnn_to_layout_39, bias = ttnn_to_layout_40);  ttnn_experimental_view_342 = ttnn_to_layout_39 = ttnn_to_layout_40 = None
    ttnn_experimental_view_513 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_164, [1, 256, 3072]);  ttnn_linear_164 = None
    ttnn_transformer_split_query_key_value_and_split_heads_19 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_513, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_513 = None
    getitem_108 = ttnn_transformer_split_query_key_value_and_split_heads_19[0]
    getitem_109 = ttnn_transformer_split_query_key_value_and_split_heads_19[1]
    getitem_110 = ttnn_transformer_split_query_key_value_and_split_heads_19[2];  ttnn_transformer_split_query_key_value_and_split_heads_19 = None
    ttnn_matmul_231 = ttnn_decorators_ttnn_matmul(getitem_108, getitem_109);  getitem_108 = getitem_109 = None
    ttnn_transformer_attention_softmax__43 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_231, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_231 = None
    ttnn_matmul_232 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__43, getitem_110);  ttnn_transformer_attention_softmax__43 = getitem_110 = None
    ttnn_transformer_concatenate_heads_19 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_232);  ttnn_matmul_232 = None
    ttnn_from_torch_319 = ttnn_decorators_ttnn_from_torch(arg315_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg315_1 = None
    ttnn_from_torch_320 = ttnn_decorators_ttnn_from_torch(arg316_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg316_1 = None
    ttnn_linear_117 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_19, ttnn_from_torch_319, transpose_b = True, bias = ttnn_from_torch_320, activation = None);  ttnn_transformer_concatenate_heads_19 = ttnn_from_torch_319 = ttnn_from_torch_320 = None
    ttnn_experimental_view_355 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_117, [1, 256, 1024]);  ttnn_linear_117 = None
    ttnn_add_205 = ttnn_decorators_ttnn_add(ttnn_experimental_view_355, ttnn_layer_norm_38);  ttnn_experimental_view_355 = ttnn_layer_norm_38 = None
    ttnn_from_torch_321 = ttnn_decorators_ttnn_from_torch(arg317_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg317_1 = None
    ttnn_from_torch_322 = ttnn_decorators_ttnn_from_torch(arg318_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg318_1 = None
    ttnn_layer_norm_39 = ttnn_decorators_ttnn_layer_norm(ttnn_add_205, epsilon = 1e-12, weight = ttnn_from_torch_321, bias = ttnn_from_torch_322);  ttnn_add_205 = ttnn_from_torch_321 = ttnn_from_torch_322 = None
    ttnn_experimental_view_356 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_39, [256, 1024])
    ttnn_from_torch_323 = ttnn_decorators_ttnn_from_torch(arg319_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg319_1 = None
    ttnn_from_torch_324 = ttnn_decorators_ttnn_from_torch(arg320_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg320_1 = None
    ttnn_linear_118 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_356, ttnn_from_torch_323, transpose_b = True, bias = ttnn_from_torch_324, activation = 'gelu');  ttnn_experimental_view_356 = ttnn_from_torch_323 = ttnn_from_torch_324 = None
    ttnn_experimental_view_358 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_118, [256, 4096]);  ttnn_linear_118 = None
    ttnn_from_torch_325 = ttnn_decorators_ttnn_from_torch(arg321_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg321_1 = None
    ttnn_from_torch_326 = ttnn_decorators_ttnn_from_torch(arg322_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg322_1 = None
    ttnn_linear_119 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_358, ttnn_from_torch_325, transpose_b = True, bias = ttnn_from_torch_326, activation = None);  ttnn_experimental_view_358 = ttnn_from_torch_325 = ttnn_from_torch_326 = None
    ttnn_experimental_view_359 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_119, [1, 256, 1024]);  ttnn_linear_119 = None
    ttnn_add_206 = ttnn_decorators_ttnn_add(ttnn_experimental_view_359, ttnn_layer_norm_39);  ttnn_experimental_view_359 = ttnn_layer_norm_39 = None
    ttnn_from_torch_327 = ttnn_decorators_ttnn_from_torch(arg323_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg323_1 = None
    ttnn_from_torch_328 = ttnn_decorators_ttnn_from_torch(arg324_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg324_1 = None
    ttnn_layer_norm_40 = ttnn_decorators_ttnn_layer_norm(ttnn_add_206, epsilon = 1e-12, weight = ttnn_from_torch_327, bias = ttnn_from_torch_328);  ttnn_add_206 = ttnn_from_torch_327 = ttnn_from_torch_328 = None
    ttnn_experimental_view_360 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_40, [256, 1024])
    ttnn_from_torch_329 = ttnn_decorators_ttnn_from_torch(arg325_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg325_1 = None
    ttnn_transpose_229 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_329, -2, -1);  ttnn_from_torch_329 = None
    ttnn_from_torch_330 = ttnn_decorators_ttnn_from_torch(arg327_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg327_1 = None
    ttnn_transpose_230 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_330, -2, -1);  ttnn_from_torch_330 = None
    ttnn_from_torch_331 = ttnn_decorators_ttnn_from_torch(arg329_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg329_1 = None
    ttnn_transpose_231 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_331, -2, -1);  ttnn_from_torch_331 = None
    ttnn_concat_40 = ttnn_decorators_ttnn_concat([ttnn_transpose_229, ttnn_transpose_230, ttnn_transpose_231], -1);  ttnn_transpose_229 = ttnn_transpose_230 = ttnn_transpose_231 = None
    ttnn_from_torch_332 = ttnn_decorators_ttnn_from_torch(arg326_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg326_1 = None
    ttnn_experimental_view_514 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_332, (1, -1));  ttnn_from_torch_332 = None
    ttnn_from_torch_333 = ttnn_decorators_ttnn_from_torch(arg328_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg328_1 = None
    ttnn_experimental_view_515 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_333, (1, -1));  ttnn_from_torch_333 = None
    ttnn_from_torch_334 = ttnn_decorators_ttnn_from_torch(arg330_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg330_1 = None
    ttnn_experimental_view_516 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_334, (1, -1));  ttnn_from_torch_334 = None
    ttnn_concat_41 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_514, ttnn_experimental_view_515, ttnn_experimental_view_516], -1);  ttnn_experimental_view_514 = ttnn_experimental_view_515 = ttnn_experimental_view_516 = None
    ttnn_to_layout_41 = ttnn_decorators_ttnn_to_layout(ttnn_concat_40, ttnn_TILE_LAYOUT);  ttnn_concat_40 = None
    ttnn_to_layout_42 = ttnn_decorators_ttnn_to_layout(ttnn_concat_41, ttnn_TILE_LAYOUT);  ttnn_concat_41 = None
    ttnn_linear_165 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_360, ttnn_to_layout_41, bias = ttnn_to_layout_42);  ttnn_experimental_view_360 = ttnn_to_layout_41 = ttnn_to_layout_42 = None
    ttnn_experimental_view_517 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_165, [1, 256, 3072]);  ttnn_linear_165 = None
    ttnn_transformer_split_query_key_value_and_split_heads_20 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_517, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_517 = None
    getitem_111 = ttnn_transformer_split_query_key_value_and_split_heads_20[0]
    getitem_112 = ttnn_transformer_split_query_key_value_and_split_heads_20[1]
    getitem_113 = ttnn_transformer_split_query_key_value_and_split_heads_20[2];  ttnn_transformer_split_query_key_value_and_split_heads_20 = None
    ttnn_matmul_233 = ttnn_decorators_ttnn_matmul(getitem_111, getitem_112);  getitem_111 = getitem_112 = None
    ttnn_transformer_attention_softmax__44 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_233, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_233 = None
    ttnn_matmul_234 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__44, getitem_113);  ttnn_transformer_attention_softmax__44 = getitem_113 = None
    ttnn_transformer_concatenate_heads_20 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_234);  ttnn_matmul_234 = None
    ttnn_from_torch_335 = ttnn_decorators_ttnn_from_torch(arg331_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg331_1 = None
    ttnn_from_torch_336 = ttnn_decorators_ttnn_from_torch(arg332_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg332_1 = None
    ttnn_linear_123 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_20, ttnn_from_torch_335, transpose_b = True, bias = ttnn_from_torch_336, activation = None);  ttnn_transformer_concatenate_heads_20 = ttnn_from_torch_335 = ttnn_from_torch_336 = None
    ttnn_experimental_view_373 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_123, [1, 256, 1024]);  ttnn_linear_123 = None
    ttnn_add_208 = ttnn_decorators_ttnn_add(ttnn_experimental_view_373, ttnn_layer_norm_40);  ttnn_experimental_view_373 = ttnn_layer_norm_40 = None
    ttnn_from_torch_337 = ttnn_decorators_ttnn_from_torch(arg333_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg333_1 = None
    ttnn_from_torch_338 = ttnn_decorators_ttnn_from_torch(arg334_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg334_1 = None
    ttnn_layer_norm_41 = ttnn_decorators_ttnn_layer_norm(ttnn_add_208, epsilon = 1e-12, weight = ttnn_from_torch_337, bias = ttnn_from_torch_338);  ttnn_add_208 = ttnn_from_torch_337 = ttnn_from_torch_338 = None
    ttnn_experimental_view_374 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_41, [256, 1024])
    ttnn_from_torch_339 = ttnn_decorators_ttnn_from_torch(arg335_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg335_1 = None
    ttnn_from_torch_340 = ttnn_decorators_ttnn_from_torch(arg336_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg336_1 = None
    ttnn_linear_124 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_374, ttnn_from_torch_339, transpose_b = True, bias = ttnn_from_torch_340, activation = 'gelu');  ttnn_experimental_view_374 = ttnn_from_torch_339 = ttnn_from_torch_340 = None
    ttnn_experimental_view_376 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_124, [256, 4096]);  ttnn_linear_124 = None
    ttnn_from_torch_341 = ttnn_decorators_ttnn_from_torch(arg337_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg337_1 = None
    ttnn_from_torch_342 = ttnn_decorators_ttnn_from_torch(arg338_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg338_1 = None
    ttnn_linear_125 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_376, ttnn_from_torch_341, transpose_b = True, bias = ttnn_from_torch_342, activation = None);  ttnn_experimental_view_376 = ttnn_from_torch_341 = ttnn_from_torch_342 = None
    ttnn_experimental_view_377 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_125, [1, 256, 1024]);  ttnn_linear_125 = None
    ttnn_add_209 = ttnn_decorators_ttnn_add(ttnn_experimental_view_377, ttnn_layer_norm_41);  ttnn_experimental_view_377 = ttnn_layer_norm_41 = None
    ttnn_from_torch_343 = ttnn_decorators_ttnn_from_torch(arg339_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg339_1 = None
    ttnn_from_torch_344 = ttnn_decorators_ttnn_from_torch(arg340_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg340_1 = None
    ttnn_layer_norm_42 = ttnn_decorators_ttnn_layer_norm(ttnn_add_209, epsilon = 1e-12, weight = ttnn_from_torch_343, bias = ttnn_from_torch_344);  ttnn_add_209 = ttnn_from_torch_343 = ttnn_from_torch_344 = None
    ttnn_experimental_view_378 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_42, [256, 1024])
    ttnn_from_torch_345 = ttnn_decorators_ttnn_from_torch(arg341_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg341_1 = None
    ttnn_transpose_232 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_345, -2, -1);  ttnn_from_torch_345 = None
    ttnn_from_torch_346 = ttnn_decorators_ttnn_from_torch(arg343_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg343_1 = None
    ttnn_transpose_233 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_346, -2, -1);  ttnn_from_torch_346 = None
    ttnn_from_torch_347 = ttnn_decorators_ttnn_from_torch(arg345_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg345_1 = None
    ttnn_transpose_234 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_347, -2, -1);  ttnn_from_torch_347 = None
    ttnn_concat_42 = ttnn_decorators_ttnn_concat([ttnn_transpose_232, ttnn_transpose_233, ttnn_transpose_234], -1);  ttnn_transpose_232 = ttnn_transpose_233 = ttnn_transpose_234 = None
    ttnn_from_torch_348 = ttnn_decorators_ttnn_from_torch(arg342_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg342_1 = None
    ttnn_experimental_view_518 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_348, (1, -1));  ttnn_from_torch_348 = None
    ttnn_from_torch_349 = ttnn_decorators_ttnn_from_torch(arg344_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg344_1 = None
    ttnn_experimental_view_519 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_349, (1, -1));  ttnn_from_torch_349 = None
    ttnn_from_torch_350 = ttnn_decorators_ttnn_from_torch(arg346_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg346_1 = None
    ttnn_experimental_view_520 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_350, (1, -1));  ttnn_from_torch_350 = None
    ttnn_concat_43 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_518, ttnn_experimental_view_519, ttnn_experimental_view_520], -1);  ttnn_experimental_view_518 = ttnn_experimental_view_519 = ttnn_experimental_view_520 = None
    ttnn_to_layout_43 = ttnn_decorators_ttnn_to_layout(ttnn_concat_42, ttnn_TILE_LAYOUT);  ttnn_concat_42 = None
    ttnn_to_layout_44 = ttnn_decorators_ttnn_to_layout(ttnn_concat_43, ttnn_TILE_LAYOUT);  ttnn_concat_43 = None
    ttnn_linear_166 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_378, ttnn_to_layout_43, bias = ttnn_to_layout_44);  ttnn_experimental_view_378 = ttnn_to_layout_43 = ttnn_to_layout_44 = None
    ttnn_experimental_view_521 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_166, [1, 256, 3072]);  ttnn_linear_166 = None
    ttnn_transformer_split_query_key_value_and_split_heads_21 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_521, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_521 = None
    getitem_114 = ttnn_transformer_split_query_key_value_and_split_heads_21[0]
    getitem_115 = ttnn_transformer_split_query_key_value_and_split_heads_21[1]
    getitem_116 = ttnn_transformer_split_query_key_value_and_split_heads_21[2];  ttnn_transformer_split_query_key_value_and_split_heads_21 = None
    ttnn_matmul_235 = ttnn_decorators_ttnn_matmul(getitem_114, getitem_115);  getitem_114 = getitem_115 = None
    ttnn_transformer_attention_softmax__45 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_235, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_235 = None
    ttnn_matmul_236 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__45, getitem_116);  ttnn_transformer_attention_softmax__45 = getitem_116 = None
    ttnn_transformer_concatenate_heads_21 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_236);  ttnn_matmul_236 = None
    ttnn_from_torch_351 = ttnn_decorators_ttnn_from_torch(arg347_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg347_1 = None
    ttnn_from_torch_352 = ttnn_decorators_ttnn_from_torch(arg348_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg348_1 = None
    ttnn_linear_129 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_21, ttnn_from_torch_351, transpose_b = True, bias = ttnn_from_torch_352, activation = None);  ttnn_transformer_concatenate_heads_21 = ttnn_from_torch_351 = ttnn_from_torch_352 = None
    ttnn_experimental_view_391 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_129, [1, 256, 1024]);  ttnn_linear_129 = None
    ttnn_add_211 = ttnn_decorators_ttnn_add(ttnn_experimental_view_391, ttnn_layer_norm_42);  ttnn_experimental_view_391 = ttnn_layer_norm_42 = None
    ttnn_from_torch_353 = ttnn_decorators_ttnn_from_torch(arg349_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg349_1 = None
    ttnn_from_torch_354 = ttnn_decorators_ttnn_from_torch(arg350_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg350_1 = None
    ttnn_layer_norm_43 = ttnn_decorators_ttnn_layer_norm(ttnn_add_211, epsilon = 1e-12, weight = ttnn_from_torch_353, bias = ttnn_from_torch_354);  ttnn_add_211 = ttnn_from_torch_353 = ttnn_from_torch_354 = None
    ttnn_experimental_view_392 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_43, [256, 1024])
    ttnn_from_torch_355 = ttnn_decorators_ttnn_from_torch(arg351_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg351_1 = None
    ttnn_from_torch_356 = ttnn_decorators_ttnn_from_torch(arg352_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg352_1 = None
    ttnn_linear_130 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_392, ttnn_from_torch_355, transpose_b = True, bias = ttnn_from_torch_356, activation = 'gelu');  ttnn_experimental_view_392 = ttnn_from_torch_355 = ttnn_from_torch_356 = None
    ttnn_experimental_view_394 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_130, [256, 4096]);  ttnn_linear_130 = None
    ttnn_from_torch_357 = ttnn_decorators_ttnn_from_torch(arg353_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg353_1 = None
    ttnn_from_torch_358 = ttnn_decorators_ttnn_from_torch(arg354_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg354_1 = None
    ttnn_linear_131 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_394, ttnn_from_torch_357, transpose_b = True, bias = ttnn_from_torch_358, activation = None);  ttnn_experimental_view_394 = ttnn_from_torch_357 = ttnn_from_torch_358 = None
    ttnn_experimental_view_395 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_131, [1, 256, 1024]);  ttnn_linear_131 = None
    ttnn_add_212 = ttnn_decorators_ttnn_add(ttnn_experimental_view_395, ttnn_layer_norm_43);  ttnn_experimental_view_395 = ttnn_layer_norm_43 = None
    ttnn_from_torch_359 = ttnn_decorators_ttnn_from_torch(arg355_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg355_1 = None
    ttnn_from_torch_360 = ttnn_decorators_ttnn_from_torch(arg356_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg356_1 = None
    ttnn_layer_norm_44 = ttnn_decorators_ttnn_layer_norm(ttnn_add_212, epsilon = 1e-12, weight = ttnn_from_torch_359, bias = ttnn_from_torch_360);  ttnn_add_212 = ttnn_from_torch_359 = ttnn_from_torch_360 = None
    ttnn_experimental_view_396 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_44, [256, 1024])
    ttnn_from_torch_361 = ttnn_decorators_ttnn_from_torch(arg357_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg357_1 = None
    ttnn_transpose_235 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_361, -2, -1);  ttnn_from_torch_361 = None
    ttnn_from_torch_362 = ttnn_decorators_ttnn_from_torch(arg359_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg359_1 = None
    ttnn_transpose_236 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_362, -2, -1);  ttnn_from_torch_362 = None
    ttnn_from_torch_363 = ttnn_decorators_ttnn_from_torch(arg361_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg361_1 = None
    ttnn_transpose_237 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_363, -2, -1);  ttnn_from_torch_363 = None
    ttnn_concat_44 = ttnn_decorators_ttnn_concat([ttnn_transpose_235, ttnn_transpose_236, ttnn_transpose_237], -1);  ttnn_transpose_235 = ttnn_transpose_236 = ttnn_transpose_237 = None
    ttnn_from_torch_364 = ttnn_decorators_ttnn_from_torch(arg358_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg358_1 = None
    ttnn_experimental_view_522 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_364, (1, -1));  ttnn_from_torch_364 = None
    ttnn_from_torch_365 = ttnn_decorators_ttnn_from_torch(arg360_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg360_1 = None
    ttnn_experimental_view_523 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_365, (1, -1));  ttnn_from_torch_365 = None
    ttnn_from_torch_366 = ttnn_decorators_ttnn_from_torch(arg362_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg362_1 = None
    ttnn_experimental_view_524 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_366, (1, -1));  ttnn_from_torch_366 = None
    ttnn_concat_45 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_522, ttnn_experimental_view_523, ttnn_experimental_view_524], -1);  ttnn_experimental_view_522 = ttnn_experimental_view_523 = ttnn_experimental_view_524 = None
    ttnn_to_layout_45 = ttnn_decorators_ttnn_to_layout(ttnn_concat_44, ttnn_TILE_LAYOUT);  ttnn_concat_44 = None
    ttnn_to_layout_46 = ttnn_decorators_ttnn_to_layout(ttnn_concat_45, ttnn_TILE_LAYOUT);  ttnn_concat_45 = None
    ttnn_linear_167 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_396, ttnn_to_layout_45, bias = ttnn_to_layout_46);  ttnn_experimental_view_396 = ttnn_to_layout_45 = ttnn_to_layout_46 = None
    ttnn_experimental_view_525 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_167, [1, 256, 3072]);  ttnn_linear_167 = None
    ttnn_transformer_split_query_key_value_and_split_heads_22 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_525, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_525 = None
    getitem_117 = ttnn_transformer_split_query_key_value_and_split_heads_22[0]
    getitem_118 = ttnn_transformer_split_query_key_value_and_split_heads_22[1]
    getitem_119 = ttnn_transformer_split_query_key_value_and_split_heads_22[2];  ttnn_transformer_split_query_key_value_and_split_heads_22 = None
    ttnn_matmul_237 = ttnn_decorators_ttnn_matmul(getitem_117, getitem_118);  getitem_117 = getitem_118 = None
    ttnn_transformer_attention_softmax__46 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_237, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_237 = None
    ttnn_matmul_238 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__46, getitem_119);  ttnn_transformer_attention_softmax__46 = getitem_119 = None
    ttnn_transformer_concatenate_heads_22 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_238);  ttnn_matmul_238 = None
    ttnn_from_torch_367 = ttnn_decorators_ttnn_from_torch(arg363_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg363_1 = None
    ttnn_from_torch_368 = ttnn_decorators_ttnn_from_torch(arg364_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg364_1 = None
    ttnn_linear_135 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_22, ttnn_from_torch_367, transpose_b = True, bias = ttnn_from_torch_368, activation = None);  ttnn_transformer_concatenate_heads_22 = ttnn_from_torch_367 = ttnn_from_torch_368 = None
    ttnn_experimental_view_409 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_135, [1, 256, 1024]);  ttnn_linear_135 = None
    ttnn_add_214 = ttnn_decorators_ttnn_add(ttnn_experimental_view_409, ttnn_layer_norm_44);  ttnn_experimental_view_409 = ttnn_layer_norm_44 = None
    ttnn_from_torch_369 = ttnn_decorators_ttnn_from_torch(arg365_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg365_1 = None
    ttnn_from_torch_370 = ttnn_decorators_ttnn_from_torch(arg366_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg366_1 = None
    ttnn_layer_norm_45 = ttnn_decorators_ttnn_layer_norm(ttnn_add_214, epsilon = 1e-12, weight = ttnn_from_torch_369, bias = ttnn_from_torch_370);  ttnn_add_214 = ttnn_from_torch_369 = ttnn_from_torch_370 = None
    ttnn_experimental_view_410 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_45, [256, 1024])
    ttnn_from_torch_371 = ttnn_decorators_ttnn_from_torch(arg367_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg367_1 = None
    ttnn_from_torch_372 = ttnn_decorators_ttnn_from_torch(arg368_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg368_1 = None
    ttnn_linear_136 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_410, ttnn_from_torch_371, transpose_b = True, bias = ttnn_from_torch_372, activation = 'gelu');  ttnn_experimental_view_410 = ttnn_from_torch_371 = ttnn_from_torch_372 = None
    ttnn_experimental_view_412 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_136, [256, 4096]);  ttnn_linear_136 = None
    ttnn_from_torch_373 = ttnn_decorators_ttnn_from_torch(arg369_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg369_1 = None
    ttnn_from_torch_374 = ttnn_decorators_ttnn_from_torch(arg370_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg370_1 = None
    ttnn_linear_137 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_412, ttnn_from_torch_373, transpose_b = True, bias = ttnn_from_torch_374, activation = None);  ttnn_experimental_view_412 = ttnn_from_torch_373 = ttnn_from_torch_374 = None
    ttnn_experimental_view_413 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_137, [1, 256, 1024]);  ttnn_linear_137 = None
    ttnn_add_215 = ttnn_decorators_ttnn_add(ttnn_experimental_view_413, ttnn_layer_norm_45);  ttnn_experimental_view_413 = ttnn_layer_norm_45 = None
    ttnn_from_torch_375 = ttnn_decorators_ttnn_from_torch(arg371_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg371_1 = None
    ttnn_from_torch_376 = ttnn_decorators_ttnn_from_torch(arg372_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg372_1 = None
    ttnn_layer_norm_46 = ttnn_decorators_ttnn_layer_norm(ttnn_add_215, epsilon = 1e-12, weight = ttnn_from_torch_375, bias = ttnn_from_torch_376);  ttnn_add_215 = ttnn_from_torch_375 = ttnn_from_torch_376 = None
    ttnn_experimental_view_414 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_46, [256, 1024])
    ttnn_from_torch_377 = ttnn_decorators_ttnn_from_torch(arg373_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg373_1 = None
    ttnn_transpose_238 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_377, -2, -1);  ttnn_from_torch_377 = None
    ttnn_from_torch_378 = ttnn_decorators_ttnn_from_torch(arg375_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg375_1 = None
    ttnn_transpose_239 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_378, -2, -1);  ttnn_from_torch_378 = None
    ttnn_from_torch_379 = ttnn_decorators_ttnn_from_torch(arg377_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg377_1 = None
    ttnn_transpose_240 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_379, -2, -1);  ttnn_from_torch_379 = None
    ttnn_concat_46 = ttnn_decorators_ttnn_concat([ttnn_transpose_238, ttnn_transpose_239, ttnn_transpose_240], -1);  ttnn_transpose_238 = ttnn_transpose_239 = ttnn_transpose_240 = None
    ttnn_from_torch_380 = ttnn_decorators_ttnn_from_torch(arg374_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg374_1 = None
    ttnn_experimental_view_526 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_380, (1, -1));  ttnn_from_torch_380 = None
    ttnn_from_torch_381 = ttnn_decorators_ttnn_from_torch(arg376_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg376_1 = None
    ttnn_experimental_view_527 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_381, (1, -1));  ttnn_from_torch_381 = None
    ttnn_from_torch_382 = ttnn_decorators_ttnn_from_torch(arg378_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg378_1 = None
    ttnn_experimental_view_528 = ttnn_decorators_ttnn_experimental_view(ttnn_from_torch_382, (1, -1));  ttnn_from_torch_382 = None
    ttnn_concat_47 = ttnn_decorators_ttnn_concat([ttnn_experimental_view_526, ttnn_experimental_view_527, ttnn_experimental_view_528], -1);  ttnn_experimental_view_526 = ttnn_experimental_view_527 = ttnn_experimental_view_528 = None
    ttnn_to_layout_47 = ttnn_decorators_ttnn_to_layout(ttnn_concat_46, ttnn_TILE_LAYOUT);  ttnn_concat_46 = None
    ttnn_to_layout_48 = ttnn_decorators_ttnn_to_layout(ttnn_concat_47, ttnn_TILE_LAYOUT);  ttnn_concat_47 = None
    ttnn_linear_168 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_414, ttnn_to_layout_47, bias = ttnn_to_layout_48);  ttnn_experimental_view_414 = ttnn_to_layout_47 = ttnn_to_layout_48 = None
    ttnn_experimental_view_529 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_168, [1, 256, 3072]);  ttnn_linear_168 = None
    ttnn_transformer_split_query_key_value_and_split_heads_23 = ttnn_decorators_ttnn_transformer_split_query_key_value_and_split_heads(ttnn_experimental_view_529, None, num_heads = 16, transpose_key = True);  ttnn_experimental_view_529 = None
    getitem_120 = ttnn_transformer_split_query_key_value_and_split_heads_23[0]
    getitem_121 = ttnn_transformer_split_query_key_value_and_split_heads_23[1]
    getitem_122 = ttnn_transformer_split_query_key_value_and_split_heads_23[2];  ttnn_transformer_split_query_key_value_and_split_heads_23 = None
    ttnn_matmul_239 = ttnn_decorators_ttnn_matmul(getitem_120, getitem_121);  getitem_120 = getitem_121 = None
    ttnn_transformer_attention_softmax__47 = ttnn_decorators_ttnn_transformer_attention_softmax_(ttnn_matmul_239, attention_mask = ttnn_multiply, head_size = 64);  ttnn_matmul_239 = ttnn_multiply = None
    ttnn_matmul_240 = ttnn_decorators_ttnn_matmul(ttnn_transformer_attention_softmax__47, getitem_122);  ttnn_transformer_attention_softmax__47 = getitem_122 = None
    ttnn_transformer_concatenate_heads_23 = ttnn_decorators_ttnn_transformer_concatenate_heads(ttnn_matmul_240);  ttnn_matmul_240 = None
    ttnn_from_torch_383 = ttnn_decorators_ttnn_from_torch(arg379_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg379_1 = None
    ttnn_from_torch_384 = ttnn_decorators_ttnn_from_torch(arg380_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg380_1 = None
    ttnn_linear_141 = ttnn_decorators_ttnn_linear(ttnn_transformer_concatenate_heads_23, ttnn_from_torch_383, transpose_b = True, bias = ttnn_from_torch_384, activation = None);  ttnn_transformer_concatenate_heads_23 = ttnn_from_torch_383 = ttnn_from_torch_384 = None
    ttnn_experimental_view_427 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_141, [1, 256, 1024]);  ttnn_linear_141 = None
    ttnn_add_217 = ttnn_decorators_ttnn_add(ttnn_experimental_view_427, ttnn_layer_norm_46);  ttnn_experimental_view_427 = ttnn_layer_norm_46 = None
    ttnn_from_torch_385 = ttnn_decorators_ttnn_from_torch(arg381_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg381_1 = None
    ttnn_from_torch_386 = ttnn_decorators_ttnn_from_torch(arg382_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg382_1 = None
    ttnn_layer_norm_47 = ttnn_decorators_ttnn_layer_norm(ttnn_add_217, epsilon = 1e-12, weight = ttnn_from_torch_385, bias = ttnn_from_torch_386);  ttnn_add_217 = ttnn_from_torch_385 = ttnn_from_torch_386 = None
    ttnn_experimental_view_428 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_47, [256, 1024])
    ttnn_from_torch_387 = ttnn_decorators_ttnn_from_torch(arg383_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg383_1 = None
    ttnn_from_torch_388 = ttnn_decorators_ttnn_from_torch(arg384_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg384_1 = None
    ttnn_linear_142 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_428, ttnn_from_torch_387, transpose_b = True, bias = ttnn_from_torch_388, activation = 'gelu');  ttnn_experimental_view_428 = ttnn_from_torch_387 = ttnn_from_torch_388 = None
    ttnn_experimental_view_430 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_142, [256, 4096]);  ttnn_linear_142 = None
    ttnn_from_torch_389 = ttnn_decorators_ttnn_from_torch(arg385_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg385_1 = None
    ttnn_from_torch_390 = ttnn_decorators_ttnn_from_torch(arg386_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg386_1 = None
    ttnn_linear_143 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_430, ttnn_from_torch_389, transpose_b = True, bias = ttnn_from_torch_390, activation = None);  ttnn_experimental_view_430 = ttnn_from_torch_389 = ttnn_from_torch_390 = None
    ttnn_experimental_view_431 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_143, [1, 256, 1024]);  ttnn_linear_143 = None
    ttnn_add_218 = ttnn_decorators_ttnn_add(ttnn_experimental_view_431, ttnn_layer_norm_47);  ttnn_experimental_view_431 = ttnn_layer_norm_47 = None
    ttnn_from_torch_391 = ttnn_decorators_ttnn_from_torch(arg387_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg387_1 = None
    ttnn_from_torch_392 = ttnn_decorators_ttnn_from_torch(arg388_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg388_1 = None
    ttnn_layer_norm_48 = ttnn_decorators_ttnn_layer_norm(ttnn_add_218, epsilon = 1e-12, weight = ttnn_from_torch_391, bias = ttnn_from_torch_392);  ttnn_add_218 = ttnn_from_torch_391 = ttnn_from_torch_392 = None
    ttnn_experimental_view_432 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_48, [256, 1024]);  ttnn_layer_norm_48 = None
    ttnn_from_torch_393 = ttnn_decorators_ttnn_from_torch(arg389_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg389_1 = None
    ttnn_from_torch_394 = ttnn_decorators_ttnn_from_torch(arg390_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg390_1 = None
    ttnn_linear_144 = ttnn_decorators_ttnn_linear(ttnn_experimental_view_432, ttnn_from_torch_393, transpose_b = True, bias = ttnn_from_torch_394, activation = None);  ttnn_experimental_view_432 = ttnn_from_torch_393 = ttnn_from_torch_394 = None
    ttnn_experimental_view_433 = ttnn_decorators_ttnn_experimental_view(ttnn_linear_144, [1, 256, 2]);  ttnn_linear_144 = None
    ttnn_to_layout_49 = ttnn_decorators_ttnn_to_layout(ttnn_experimental_view_433, ttnn_ROW_MAJOR_LAYOUT);  ttnn_experimental_view_433 = None
    ttnn_split = ttnn_decorators_ttnn_split(ttnn_to_layout_49, 1, 2);  ttnn_to_layout_49 = None
    getitem_49 = ttnn_split[0]
    getitem_50 = ttnn_split[1];  ttnn_split = None
    ttnn_to_layout_50 = ttnn_decorators_ttnn_to_layout(getitem_49, ttnn_TILE_LAYOUT);  getitem_49 = None
    ttnn_squeeze = ttnn_decorators_ttnn_squeeze(ttnn_to_layout_50, -1);  ttnn_to_layout_50 = None
    ttnn_to_layout_51 = ttnn_decorators_ttnn_to_layout(getitem_50, ttnn_TILE_LAYOUT);  getitem_50 = None
    ttnn_squeeze_1 = ttnn_decorators_ttnn_squeeze(ttnn_to_layout_51, -1);  ttnn_to_layout_51 = None
    ttnn_to_torch = ttnn_decorators_ttnn_to_torch(ttnn_squeeze, dtype = torch.bfloat16);  ttnn_squeeze = None
    ttnn_to_torch_1 = ttnn_decorators_ttnn_to_torch(ttnn_squeeze_1, dtype = torch.bfloat16);  ttnn_squeeze_1 = None
    return (ttnn_to_torch, ttnn_to_torch_1)

# Original model we started to optimize
def original(ttnn_Specified_Device, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1):
    ttnn_from_torch = ttnn_decorators_ttnn_from_torch(arg393_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_uint32);  arg393_1 = None
    ttnn_reshape = ttnn_decorators_ttnn_reshape(ttnn_from_torch, [1, 1, 256]);  ttnn_from_torch = None
    ttnn_reshape_1 = ttnn_decorators_ttnn_reshape(ttnn_reshape, [1, 1, 1, 256]);  ttnn_reshape = None
    ttnn_typecast = ttnn_decorators_ttnn_typecast(ttnn_reshape_1, ttnn_bfloat16);  ttnn_reshape_1 = None
    ttnn_rsub = ttnn_decorators_ttnn_rsub(ttnn_typecast, 1.0);  ttnn_typecast = None
    ttnn_multiply = ttnn_decorators_ttnn_multiply(ttnn_rsub, -3.3895313892515355e+38);  ttnn_rsub = None
    ttnn_from_torch_1 = ttnn_decorators_ttnn_from_torch(arg391_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg391_1 = None
    ttnn_slice = ttnn_decorators_ttnn_slice(ttnn_from_torch_1, [0, 0], [1, 256]);  ttnn_from_torch_1 = None
    ttnn_from_torch_2 = ttnn_decorators_ttnn_from_torch(arg392_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg392_1 = None
    ttnn_from_torch_3 = ttnn_decorators_ttnn_from_torch(arg0_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg0_1 = None
    ttnn_embedding = ttnn_decorators_ttnn_embedding(ttnn_from_torch_2, ttnn_from_torch_3, layout = ttnn_TILE_LAYOUT);  ttnn_from_torch_2 = ttnn_from_torch_3 = None
    ttnn_from_torch_4 = ttnn_decorators_ttnn_from_torch(arg394_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32);  arg394_1 = None
    ttnn_from_torch_5 = ttnn_decorators_ttnn_from_torch(arg1_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg1_1 = None
    ttnn_embedding_1 = ttnn_decorators_ttnn_embedding(ttnn_from_torch_4, ttnn_from_torch_5, layout = ttnn_TILE_LAYOUT);  ttnn_from_torch_4 = ttnn_from_torch_5 = None
    ttnn_add_145 = ttnn_decorators_ttnn_add(ttnn_embedding, ttnn_embedding_1);  ttnn_embedding = ttnn_embedding_1 = None
    ttnn_to_layout = ttnn_decorators_ttnn_to_layout(ttnn_slice, ttnn_ROW_MAJOR_LAYOUT);  ttnn_slice = None
    ttnn_from_torch_6 = ttnn_decorators_ttnn_from_torch(arg2_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg2_1 = None
    ttnn_embedding_2 = ttnn_decorators_ttnn_embedding(ttnn_to_layout, ttnn_from_torch_6, layout = ttnn_TILE_LAYOUT);  ttnn_to_layout = ttnn_from_torch_6 = None
    ttnn_add_146 = ttnn_decorators_ttnn_add(ttnn_add_145, ttnn_embedding_2);  ttnn_add_145 = ttnn_embedding_2 = None
    ttnn_from_torch_7 = ttnn_decorators_ttnn_from_torch(arg3_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg3_1 = None
    ttnn_from_torch_8 = ttnn_decorators_ttnn_from_torch(arg4_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg4_1 = None
    ttnn_layer_norm = ttnn_decorators_ttnn_layer_norm(ttnn_add_146, epsilon = 1e-12, weight = ttnn_from_torch_7, bias = ttnn_from_torch_8);  ttnn_add_146 = ttnn_from_torch_7 = ttnn_from_torch_8 = None
    ttnn_experimental_view = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm, [256, 1024])
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
    ttnn_add_147 = ttnn_decorators_ttnn_add(ttnn_multiply_1, ttnn_multiply);  ttnn_multiply_1 = None
    ttnn_softmax = ttnn_decorators_ttnn_softmax(ttnn_add_147, -1, numeric_stable = True);  ttnn_add_147 = None
    ttnn_experimental_view_9 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax, [16, 256, 256]);  ttnn_softmax = None
    ttnn_experimental_view_10 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_1, [16, 256, 64]);  ttnn_permute_1 = None
    ttnn_matmul_4 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_9, ttnn_experimental_view_10);  ttnn_experimental_view_9 = ttnn_experimental_view_10 = None
    ttnn_experimental_view_11 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_4, [1, 16, 256, 64]);  ttnn_matmul_4 = None
    ttnn_permute_3 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_11, [0, 2, 1, 3]);  ttnn_experimental_view_11 = None
    ttnn_reshape_5 = ttnn_decorators_ttnn_reshape(ttnn_permute_3, [1, 256, 1024]);  ttnn_permute_3 = None
    ttnn_experimental_view_12 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_5, [256, 1024]);  ttnn_reshape_5 = None
    ttnn_from_torch_15 = ttnn_decorators_ttnn_from_torch(arg11_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg11_1 = None
    ttnn_transpose_4 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_15, 0, 1);  ttnn_from_torch_15 = None
    ttnn_matmul_5 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_12, ttnn_transpose_4);  ttnn_experimental_view_12 = ttnn_transpose_4 = None
    ttnn_from_torch_16 = ttnn_decorators_ttnn_from_torch(arg12_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg12_1 = None
    ttnn_add_3 = ttnn_decorators_ttnn_add(ttnn_from_torch_16, ttnn_matmul_5);  ttnn_from_torch_16 = ttnn_matmul_5 = None
    ttnn_experimental_view_13 = ttnn_decorators_ttnn_experimental_view(ttnn_add_3, [1, 256, 1024]);  ttnn_add_3 = None
    ttnn_add_148 = ttnn_decorators_ttnn_add(ttnn_experimental_view_13, ttnn_layer_norm);  ttnn_experimental_view_13 = ttnn_layer_norm = None
    ttnn_from_torch_17 = ttnn_decorators_ttnn_from_torch(arg13_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg13_1 = None
    ttnn_from_torch_18 = ttnn_decorators_ttnn_from_torch(arg14_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg14_1 = None
    ttnn_layer_norm_1 = ttnn_decorators_ttnn_layer_norm(ttnn_add_148, epsilon = 1e-12, weight = ttnn_from_torch_17, bias = ttnn_from_torch_18);  ttnn_add_148 = ttnn_from_torch_17 = ttnn_from_torch_18 = None
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
    ttnn_add_149 = ttnn_decorators_ttnn_add(ttnn_experimental_view_17, ttnn_layer_norm_1);  ttnn_experimental_view_17 = ttnn_layer_norm_1 = None
    ttnn_from_torch_23 = ttnn_decorators_ttnn_from_torch(arg19_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg19_1 = None
    ttnn_from_torch_24 = ttnn_decorators_ttnn_from_torch(arg20_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg20_1 = None
    ttnn_layer_norm_2 = ttnn_decorators_ttnn_layer_norm(ttnn_add_149, epsilon = 1e-12, weight = ttnn_from_torch_23, bias = ttnn_from_torch_24);  ttnn_add_149 = ttnn_from_torch_23 = ttnn_from_torch_24 = None
    ttnn_experimental_view_18 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_2, [256, 1024])
    ttnn_from_torch_25 = ttnn_decorators_ttnn_from_torch(arg21_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg21_1 = None
    ttnn_transpose_7 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_25, 0, 1);  ttnn_from_torch_25 = None
    ttnn_matmul_8 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_18, ttnn_transpose_7);  ttnn_transpose_7 = None
    ttnn_from_torch_26 = ttnn_decorators_ttnn_from_torch(arg22_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg22_1 = None
    ttnn_add_6 = ttnn_decorators_ttnn_add(ttnn_from_torch_26, ttnn_matmul_8);  ttnn_from_torch_26 = ttnn_matmul_8 = None
    ttnn_experimental_view_19 = ttnn_decorators_ttnn_experimental_view(ttnn_add_6, [1, 256, 1024]);  ttnn_add_6 = None
    ttnn_from_torch_27 = ttnn_decorators_ttnn_from_torch(arg23_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg23_1 = None
    ttnn_transpose_8 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_27, 0, 1);  ttnn_from_torch_27 = None
    ttnn_matmul_9 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_18, ttnn_transpose_8);  ttnn_transpose_8 = None
    ttnn_from_torch_28 = ttnn_decorators_ttnn_from_torch(arg24_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg24_1 = None
    ttnn_add_7 = ttnn_decorators_ttnn_add(ttnn_from_torch_28, ttnn_matmul_9);  ttnn_from_torch_28 = ttnn_matmul_9 = None
    ttnn_experimental_view_21 = ttnn_decorators_ttnn_experimental_view(ttnn_add_7, [1, 256, 1024]);  ttnn_add_7 = None
    ttnn_reshape_6 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_21, [1, 256, 16, 64]);  ttnn_experimental_view_21 = None
    ttnn_permute_4 = ttnn_decorators_ttnn_permute(ttnn_reshape_6, [0, 2, 1, 3]);  ttnn_reshape_6 = None
    ttnn_from_torch_29 = ttnn_decorators_ttnn_from_torch(arg25_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg25_1 = None
    ttnn_transpose_9 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_29, 0, 1);  ttnn_from_torch_29 = None
    ttnn_matmul_10 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_18, ttnn_transpose_9);  ttnn_experimental_view_18 = ttnn_transpose_9 = None
    ttnn_from_torch_30 = ttnn_decorators_ttnn_from_torch(arg26_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg26_1 = None
    ttnn_add_8 = ttnn_decorators_ttnn_add(ttnn_from_torch_30, ttnn_matmul_10);  ttnn_from_torch_30 = ttnn_matmul_10 = None
    ttnn_experimental_view_23 = ttnn_decorators_ttnn_experimental_view(ttnn_add_8, [1, 256, 1024]);  ttnn_add_8 = None
    ttnn_reshape_7 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_23, [1, 256, 16, 64]);  ttnn_experimental_view_23 = None
    ttnn_permute_5 = ttnn_decorators_ttnn_permute(ttnn_reshape_7, [0, 2, 1, 3]);  ttnn_reshape_7 = None
    ttnn_reshape_8 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_19, [1, 256, 16, 64]);  ttnn_experimental_view_19 = None
    ttnn_permute_6 = ttnn_decorators_ttnn_permute(ttnn_reshape_8, [0, 2, 1, 3]);  ttnn_reshape_8 = None
    ttnn_transpose_10 = ttnn_decorators_ttnn_transpose(ttnn_permute_4, 3, 2);  ttnn_permute_4 = None
    ttnn_experimental_view_24 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_6, [16, 256, 64]);  ttnn_permute_6 = None
    ttnn_experimental_view_25 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_10, [16, 64, 256]);  ttnn_transpose_10 = None
    ttnn_matmul_11 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_24, ttnn_experimental_view_25);  ttnn_experimental_view_24 = ttnn_experimental_view_25 = None
    ttnn_experimental_view_26 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_11, [1, 16, 256, 256]);  ttnn_matmul_11 = None
    ttnn_multiply_2 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_26, 0.125);  ttnn_experimental_view_26 = None
    ttnn_add_150 = ttnn_decorators_ttnn_add(ttnn_multiply_2, ttnn_multiply);  ttnn_multiply_2 = None
    ttnn_softmax_1 = ttnn_decorators_ttnn_softmax(ttnn_add_150, -1, numeric_stable = True);  ttnn_add_150 = None
    ttnn_experimental_view_27 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_1, [16, 256, 256]);  ttnn_softmax_1 = None
    ttnn_experimental_view_28 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_5, [16, 256, 64]);  ttnn_permute_5 = None
    ttnn_matmul_12 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_27, ttnn_experimental_view_28);  ttnn_experimental_view_27 = ttnn_experimental_view_28 = None
    ttnn_experimental_view_29 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_12, [1, 16, 256, 64]);  ttnn_matmul_12 = None
    ttnn_permute_7 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_29, [0, 2, 1, 3]);  ttnn_experimental_view_29 = None
    ttnn_reshape_9 = ttnn_decorators_ttnn_reshape(ttnn_permute_7, [1, 256, 1024]);  ttnn_permute_7 = None
    ttnn_experimental_view_30 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_9, [256, 1024]);  ttnn_reshape_9 = None
    ttnn_from_torch_31 = ttnn_decorators_ttnn_from_torch(arg27_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg27_1 = None
    ttnn_transpose_11 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_31, 0, 1);  ttnn_from_torch_31 = None
    ttnn_matmul_13 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_30, ttnn_transpose_11);  ttnn_experimental_view_30 = ttnn_transpose_11 = None
    ttnn_from_torch_32 = ttnn_decorators_ttnn_from_torch(arg28_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg28_1 = None
    ttnn_add_9 = ttnn_decorators_ttnn_add(ttnn_from_torch_32, ttnn_matmul_13);  ttnn_from_torch_32 = ttnn_matmul_13 = None
    ttnn_experimental_view_31 = ttnn_decorators_ttnn_experimental_view(ttnn_add_9, [1, 256, 1024]);  ttnn_add_9 = None
    ttnn_add_151 = ttnn_decorators_ttnn_add(ttnn_experimental_view_31, ttnn_layer_norm_2);  ttnn_experimental_view_31 = ttnn_layer_norm_2 = None
    ttnn_from_torch_33 = ttnn_decorators_ttnn_from_torch(arg29_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg29_1 = None
    ttnn_from_torch_34 = ttnn_decorators_ttnn_from_torch(arg30_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg30_1 = None
    ttnn_layer_norm_3 = ttnn_decorators_ttnn_layer_norm(ttnn_add_151, epsilon = 1e-12, weight = ttnn_from_torch_33, bias = ttnn_from_torch_34);  ttnn_add_151 = ttnn_from_torch_33 = ttnn_from_torch_34 = None
    ttnn_experimental_view_32 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_3, [256, 1024])
    ttnn_from_torch_35 = ttnn_decorators_ttnn_from_torch(arg31_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg31_1 = None
    ttnn_transpose_12 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_35, 0, 1);  ttnn_from_torch_35 = None
    ttnn_matmul_14 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_32, ttnn_transpose_12);  ttnn_experimental_view_32 = ttnn_transpose_12 = None
    ttnn_from_torch_36 = ttnn_decorators_ttnn_from_torch(arg32_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg32_1 = None
    ttnn_add_10 = ttnn_decorators_ttnn_add(ttnn_from_torch_36, ttnn_matmul_14);  ttnn_from_torch_36 = ttnn_matmul_14 = None
    ttnn_experimental_view_33 = ttnn_decorators_ttnn_experimental_view(ttnn_add_10, [1, 256, 4096]);  ttnn_add_10 = None
    ttnn_gelu_1 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_33);  ttnn_experimental_view_33 = None
    ttnn_experimental_view_34 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_1, [256, 4096]);  ttnn_gelu_1 = None
    ttnn_from_torch_37 = ttnn_decorators_ttnn_from_torch(arg33_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg33_1 = None
    ttnn_transpose_13 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_37, 0, 1);  ttnn_from_torch_37 = None
    ttnn_matmul_15 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_34, ttnn_transpose_13);  ttnn_experimental_view_34 = ttnn_transpose_13 = None
    ttnn_from_torch_38 = ttnn_decorators_ttnn_from_torch(arg34_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg34_1 = None
    ttnn_add_11 = ttnn_decorators_ttnn_add(ttnn_from_torch_38, ttnn_matmul_15);  ttnn_from_torch_38 = ttnn_matmul_15 = None
    ttnn_experimental_view_35 = ttnn_decorators_ttnn_experimental_view(ttnn_add_11, [1, 256, 1024]);  ttnn_add_11 = None
    ttnn_add_152 = ttnn_decorators_ttnn_add(ttnn_experimental_view_35, ttnn_layer_norm_3);  ttnn_experimental_view_35 = ttnn_layer_norm_3 = None
    ttnn_from_torch_39 = ttnn_decorators_ttnn_from_torch(arg35_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg35_1 = None
    ttnn_from_torch_40 = ttnn_decorators_ttnn_from_torch(arg36_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg36_1 = None
    ttnn_layer_norm_4 = ttnn_decorators_ttnn_layer_norm(ttnn_add_152, epsilon = 1e-12, weight = ttnn_from_torch_39, bias = ttnn_from_torch_40);  ttnn_add_152 = ttnn_from_torch_39 = ttnn_from_torch_40 = None
    ttnn_experimental_view_36 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_4, [256, 1024])
    ttnn_from_torch_41 = ttnn_decorators_ttnn_from_torch(arg37_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg37_1 = None
    ttnn_transpose_14 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_41, 0, 1);  ttnn_from_torch_41 = None
    ttnn_matmul_16 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_36, ttnn_transpose_14);  ttnn_transpose_14 = None
    ttnn_from_torch_42 = ttnn_decorators_ttnn_from_torch(arg38_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg38_1 = None
    ttnn_add_12 = ttnn_decorators_ttnn_add(ttnn_from_torch_42, ttnn_matmul_16);  ttnn_from_torch_42 = ttnn_matmul_16 = None
    ttnn_experimental_view_37 = ttnn_decorators_ttnn_experimental_view(ttnn_add_12, [1, 256, 1024]);  ttnn_add_12 = None
    ttnn_from_torch_43 = ttnn_decorators_ttnn_from_torch(arg39_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg39_1 = None
    ttnn_transpose_15 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_43, 0, 1);  ttnn_from_torch_43 = None
    ttnn_matmul_17 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_36, ttnn_transpose_15);  ttnn_transpose_15 = None
    ttnn_from_torch_44 = ttnn_decorators_ttnn_from_torch(arg40_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg40_1 = None
    ttnn_add_13 = ttnn_decorators_ttnn_add(ttnn_from_torch_44, ttnn_matmul_17);  ttnn_from_torch_44 = ttnn_matmul_17 = None
    ttnn_experimental_view_39 = ttnn_decorators_ttnn_experimental_view(ttnn_add_13, [1, 256, 1024]);  ttnn_add_13 = None
    ttnn_reshape_10 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_39, [1, 256, 16, 64]);  ttnn_experimental_view_39 = None
    ttnn_permute_8 = ttnn_decorators_ttnn_permute(ttnn_reshape_10, [0, 2, 1, 3]);  ttnn_reshape_10 = None
    ttnn_from_torch_45 = ttnn_decorators_ttnn_from_torch(arg41_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg41_1 = None
    ttnn_transpose_16 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_45, 0, 1);  ttnn_from_torch_45 = None
    ttnn_matmul_18 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_36, ttnn_transpose_16);  ttnn_experimental_view_36 = ttnn_transpose_16 = None
    ttnn_from_torch_46 = ttnn_decorators_ttnn_from_torch(arg42_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg42_1 = None
    ttnn_add_14 = ttnn_decorators_ttnn_add(ttnn_from_torch_46, ttnn_matmul_18);  ttnn_from_torch_46 = ttnn_matmul_18 = None
    ttnn_experimental_view_41 = ttnn_decorators_ttnn_experimental_view(ttnn_add_14, [1, 256, 1024]);  ttnn_add_14 = None
    ttnn_reshape_11 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_41, [1, 256, 16, 64]);  ttnn_experimental_view_41 = None
    ttnn_permute_9 = ttnn_decorators_ttnn_permute(ttnn_reshape_11, [0, 2, 1, 3]);  ttnn_reshape_11 = None
    ttnn_reshape_12 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_37, [1, 256, 16, 64]);  ttnn_experimental_view_37 = None
    ttnn_permute_10 = ttnn_decorators_ttnn_permute(ttnn_reshape_12, [0, 2, 1, 3]);  ttnn_reshape_12 = None
    ttnn_transpose_17 = ttnn_decorators_ttnn_transpose(ttnn_permute_8, 3, 2);  ttnn_permute_8 = None
    ttnn_experimental_view_42 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_10, [16, 256, 64]);  ttnn_permute_10 = None
    ttnn_experimental_view_43 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_17, [16, 64, 256]);  ttnn_transpose_17 = None
    ttnn_matmul_19 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_42, ttnn_experimental_view_43);  ttnn_experimental_view_42 = ttnn_experimental_view_43 = None
    ttnn_experimental_view_44 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_19, [1, 16, 256, 256]);  ttnn_matmul_19 = None
    ttnn_multiply_3 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_44, 0.125);  ttnn_experimental_view_44 = None
    ttnn_add_153 = ttnn_decorators_ttnn_add(ttnn_multiply_3, ttnn_multiply);  ttnn_multiply_3 = None
    ttnn_softmax_2 = ttnn_decorators_ttnn_softmax(ttnn_add_153, -1, numeric_stable = True);  ttnn_add_153 = None
    ttnn_experimental_view_45 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_2, [16, 256, 256]);  ttnn_softmax_2 = None
    ttnn_experimental_view_46 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_9, [16, 256, 64]);  ttnn_permute_9 = None
    ttnn_matmul_20 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_45, ttnn_experimental_view_46);  ttnn_experimental_view_45 = ttnn_experimental_view_46 = None
    ttnn_experimental_view_47 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_20, [1, 16, 256, 64]);  ttnn_matmul_20 = None
    ttnn_permute_11 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_47, [0, 2, 1, 3]);  ttnn_experimental_view_47 = None
    ttnn_reshape_13 = ttnn_decorators_ttnn_reshape(ttnn_permute_11, [1, 256, 1024]);  ttnn_permute_11 = None
    ttnn_experimental_view_48 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_13, [256, 1024]);  ttnn_reshape_13 = None
    ttnn_from_torch_47 = ttnn_decorators_ttnn_from_torch(arg43_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg43_1 = None
    ttnn_transpose_18 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_47, 0, 1);  ttnn_from_torch_47 = None
    ttnn_matmul_21 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_48, ttnn_transpose_18);  ttnn_experimental_view_48 = ttnn_transpose_18 = None
    ttnn_from_torch_48 = ttnn_decorators_ttnn_from_torch(arg44_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg44_1 = None
    ttnn_add_15 = ttnn_decorators_ttnn_add(ttnn_from_torch_48, ttnn_matmul_21);  ttnn_from_torch_48 = ttnn_matmul_21 = None
    ttnn_experimental_view_49 = ttnn_decorators_ttnn_experimental_view(ttnn_add_15, [1, 256, 1024]);  ttnn_add_15 = None
    ttnn_add_154 = ttnn_decorators_ttnn_add(ttnn_experimental_view_49, ttnn_layer_norm_4);  ttnn_experimental_view_49 = ttnn_layer_norm_4 = None
    ttnn_from_torch_49 = ttnn_decorators_ttnn_from_torch(arg45_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg45_1 = None
    ttnn_from_torch_50 = ttnn_decorators_ttnn_from_torch(arg46_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg46_1 = None
    ttnn_layer_norm_5 = ttnn_decorators_ttnn_layer_norm(ttnn_add_154, epsilon = 1e-12, weight = ttnn_from_torch_49, bias = ttnn_from_torch_50);  ttnn_add_154 = ttnn_from_torch_49 = ttnn_from_torch_50 = None
    ttnn_experimental_view_50 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_5, [256, 1024])
    ttnn_from_torch_51 = ttnn_decorators_ttnn_from_torch(arg47_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg47_1 = None
    ttnn_transpose_19 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_51, 0, 1);  ttnn_from_torch_51 = None
    ttnn_matmul_22 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_50, ttnn_transpose_19);  ttnn_experimental_view_50 = ttnn_transpose_19 = None
    ttnn_from_torch_52 = ttnn_decorators_ttnn_from_torch(arg48_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg48_1 = None
    ttnn_add_16 = ttnn_decorators_ttnn_add(ttnn_from_torch_52, ttnn_matmul_22);  ttnn_from_torch_52 = ttnn_matmul_22 = None
    ttnn_experimental_view_51 = ttnn_decorators_ttnn_experimental_view(ttnn_add_16, [1, 256, 4096]);  ttnn_add_16 = None
    ttnn_gelu_2 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_51);  ttnn_experimental_view_51 = None
    ttnn_experimental_view_52 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_2, [256, 4096]);  ttnn_gelu_2 = None
    ttnn_from_torch_53 = ttnn_decorators_ttnn_from_torch(arg49_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg49_1 = None
    ttnn_transpose_20 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_53, 0, 1);  ttnn_from_torch_53 = None
    ttnn_matmul_23 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_52, ttnn_transpose_20);  ttnn_experimental_view_52 = ttnn_transpose_20 = None
    ttnn_from_torch_54 = ttnn_decorators_ttnn_from_torch(arg50_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg50_1 = None
    ttnn_add_17 = ttnn_decorators_ttnn_add(ttnn_from_torch_54, ttnn_matmul_23);  ttnn_from_torch_54 = ttnn_matmul_23 = None
    ttnn_experimental_view_53 = ttnn_decorators_ttnn_experimental_view(ttnn_add_17, [1, 256, 1024]);  ttnn_add_17 = None
    ttnn_add_155 = ttnn_decorators_ttnn_add(ttnn_experimental_view_53, ttnn_layer_norm_5);  ttnn_experimental_view_53 = ttnn_layer_norm_5 = None
    ttnn_from_torch_55 = ttnn_decorators_ttnn_from_torch(arg51_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg51_1 = None
    ttnn_from_torch_56 = ttnn_decorators_ttnn_from_torch(arg52_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg52_1 = None
    ttnn_layer_norm_6 = ttnn_decorators_ttnn_layer_norm(ttnn_add_155, epsilon = 1e-12, weight = ttnn_from_torch_55, bias = ttnn_from_torch_56);  ttnn_add_155 = ttnn_from_torch_55 = ttnn_from_torch_56 = None
    ttnn_experimental_view_54 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_6, [256, 1024])
    ttnn_from_torch_57 = ttnn_decorators_ttnn_from_torch(arg53_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg53_1 = None
    ttnn_transpose_21 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_57, 0, 1);  ttnn_from_torch_57 = None
    ttnn_matmul_24 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_54, ttnn_transpose_21);  ttnn_transpose_21 = None
    ttnn_from_torch_58 = ttnn_decorators_ttnn_from_torch(arg54_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg54_1 = None
    ttnn_add_18 = ttnn_decorators_ttnn_add(ttnn_from_torch_58, ttnn_matmul_24);  ttnn_from_torch_58 = ttnn_matmul_24 = None
    ttnn_experimental_view_55 = ttnn_decorators_ttnn_experimental_view(ttnn_add_18, [1, 256, 1024]);  ttnn_add_18 = None
    ttnn_from_torch_59 = ttnn_decorators_ttnn_from_torch(arg55_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg55_1 = None
    ttnn_transpose_22 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_59, 0, 1);  ttnn_from_torch_59 = None
    ttnn_matmul_25 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_54, ttnn_transpose_22);  ttnn_transpose_22 = None
    ttnn_from_torch_60 = ttnn_decorators_ttnn_from_torch(arg56_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg56_1 = None
    ttnn_add_19 = ttnn_decorators_ttnn_add(ttnn_from_torch_60, ttnn_matmul_25);  ttnn_from_torch_60 = ttnn_matmul_25 = None
    ttnn_experimental_view_57 = ttnn_decorators_ttnn_experimental_view(ttnn_add_19, [1, 256, 1024]);  ttnn_add_19 = None
    ttnn_reshape_14 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_57, [1, 256, 16, 64]);  ttnn_experimental_view_57 = None
    ttnn_permute_12 = ttnn_decorators_ttnn_permute(ttnn_reshape_14, [0, 2, 1, 3]);  ttnn_reshape_14 = None
    ttnn_from_torch_61 = ttnn_decorators_ttnn_from_torch(arg57_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg57_1 = None
    ttnn_transpose_23 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_61, 0, 1);  ttnn_from_torch_61 = None
    ttnn_matmul_26 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_54, ttnn_transpose_23);  ttnn_experimental_view_54 = ttnn_transpose_23 = None
    ttnn_from_torch_62 = ttnn_decorators_ttnn_from_torch(arg58_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg58_1 = None
    ttnn_add_20 = ttnn_decorators_ttnn_add(ttnn_from_torch_62, ttnn_matmul_26);  ttnn_from_torch_62 = ttnn_matmul_26 = None
    ttnn_experimental_view_59 = ttnn_decorators_ttnn_experimental_view(ttnn_add_20, [1, 256, 1024]);  ttnn_add_20 = None
    ttnn_reshape_15 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_59, [1, 256, 16, 64]);  ttnn_experimental_view_59 = None
    ttnn_permute_13 = ttnn_decorators_ttnn_permute(ttnn_reshape_15, [0, 2, 1, 3]);  ttnn_reshape_15 = None
    ttnn_reshape_16 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_55, [1, 256, 16, 64]);  ttnn_experimental_view_55 = None
    ttnn_permute_14 = ttnn_decorators_ttnn_permute(ttnn_reshape_16, [0, 2, 1, 3]);  ttnn_reshape_16 = None
    ttnn_transpose_24 = ttnn_decorators_ttnn_transpose(ttnn_permute_12, 3, 2);  ttnn_permute_12 = None
    ttnn_experimental_view_60 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_14, [16, 256, 64]);  ttnn_permute_14 = None
    ttnn_experimental_view_61 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_24, [16, 64, 256]);  ttnn_transpose_24 = None
    ttnn_matmul_27 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_60, ttnn_experimental_view_61);  ttnn_experimental_view_60 = ttnn_experimental_view_61 = None
    ttnn_experimental_view_62 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_27, [1, 16, 256, 256]);  ttnn_matmul_27 = None
    ttnn_multiply_4 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_62, 0.125);  ttnn_experimental_view_62 = None
    ttnn_add_156 = ttnn_decorators_ttnn_add(ttnn_multiply_4, ttnn_multiply);  ttnn_multiply_4 = None
    ttnn_softmax_3 = ttnn_decorators_ttnn_softmax(ttnn_add_156, -1, numeric_stable = True);  ttnn_add_156 = None
    ttnn_experimental_view_63 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_3, [16, 256, 256]);  ttnn_softmax_3 = None
    ttnn_experimental_view_64 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_13, [16, 256, 64]);  ttnn_permute_13 = None
    ttnn_matmul_28 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_63, ttnn_experimental_view_64);  ttnn_experimental_view_63 = ttnn_experimental_view_64 = None
    ttnn_experimental_view_65 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_28, [1, 16, 256, 64]);  ttnn_matmul_28 = None
    ttnn_permute_15 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_65, [0, 2, 1, 3]);  ttnn_experimental_view_65 = None
    ttnn_reshape_17 = ttnn_decorators_ttnn_reshape(ttnn_permute_15, [1, 256, 1024]);  ttnn_permute_15 = None
    ttnn_experimental_view_66 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_17, [256, 1024]);  ttnn_reshape_17 = None
    ttnn_from_torch_63 = ttnn_decorators_ttnn_from_torch(arg59_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg59_1 = None
    ttnn_transpose_25 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_63, 0, 1);  ttnn_from_torch_63 = None
    ttnn_matmul_29 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_66, ttnn_transpose_25);  ttnn_experimental_view_66 = ttnn_transpose_25 = None
    ttnn_from_torch_64 = ttnn_decorators_ttnn_from_torch(arg60_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg60_1 = None
    ttnn_add_21 = ttnn_decorators_ttnn_add(ttnn_from_torch_64, ttnn_matmul_29);  ttnn_from_torch_64 = ttnn_matmul_29 = None
    ttnn_experimental_view_67 = ttnn_decorators_ttnn_experimental_view(ttnn_add_21, [1, 256, 1024]);  ttnn_add_21 = None
    ttnn_add_157 = ttnn_decorators_ttnn_add(ttnn_experimental_view_67, ttnn_layer_norm_6);  ttnn_experimental_view_67 = ttnn_layer_norm_6 = None
    ttnn_from_torch_65 = ttnn_decorators_ttnn_from_torch(arg61_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg61_1 = None
    ttnn_from_torch_66 = ttnn_decorators_ttnn_from_torch(arg62_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg62_1 = None
    ttnn_layer_norm_7 = ttnn_decorators_ttnn_layer_norm(ttnn_add_157, epsilon = 1e-12, weight = ttnn_from_torch_65, bias = ttnn_from_torch_66);  ttnn_add_157 = ttnn_from_torch_65 = ttnn_from_torch_66 = None
    ttnn_experimental_view_68 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_7, [256, 1024])
    ttnn_from_torch_67 = ttnn_decorators_ttnn_from_torch(arg63_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg63_1 = None
    ttnn_transpose_26 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_67, 0, 1);  ttnn_from_torch_67 = None
    ttnn_matmul_30 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_68, ttnn_transpose_26);  ttnn_experimental_view_68 = ttnn_transpose_26 = None
    ttnn_from_torch_68 = ttnn_decorators_ttnn_from_torch(arg64_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg64_1 = None
    ttnn_add_22 = ttnn_decorators_ttnn_add(ttnn_from_torch_68, ttnn_matmul_30);  ttnn_from_torch_68 = ttnn_matmul_30 = None
    ttnn_experimental_view_69 = ttnn_decorators_ttnn_experimental_view(ttnn_add_22, [1, 256, 4096]);  ttnn_add_22 = None
    ttnn_gelu_3 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_69);  ttnn_experimental_view_69 = None
    ttnn_experimental_view_70 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_3, [256, 4096]);  ttnn_gelu_3 = None
    ttnn_from_torch_69 = ttnn_decorators_ttnn_from_torch(arg65_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg65_1 = None
    ttnn_transpose_27 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_69, 0, 1);  ttnn_from_torch_69 = None
    ttnn_matmul_31 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_70, ttnn_transpose_27);  ttnn_experimental_view_70 = ttnn_transpose_27 = None
    ttnn_from_torch_70 = ttnn_decorators_ttnn_from_torch(arg66_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg66_1 = None
    ttnn_add_23 = ttnn_decorators_ttnn_add(ttnn_from_torch_70, ttnn_matmul_31);  ttnn_from_torch_70 = ttnn_matmul_31 = None
    ttnn_experimental_view_71 = ttnn_decorators_ttnn_experimental_view(ttnn_add_23, [1, 256, 1024]);  ttnn_add_23 = None
    ttnn_add_158 = ttnn_decorators_ttnn_add(ttnn_experimental_view_71, ttnn_layer_norm_7);  ttnn_experimental_view_71 = ttnn_layer_norm_7 = None
    ttnn_from_torch_71 = ttnn_decorators_ttnn_from_torch(arg67_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg67_1 = None
    ttnn_from_torch_72 = ttnn_decorators_ttnn_from_torch(arg68_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg68_1 = None
    ttnn_layer_norm_8 = ttnn_decorators_ttnn_layer_norm(ttnn_add_158, epsilon = 1e-12, weight = ttnn_from_torch_71, bias = ttnn_from_torch_72);  ttnn_add_158 = ttnn_from_torch_71 = ttnn_from_torch_72 = None
    ttnn_experimental_view_72 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_8, [256, 1024])
    ttnn_from_torch_73 = ttnn_decorators_ttnn_from_torch(arg69_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg69_1 = None
    ttnn_transpose_28 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_73, 0, 1);  ttnn_from_torch_73 = None
    ttnn_matmul_32 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_72, ttnn_transpose_28);  ttnn_transpose_28 = None
    ttnn_from_torch_74 = ttnn_decorators_ttnn_from_torch(arg70_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg70_1 = None
    ttnn_add_24 = ttnn_decorators_ttnn_add(ttnn_from_torch_74, ttnn_matmul_32);  ttnn_from_torch_74 = ttnn_matmul_32 = None
    ttnn_experimental_view_73 = ttnn_decorators_ttnn_experimental_view(ttnn_add_24, [1, 256, 1024]);  ttnn_add_24 = None
    ttnn_from_torch_75 = ttnn_decorators_ttnn_from_torch(arg71_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg71_1 = None
    ttnn_transpose_29 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_75, 0, 1);  ttnn_from_torch_75 = None
    ttnn_matmul_33 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_72, ttnn_transpose_29);  ttnn_transpose_29 = None
    ttnn_from_torch_76 = ttnn_decorators_ttnn_from_torch(arg72_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg72_1 = None
    ttnn_add_25 = ttnn_decorators_ttnn_add(ttnn_from_torch_76, ttnn_matmul_33);  ttnn_from_torch_76 = ttnn_matmul_33 = None
    ttnn_experimental_view_75 = ttnn_decorators_ttnn_experimental_view(ttnn_add_25, [1, 256, 1024]);  ttnn_add_25 = None
    ttnn_reshape_18 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_75, [1, 256, 16, 64]);  ttnn_experimental_view_75 = None
    ttnn_permute_16 = ttnn_decorators_ttnn_permute(ttnn_reshape_18, [0, 2, 1, 3]);  ttnn_reshape_18 = None
    ttnn_from_torch_77 = ttnn_decorators_ttnn_from_torch(arg73_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg73_1 = None
    ttnn_transpose_30 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_77, 0, 1);  ttnn_from_torch_77 = None
    ttnn_matmul_34 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_72, ttnn_transpose_30);  ttnn_experimental_view_72 = ttnn_transpose_30 = None
    ttnn_from_torch_78 = ttnn_decorators_ttnn_from_torch(arg74_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg74_1 = None
    ttnn_add_26 = ttnn_decorators_ttnn_add(ttnn_from_torch_78, ttnn_matmul_34);  ttnn_from_torch_78 = ttnn_matmul_34 = None
    ttnn_experimental_view_77 = ttnn_decorators_ttnn_experimental_view(ttnn_add_26, [1, 256, 1024]);  ttnn_add_26 = None
    ttnn_reshape_19 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_77, [1, 256, 16, 64]);  ttnn_experimental_view_77 = None
    ttnn_permute_17 = ttnn_decorators_ttnn_permute(ttnn_reshape_19, [0, 2, 1, 3]);  ttnn_reshape_19 = None
    ttnn_reshape_20 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_73, [1, 256, 16, 64]);  ttnn_experimental_view_73 = None
    ttnn_permute_18 = ttnn_decorators_ttnn_permute(ttnn_reshape_20, [0, 2, 1, 3]);  ttnn_reshape_20 = None
    ttnn_transpose_31 = ttnn_decorators_ttnn_transpose(ttnn_permute_16, 3, 2);  ttnn_permute_16 = None
    ttnn_experimental_view_78 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_18, [16, 256, 64]);  ttnn_permute_18 = None
    ttnn_experimental_view_79 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_31, [16, 64, 256]);  ttnn_transpose_31 = None
    ttnn_matmul_35 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_78, ttnn_experimental_view_79);  ttnn_experimental_view_78 = ttnn_experimental_view_79 = None
    ttnn_experimental_view_80 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_35, [1, 16, 256, 256]);  ttnn_matmul_35 = None
    ttnn_multiply_5 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_80, 0.125);  ttnn_experimental_view_80 = None
    ttnn_add_159 = ttnn_decorators_ttnn_add(ttnn_multiply_5, ttnn_multiply);  ttnn_multiply_5 = None
    ttnn_softmax_4 = ttnn_decorators_ttnn_softmax(ttnn_add_159, -1, numeric_stable = True);  ttnn_add_159 = None
    ttnn_experimental_view_81 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_4, [16, 256, 256]);  ttnn_softmax_4 = None
    ttnn_experimental_view_82 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_17, [16, 256, 64]);  ttnn_permute_17 = None
    ttnn_matmul_36 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_81, ttnn_experimental_view_82);  ttnn_experimental_view_81 = ttnn_experimental_view_82 = None
    ttnn_experimental_view_83 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_36, [1, 16, 256, 64]);  ttnn_matmul_36 = None
    ttnn_permute_19 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_83, [0, 2, 1, 3]);  ttnn_experimental_view_83 = None
    ttnn_reshape_21 = ttnn_decorators_ttnn_reshape(ttnn_permute_19, [1, 256, 1024]);  ttnn_permute_19 = None
    ttnn_experimental_view_84 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_21, [256, 1024]);  ttnn_reshape_21 = None
    ttnn_from_torch_79 = ttnn_decorators_ttnn_from_torch(arg75_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg75_1 = None
    ttnn_transpose_32 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_79, 0, 1);  ttnn_from_torch_79 = None
    ttnn_matmul_37 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_84, ttnn_transpose_32);  ttnn_experimental_view_84 = ttnn_transpose_32 = None
    ttnn_from_torch_80 = ttnn_decorators_ttnn_from_torch(arg76_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg76_1 = None
    ttnn_add_27 = ttnn_decorators_ttnn_add(ttnn_from_torch_80, ttnn_matmul_37);  ttnn_from_torch_80 = ttnn_matmul_37 = None
    ttnn_experimental_view_85 = ttnn_decorators_ttnn_experimental_view(ttnn_add_27, [1, 256, 1024]);  ttnn_add_27 = None
    ttnn_add_160 = ttnn_decorators_ttnn_add(ttnn_experimental_view_85, ttnn_layer_norm_8);  ttnn_experimental_view_85 = ttnn_layer_norm_8 = None
    ttnn_from_torch_81 = ttnn_decorators_ttnn_from_torch(arg77_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg77_1 = None
    ttnn_from_torch_82 = ttnn_decorators_ttnn_from_torch(arg78_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg78_1 = None
    ttnn_layer_norm_9 = ttnn_decorators_ttnn_layer_norm(ttnn_add_160, epsilon = 1e-12, weight = ttnn_from_torch_81, bias = ttnn_from_torch_82);  ttnn_add_160 = ttnn_from_torch_81 = ttnn_from_torch_82 = None
    ttnn_experimental_view_86 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_9, [256, 1024])
    ttnn_from_torch_83 = ttnn_decorators_ttnn_from_torch(arg79_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg79_1 = None
    ttnn_transpose_33 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_83, 0, 1);  ttnn_from_torch_83 = None
    ttnn_matmul_38 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_86, ttnn_transpose_33);  ttnn_experimental_view_86 = ttnn_transpose_33 = None
    ttnn_from_torch_84 = ttnn_decorators_ttnn_from_torch(arg80_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg80_1 = None
    ttnn_add_28 = ttnn_decorators_ttnn_add(ttnn_from_torch_84, ttnn_matmul_38);  ttnn_from_torch_84 = ttnn_matmul_38 = None
    ttnn_experimental_view_87 = ttnn_decorators_ttnn_experimental_view(ttnn_add_28, [1, 256, 4096]);  ttnn_add_28 = None
    ttnn_gelu_4 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_87);  ttnn_experimental_view_87 = None
    ttnn_experimental_view_88 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_4, [256, 4096]);  ttnn_gelu_4 = None
    ttnn_from_torch_85 = ttnn_decorators_ttnn_from_torch(arg81_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg81_1 = None
    ttnn_transpose_34 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_85, 0, 1);  ttnn_from_torch_85 = None
    ttnn_matmul_39 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_88, ttnn_transpose_34);  ttnn_experimental_view_88 = ttnn_transpose_34 = None
    ttnn_from_torch_86 = ttnn_decorators_ttnn_from_torch(arg82_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg82_1 = None
    ttnn_add_29 = ttnn_decorators_ttnn_add(ttnn_from_torch_86, ttnn_matmul_39);  ttnn_from_torch_86 = ttnn_matmul_39 = None
    ttnn_experimental_view_89 = ttnn_decorators_ttnn_experimental_view(ttnn_add_29, [1, 256, 1024]);  ttnn_add_29 = None
    ttnn_add_161 = ttnn_decorators_ttnn_add(ttnn_experimental_view_89, ttnn_layer_norm_9);  ttnn_experimental_view_89 = ttnn_layer_norm_9 = None
    ttnn_from_torch_87 = ttnn_decorators_ttnn_from_torch(arg83_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg83_1 = None
    ttnn_from_torch_88 = ttnn_decorators_ttnn_from_torch(arg84_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg84_1 = None
    ttnn_layer_norm_10 = ttnn_decorators_ttnn_layer_norm(ttnn_add_161, epsilon = 1e-12, weight = ttnn_from_torch_87, bias = ttnn_from_torch_88);  ttnn_add_161 = ttnn_from_torch_87 = ttnn_from_torch_88 = None
    ttnn_experimental_view_90 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_10, [256, 1024])
    ttnn_from_torch_89 = ttnn_decorators_ttnn_from_torch(arg85_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg85_1 = None
    ttnn_transpose_35 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_89, 0, 1);  ttnn_from_torch_89 = None
    ttnn_matmul_40 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_90, ttnn_transpose_35);  ttnn_transpose_35 = None
    ttnn_from_torch_90 = ttnn_decorators_ttnn_from_torch(arg86_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg86_1 = None
    ttnn_add_30 = ttnn_decorators_ttnn_add(ttnn_from_torch_90, ttnn_matmul_40);  ttnn_from_torch_90 = ttnn_matmul_40 = None
    ttnn_experimental_view_91 = ttnn_decorators_ttnn_experimental_view(ttnn_add_30, [1, 256, 1024]);  ttnn_add_30 = None
    ttnn_from_torch_91 = ttnn_decorators_ttnn_from_torch(arg87_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg87_1 = None
    ttnn_transpose_36 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_91, 0, 1);  ttnn_from_torch_91 = None
    ttnn_matmul_41 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_90, ttnn_transpose_36);  ttnn_transpose_36 = None
    ttnn_from_torch_92 = ttnn_decorators_ttnn_from_torch(arg88_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg88_1 = None
    ttnn_add_31 = ttnn_decorators_ttnn_add(ttnn_from_torch_92, ttnn_matmul_41);  ttnn_from_torch_92 = ttnn_matmul_41 = None
    ttnn_experimental_view_93 = ttnn_decorators_ttnn_experimental_view(ttnn_add_31, [1, 256, 1024]);  ttnn_add_31 = None
    ttnn_reshape_22 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_93, [1, 256, 16, 64]);  ttnn_experimental_view_93 = None
    ttnn_permute_20 = ttnn_decorators_ttnn_permute(ttnn_reshape_22, [0, 2, 1, 3]);  ttnn_reshape_22 = None
    ttnn_from_torch_93 = ttnn_decorators_ttnn_from_torch(arg89_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg89_1 = None
    ttnn_transpose_37 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_93, 0, 1);  ttnn_from_torch_93 = None
    ttnn_matmul_42 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_90, ttnn_transpose_37);  ttnn_experimental_view_90 = ttnn_transpose_37 = None
    ttnn_from_torch_94 = ttnn_decorators_ttnn_from_torch(arg90_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg90_1 = None
    ttnn_add_32 = ttnn_decorators_ttnn_add(ttnn_from_torch_94, ttnn_matmul_42);  ttnn_from_torch_94 = ttnn_matmul_42 = None
    ttnn_experimental_view_95 = ttnn_decorators_ttnn_experimental_view(ttnn_add_32, [1, 256, 1024]);  ttnn_add_32 = None
    ttnn_reshape_23 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_95, [1, 256, 16, 64]);  ttnn_experimental_view_95 = None
    ttnn_permute_21 = ttnn_decorators_ttnn_permute(ttnn_reshape_23, [0, 2, 1, 3]);  ttnn_reshape_23 = None
    ttnn_reshape_24 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_91, [1, 256, 16, 64]);  ttnn_experimental_view_91 = None
    ttnn_permute_22 = ttnn_decorators_ttnn_permute(ttnn_reshape_24, [0, 2, 1, 3]);  ttnn_reshape_24 = None
    ttnn_transpose_38 = ttnn_decorators_ttnn_transpose(ttnn_permute_20, 3, 2);  ttnn_permute_20 = None
    ttnn_experimental_view_96 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_22, [16, 256, 64]);  ttnn_permute_22 = None
    ttnn_experimental_view_97 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_38, [16, 64, 256]);  ttnn_transpose_38 = None
    ttnn_matmul_43 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_96, ttnn_experimental_view_97);  ttnn_experimental_view_96 = ttnn_experimental_view_97 = None
    ttnn_experimental_view_98 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_43, [1, 16, 256, 256]);  ttnn_matmul_43 = None
    ttnn_multiply_6 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_98, 0.125);  ttnn_experimental_view_98 = None
    ttnn_add_162 = ttnn_decorators_ttnn_add(ttnn_multiply_6, ttnn_multiply);  ttnn_multiply_6 = None
    ttnn_softmax_5 = ttnn_decorators_ttnn_softmax(ttnn_add_162, -1, numeric_stable = True);  ttnn_add_162 = None
    ttnn_experimental_view_99 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_5, [16, 256, 256]);  ttnn_softmax_5 = None
    ttnn_experimental_view_100 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_21, [16, 256, 64]);  ttnn_permute_21 = None
    ttnn_matmul_44 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_99, ttnn_experimental_view_100);  ttnn_experimental_view_99 = ttnn_experimental_view_100 = None
    ttnn_experimental_view_101 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_44, [1, 16, 256, 64]);  ttnn_matmul_44 = None
    ttnn_permute_23 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_101, [0, 2, 1, 3]);  ttnn_experimental_view_101 = None
    ttnn_reshape_25 = ttnn_decorators_ttnn_reshape(ttnn_permute_23, [1, 256, 1024]);  ttnn_permute_23 = None
    ttnn_experimental_view_102 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_25, [256, 1024]);  ttnn_reshape_25 = None
    ttnn_from_torch_95 = ttnn_decorators_ttnn_from_torch(arg91_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg91_1 = None
    ttnn_transpose_39 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_95, 0, 1);  ttnn_from_torch_95 = None
    ttnn_matmul_45 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_102, ttnn_transpose_39);  ttnn_experimental_view_102 = ttnn_transpose_39 = None
    ttnn_from_torch_96 = ttnn_decorators_ttnn_from_torch(arg92_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg92_1 = None
    ttnn_add_33 = ttnn_decorators_ttnn_add(ttnn_from_torch_96, ttnn_matmul_45);  ttnn_from_torch_96 = ttnn_matmul_45 = None
    ttnn_experimental_view_103 = ttnn_decorators_ttnn_experimental_view(ttnn_add_33, [1, 256, 1024]);  ttnn_add_33 = None
    ttnn_add_163 = ttnn_decorators_ttnn_add(ttnn_experimental_view_103, ttnn_layer_norm_10);  ttnn_experimental_view_103 = ttnn_layer_norm_10 = None
    ttnn_from_torch_97 = ttnn_decorators_ttnn_from_torch(arg93_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg93_1 = None
    ttnn_from_torch_98 = ttnn_decorators_ttnn_from_torch(arg94_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg94_1 = None
    ttnn_layer_norm_11 = ttnn_decorators_ttnn_layer_norm(ttnn_add_163, epsilon = 1e-12, weight = ttnn_from_torch_97, bias = ttnn_from_torch_98);  ttnn_add_163 = ttnn_from_torch_97 = ttnn_from_torch_98 = None
    ttnn_experimental_view_104 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_11, [256, 1024])
    ttnn_from_torch_99 = ttnn_decorators_ttnn_from_torch(arg95_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg95_1 = None
    ttnn_transpose_40 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_99, 0, 1);  ttnn_from_torch_99 = None
    ttnn_matmul_46 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_104, ttnn_transpose_40);  ttnn_experimental_view_104 = ttnn_transpose_40 = None
    ttnn_from_torch_100 = ttnn_decorators_ttnn_from_torch(arg96_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg96_1 = None
    ttnn_add_34 = ttnn_decorators_ttnn_add(ttnn_from_torch_100, ttnn_matmul_46);  ttnn_from_torch_100 = ttnn_matmul_46 = None
    ttnn_experimental_view_105 = ttnn_decorators_ttnn_experimental_view(ttnn_add_34, [1, 256, 4096]);  ttnn_add_34 = None
    ttnn_gelu_5 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_105);  ttnn_experimental_view_105 = None
    ttnn_experimental_view_106 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_5, [256, 4096]);  ttnn_gelu_5 = None
    ttnn_from_torch_101 = ttnn_decorators_ttnn_from_torch(arg97_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg97_1 = None
    ttnn_transpose_41 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_101, 0, 1);  ttnn_from_torch_101 = None
    ttnn_matmul_47 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_106, ttnn_transpose_41);  ttnn_experimental_view_106 = ttnn_transpose_41 = None
    ttnn_from_torch_102 = ttnn_decorators_ttnn_from_torch(arg98_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg98_1 = None
    ttnn_add_35 = ttnn_decorators_ttnn_add(ttnn_from_torch_102, ttnn_matmul_47);  ttnn_from_torch_102 = ttnn_matmul_47 = None
    ttnn_experimental_view_107 = ttnn_decorators_ttnn_experimental_view(ttnn_add_35, [1, 256, 1024]);  ttnn_add_35 = None
    ttnn_add_164 = ttnn_decorators_ttnn_add(ttnn_experimental_view_107, ttnn_layer_norm_11);  ttnn_experimental_view_107 = ttnn_layer_norm_11 = None
    ttnn_from_torch_103 = ttnn_decorators_ttnn_from_torch(arg99_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg99_1 = None
    ttnn_from_torch_104 = ttnn_decorators_ttnn_from_torch(arg100_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg100_1 = None
    ttnn_layer_norm_12 = ttnn_decorators_ttnn_layer_norm(ttnn_add_164, epsilon = 1e-12, weight = ttnn_from_torch_103, bias = ttnn_from_torch_104);  ttnn_add_164 = ttnn_from_torch_103 = ttnn_from_torch_104 = None
    ttnn_experimental_view_108 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_12, [256, 1024])
    ttnn_from_torch_105 = ttnn_decorators_ttnn_from_torch(arg101_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg101_1 = None
    ttnn_transpose_42 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_105, 0, 1);  ttnn_from_torch_105 = None
    ttnn_matmul_48 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_108, ttnn_transpose_42);  ttnn_transpose_42 = None
    ttnn_from_torch_106 = ttnn_decorators_ttnn_from_torch(arg102_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg102_1 = None
    ttnn_add_36 = ttnn_decorators_ttnn_add(ttnn_from_torch_106, ttnn_matmul_48);  ttnn_from_torch_106 = ttnn_matmul_48 = None
    ttnn_experimental_view_109 = ttnn_decorators_ttnn_experimental_view(ttnn_add_36, [1, 256, 1024]);  ttnn_add_36 = None
    ttnn_from_torch_107 = ttnn_decorators_ttnn_from_torch(arg103_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg103_1 = None
    ttnn_transpose_43 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_107, 0, 1);  ttnn_from_torch_107 = None
    ttnn_matmul_49 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_108, ttnn_transpose_43);  ttnn_transpose_43 = None
    ttnn_from_torch_108 = ttnn_decorators_ttnn_from_torch(arg104_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg104_1 = None
    ttnn_add_37 = ttnn_decorators_ttnn_add(ttnn_from_torch_108, ttnn_matmul_49);  ttnn_from_torch_108 = ttnn_matmul_49 = None
    ttnn_experimental_view_111 = ttnn_decorators_ttnn_experimental_view(ttnn_add_37, [1, 256, 1024]);  ttnn_add_37 = None
    ttnn_reshape_26 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_111, [1, 256, 16, 64]);  ttnn_experimental_view_111 = None
    ttnn_permute_24 = ttnn_decorators_ttnn_permute(ttnn_reshape_26, [0, 2, 1, 3]);  ttnn_reshape_26 = None
    ttnn_from_torch_109 = ttnn_decorators_ttnn_from_torch(arg105_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg105_1 = None
    ttnn_transpose_44 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_109, 0, 1);  ttnn_from_torch_109 = None
    ttnn_matmul_50 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_108, ttnn_transpose_44);  ttnn_experimental_view_108 = ttnn_transpose_44 = None
    ttnn_from_torch_110 = ttnn_decorators_ttnn_from_torch(arg106_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg106_1 = None
    ttnn_add_38 = ttnn_decorators_ttnn_add(ttnn_from_torch_110, ttnn_matmul_50);  ttnn_from_torch_110 = ttnn_matmul_50 = None
    ttnn_experimental_view_113 = ttnn_decorators_ttnn_experimental_view(ttnn_add_38, [1, 256, 1024]);  ttnn_add_38 = None
    ttnn_reshape_27 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_113, [1, 256, 16, 64]);  ttnn_experimental_view_113 = None
    ttnn_permute_25 = ttnn_decorators_ttnn_permute(ttnn_reshape_27, [0, 2, 1, 3]);  ttnn_reshape_27 = None
    ttnn_reshape_28 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_109, [1, 256, 16, 64]);  ttnn_experimental_view_109 = None
    ttnn_permute_26 = ttnn_decorators_ttnn_permute(ttnn_reshape_28, [0, 2, 1, 3]);  ttnn_reshape_28 = None
    ttnn_transpose_45 = ttnn_decorators_ttnn_transpose(ttnn_permute_24, 3, 2);  ttnn_permute_24 = None
    ttnn_experimental_view_114 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_26, [16, 256, 64]);  ttnn_permute_26 = None
    ttnn_experimental_view_115 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_45, [16, 64, 256]);  ttnn_transpose_45 = None
    ttnn_matmul_51 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_114, ttnn_experimental_view_115);  ttnn_experimental_view_114 = ttnn_experimental_view_115 = None
    ttnn_experimental_view_116 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_51, [1, 16, 256, 256]);  ttnn_matmul_51 = None
    ttnn_multiply_7 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_116, 0.125);  ttnn_experimental_view_116 = None
    ttnn_add_165 = ttnn_decorators_ttnn_add(ttnn_multiply_7, ttnn_multiply);  ttnn_multiply_7 = None
    ttnn_softmax_6 = ttnn_decorators_ttnn_softmax(ttnn_add_165, -1, numeric_stable = True);  ttnn_add_165 = None
    ttnn_experimental_view_117 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_6, [16, 256, 256]);  ttnn_softmax_6 = None
    ttnn_experimental_view_118 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_25, [16, 256, 64]);  ttnn_permute_25 = None
    ttnn_matmul_52 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_117, ttnn_experimental_view_118);  ttnn_experimental_view_117 = ttnn_experimental_view_118 = None
    ttnn_experimental_view_119 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_52, [1, 16, 256, 64]);  ttnn_matmul_52 = None
    ttnn_permute_27 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_119, [0, 2, 1, 3]);  ttnn_experimental_view_119 = None
    ttnn_reshape_29 = ttnn_decorators_ttnn_reshape(ttnn_permute_27, [1, 256, 1024]);  ttnn_permute_27 = None
    ttnn_experimental_view_120 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_29, [256, 1024]);  ttnn_reshape_29 = None
    ttnn_from_torch_111 = ttnn_decorators_ttnn_from_torch(arg107_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg107_1 = None
    ttnn_transpose_46 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_111, 0, 1);  ttnn_from_torch_111 = None
    ttnn_matmul_53 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_120, ttnn_transpose_46);  ttnn_experimental_view_120 = ttnn_transpose_46 = None
    ttnn_from_torch_112 = ttnn_decorators_ttnn_from_torch(arg108_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg108_1 = None
    ttnn_add_39 = ttnn_decorators_ttnn_add(ttnn_from_torch_112, ttnn_matmul_53);  ttnn_from_torch_112 = ttnn_matmul_53 = None
    ttnn_experimental_view_121 = ttnn_decorators_ttnn_experimental_view(ttnn_add_39, [1, 256, 1024]);  ttnn_add_39 = None
    ttnn_add_166 = ttnn_decorators_ttnn_add(ttnn_experimental_view_121, ttnn_layer_norm_12);  ttnn_experimental_view_121 = ttnn_layer_norm_12 = None
    ttnn_from_torch_113 = ttnn_decorators_ttnn_from_torch(arg109_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg109_1 = None
    ttnn_from_torch_114 = ttnn_decorators_ttnn_from_torch(arg110_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg110_1 = None
    ttnn_layer_norm_13 = ttnn_decorators_ttnn_layer_norm(ttnn_add_166, epsilon = 1e-12, weight = ttnn_from_torch_113, bias = ttnn_from_torch_114);  ttnn_add_166 = ttnn_from_torch_113 = ttnn_from_torch_114 = None
    ttnn_experimental_view_122 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_13, [256, 1024])
    ttnn_from_torch_115 = ttnn_decorators_ttnn_from_torch(arg111_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg111_1 = None
    ttnn_transpose_47 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_115, 0, 1);  ttnn_from_torch_115 = None
    ttnn_matmul_54 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_122, ttnn_transpose_47);  ttnn_experimental_view_122 = ttnn_transpose_47 = None
    ttnn_from_torch_116 = ttnn_decorators_ttnn_from_torch(arg112_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg112_1 = None
    ttnn_add_40 = ttnn_decorators_ttnn_add(ttnn_from_torch_116, ttnn_matmul_54);  ttnn_from_torch_116 = ttnn_matmul_54 = None
    ttnn_experimental_view_123 = ttnn_decorators_ttnn_experimental_view(ttnn_add_40, [1, 256, 4096]);  ttnn_add_40 = None
    ttnn_gelu_6 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_123);  ttnn_experimental_view_123 = None
    ttnn_experimental_view_124 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_6, [256, 4096]);  ttnn_gelu_6 = None
    ttnn_from_torch_117 = ttnn_decorators_ttnn_from_torch(arg113_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg113_1 = None
    ttnn_transpose_48 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_117, 0, 1);  ttnn_from_torch_117 = None
    ttnn_matmul_55 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_124, ttnn_transpose_48);  ttnn_experimental_view_124 = ttnn_transpose_48 = None
    ttnn_from_torch_118 = ttnn_decorators_ttnn_from_torch(arg114_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg114_1 = None
    ttnn_add_41 = ttnn_decorators_ttnn_add(ttnn_from_torch_118, ttnn_matmul_55);  ttnn_from_torch_118 = ttnn_matmul_55 = None
    ttnn_experimental_view_125 = ttnn_decorators_ttnn_experimental_view(ttnn_add_41, [1, 256, 1024]);  ttnn_add_41 = None
    ttnn_add_167 = ttnn_decorators_ttnn_add(ttnn_experimental_view_125, ttnn_layer_norm_13);  ttnn_experimental_view_125 = ttnn_layer_norm_13 = None
    ttnn_from_torch_119 = ttnn_decorators_ttnn_from_torch(arg115_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg115_1 = None
    ttnn_from_torch_120 = ttnn_decorators_ttnn_from_torch(arg116_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg116_1 = None
    ttnn_layer_norm_14 = ttnn_decorators_ttnn_layer_norm(ttnn_add_167, epsilon = 1e-12, weight = ttnn_from_torch_119, bias = ttnn_from_torch_120);  ttnn_add_167 = ttnn_from_torch_119 = ttnn_from_torch_120 = None
    ttnn_experimental_view_126 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_14, [256, 1024])
    ttnn_from_torch_121 = ttnn_decorators_ttnn_from_torch(arg117_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg117_1 = None
    ttnn_transpose_49 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_121, 0, 1);  ttnn_from_torch_121 = None
    ttnn_matmul_56 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_126, ttnn_transpose_49);  ttnn_transpose_49 = None
    ttnn_from_torch_122 = ttnn_decorators_ttnn_from_torch(arg118_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg118_1 = None
    ttnn_add_42 = ttnn_decorators_ttnn_add(ttnn_from_torch_122, ttnn_matmul_56);  ttnn_from_torch_122 = ttnn_matmul_56 = None
    ttnn_experimental_view_127 = ttnn_decorators_ttnn_experimental_view(ttnn_add_42, [1, 256, 1024]);  ttnn_add_42 = None
    ttnn_from_torch_123 = ttnn_decorators_ttnn_from_torch(arg119_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg119_1 = None
    ttnn_transpose_50 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_123, 0, 1);  ttnn_from_torch_123 = None
    ttnn_matmul_57 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_126, ttnn_transpose_50);  ttnn_transpose_50 = None
    ttnn_from_torch_124 = ttnn_decorators_ttnn_from_torch(arg120_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg120_1 = None
    ttnn_add_43 = ttnn_decorators_ttnn_add(ttnn_from_torch_124, ttnn_matmul_57);  ttnn_from_torch_124 = ttnn_matmul_57 = None
    ttnn_experimental_view_129 = ttnn_decorators_ttnn_experimental_view(ttnn_add_43, [1, 256, 1024]);  ttnn_add_43 = None
    ttnn_reshape_30 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_129, [1, 256, 16, 64]);  ttnn_experimental_view_129 = None
    ttnn_permute_28 = ttnn_decorators_ttnn_permute(ttnn_reshape_30, [0, 2, 1, 3]);  ttnn_reshape_30 = None
    ttnn_from_torch_125 = ttnn_decorators_ttnn_from_torch(arg121_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg121_1 = None
    ttnn_transpose_51 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_125, 0, 1);  ttnn_from_torch_125 = None
    ttnn_matmul_58 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_126, ttnn_transpose_51);  ttnn_experimental_view_126 = ttnn_transpose_51 = None
    ttnn_from_torch_126 = ttnn_decorators_ttnn_from_torch(arg122_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg122_1 = None
    ttnn_add_44 = ttnn_decorators_ttnn_add(ttnn_from_torch_126, ttnn_matmul_58);  ttnn_from_torch_126 = ttnn_matmul_58 = None
    ttnn_experimental_view_131 = ttnn_decorators_ttnn_experimental_view(ttnn_add_44, [1, 256, 1024]);  ttnn_add_44 = None
    ttnn_reshape_31 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_131, [1, 256, 16, 64]);  ttnn_experimental_view_131 = None
    ttnn_permute_29 = ttnn_decorators_ttnn_permute(ttnn_reshape_31, [0, 2, 1, 3]);  ttnn_reshape_31 = None
    ttnn_reshape_32 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_127, [1, 256, 16, 64]);  ttnn_experimental_view_127 = None
    ttnn_permute_30 = ttnn_decorators_ttnn_permute(ttnn_reshape_32, [0, 2, 1, 3]);  ttnn_reshape_32 = None
    ttnn_transpose_52 = ttnn_decorators_ttnn_transpose(ttnn_permute_28, 3, 2);  ttnn_permute_28 = None
    ttnn_experimental_view_132 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_30, [16, 256, 64]);  ttnn_permute_30 = None
    ttnn_experimental_view_133 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_52, [16, 64, 256]);  ttnn_transpose_52 = None
    ttnn_matmul_59 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_132, ttnn_experimental_view_133);  ttnn_experimental_view_132 = ttnn_experimental_view_133 = None
    ttnn_experimental_view_134 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_59, [1, 16, 256, 256]);  ttnn_matmul_59 = None
    ttnn_multiply_8 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_134, 0.125);  ttnn_experimental_view_134 = None
    ttnn_add_168 = ttnn_decorators_ttnn_add(ttnn_multiply_8, ttnn_multiply);  ttnn_multiply_8 = None
    ttnn_softmax_7 = ttnn_decorators_ttnn_softmax(ttnn_add_168, -1, numeric_stable = True);  ttnn_add_168 = None
    ttnn_experimental_view_135 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_7, [16, 256, 256]);  ttnn_softmax_7 = None
    ttnn_experimental_view_136 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_29, [16, 256, 64]);  ttnn_permute_29 = None
    ttnn_matmul_60 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_135, ttnn_experimental_view_136);  ttnn_experimental_view_135 = ttnn_experimental_view_136 = None
    ttnn_experimental_view_137 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_60, [1, 16, 256, 64]);  ttnn_matmul_60 = None
    ttnn_permute_31 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_137, [0, 2, 1, 3]);  ttnn_experimental_view_137 = None
    ttnn_reshape_33 = ttnn_decorators_ttnn_reshape(ttnn_permute_31, [1, 256, 1024]);  ttnn_permute_31 = None
    ttnn_experimental_view_138 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_33, [256, 1024]);  ttnn_reshape_33 = None
    ttnn_from_torch_127 = ttnn_decorators_ttnn_from_torch(arg123_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg123_1 = None
    ttnn_transpose_53 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_127, 0, 1);  ttnn_from_torch_127 = None
    ttnn_matmul_61 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_138, ttnn_transpose_53);  ttnn_experimental_view_138 = ttnn_transpose_53 = None
    ttnn_from_torch_128 = ttnn_decorators_ttnn_from_torch(arg124_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg124_1 = None
    ttnn_add_45 = ttnn_decorators_ttnn_add(ttnn_from_torch_128, ttnn_matmul_61);  ttnn_from_torch_128 = ttnn_matmul_61 = None
    ttnn_experimental_view_139 = ttnn_decorators_ttnn_experimental_view(ttnn_add_45, [1, 256, 1024]);  ttnn_add_45 = None
    ttnn_add_169 = ttnn_decorators_ttnn_add(ttnn_experimental_view_139, ttnn_layer_norm_14);  ttnn_experimental_view_139 = ttnn_layer_norm_14 = None
    ttnn_from_torch_129 = ttnn_decorators_ttnn_from_torch(arg125_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg125_1 = None
    ttnn_from_torch_130 = ttnn_decorators_ttnn_from_torch(arg126_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg126_1 = None
    ttnn_layer_norm_15 = ttnn_decorators_ttnn_layer_norm(ttnn_add_169, epsilon = 1e-12, weight = ttnn_from_torch_129, bias = ttnn_from_torch_130);  ttnn_add_169 = ttnn_from_torch_129 = ttnn_from_torch_130 = None
    ttnn_experimental_view_140 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_15, [256, 1024])
    ttnn_from_torch_131 = ttnn_decorators_ttnn_from_torch(arg127_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg127_1 = None
    ttnn_transpose_54 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_131, 0, 1);  ttnn_from_torch_131 = None
    ttnn_matmul_62 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_140, ttnn_transpose_54);  ttnn_experimental_view_140 = ttnn_transpose_54 = None
    ttnn_from_torch_132 = ttnn_decorators_ttnn_from_torch(arg128_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg128_1 = None
    ttnn_add_46 = ttnn_decorators_ttnn_add(ttnn_from_torch_132, ttnn_matmul_62);  ttnn_from_torch_132 = ttnn_matmul_62 = None
    ttnn_experimental_view_141 = ttnn_decorators_ttnn_experimental_view(ttnn_add_46, [1, 256, 4096]);  ttnn_add_46 = None
    ttnn_gelu_7 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_141);  ttnn_experimental_view_141 = None
    ttnn_experimental_view_142 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_7, [256, 4096]);  ttnn_gelu_7 = None
    ttnn_from_torch_133 = ttnn_decorators_ttnn_from_torch(arg129_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg129_1 = None
    ttnn_transpose_55 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_133, 0, 1);  ttnn_from_torch_133 = None
    ttnn_matmul_63 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_142, ttnn_transpose_55);  ttnn_experimental_view_142 = ttnn_transpose_55 = None
    ttnn_from_torch_134 = ttnn_decorators_ttnn_from_torch(arg130_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg130_1 = None
    ttnn_add_47 = ttnn_decorators_ttnn_add(ttnn_from_torch_134, ttnn_matmul_63);  ttnn_from_torch_134 = ttnn_matmul_63 = None
    ttnn_experimental_view_143 = ttnn_decorators_ttnn_experimental_view(ttnn_add_47, [1, 256, 1024]);  ttnn_add_47 = None
    ttnn_add_170 = ttnn_decorators_ttnn_add(ttnn_experimental_view_143, ttnn_layer_norm_15);  ttnn_experimental_view_143 = ttnn_layer_norm_15 = None
    ttnn_from_torch_135 = ttnn_decorators_ttnn_from_torch(arg131_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg131_1 = None
    ttnn_from_torch_136 = ttnn_decorators_ttnn_from_torch(arg132_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg132_1 = None
    ttnn_layer_norm_16 = ttnn_decorators_ttnn_layer_norm(ttnn_add_170, epsilon = 1e-12, weight = ttnn_from_torch_135, bias = ttnn_from_torch_136);  ttnn_add_170 = ttnn_from_torch_135 = ttnn_from_torch_136 = None
    ttnn_experimental_view_144 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_16, [256, 1024])
    ttnn_from_torch_137 = ttnn_decorators_ttnn_from_torch(arg133_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg133_1 = None
    ttnn_transpose_56 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_137, 0, 1);  ttnn_from_torch_137 = None
    ttnn_matmul_64 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_144, ttnn_transpose_56);  ttnn_transpose_56 = None
    ttnn_from_torch_138 = ttnn_decorators_ttnn_from_torch(arg134_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg134_1 = None
    ttnn_add_48 = ttnn_decorators_ttnn_add(ttnn_from_torch_138, ttnn_matmul_64);  ttnn_from_torch_138 = ttnn_matmul_64 = None
    ttnn_experimental_view_145 = ttnn_decorators_ttnn_experimental_view(ttnn_add_48, [1, 256, 1024]);  ttnn_add_48 = None
    ttnn_from_torch_139 = ttnn_decorators_ttnn_from_torch(arg135_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg135_1 = None
    ttnn_transpose_57 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_139, 0, 1);  ttnn_from_torch_139 = None
    ttnn_matmul_65 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_144, ttnn_transpose_57);  ttnn_transpose_57 = None
    ttnn_from_torch_140 = ttnn_decorators_ttnn_from_torch(arg136_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg136_1 = None
    ttnn_add_49 = ttnn_decorators_ttnn_add(ttnn_from_torch_140, ttnn_matmul_65);  ttnn_from_torch_140 = ttnn_matmul_65 = None
    ttnn_experimental_view_147 = ttnn_decorators_ttnn_experimental_view(ttnn_add_49, [1, 256, 1024]);  ttnn_add_49 = None
    ttnn_reshape_34 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_147, [1, 256, 16, 64]);  ttnn_experimental_view_147 = None
    ttnn_permute_32 = ttnn_decorators_ttnn_permute(ttnn_reshape_34, [0, 2, 1, 3]);  ttnn_reshape_34 = None
    ttnn_from_torch_141 = ttnn_decorators_ttnn_from_torch(arg137_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg137_1 = None
    ttnn_transpose_58 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_141, 0, 1);  ttnn_from_torch_141 = None
    ttnn_matmul_66 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_144, ttnn_transpose_58);  ttnn_experimental_view_144 = ttnn_transpose_58 = None
    ttnn_from_torch_142 = ttnn_decorators_ttnn_from_torch(arg138_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg138_1 = None
    ttnn_add_50 = ttnn_decorators_ttnn_add(ttnn_from_torch_142, ttnn_matmul_66);  ttnn_from_torch_142 = ttnn_matmul_66 = None
    ttnn_experimental_view_149 = ttnn_decorators_ttnn_experimental_view(ttnn_add_50, [1, 256, 1024]);  ttnn_add_50 = None
    ttnn_reshape_35 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_149, [1, 256, 16, 64]);  ttnn_experimental_view_149 = None
    ttnn_permute_33 = ttnn_decorators_ttnn_permute(ttnn_reshape_35, [0, 2, 1, 3]);  ttnn_reshape_35 = None
    ttnn_reshape_36 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_145, [1, 256, 16, 64]);  ttnn_experimental_view_145 = None
    ttnn_permute_34 = ttnn_decorators_ttnn_permute(ttnn_reshape_36, [0, 2, 1, 3]);  ttnn_reshape_36 = None
    ttnn_transpose_59 = ttnn_decorators_ttnn_transpose(ttnn_permute_32, 3, 2);  ttnn_permute_32 = None
    ttnn_experimental_view_150 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_34, [16, 256, 64]);  ttnn_permute_34 = None
    ttnn_experimental_view_151 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_59, [16, 64, 256]);  ttnn_transpose_59 = None
    ttnn_matmul_67 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_150, ttnn_experimental_view_151);  ttnn_experimental_view_150 = ttnn_experimental_view_151 = None
    ttnn_experimental_view_152 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_67, [1, 16, 256, 256]);  ttnn_matmul_67 = None
    ttnn_multiply_9 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_152, 0.125);  ttnn_experimental_view_152 = None
    ttnn_add_171 = ttnn_decorators_ttnn_add(ttnn_multiply_9, ttnn_multiply);  ttnn_multiply_9 = None
    ttnn_softmax_8 = ttnn_decorators_ttnn_softmax(ttnn_add_171, -1, numeric_stable = True);  ttnn_add_171 = None
    ttnn_experimental_view_153 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_8, [16, 256, 256]);  ttnn_softmax_8 = None
    ttnn_experimental_view_154 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_33, [16, 256, 64]);  ttnn_permute_33 = None
    ttnn_matmul_68 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_153, ttnn_experimental_view_154);  ttnn_experimental_view_153 = ttnn_experimental_view_154 = None
    ttnn_experimental_view_155 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_68, [1, 16, 256, 64]);  ttnn_matmul_68 = None
    ttnn_permute_35 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_155, [0, 2, 1, 3]);  ttnn_experimental_view_155 = None
    ttnn_reshape_37 = ttnn_decorators_ttnn_reshape(ttnn_permute_35, [1, 256, 1024]);  ttnn_permute_35 = None
    ttnn_experimental_view_156 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_37, [256, 1024]);  ttnn_reshape_37 = None
    ttnn_from_torch_143 = ttnn_decorators_ttnn_from_torch(arg139_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg139_1 = None
    ttnn_transpose_60 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_143, 0, 1);  ttnn_from_torch_143 = None
    ttnn_matmul_69 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_156, ttnn_transpose_60);  ttnn_experimental_view_156 = ttnn_transpose_60 = None
    ttnn_from_torch_144 = ttnn_decorators_ttnn_from_torch(arg140_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg140_1 = None
    ttnn_add_51 = ttnn_decorators_ttnn_add(ttnn_from_torch_144, ttnn_matmul_69);  ttnn_from_torch_144 = ttnn_matmul_69 = None
    ttnn_experimental_view_157 = ttnn_decorators_ttnn_experimental_view(ttnn_add_51, [1, 256, 1024]);  ttnn_add_51 = None
    ttnn_add_172 = ttnn_decorators_ttnn_add(ttnn_experimental_view_157, ttnn_layer_norm_16);  ttnn_experimental_view_157 = ttnn_layer_norm_16 = None
    ttnn_from_torch_145 = ttnn_decorators_ttnn_from_torch(arg141_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg141_1 = None
    ttnn_from_torch_146 = ttnn_decorators_ttnn_from_torch(arg142_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg142_1 = None
    ttnn_layer_norm_17 = ttnn_decorators_ttnn_layer_norm(ttnn_add_172, epsilon = 1e-12, weight = ttnn_from_torch_145, bias = ttnn_from_torch_146);  ttnn_add_172 = ttnn_from_torch_145 = ttnn_from_torch_146 = None
    ttnn_experimental_view_158 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_17, [256, 1024])
    ttnn_from_torch_147 = ttnn_decorators_ttnn_from_torch(arg143_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg143_1 = None
    ttnn_transpose_61 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_147, 0, 1);  ttnn_from_torch_147 = None
    ttnn_matmul_70 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_158, ttnn_transpose_61);  ttnn_experimental_view_158 = ttnn_transpose_61 = None
    ttnn_from_torch_148 = ttnn_decorators_ttnn_from_torch(arg144_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg144_1 = None
    ttnn_add_52 = ttnn_decorators_ttnn_add(ttnn_from_torch_148, ttnn_matmul_70);  ttnn_from_torch_148 = ttnn_matmul_70 = None
    ttnn_experimental_view_159 = ttnn_decorators_ttnn_experimental_view(ttnn_add_52, [1, 256, 4096]);  ttnn_add_52 = None
    ttnn_gelu_8 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_159);  ttnn_experimental_view_159 = None
    ttnn_experimental_view_160 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_8, [256, 4096]);  ttnn_gelu_8 = None
    ttnn_from_torch_149 = ttnn_decorators_ttnn_from_torch(arg145_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg145_1 = None
    ttnn_transpose_62 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_149, 0, 1);  ttnn_from_torch_149 = None
    ttnn_matmul_71 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_160, ttnn_transpose_62);  ttnn_experimental_view_160 = ttnn_transpose_62 = None
    ttnn_from_torch_150 = ttnn_decorators_ttnn_from_torch(arg146_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg146_1 = None
    ttnn_add_53 = ttnn_decorators_ttnn_add(ttnn_from_torch_150, ttnn_matmul_71);  ttnn_from_torch_150 = ttnn_matmul_71 = None
    ttnn_experimental_view_161 = ttnn_decorators_ttnn_experimental_view(ttnn_add_53, [1, 256, 1024]);  ttnn_add_53 = None
    ttnn_add_173 = ttnn_decorators_ttnn_add(ttnn_experimental_view_161, ttnn_layer_norm_17);  ttnn_experimental_view_161 = ttnn_layer_norm_17 = None
    ttnn_from_torch_151 = ttnn_decorators_ttnn_from_torch(arg147_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg147_1 = None
    ttnn_from_torch_152 = ttnn_decorators_ttnn_from_torch(arg148_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg148_1 = None
    ttnn_layer_norm_18 = ttnn_decorators_ttnn_layer_norm(ttnn_add_173, epsilon = 1e-12, weight = ttnn_from_torch_151, bias = ttnn_from_torch_152);  ttnn_add_173 = ttnn_from_torch_151 = ttnn_from_torch_152 = None
    ttnn_experimental_view_162 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_18, [256, 1024])
    ttnn_from_torch_153 = ttnn_decorators_ttnn_from_torch(arg149_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg149_1 = None
    ttnn_transpose_63 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_153, 0, 1);  ttnn_from_torch_153 = None
    ttnn_matmul_72 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_162, ttnn_transpose_63);  ttnn_transpose_63 = None
    ttnn_from_torch_154 = ttnn_decorators_ttnn_from_torch(arg150_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg150_1 = None
    ttnn_add_54 = ttnn_decorators_ttnn_add(ttnn_from_torch_154, ttnn_matmul_72);  ttnn_from_torch_154 = ttnn_matmul_72 = None
    ttnn_experimental_view_163 = ttnn_decorators_ttnn_experimental_view(ttnn_add_54, [1, 256, 1024]);  ttnn_add_54 = None
    ttnn_from_torch_155 = ttnn_decorators_ttnn_from_torch(arg151_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg151_1 = None
    ttnn_transpose_64 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_155, 0, 1);  ttnn_from_torch_155 = None
    ttnn_matmul_73 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_162, ttnn_transpose_64);  ttnn_transpose_64 = None
    ttnn_from_torch_156 = ttnn_decorators_ttnn_from_torch(arg152_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg152_1 = None
    ttnn_add_55 = ttnn_decorators_ttnn_add(ttnn_from_torch_156, ttnn_matmul_73);  ttnn_from_torch_156 = ttnn_matmul_73 = None
    ttnn_experimental_view_165 = ttnn_decorators_ttnn_experimental_view(ttnn_add_55, [1, 256, 1024]);  ttnn_add_55 = None
    ttnn_reshape_38 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_165, [1, 256, 16, 64]);  ttnn_experimental_view_165 = None
    ttnn_permute_36 = ttnn_decorators_ttnn_permute(ttnn_reshape_38, [0, 2, 1, 3]);  ttnn_reshape_38 = None
    ttnn_from_torch_157 = ttnn_decorators_ttnn_from_torch(arg153_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg153_1 = None
    ttnn_transpose_65 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_157, 0, 1);  ttnn_from_torch_157 = None
    ttnn_matmul_74 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_162, ttnn_transpose_65);  ttnn_experimental_view_162 = ttnn_transpose_65 = None
    ttnn_from_torch_158 = ttnn_decorators_ttnn_from_torch(arg154_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg154_1 = None
    ttnn_add_56 = ttnn_decorators_ttnn_add(ttnn_from_torch_158, ttnn_matmul_74);  ttnn_from_torch_158 = ttnn_matmul_74 = None
    ttnn_experimental_view_167 = ttnn_decorators_ttnn_experimental_view(ttnn_add_56, [1, 256, 1024]);  ttnn_add_56 = None
    ttnn_reshape_39 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_167, [1, 256, 16, 64]);  ttnn_experimental_view_167 = None
    ttnn_permute_37 = ttnn_decorators_ttnn_permute(ttnn_reshape_39, [0, 2, 1, 3]);  ttnn_reshape_39 = None
    ttnn_reshape_40 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_163, [1, 256, 16, 64]);  ttnn_experimental_view_163 = None
    ttnn_permute_38 = ttnn_decorators_ttnn_permute(ttnn_reshape_40, [0, 2, 1, 3]);  ttnn_reshape_40 = None
    ttnn_transpose_66 = ttnn_decorators_ttnn_transpose(ttnn_permute_36, 3, 2);  ttnn_permute_36 = None
    ttnn_experimental_view_168 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_38, [16, 256, 64]);  ttnn_permute_38 = None
    ttnn_experimental_view_169 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_66, [16, 64, 256]);  ttnn_transpose_66 = None
    ttnn_matmul_75 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_168, ttnn_experimental_view_169);  ttnn_experimental_view_168 = ttnn_experimental_view_169 = None
    ttnn_experimental_view_170 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_75, [1, 16, 256, 256]);  ttnn_matmul_75 = None
    ttnn_multiply_10 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_170, 0.125);  ttnn_experimental_view_170 = None
    ttnn_add_174 = ttnn_decorators_ttnn_add(ttnn_multiply_10, ttnn_multiply);  ttnn_multiply_10 = None
    ttnn_softmax_9 = ttnn_decorators_ttnn_softmax(ttnn_add_174, -1, numeric_stable = True);  ttnn_add_174 = None
    ttnn_experimental_view_171 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_9, [16, 256, 256]);  ttnn_softmax_9 = None
    ttnn_experimental_view_172 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_37, [16, 256, 64]);  ttnn_permute_37 = None
    ttnn_matmul_76 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_171, ttnn_experimental_view_172);  ttnn_experimental_view_171 = ttnn_experimental_view_172 = None
    ttnn_experimental_view_173 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_76, [1, 16, 256, 64]);  ttnn_matmul_76 = None
    ttnn_permute_39 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_173, [0, 2, 1, 3]);  ttnn_experimental_view_173 = None
    ttnn_reshape_41 = ttnn_decorators_ttnn_reshape(ttnn_permute_39, [1, 256, 1024]);  ttnn_permute_39 = None
    ttnn_experimental_view_174 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_41, [256, 1024]);  ttnn_reshape_41 = None
    ttnn_from_torch_159 = ttnn_decorators_ttnn_from_torch(arg155_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg155_1 = None
    ttnn_transpose_67 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_159, 0, 1);  ttnn_from_torch_159 = None
    ttnn_matmul_77 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_174, ttnn_transpose_67);  ttnn_experimental_view_174 = ttnn_transpose_67 = None
    ttnn_from_torch_160 = ttnn_decorators_ttnn_from_torch(arg156_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg156_1 = None
    ttnn_add_57 = ttnn_decorators_ttnn_add(ttnn_from_torch_160, ttnn_matmul_77);  ttnn_from_torch_160 = ttnn_matmul_77 = None
    ttnn_experimental_view_175 = ttnn_decorators_ttnn_experimental_view(ttnn_add_57, [1, 256, 1024]);  ttnn_add_57 = None
    ttnn_add_175 = ttnn_decorators_ttnn_add(ttnn_experimental_view_175, ttnn_layer_norm_18);  ttnn_experimental_view_175 = ttnn_layer_norm_18 = None
    ttnn_from_torch_161 = ttnn_decorators_ttnn_from_torch(arg157_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg157_1 = None
    ttnn_from_torch_162 = ttnn_decorators_ttnn_from_torch(arg158_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg158_1 = None
    ttnn_layer_norm_19 = ttnn_decorators_ttnn_layer_norm(ttnn_add_175, epsilon = 1e-12, weight = ttnn_from_torch_161, bias = ttnn_from_torch_162);  ttnn_add_175 = ttnn_from_torch_161 = ttnn_from_torch_162 = None
    ttnn_experimental_view_176 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_19, [256, 1024])
    ttnn_from_torch_163 = ttnn_decorators_ttnn_from_torch(arg159_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg159_1 = None
    ttnn_transpose_68 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_163, 0, 1);  ttnn_from_torch_163 = None
    ttnn_matmul_78 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_176, ttnn_transpose_68);  ttnn_experimental_view_176 = ttnn_transpose_68 = None
    ttnn_from_torch_164 = ttnn_decorators_ttnn_from_torch(arg160_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg160_1 = None
    ttnn_add_58 = ttnn_decorators_ttnn_add(ttnn_from_torch_164, ttnn_matmul_78);  ttnn_from_torch_164 = ttnn_matmul_78 = None
    ttnn_experimental_view_177 = ttnn_decorators_ttnn_experimental_view(ttnn_add_58, [1, 256, 4096]);  ttnn_add_58 = None
    ttnn_gelu_9 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_177);  ttnn_experimental_view_177 = None
    ttnn_experimental_view_178 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_9, [256, 4096]);  ttnn_gelu_9 = None
    ttnn_from_torch_165 = ttnn_decorators_ttnn_from_torch(arg161_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg161_1 = None
    ttnn_transpose_69 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_165, 0, 1);  ttnn_from_torch_165 = None
    ttnn_matmul_79 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_178, ttnn_transpose_69);  ttnn_experimental_view_178 = ttnn_transpose_69 = None
    ttnn_from_torch_166 = ttnn_decorators_ttnn_from_torch(arg162_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg162_1 = None
    ttnn_add_59 = ttnn_decorators_ttnn_add(ttnn_from_torch_166, ttnn_matmul_79);  ttnn_from_torch_166 = ttnn_matmul_79 = None
    ttnn_experimental_view_179 = ttnn_decorators_ttnn_experimental_view(ttnn_add_59, [1, 256, 1024]);  ttnn_add_59 = None
    ttnn_add_176 = ttnn_decorators_ttnn_add(ttnn_experimental_view_179, ttnn_layer_norm_19);  ttnn_experimental_view_179 = ttnn_layer_norm_19 = None
    ttnn_from_torch_167 = ttnn_decorators_ttnn_from_torch(arg163_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg163_1 = None
    ttnn_from_torch_168 = ttnn_decorators_ttnn_from_torch(arg164_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg164_1 = None
    ttnn_layer_norm_20 = ttnn_decorators_ttnn_layer_norm(ttnn_add_176, epsilon = 1e-12, weight = ttnn_from_torch_167, bias = ttnn_from_torch_168);  ttnn_add_176 = ttnn_from_torch_167 = ttnn_from_torch_168 = None
    ttnn_experimental_view_180 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_20, [256, 1024])
    ttnn_from_torch_169 = ttnn_decorators_ttnn_from_torch(arg165_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg165_1 = None
    ttnn_transpose_70 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_169, 0, 1);  ttnn_from_torch_169 = None
    ttnn_matmul_80 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_180, ttnn_transpose_70);  ttnn_transpose_70 = None
    ttnn_from_torch_170 = ttnn_decorators_ttnn_from_torch(arg166_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg166_1 = None
    ttnn_add_60 = ttnn_decorators_ttnn_add(ttnn_from_torch_170, ttnn_matmul_80);  ttnn_from_torch_170 = ttnn_matmul_80 = None
    ttnn_experimental_view_181 = ttnn_decorators_ttnn_experimental_view(ttnn_add_60, [1, 256, 1024]);  ttnn_add_60 = None
    ttnn_from_torch_171 = ttnn_decorators_ttnn_from_torch(arg167_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg167_1 = None
    ttnn_transpose_71 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_171, 0, 1);  ttnn_from_torch_171 = None
    ttnn_matmul_81 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_180, ttnn_transpose_71);  ttnn_transpose_71 = None
    ttnn_from_torch_172 = ttnn_decorators_ttnn_from_torch(arg168_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg168_1 = None
    ttnn_add_61 = ttnn_decorators_ttnn_add(ttnn_from_torch_172, ttnn_matmul_81);  ttnn_from_torch_172 = ttnn_matmul_81 = None
    ttnn_experimental_view_183 = ttnn_decorators_ttnn_experimental_view(ttnn_add_61, [1, 256, 1024]);  ttnn_add_61 = None
    ttnn_reshape_42 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_183, [1, 256, 16, 64]);  ttnn_experimental_view_183 = None
    ttnn_permute_40 = ttnn_decorators_ttnn_permute(ttnn_reshape_42, [0, 2, 1, 3]);  ttnn_reshape_42 = None
    ttnn_from_torch_173 = ttnn_decorators_ttnn_from_torch(arg169_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg169_1 = None
    ttnn_transpose_72 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_173, 0, 1);  ttnn_from_torch_173 = None
    ttnn_matmul_82 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_180, ttnn_transpose_72);  ttnn_experimental_view_180 = ttnn_transpose_72 = None
    ttnn_from_torch_174 = ttnn_decorators_ttnn_from_torch(arg170_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg170_1 = None
    ttnn_add_62 = ttnn_decorators_ttnn_add(ttnn_from_torch_174, ttnn_matmul_82);  ttnn_from_torch_174 = ttnn_matmul_82 = None
    ttnn_experimental_view_185 = ttnn_decorators_ttnn_experimental_view(ttnn_add_62, [1, 256, 1024]);  ttnn_add_62 = None
    ttnn_reshape_43 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_185, [1, 256, 16, 64]);  ttnn_experimental_view_185 = None
    ttnn_permute_41 = ttnn_decorators_ttnn_permute(ttnn_reshape_43, [0, 2, 1, 3]);  ttnn_reshape_43 = None
    ttnn_reshape_44 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_181, [1, 256, 16, 64]);  ttnn_experimental_view_181 = None
    ttnn_permute_42 = ttnn_decorators_ttnn_permute(ttnn_reshape_44, [0, 2, 1, 3]);  ttnn_reshape_44 = None
    ttnn_transpose_73 = ttnn_decorators_ttnn_transpose(ttnn_permute_40, 3, 2);  ttnn_permute_40 = None
    ttnn_experimental_view_186 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_42, [16, 256, 64]);  ttnn_permute_42 = None
    ttnn_experimental_view_187 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_73, [16, 64, 256]);  ttnn_transpose_73 = None
    ttnn_matmul_83 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_186, ttnn_experimental_view_187);  ttnn_experimental_view_186 = ttnn_experimental_view_187 = None
    ttnn_experimental_view_188 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_83, [1, 16, 256, 256]);  ttnn_matmul_83 = None
    ttnn_multiply_11 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_188, 0.125);  ttnn_experimental_view_188 = None
    ttnn_add_177 = ttnn_decorators_ttnn_add(ttnn_multiply_11, ttnn_multiply);  ttnn_multiply_11 = None
    ttnn_softmax_10 = ttnn_decorators_ttnn_softmax(ttnn_add_177, -1, numeric_stable = True);  ttnn_add_177 = None
    ttnn_experimental_view_189 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_10, [16, 256, 256]);  ttnn_softmax_10 = None
    ttnn_experimental_view_190 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_41, [16, 256, 64]);  ttnn_permute_41 = None
    ttnn_matmul_84 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_189, ttnn_experimental_view_190);  ttnn_experimental_view_189 = ttnn_experimental_view_190 = None
    ttnn_experimental_view_191 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_84, [1, 16, 256, 64]);  ttnn_matmul_84 = None
    ttnn_permute_43 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_191, [0, 2, 1, 3]);  ttnn_experimental_view_191 = None
    ttnn_reshape_45 = ttnn_decorators_ttnn_reshape(ttnn_permute_43, [1, 256, 1024]);  ttnn_permute_43 = None
    ttnn_experimental_view_192 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_45, [256, 1024]);  ttnn_reshape_45 = None
    ttnn_from_torch_175 = ttnn_decorators_ttnn_from_torch(arg171_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg171_1 = None
    ttnn_transpose_74 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_175, 0, 1);  ttnn_from_torch_175 = None
    ttnn_matmul_85 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_192, ttnn_transpose_74);  ttnn_experimental_view_192 = ttnn_transpose_74 = None
    ttnn_from_torch_176 = ttnn_decorators_ttnn_from_torch(arg172_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg172_1 = None
    ttnn_add_63 = ttnn_decorators_ttnn_add(ttnn_from_torch_176, ttnn_matmul_85);  ttnn_from_torch_176 = ttnn_matmul_85 = None
    ttnn_experimental_view_193 = ttnn_decorators_ttnn_experimental_view(ttnn_add_63, [1, 256, 1024]);  ttnn_add_63 = None
    ttnn_add_178 = ttnn_decorators_ttnn_add(ttnn_experimental_view_193, ttnn_layer_norm_20);  ttnn_experimental_view_193 = ttnn_layer_norm_20 = None
    ttnn_from_torch_177 = ttnn_decorators_ttnn_from_torch(arg173_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg173_1 = None
    ttnn_from_torch_178 = ttnn_decorators_ttnn_from_torch(arg174_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg174_1 = None
    ttnn_layer_norm_21 = ttnn_decorators_ttnn_layer_norm(ttnn_add_178, epsilon = 1e-12, weight = ttnn_from_torch_177, bias = ttnn_from_torch_178);  ttnn_add_178 = ttnn_from_torch_177 = ttnn_from_torch_178 = None
    ttnn_experimental_view_194 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_21, [256, 1024])
    ttnn_from_torch_179 = ttnn_decorators_ttnn_from_torch(arg175_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg175_1 = None
    ttnn_transpose_75 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_179, 0, 1);  ttnn_from_torch_179 = None
    ttnn_matmul_86 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_194, ttnn_transpose_75);  ttnn_experimental_view_194 = ttnn_transpose_75 = None
    ttnn_from_torch_180 = ttnn_decorators_ttnn_from_torch(arg176_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg176_1 = None
    ttnn_add_64 = ttnn_decorators_ttnn_add(ttnn_from_torch_180, ttnn_matmul_86);  ttnn_from_torch_180 = ttnn_matmul_86 = None
    ttnn_experimental_view_195 = ttnn_decorators_ttnn_experimental_view(ttnn_add_64, [1, 256, 4096]);  ttnn_add_64 = None
    ttnn_gelu_10 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_195);  ttnn_experimental_view_195 = None
    ttnn_experimental_view_196 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_10, [256, 4096]);  ttnn_gelu_10 = None
    ttnn_from_torch_181 = ttnn_decorators_ttnn_from_torch(arg177_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg177_1 = None
    ttnn_transpose_76 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_181, 0, 1);  ttnn_from_torch_181 = None
    ttnn_matmul_87 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_196, ttnn_transpose_76);  ttnn_experimental_view_196 = ttnn_transpose_76 = None
    ttnn_from_torch_182 = ttnn_decorators_ttnn_from_torch(arg178_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg178_1 = None
    ttnn_add_65 = ttnn_decorators_ttnn_add(ttnn_from_torch_182, ttnn_matmul_87);  ttnn_from_torch_182 = ttnn_matmul_87 = None
    ttnn_experimental_view_197 = ttnn_decorators_ttnn_experimental_view(ttnn_add_65, [1, 256, 1024]);  ttnn_add_65 = None
    ttnn_add_179 = ttnn_decorators_ttnn_add(ttnn_experimental_view_197, ttnn_layer_norm_21);  ttnn_experimental_view_197 = ttnn_layer_norm_21 = None
    ttnn_from_torch_183 = ttnn_decorators_ttnn_from_torch(arg179_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg179_1 = None
    ttnn_from_torch_184 = ttnn_decorators_ttnn_from_torch(arg180_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg180_1 = None
    ttnn_layer_norm_22 = ttnn_decorators_ttnn_layer_norm(ttnn_add_179, epsilon = 1e-12, weight = ttnn_from_torch_183, bias = ttnn_from_torch_184);  ttnn_add_179 = ttnn_from_torch_183 = ttnn_from_torch_184 = None
    ttnn_experimental_view_198 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_22, [256, 1024])
    ttnn_from_torch_185 = ttnn_decorators_ttnn_from_torch(arg181_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg181_1 = None
    ttnn_transpose_77 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_185, 0, 1);  ttnn_from_torch_185 = None
    ttnn_matmul_88 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_198, ttnn_transpose_77);  ttnn_transpose_77 = None
    ttnn_from_torch_186 = ttnn_decorators_ttnn_from_torch(arg182_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg182_1 = None
    ttnn_add_66 = ttnn_decorators_ttnn_add(ttnn_from_torch_186, ttnn_matmul_88);  ttnn_from_torch_186 = ttnn_matmul_88 = None
    ttnn_experimental_view_199 = ttnn_decorators_ttnn_experimental_view(ttnn_add_66, [1, 256, 1024]);  ttnn_add_66 = None
    ttnn_from_torch_187 = ttnn_decorators_ttnn_from_torch(arg183_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg183_1 = None
    ttnn_transpose_78 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_187, 0, 1);  ttnn_from_torch_187 = None
    ttnn_matmul_89 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_198, ttnn_transpose_78);  ttnn_transpose_78 = None
    ttnn_from_torch_188 = ttnn_decorators_ttnn_from_torch(arg184_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg184_1 = None
    ttnn_add_67 = ttnn_decorators_ttnn_add(ttnn_from_torch_188, ttnn_matmul_89);  ttnn_from_torch_188 = ttnn_matmul_89 = None
    ttnn_experimental_view_201 = ttnn_decorators_ttnn_experimental_view(ttnn_add_67, [1, 256, 1024]);  ttnn_add_67 = None
    ttnn_reshape_46 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_201, [1, 256, 16, 64]);  ttnn_experimental_view_201 = None
    ttnn_permute_44 = ttnn_decorators_ttnn_permute(ttnn_reshape_46, [0, 2, 1, 3]);  ttnn_reshape_46 = None
    ttnn_from_torch_189 = ttnn_decorators_ttnn_from_torch(arg185_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg185_1 = None
    ttnn_transpose_79 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_189, 0, 1);  ttnn_from_torch_189 = None
    ttnn_matmul_90 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_198, ttnn_transpose_79);  ttnn_experimental_view_198 = ttnn_transpose_79 = None
    ttnn_from_torch_190 = ttnn_decorators_ttnn_from_torch(arg186_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg186_1 = None
    ttnn_add_68 = ttnn_decorators_ttnn_add(ttnn_from_torch_190, ttnn_matmul_90);  ttnn_from_torch_190 = ttnn_matmul_90 = None
    ttnn_experimental_view_203 = ttnn_decorators_ttnn_experimental_view(ttnn_add_68, [1, 256, 1024]);  ttnn_add_68 = None
    ttnn_reshape_47 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_203, [1, 256, 16, 64]);  ttnn_experimental_view_203 = None
    ttnn_permute_45 = ttnn_decorators_ttnn_permute(ttnn_reshape_47, [0, 2, 1, 3]);  ttnn_reshape_47 = None
    ttnn_reshape_48 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_199, [1, 256, 16, 64]);  ttnn_experimental_view_199 = None
    ttnn_permute_46 = ttnn_decorators_ttnn_permute(ttnn_reshape_48, [0, 2, 1, 3]);  ttnn_reshape_48 = None
    ttnn_transpose_80 = ttnn_decorators_ttnn_transpose(ttnn_permute_44, 3, 2);  ttnn_permute_44 = None
    ttnn_experimental_view_204 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_46, [16, 256, 64]);  ttnn_permute_46 = None
    ttnn_experimental_view_205 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_80, [16, 64, 256]);  ttnn_transpose_80 = None
    ttnn_matmul_91 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_204, ttnn_experimental_view_205);  ttnn_experimental_view_204 = ttnn_experimental_view_205 = None
    ttnn_experimental_view_206 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_91, [1, 16, 256, 256]);  ttnn_matmul_91 = None
    ttnn_multiply_12 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_206, 0.125);  ttnn_experimental_view_206 = None
    ttnn_add_180 = ttnn_decorators_ttnn_add(ttnn_multiply_12, ttnn_multiply);  ttnn_multiply_12 = None
    ttnn_softmax_11 = ttnn_decorators_ttnn_softmax(ttnn_add_180, -1, numeric_stable = True);  ttnn_add_180 = None
    ttnn_experimental_view_207 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_11, [16, 256, 256]);  ttnn_softmax_11 = None
    ttnn_experimental_view_208 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_45, [16, 256, 64]);  ttnn_permute_45 = None
    ttnn_matmul_92 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_207, ttnn_experimental_view_208);  ttnn_experimental_view_207 = ttnn_experimental_view_208 = None
    ttnn_experimental_view_209 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_92, [1, 16, 256, 64]);  ttnn_matmul_92 = None
    ttnn_permute_47 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_209, [0, 2, 1, 3]);  ttnn_experimental_view_209 = None
    ttnn_reshape_49 = ttnn_decorators_ttnn_reshape(ttnn_permute_47, [1, 256, 1024]);  ttnn_permute_47 = None
    ttnn_experimental_view_210 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_49, [256, 1024]);  ttnn_reshape_49 = None
    ttnn_from_torch_191 = ttnn_decorators_ttnn_from_torch(arg187_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg187_1 = None
    ttnn_transpose_81 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_191, 0, 1);  ttnn_from_torch_191 = None
    ttnn_matmul_93 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_210, ttnn_transpose_81);  ttnn_experimental_view_210 = ttnn_transpose_81 = None
    ttnn_from_torch_192 = ttnn_decorators_ttnn_from_torch(arg188_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg188_1 = None
    ttnn_add_69 = ttnn_decorators_ttnn_add(ttnn_from_torch_192, ttnn_matmul_93);  ttnn_from_torch_192 = ttnn_matmul_93 = None
    ttnn_experimental_view_211 = ttnn_decorators_ttnn_experimental_view(ttnn_add_69, [1, 256, 1024]);  ttnn_add_69 = None
    ttnn_add_181 = ttnn_decorators_ttnn_add(ttnn_experimental_view_211, ttnn_layer_norm_22);  ttnn_experimental_view_211 = ttnn_layer_norm_22 = None
    ttnn_from_torch_193 = ttnn_decorators_ttnn_from_torch(arg189_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg189_1 = None
    ttnn_from_torch_194 = ttnn_decorators_ttnn_from_torch(arg190_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg190_1 = None
    ttnn_layer_norm_23 = ttnn_decorators_ttnn_layer_norm(ttnn_add_181, epsilon = 1e-12, weight = ttnn_from_torch_193, bias = ttnn_from_torch_194);  ttnn_add_181 = ttnn_from_torch_193 = ttnn_from_torch_194 = None
    ttnn_experimental_view_212 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_23, [256, 1024])
    ttnn_from_torch_195 = ttnn_decorators_ttnn_from_torch(arg191_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg191_1 = None
    ttnn_transpose_82 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_195, 0, 1);  ttnn_from_torch_195 = None
    ttnn_matmul_94 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_212, ttnn_transpose_82);  ttnn_experimental_view_212 = ttnn_transpose_82 = None
    ttnn_from_torch_196 = ttnn_decorators_ttnn_from_torch(arg192_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg192_1 = None
    ttnn_add_70 = ttnn_decorators_ttnn_add(ttnn_from_torch_196, ttnn_matmul_94);  ttnn_from_torch_196 = ttnn_matmul_94 = None
    ttnn_experimental_view_213 = ttnn_decorators_ttnn_experimental_view(ttnn_add_70, [1, 256, 4096]);  ttnn_add_70 = None
    ttnn_gelu_11 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_213);  ttnn_experimental_view_213 = None
    ttnn_experimental_view_214 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_11, [256, 4096]);  ttnn_gelu_11 = None
    ttnn_from_torch_197 = ttnn_decorators_ttnn_from_torch(arg193_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg193_1 = None
    ttnn_transpose_83 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_197, 0, 1);  ttnn_from_torch_197 = None
    ttnn_matmul_95 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_214, ttnn_transpose_83);  ttnn_experimental_view_214 = ttnn_transpose_83 = None
    ttnn_from_torch_198 = ttnn_decorators_ttnn_from_torch(arg194_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg194_1 = None
    ttnn_add_71 = ttnn_decorators_ttnn_add(ttnn_from_torch_198, ttnn_matmul_95);  ttnn_from_torch_198 = ttnn_matmul_95 = None
    ttnn_experimental_view_215 = ttnn_decorators_ttnn_experimental_view(ttnn_add_71, [1, 256, 1024]);  ttnn_add_71 = None
    ttnn_add_182 = ttnn_decorators_ttnn_add(ttnn_experimental_view_215, ttnn_layer_norm_23);  ttnn_experimental_view_215 = ttnn_layer_norm_23 = None
    ttnn_from_torch_199 = ttnn_decorators_ttnn_from_torch(arg195_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg195_1 = None
    ttnn_from_torch_200 = ttnn_decorators_ttnn_from_torch(arg196_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg196_1 = None
    ttnn_layer_norm_24 = ttnn_decorators_ttnn_layer_norm(ttnn_add_182, epsilon = 1e-12, weight = ttnn_from_torch_199, bias = ttnn_from_torch_200);  ttnn_add_182 = ttnn_from_torch_199 = ttnn_from_torch_200 = None
    ttnn_experimental_view_216 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_24, [256, 1024])
    ttnn_from_torch_201 = ttnn_decorators_ttnn_from_torch(arg197_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg197_1 = None
    ttnn_transpose_84 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_201, 0, 1);  ttnn_from_torch_201 = None
    ttnn_matmul_96 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_216, ttnn_transpose_84);  ttnn_transpose_84 = None
    ttnn_from_torch_202 = ttnn_decorators_ttnn_from_torch(arg198_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg198_1 = None
    ttnn_add_72 = ttnn_decorators_ttnn_add(ttnn_from_torch_202, ttnn_matmul_96);  ttnn_from_torch_202 = ttnn_matmul_96 = None
    ttnn_experimental_view_217 = ttnn_decorators_ttnn_experimental_view(ttnn_add_72, [1, 256, 1024]);  ttnn_add_72 = None
    ttnn_from_torch_203 = ttnn_decorators_ttnn_from_torch(arg199_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg199_1 = None
    ttnn_transpose_85 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_203, 0, 1);  ttnn_from_torch_203 = None
    ttnn_matmul_97 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_216, ttnn_transpose_85);  ttnn_transpose_85 = None
    ttnn_from_torch_204 = ttnn_decorators_ttnn_from_torch(arg200_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg200_1 = None
    ttnn_add_73 = ttnn_decorators_ttnn_add(ttnn_from_torch_204, ttnn_matmul_97);  ttnn_from_torch_204 = ttnn_matmul_97 = None
    ttnn_experimental_view_219 = ttnn_decorators_ttnn_experimental_view(ttnn_add_73, [1, 256, 1024]);  ttnn_add_73 = None
    ttnn_reshape_50 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_219, [1, 256, 16, 64]);  ttnn_experimental_view_219 = None
    ttnn_permute_48 = ttnn_decorators_ttnn_permute(ttnn_reshape_50, [0, 2, 1, 3]);  ttnn_reshape_50 = None
    ttnn_from_torch_205 = ttnn_decorators_ttnn_from_torch(arg201_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg201_1 = None
    ttnn_transpose_86 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_205, 0, 1);  ttnn_from_torch_205 = None
    ttnn_matmul_98 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_216, ttnn_transpose_86);  ttnn_experimental_view_216 = ttnn_transpose_86 = None
    ttnn_from_torch_206 = ttnn_decorators_ttnn_from_torch(arg202_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg202_1 = None
    ttnn_add_74 = ttnn_decorators_ttnn_add(ttnn_from_torch_206, ttnn_matmul_98);  ttnn_from_torch_206 = ttnn_matmul_98 = None
    ttnn_experimental_view_221 = ttnn_decorators_ttnn_experimental_view(ttnn_add_74, [1, 256, 1024]);  ttnn_add_74 = None
    ttnn_reshape_51 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_221, [1, 256, 16, 64]);  ttnn_experimental_view_221 = None
    ttnn_permute_49 = ttnn_decorators_ttnn_permute(ttnn_reshape_51, [0, 2, 1, 3]);  ttnn_reshape_51 = None
    ttnn_reshape_52 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_217, [1, 256, 16, 64]);  ttnn_experimental_view_217 = None
    ttnn_permute_50 = ttnn_decorators_ttnn_permute(ttnn_reshape_52, [0, 2, 1, 3]);  ttnn_reshape_52 = None
    ttnn_transpose_87 = ttnn_decorators_ttnn_transpose(ttnn_permute_48, 3, 2);  ttnn_permute_48 = None
    ttnn_experimental_view_222 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_50, [16, 256, 64]);  ttnn_permute_50 = None
    ttnn_experimental_view_223 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_87, [16, 64, 256]);  ttnn_transpose_87 = None
    ttnn_matmul_99 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_222, ttnn_experimental_view_223);  ttnn_experimental_view_222 = ttnn_experimental_view_223 = None
    ttnn_experimental_view_224 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_99, [1, 16, 256, 256]);  ttnn_matmul_99 = None
    ttnn_multiply_13 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_224, 0.125);  ttnn_experimental_view_224 = None
    ttnn_add_183 = ttnn_decorators_ttnn_add(ttnn_multiply_13, ttnn_multiply);  ttnn_multiply_13 = None
    ttnn_softmax_12 = ttnn_decorators_ttnn_softmax(ttnn_add_183, -1, numeric_stable = True);  ttnn_add_183 = None
    ttnn_experimental_view_225 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_12, [16, 256, 256]);  ttnn_softmax_12 = None
    ttnn_experimental_view_226 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_49, [16, 256, 64]);  ttnn_permute_49 = None
    ttnn_matmul_100 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_225, ttnn_experimental_view_226);  ttnn_experimental_view_225 = ttnn_experimental_view_226 = None
    ttnn_experimental_view_227 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_100, [1, 16, 256, 64]);  ttnn_matmul_100 = None
    ttnn_permute_51 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_227, [0, 2, 1, 3]);  ttnn_experimental_view_227 = None
    ttnn_reshape_53 = ttnn_decorators_ttnn_reshape(ttnn_permute_51, [1, 256, 1024]);  ttnn_permute_51 = None
    ttnn_experimental_view_228 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_53, [256, 1024]);  ttnn_reshape_53 = None
    ttnn_from_torch_207 = ttnn_decorators_ttnn_from_torch(arg203_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg203_1 = None
    ttnn_transpose_88 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_207, 0, 1);  ttnn_from_torch_207 = None
    ttnn_matmul_101 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_228, ttnn_transpose_88);  ttnn_experimental_view_228 = ttnn_transpose_88 = None
    ttnn_from_torch_208 = ttnn_decorators_ttnn_from_torch(arg204_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg204_1 = None
    ttnn_add_75 = ttnn_decorators_ttnn_add(ttnn_from_torch_208, ttnn_matmul_101);  ttnn_from_torch_208 = ttnn_matmul_101 = None
    ttnn_experimental_view_229 = ttnn_decorators_ttnn_experimental_view(ttnn_add_75, [1, 256, 1024]);  ttnn_add_75 = None
    ttnn_add_184 = ttnn_decorators_ttnn_add(ttnn_experimental_view_229, ttnn_layer_norm_24);  ttnn_experimental_view_229 = ttnn_layer_norm_24 = None
    ttnn_from_torch_209 = ttnn_decorators_ttnn_from_torch(arg205_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg205_1 = None
    ttnn_from_torch_210 = ttnn_decorators_ttnn_from_torch(arg206_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg206_1 = None
    ttnn_layer_norm_25 = ttnn_decorators_ttnn_layer_norm(ttnn_add_184, epsilon = 1e-12, weight = ttnn_from_torch_209, bias = ttnn_from_torch_210);  ttnn_add_184 = ttnn_from_torch_209 = ttnn_from_torch_210 = None
    ttnn_experimental_view_230 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_25, [256, 1024])
    ttnn_from_torch_211 = ttnn_decorators_ttnn_from_torch(arg207_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg207_1 = None
    ttnn_transpose_89 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_211, 0, 1);  ttnn_from_torch_211 = None
    ttnn_matmul_102 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_230, ttnn_transpose_89);  ttnn_experimental_view_230 = ttnn_transpose_89 = None
    ttnn_from_torch_212 = ttnn_decorators_ttnn_from_torch(arg208_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg208_1 = None
    ttnn_add_76 = ttnn_decorators_ttnn_add(ttnn_from_torch_212, ttnn_matmul_102);  ttnn_from_torch_212 = ttnn_matmul_102 = None
    ttnn_experimental_view_231 = ttnn_decorators_ttnn_experimental_view(ttnn_add_76, [1, 256, 4096]);  ttnn_add_76 = None
    ttnn_gelu_12 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_231);  ttnn_experimental_view_231 = None
    ttnn_experimental_view_232 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_12, [256, 4096]);  ttnn_gelu_12 = None
    ttnn_from_torch_213 = ttnn_decorators_ttnn_from_torch(arg209_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg209_1 = None
    ttnn_transpose_90 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_213, 0, 1);  ttnn_from_torch_213 = None
    ttnn_matmul_103 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_232, ttnn_transpose_90);  ttnn_experimental_view_232 = ttnn_transpose_90 = None
    ttnn_from_torch_214 = ttnn_decorators_ttnn_from_torch(arg210_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg210_1 = None
    ttnn_add_77 = ttnn_decorators_ttnn_add(ttnn_from_torch_214, ttnn_matmul_103);  ttnn_from_torch_214 = ttnn_matmul_103 = None
    ttnn_experimental_view_233 = ttnn_decorators_ttnn_experimental_view(ttnn_add_77, [1, 256, 1024]);  ttnn_add_77 = None
    ttnn_add_185 = ttnn_decorators_ttnn_add(ttnn_experimental_view_233, ttnn_layer_norm_25);  ttnn_experimental_view_233 = ttnn_layer_norm_25 = None
    ttnn_from_torch_215 = ttnn_decorators_ttnn_from_torch(arg211_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg211_1 = None
    ttnn_from_torch_216 = ttnn_decorators_ttnn_from_torch(arg212_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg212_1 = None
    ttnn_layer_norm_26 = ttnn_decorators_ttnn_layer_norm(ttnn_add_185, epsilon = 1e-12, weight = ttnn_from_torch_215, bias = ttnn_from_torch_216);  ttnn_add_185 = ttnn_from_torch_215 = ttnn_from_torch_216 = None
    ttnn_experimental_view_234 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_26, [256, 1024])
    ttnn_from_torch_217 = ttnn_decorators_ttnn_from_torch(arg213_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg213_1 = None
    ttnn_transpose_91 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_217, 0, 1);  ttnn_from_torch_217 = None
    ttnn_matmul_104 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_234, ttnn_transpose_91);  ttnn_transpose_91 = None
    ttnn_from_torch_218 = ttnn_decorators_ttnn_from_torch(arg214_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg214_1 = None
    ttnn_add_78 = ttnn_decorators_ttnn_add(ttnn_from_torch_218, ttnn_matmul_104);  ttnn_from_torch_218 = ttnn_matmul_104 = None
    ttnn_experimental_view_235 = ttnn_decorators_ttnn_experimental_view(ttnn_add_78, [1, 256, 1024]);  ttnn_add_78 = None
    ttnn_from_torch_219 = ttnn_decorators_ttnn_from_torch(arg215_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg215_1 = None
    ttnn_transpose_92 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_219, 0, 1);  ttnn_from_torch_219 = None
    ttnn_matmul_105 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_234, ttnn_transpose_92);  ttnn_transpose_92 = None
    ttnn_from_torch_220 = ttnn_decorators_ttnn_from_torch(arg216_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg216_1 = None
    ttnn_add_79 = ttnn_decorators_ttnn_add(ttnn_from_torch_220, ttnn_matmul_105);  ttnn_from_torch_220 = ttnn_matmul_105 = None
    ttnn_experimental_view_237 = ttnn_decorators_ttnn_experimental_view(ttnn_add_79, [1, 256, 1024]);  ttnn_add_79 = None
    ttnn_reshape_54 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_237, [1, 256, 16, 64]);  ttnn_experimental_view_237 = None
    ttnn_permute_52 = ttnn_decorators_ttnn_permute(ttnn_reshape_54, [0, 2, 1, 3]);  ttnn_reshape_54 = None
    ttnn_from_torch_221 = ttnn_decorators_ttnn_from_torch(arg217_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg217_1 = None
    ttnn_transpose_93 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_221, 0, 1);  ttnn_from_torch_221 = None
    ttnn_matmul_106 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_234, ttnn_transpose_93);  ttnn_experimental_view_234 = ttnn_transpose_93 = None
    ttnn_from_torch_222 = ttnn_decorators_ttnn_from_torch(arg218_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg218_1 = None
    ttnn_add_80 = ttnn_decorators_ttnn_add(ttnn_from_torch_222, ttnn_matmul_106);  ttnn_from_torch_222 = ttnn_matmul_106 = None
    ttnn_experimental_view_239 = ttnn_decorators_ttnn_experimental_view(ttnn_add_80, [1, 256, 1024]);  ttnn_add_80 = None
    ttnn_reshape_55 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_239, [1, 256, 16, 64]);  ttnn_experimental_view_239 = None
    ttnn_permute_53 = ttnn_decorators_ttnn_permute(ttnn_reshape_55, [0, 2, 1, 3]);  ttnn_reshape_55 = None
    ttnn_reshape_56 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_235, [1, 256, 16, 64]);  ttnn_experimental_view_235 = None
    ttnn_permute_54 = ttnn_decorators_ttnn_permute(ttnn_reshape_56, [0, 2, 1, 3]);  ttnn_reshape_56 = None
    ttnn_transpose_94 = ttnn_decorators_ttnn_transpose(ttnn_permute_52, 3, 2);  ttnn_permute_52 = None
    ttnn_experimental_view_240 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_54, [16, 256, 64]);  ttnn_permute_54 = None
    ttnn_experimental_view_241 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_94, [16, 64, 256]);  ttnn_transpose_94 = None
    ttnn_matmul_107 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_240, ttnn_experimental_view_241);  ttnn_experimental_view_240 = ttnn_experimental_view_241 = None
    ttnn_experimental_view_242 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_107, [1, 16, 256, 256]);  ttnn_matmul_107 = None
    ttnn_multiply_14 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_242, 0.125);  ttnn_experimental_view_242 = None
    ttnn_add_186 = ttnn_decorators_ttnn_add(ttnn_multiply_14, ttnn_multiply);  ttnn_multiply_14 = None
    ttnn_softmax_13 = ttnn_decorators_ttnn_softmax(ttnn_add_186, -1, numeric_stable = True);  ttnn_add_186 = None
    ttnn_experimental_view_243 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_13, [16, 256, 256]);  ttnn_softmax_13 = None
    ttnn_experimental_view_244 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_53, [16, 256, 64]);  ttnn_permute_53 = None
    ttnn_matmul_108 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_243, ttnn_experimental_view_244);  ttnn_experimental_view_243 = ttnn_experimental_view_244 = None
    ttnn_experimental_view_245 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_108, [1, 16, 256, 64]);  ttnn_matmul_108 = None
    ttnn_permute_55 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_245, [0, 2, 1, 3]);  ttnn_experimental_view_245 = None
    ttnn_reshape_57 = ttnn_decorators_ttnn_reshape(ttnn_permute_55, [1, 256, 1024]);  ttnn_permute_55 = None
    ttnn_experimental_view_246 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_57, [256, 1024]);  ttnn_reshape_57 = None
    ttnn_from_torch_223 = ttnn_decorators_ttnn_from_torch(arg219_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg219_1 = None
    ttnn_transpose_95 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_223, 0, 1);  ttnn_from_torch_223 = None
    ttnn_matmul_109 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_246, ttnn_transpose_95);  ttnn_experimental_view_246 = ttnn_transpose_95 = None
    ttnn_from_torch_224 = ttnn_decorators_ttnn_from_torch(arg220_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg220_1 = None
    ttnn_add_81 = ttnn_decorators_ttnn_add(ttnn_from_torch_224, ttnn_matmul_109);  ttnn_from_torch_224 = ttnn_matmul_109 = None
    ttnn_experimental_view_247 = ttnn_decorators_ttnn_experimental_view(ttnn_add_81, [1, 256, 1024]);  ttnn_add_81 = None
    ttnn_add_187 = ttnn_decorators_ttnn_add(ttnn_experimental_view_247, ttnn_layer_norm_26);  ttnn_experimental_view_247 = ttnn_layer_norm_26 = None
    ttnn_from_torch_225 = ttnn_decorators_ttnn_from_torch(arg221_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg221_1 = None
    ttnn_from_torch_226 = ttnn_decorators_ttnn_from_torch(arg222_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg222_1 = None
    ttnn_layer_norm_27 = ttnn_decorators_ttnn_layer_norm(ttnn_add_187, epsilon = 1e-12, weight = ttnn_from_torch_225, bias = ttnn_from_torch_226);  ttnn_add_187 = ttnn_from_torch_225 = ttnn_from_torch_226 = None
    ttnn_experimental_view_248 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_27, [256, 1024])
    ttnn_from_torch_227 = ttnn_decorators_ttnn_from_torch(arg223_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg223_1 = None
    ttnn_transpose_96 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_227, 0, 1);  ttnn_from_torch_227 = None
    ttnn_matmul_110 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_248, ttnn_transpose_96);  ttnn_experimental_view_248 = ttnn_transpose_96 = None
    ttnn_from_torch_228 = ttnn_decorators_ttnn_from_torch(arg224_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg224_1 = None
    ttnn_add_82 = ttnn_decorators_ttnn_add(ttnn_from_torch_228, ttnn_matmul_110);  ttnn_from_torch_228 = ttnn_matmul_110 = None
    ttnn_experimental_view_249 = ttnn_decorators_ttnn_experimental_view(ttnn_add_82, [1, 256, 4096]);  ttnn_add_82 = None
    ttnn_gelu_13 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_249);  ttnn_experimental_view_249 = None
    ttnn_experimental_view_250 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_13, [256, 4096]);  ttnn_gelu_13 = None
    ttnn_from_torch_229 = ttnn_decorators_ttnn_from_torch(arg225_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg225_1 = None
    ttnn_transpose_97 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_229, 0, 1);  ttnn_from_torch_229 = None
    ttnn_matmul_111 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_250, ttnn_transpose_97);  ttnn_experimental_view_250 = ttnn_transpose_97 = None
    ttnn_from_torch_230 = ttnn_decorators_ttnn_from_torch(arg226_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg226_1 = None
    ttnn_add_83 = ttnn_decorators_ttnn_add(ttnn_from_torch_230, ttnn_matmul_111);  ttnn_from_torch_230 = ttnn_matmul_111 = None
    ttnn_experimental_view_251 = ttnn_decorators_ttnn_experimental_view(ttnn_add_83, [1, 256, 1024]);  ttnn_add_83 = None
    ttnn_add_188 = ttnn_decorators_ttnn_add(ttnn_experimental_view_251, ttnn_layer_norm_27);  ttnn_experimental_view_251 = ttnn_layer_norm_27 = None
    ttnn_from_torch_231 = ttnn_decorators_ttnn_from_torch(arg227_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg227_1 = None
    ttnn_from_torch_232 = ttnn_decorators_ttnn_from_torch(arg228_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg228_1 = None
    ttnn_layer_norm_28 = ttnn_decorators_ttnn_layer_norm(ttnn_add_188, epsilon = 1e-12, weight = ttnn_from_torch_231, bias = ttnn_from_torch_232);  ttnn_add_188 = ttnn_from_torch_231 = ttnn_from_torch_232 = None
    ttnn_experimental_view_252 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_28, [256, 1024])
    ttnn_from_torch_233 = ttnn_decorators_ttnn_from_torch(arg229_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg229_1 = None
    ttnn_transpose_98 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_233, 0, 1);  ttnn_from_torch_233 = None
    ttnn_matmul_112 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_252, ttnn_transpose_98);  ttnn_transpose_98 = None
    ttnn_from_torch_234 = ttnn_decorators_ttnn_from_torch(arg230_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg230_1 = None
    ttnn_add_84 = ttnn_decorators_ttnn_add(ttnn_from_torch_234, ttnn_matmul_112);  ttnn_from_torch_234 = ttnn_matmul_112 = None
    ttnn_experimental_view_253 = ttnn_decorators_ttnn_experimental_view(ttnn_add_84, [1, 256, 1024]);  ttnn_add_84 = None
    ttnn_from_torch_235 = ttnn_decorators_ttnn_from_torch(arg231_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg231_1 = None
    ttnn_transpose_99 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_235, 0, 1);  ttnn_from_torch_235 = None
    ttnn_matmul_113 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_252, ttnn_transpose_99);  ttnn_transpose_99 = None
    ttnn_from_torch_236 = ttnn_decorators_ttnn_from_torch(arg232_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg232_1 = None
    ttnn_add_85 = ttnn_decorators_ttnn_add(ttnn_from_torch_236, ttnn_matmul_113);  ttnn_from_torch_236 = ttnn_matmul_113 = None
    ttnn_experimental_view_255 = ttnn_decorators_ttnn_experimental_view(ttnn_add_85, [1, 256, 1024]);  ttnn_add_85 = None
    ttnn_reshape_58 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_255, [1, 256, 16, 64]);  ttnn_experimental_view_255 = None
    ttnn_permute_56 = ttnn_decorators_ttnn_permute(ttnn_reshape_58, [0, 2, 1, 3]);  ttnn_reshape_58 = None
    ttnn_from_torch_237 = ttnn_decorators_ttnn_from_torch(arg233_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg233_1 = None
    ttnn_transpose_100 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_237, 0, 1);  ttnn_from_torch_237 = None
    ttnn_matmul_114 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_252, ttnn_transpose_100);  ttnn_experimental_view_252 = ttnn_transpose_100 = None
    ttnn_from_torch_238 = ttnn_decorators_ttnn_from_torch(arg234_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg234_1 = None
    ttnn_add_86 = ttnn_decorators_ttnn_add(ttnn_from_torch_238, ttnn_matmul_114);  ttnn_from_torch_238 = ttnn_matmul_114 = None
    ttnn_experimental_view_257 = ttnn_decorators_ttnn_experimental_view(ttnn_add_86, [1, 256, 1024]);  ttnn_add_86 = None
    ttnn_reshape_59 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_257, [1, 256, 16, 64]);  ttnn_experimental_view_257 = None
    ttnn_permute_57 = ttnn_decorators_ttnn_permute(ttnn_reshape_59, [0, 2, 1, 3]);  ttnn_reshape_59 = None
    ttnn_reshape_60 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_253, [1, 256, 16, 64]);  ttnn_experimental_view_253 = None
    ttnn_permute_58 = ttnn_decorators_ttnn_permute(ttnn_reshape_60, [0, 2, 1, 3]);  ttnn_reshape_60 = None
    ttnn_transpose_101 = ttnn_decorators_ttnn_transpose(ttnn_permute_56, 3, 2);  ttnn_permute_56 = None
    ttnn_experimental_view_258 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_58, [16, 256, 64]);  ttnn_permute_58 = None
    ttnn_experimental_view_259 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_101, [16, 64, 256]);  ttnn_transpose_101 = None
    ttnn_matmul_115 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_258, ttnn_experimental_view_259);  ttnn_experimental_view_258 = ttnn_experimental_view_259 = None
    ttnn_experimental_view_260 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_115, [1, 16, 256, 256]);  ttnn_matmul_115 = None
    ttnn_multiply_15 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_260, 0.125);  ttnn_experimental_view_260 = None
    ttnn_add_189 = ttnn_decorators_ttnn_add(ttnn_multiply_15, ttnn_multiply);  ttnn_multiply_15 = None
    ttnn_softmax_14 = ttnn_decorators_ttnn_softmax(ttnn_add_189, -1, numeric_stable = True);  ttnn_add_189 = None
    ttnn_experimental_view_261 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_14, [16, 256, 256]);  ttnn_softmax_14 = None
    ttnn_experimental_view_262 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_57, [16, 256, 64]);  ttnn_permute_57 = None
    ttnn_matmul_116 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_261, ttnn_experimental_view_262);  ttnn_experimental_view_261 = ttnn_experimental_view_262 = None
    ttnn_experimental_view_263 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_116, [1, 16, 256, 64]);  ttnn_matmul_116 = None
    ttnn_permute_59 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_263, [0, 2, 1, 3]);  ttnn_experimental_view_263 = None
    ttnn_reshape_61 = ttnn_decorators_ttnn_reshape(ttnn_permute_59, [1, 256, 1024]);  ttnn_permute_59 = None
    ttnn_experimental_view_264 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_61, [256, 1024]);  ttnn_reshape_61 = None
    ttnn_from_torch_239 = ttnn_decorators_ttnn_from_torch(arg235_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg235_1 = None
    ttnn_transpose_102 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_239, 0, 1);  ttnn_from_torch_239 = None
    ttnn_matmul_117 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_264, ttnn_transpose_102);  ttnn_experimental_view_264 = ttnn_transpose_102 = None
    ttnn_from_torch_240 = ttnn_decorators_ttnn_from_torch(arg236_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg236_1 = None
    ttnn_add_87 = ttnn_decorators_ttnn_add(ttnn_from_torch_240, ttnn_matmul_117);  ttnn_from_torch_240 = ttnn_matmul_117 = None
    ttnn_experimental_view_265 = ttnn_decorators_ttnn_experimental_view(ttnn_add_87, [1, 256, 1024]);  ttnn_add_87 = None
    ttnn_add_190 = ttnn_decorators_ttnn_add(ttnn_experimental_view_265, ttnn_layer_norm_28);  ttnn_experimental_view_265 = ttnn_layer_norm_28 = None
    ttnn_from_torch_241 = ttnn_decorators_ttnn_from_torch(arg237_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg237_1 = None
    ttnn_from_torch_242 = ttnn_decorators_ttnn_from_torch(arg238_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg238_1 = None
    ttnn_layer_norm_29 = ttnn_decorators_ttnn_layer_norm(ttnn_add_190, epsilon = 1e-12, weight = ttnn_from_torch_241, bias = ttnn_from_torch_242);  ttnn_add_190 = ttnn_from_torch_241 = ttnn_from_torch_242 = None
    ttnn_experimental_view_266 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_29, [256, 1024])
    ttnn_from_torch_243 = ttnn_decorators_ttnn_from_torch(arg239_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg239_1 = None
    ttnn_transpose_103 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_243, 0, 1);  ttnn_from_torch_243 = None
    ttnn_matmul_118 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_266, ttnn_transpose_103);  ttnn_experimental_view_266 = ttnn_transpose_103 = None
    ttnn_from_torch_244 = ttnn_decorators_ttnn_from_torch(arg240_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg240_1 = None
    ttnn_add_88 = ttnn_decorators_ttnn_add(ttnn_from_torch_244, ttnn_matmul_118);  ttnn_from_torch_244 = ttnn_matmul_118 = None
    ttnn_experimental_view_267 = ttnn_decorators_ttnn_experimental_view(ttnn_add_88, [1, 256, 4096]);  ttnn_add_88 = None
    ttnn_gelu_14 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_267);  ttnn_experimental_view_267 = None
    ttnn_experimental_view_268 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_14, [256, 4096]);  ttnn_gelu_14 = None
    ttnn_from_torch_245 = ttnn_decorators_ttnn_from_torch(arg241_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg241_1 = None
    ttnn_transpose_104 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_245, 0, 1);  ttnn_from_torch_245 = None
    ttnn_matmul_119 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_268, ttnn_transpose_104);  ttnn_experimental_view_268 = ttnn_transpose_104 = None
    ttnn_from_torch_246 = ttnn_decorators_ttnn_from_torch(arg242_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg242_1 = None
    ttnn_add_89 = ttnn_decorators_ttnn_add(ttnn_from_torch_246, ttnn_matmul_119);  ttnn_from_torch_246 = ttnn_matmul_119 = None
    ttnn_experimental_view_269 = ttnn_decorators_ttnn_experimental_view(ttnn_add_89, [1, 256, 1024]);  ttnn_add_89 = None
    ttnn_add_191 = ttnn_decorators_ttnn_add(ttnn_experimental_view_269, ttnn_layer_norm_29);  ttnn_experimental_view_269 = ttnn_layer_norm_29 = None
    ttnn_from_torch_247 = ttnn_decorators_ttnn_from_torch(arg243_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg243_1 = None
    ttnn_from_torch_248 = ttnn_decorators_ttnn_from_torch(arg244_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg244_1 = None
    ttnn_layer_norm_30 = ttnn_decorators_ttnn_layer_norm(ttnn_add_191, epsilon = 1e-12, weight = ttnn_from_torch_247, bias = ttnn_from_torch_248);  ttnn_add_191 = ttnn_from_torch_247 = ttnn_from_torch_248 = None
    ttnn_experimental_view_270 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_30, [256, 1024])
    ttnn_from_torch_249 = ttnn_decorators_ttnn_from_torch(arg245_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg245_1 = None
    ttnn_transpose_105 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_249, 0, 1);  ttnn_from_torch_249 = None
    ttnn_matmul_120 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_270, ttnn_transpose_105);  ttnn_transpose_105 = None
    ttnn_from_torch_250 = ttnn_decorators_ttnn_from_torch(arg246_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg246_1 = None
    ttnn_add_90 = ttnn_decorators_ttnn_add(ttnn_from_torch_250, ttnn_matmul_120);  ttnn_from_torch_250 = ttnn_matmul_120 = None
    ttnn_experimental_view_271 = ttnn_decorators_ttnn_experimental_view(ttnn_add_90, [1, 256, 1024]);  ttnn_add_90 = None
    ttnn_from_torch_251 = ttnn_decorators_ttnn_from_torch(arg247_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg247_1 = None
    ttnn_transpose_106 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_251, 0, 1);  ttnn_from_torch_251 = None
    ttnn_matmul_121 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_270, ttnn_transpose_106);  ttnn_transpose_106 = None
    ttnn_from_torch_252 = ttnn_decorators_ttnn_from_torch(arg248_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg248_1 = None
    ttnn_add_91 = ttnn_decorators_ttnn_add(ttnn_from_torch_252, ttnn_matmul_121);  ttnn_from_torch_252 = ttnn_matmul_121 = None
    ttnn_experimental_view_273 = ttnn_decorators_ttnn_experimental_view(ttnn_add_91, [1, 256, 1024]);  ttnn_add_91 = None
    ttnn_reshape_62 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_273, [1, 256, 16, 64]);  ttnn_experimental_view_273 = None
    ttnn_permute_60 = ttnn_decorators_ttnn_permute(ttnn_reshape_62, [0, 2, 1, 3]);  ttnn_reshape_62 = None
    ttnn_from_torch_253 = ttnn_decorators_ttnn_from_torch(arg249_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg249_1 = None
    ttnn_transpose_107 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_253, 0, 1);  ttnn_from_torch_253 = None
    ttnn_matmul_122 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_270, ttnn_transpose_107);  ttnn_experimental_view_270 = ttnn_transpose_107 = None
    ttnn_from_torch_254 = ttnn_decorators_ttnn_from_torch(arg250_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg250_1 = None
    ttnn_add_92 = ttnn_decorators_ttnn_add(ttnn_from_torch_254, ttnn_matmul_122);  ttnn_from_torch_254 = ttnn_matmul_122 = None
    ttnn_experimental_view_275 = ttnn_decorators_ttnn_experimental_view(ttnn_add_92, [1, 256, 1024]);  ttnn_add_92 = None
    ttnn_reshape_63 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_275, [1, 256, 16, 64]);  ttnn_experimental_view_275 = None
    ttnn_permute_61 = ttnn_decorators_ttnn_permute(ttnn_reshape_63, [0, 2, 1, 3]);  ttnn_reshape_63 = None
    ttnn_reshape_64 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_271, [1, 256, 16, 64]);  ttnn_experimental_view_271 = None
    ttnn_permute_62 = ttnn_decorators_ttnn_permute(ttnn_reshape_64, [0, 2, 1, 3]);  ttnn_reshape_64 = None
    ttnn_transpose_108 = ttnn_decorators_ttnn_transpose(ttnn_permute_60, 3, 2);  ttnn_permute_60 = None
    ttnn_experimental_view_276 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_62, [16, 256, 64]);  ttnn_permute_62 = None
    ttnn_experimental_view_277 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_108, [16, 64, 256]);  ttnn_transpose_108 = None
    ttnn_matmul_123 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_276, ttnn_experimental_view_277);  ttnn_experimental_view_276 = ttnn_experimental_view_277 = None
    ttnn_experimental_view_278 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_123, [1, 16, 256, 256]);  ttnn_matmul_123 = None
    ttnn_multiply_16 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_278, 0.125);  ttnn_experimental_view_278 = None
    ttnn_add_192 = ttnn_decorators_ttnn_add(ttnn_multiply_16, ttnn_multiply);  ttnn_multiply_16 = None
    ttnn_softmax_15 = ttnn_decorators_ttnn_softmax(ttnn_add_192, -1, numeric_stable = True);  ttnn_add_192 = None
    ttnn_experimental_view_279 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_15, [16, 256, 256]);  ttnn_softmax_15 = None
    ttnn_experimental_view_280 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_61, [16, 256, 64]);  ttnn_permute_61 = None
    ttnn_matmul_124 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_279, ttnn_experimental_view_280);  ttnn_experimental_view_279 = ttnn_experimental_view_280 = None
    ttnn_experimental_view_281 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_124, [1, 16, 256, 64]);  ttnn_matmul_124 = None
    ttnn_permute_63 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_281, [0, 2, 1, 3]);  ttnn_experimental_view_281 = None
    ttnn_reshape_65 = ttnn_decorators_ttnn_reshape(ttnn_permute_63, [1, 256, 1024]);  ttnn_permute_63 = None
    ttnn_experimental_view_282 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_65, [256, 1024]);  ttnn_reshape_65 = None
    ttnn_from_torch_255 = ttnn_decorators_ttnn_from_torch(arg251_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg251_1 = None
    ttnn_transpose_109 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_255, 0, 1);  ttnn_from_torch_255 = None
    ttnn_matmul_125 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_282, ttnn_transpose_109);  ttnn_experimental_view_282 = ttnn_transpose_109 = None
    ttnn_from_torch_256 = ttnn_decorators_ttnn_from_torch(arg252_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg252_1 = None
    ttnn_add_93 = ttnn_decorators_ttnn_add(ttnn_from_torch_256, ttnn_matmul_125);  ttnn_from_torch_256 = ttnn_matmul_125 = None
    ttnn_experimental_view_283 = ttnn_decorators_ttnn_experimental_view(ttnn_add_93, [1, 256, 1024]);  ttnn_add_93 = None
    ttnn_add_193 = ttnn_decorators_ttnn_add(ttnn_experimental_view_283, ttnn_layer_norm_30);  ttnn_experimental_view_283 = ttnn_layer_norm_30 = None
    ttnn_from_torch_257 = ttnn_decorators_ttnn_from_torch(arg253_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg253_1 = None
    ttnn_from_torch_258 = ttnn_decorators_ttnn_from_torch(arg254_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg254_1 = None
    ttnn_layer_norm_31 = ttnn_decorators_ttnn_layer_norm(ttnn_add_193, epsilon = 1e-12, weight = ttnn_from_torch_257, bias = ttnn_from_torch_258);  ttnn_add_193 = ttnn_from_torch_257 = ttnn_from_torch_258 = None
    ttnn_experimental_view_284 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_31, [256, 1024])
    ttnn_from_torch_259 = ttnn_decorators_ttnn_from_torch(arg255_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg255_1 = None
    ttnn_transpose_110 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_259, 0, 1);  ttnn_from_torch_259 = None
    ttnn_matmul_126 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_284, ttnn_transpose_110);  ttnn_experimental_view_284 = ttnn_transpose_110 = None
    ttnn_from_torch_260 = ttnn_decorators_ttnn_from_torch(arg256_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg256_1 = None
    ttnn_add_94 = ttnn_decorators_ttnn_add(ttnn_from_torch_260, ttnn_matmul_126);  ttnn_from_torch_260 = ttnn_matmul_126 = None
    ttnn_experimental_view_285 = ttnn_decorators_ttnn_experimental_view(ttnn_add_94, [1, 256, 4096]);  ttnn_add_94 = None
    ttnn_gelu_15 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_285);  ttnn_experimental_view_285 = None
    ttnn_experimental_view_286 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_15, [256, 4096]);  ttnn_gelu_15 = None
    ttnn_from_torch_261 = ttnn_decorators_ttnn_from_torch(arg257_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg257_1 = None
    ttnn_transpose_111 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_261, 0, 1);  ttnn_from_torch_261 = None
    ttnn_matmul_127 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_286, ttnn_transpose_111);  ttnn_experimental_view_286 = ttnn_transpose_111 = None
    ttnn_from_torch_262 = ttnn_decorators_ttnn_from_torch(arg258_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg258_1 = None
    ttnn_add_95 = ttnn_decorators_ttnn_add(ttnn_from_torch_262, ttnn_matmul_127);  ttnn_from_torch_262 = ttnn_matmul_127 = None
    ttnn_experimental_view_287 = ttnn_decorators_ttnn_experimental_view(ttnn_add_95, [1, 256, 1024]);  ttnn_add_95 = None
    ttnn_add_194 = ttnn_decorators_ttnn_add(ttnn_experimental_view_287, ttnn_layer_norm_31);  ttnn_experimental_view_287 = ttnn_layer_norm_31 = None
    ttnn_from_torch_263 = ttnn_decorators_ttnn_from_torch(arg259_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg259_1 = None
    ttnn_from_torch_264 = ttnn_decorators_ttnn_from_torch(arg260_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg260_1 = None
    ttnn_layer_norm_32 = ttnn_decorators_ttnn_layer_norm(ttnn_add_194, epsilon = 1e-12, weight = ttnn_from_torch_263, bias = ttnn_from_torch_264);  ttnn_add_194 = ttnn_from_torch_263 = ttnn_from_torch_264 = None
    ttnn_experimental_view_288 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_32, [256, 1024])
    ttnn_from_torch_265 = ttnn_decorators_ttnn_from_torch(arg261_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg261_1 = None
    ttnn_transpose_112 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_265, 0, 1);  ttnn_from_torch_265 = None
    ttnn_matmul_128 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_288, ttnn_transpose_112);  ttnn_transpose_112 = None
    ttnn_from_torch_266 = ttnn_decorators_ttnn_from_torch(arg262_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg262_1 = None
    ttnn_add_96 = ttnn_decorators_ttnn_add(ttnn_from_torch_266, ttnn_matmul_128);  ttnn_from_torch_266 = ttnn_matmul_128 = None
    ttnn_experimental_view_289 = ttnn_decorators_ttnn_experimental_view(ttnn_add_96, [1, 256, 1024]);  ttnn_add_96 = None
    ttnn_from_torch_267 = ttnn_decorators_ttnn_from_torch(arg263_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg263_1 = None
    ttnn_transpose_113 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_267, 0, 1);  ttnn_from_torch_267 = None
    ttnn_matmul_129 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_288, ttnn_transpose_113);  ttnn_transpose_113 = None
    ttnn_from_torch_268 = ttnn_decorators_ttnn_from_torch(arg264_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg264_1 = None
    ttnn_add_97 = ttnn_decorators_ttnn_add(ttnn_from_torch_268, ttnn_matmul_129);  ttnn_from_torch_268 = ttnn_matmul_129 = None
    ttnn_experimental_view_291 = ttnn_decorators_ttnn_experimental_view(ttnn_add_97, [1, 256, 1024]);  ttnn_add_97 = None
    ttnn_reshape_66 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_291, [1, 256, 16, 64]);  ttnn_experimental_view_291 = None
    ttnn_permute_64 = ttnn_decorators_ttnn_permute(ttnn_reshape_66, [0, 2, 1, 3]);  ttnn_reshape_66 = None
    ttnn_from_torch_269 = ttnn_decorators_ttnn_from_torch(arg265_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg265_1 = None
    ttnn_transpose_114 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_269, 0, 1);  ttnn_from_torch_269 = None
    ttnn_matmul_130 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_288, ttnn_transpose_114);  ttnn_experimental_view_288 = ttnn_transpose_114 = None
    ttnn_from_torch_270 = ttnn_decorators_ttnn_from_torch(arg266_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg266_1 = None
    ttnn_add_98 = ttnn_decorators_ttnn_add(ttnn_from_torch_270, ttnn_matmul_130);  ttnn_from_torch_270 = ttnn_matmul_130 = None
    ttnn_experimental_view_293 = ttnn_decorators_ttnn_experimental_view(ttnn_add_98, [1, 256, 1024]);  ttnn_add_98 = None
    ttnn_reshape_67 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_293, [1, 256, 16, 64]);  ttnn_experimental_view_293 = None
    ttnn_permute_65 = ttnn_decorators_ttnn_permute(ttnn_reshape_67, [0, 2, 1, 3]);  ttnn_reshape_67 = None
    ttnn_reshape_68 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_289, [1, 256, 16, 64]);  ttnn_experimental_view_289 = None
    ttnn_permute_66 = ttnn_decorators_ttnn_permute(ttnn_reshape_68, [0, 2, 1, 3]);  ttnn_reshape_68 = None
    ttnn_transpose_115 = ttnn_decorators_ttnn_transpose(ttnn_permute_64, 3, 2);  ttnn_permute_64 = None
    ttnn_experimental_view_294 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_66, [16, 256, 64]);  ttnn_permute_66 = None
    ttnn_experimental_view_295 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_115, [16, 64, 256]);  ttnn_transpose_115 = None
    ttnn_matmul_131 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_294, ttnn_experimental_view_295);  ttnn_experimental_view_294 = ttnn_experimental_view_295 = None
    ttnn_experimental_view_296 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_131, [1, 16, 256, 256]);  ttnn_matmul_131 = None
    ttnn_multiply_17 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_296, 0.125);  ttnn_experimental_view_296 = None
    ttnn_add_195 = ttnn_decorators_ttnn_add(ttnn_multiply_17, ttnn_multiply);  ttnn_multiply_17 = None
    ttnn_softmax_16 = ttnn_decorators_ttnn_softmax(ttnn_add_195, -1, numeric_stable = True);  ttnn_add_195 = None
    ttnn_experimental_view_297 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_16, [16, 256, 256]);  ttnn_softmax_16 = None
    ttnn_experimental_view_298 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_65, [16, 256, 64]);  ttnn_permute_65 = None
    ttnn_matmul_132 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_297, ttnn_experimental_view_298);  ttnn_experimental_view_297 = ttnn_experimental_view_298 = None
    ttnn_experimental_view_299 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_132, [1, 16, 256, 64]);  ttnn_matmul_132 = None
    ttnn_permute_67 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_299, [0, 2, 1, 3]);  ttnn_experimental_view_299 = None
    ttnn_reshape_69 = ttnn_decorators_ttnn_reshape(ttnn_permute_67, [1, 256, 1024]);  ttnn_permute_67 = None
    ttnn_experimental_view_300 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_69, [256, 1024]);  ttnn_reshape_69 = None
    ttnn_from_torch_271 = ttnn_decorators_ttnn_from_torch(arg267_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg267_1 = None
    ttnn_transpose_116 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_271, 0, 1);  ttnn_from_torch_271 = None
    ttnn_matmul_133 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_300, ttnn_transpose_116);  ttnn_experimental_view_300 = ttnn_transpose_116 = None
    ttnn_from_torch_272 = ttnn_decorators_ttnn_from_torch(arg268_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg268_1 = None
    ttnn_add_99 = ttnn_decorators_ttnn_add(ttnn_from_torch_272, ttnn_matmul_133);  ttnn_from_torch_272 = ttnn_matmul_133 = None
    ttnn_experimental_view_301 = ttnn_decorators_ttnn_experimental_view(ttnn_add_99, [1, 256, 1024]);  ttnn_add_99 = None
    ttnn_add_196 = ttnn_decorators_ttnn_add(ttnn_experimental_view_301, ttnn_layer_norm_32);  ttnn_experimental_view_301 = ttnn_layer_norm_32 = None
    ttnn_from_torch_273 = ttnn_decorators_ttnn_from_torch(arg269_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg269_1 = None
    ttnn_from_torch_274 = ttnn_decorators_ttnn_from_torch(arg270_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg270_1 = None
    ttnn_layer_norm_33 = ttnn_decorators_ttnn_layer_norm(ttnn_add_196, epsilon = 1e-12, weight = ttnn_from_torch_273, bias = ttnn_from_torch_274);  ttnn_add_196 = ttnn_from_torch_273 = ttnn_from_torch_274 = None
    ttnn_experimental_view_302 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_33, [256, 1024])
    ttnn_from_torch_275 = ttnn_decorators_ttnn_from_torch(arg271_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg271_1 = None
    ttnn_transpose_117 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_275, 0, 1);  ttnn_from_torch_275 = None
    ttnn_matmul_134 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_302, ttnn_transpose_117);  ttnn_experimental_view_302 = ttnn_transpose_117 = None
    ttnn_from_torch_276 = ttnn_decorators_ttnn_from_torch(arg272_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg272_1 = None
    ttnn_add_100 = ttnn_decorators_ttnn_add(ttnn_from_torch_276, ttnn_matmul_134);  ttnn_from_torch_276 = ttnn_matmul_134 = None
    ttnn_experimental_view_303 = ttnn_decorators_ttnn_experimental_view(ttnn_add_100, [1, 256, 4096]);  ttnn_add_100 = None
    ttnn_gelu_16 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_303);  ttnn_experimental_view_303 = None
    ttnn_experimental_view_304 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_16, [256, 4096]);  ttnn_gelu_16 = None
    ttnn_from_torch_277 = ttnn_decorators_ttnn_from_torch(arg273_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg273_1 = None
    ttnn_transpose_118 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_277, 0, 1);  ttnn_from_torch_277 = None
    ttnn_matmul_135 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_304, ttnn_transpose_118);  ttnn_experimental_view_304 = ttnn_transpose_118 = None
    ttnn_from_torch_278 = ttnn_decorators_ttnn_from_torch(arg274_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg274_1 = None
    ttnn_add_101 = ttnn_decorators_ttnn_add(ttnn_from_torch_278, ttnn_matmul_135);  ttnn_from_torch_278 = ttnn_matmul_135 = None
    ttnn_experimental_view_305 = ttnn_decorators_ttnn_experimental_view(ttnn_add_101, [1, 256, 1024]);  ttnn_add_101 = None
    ttnn_add_197 = ttnn_decorators_ttnn_add(ttnn_experimental_view_305, ttnn_layer_norm_33);  ttnn_experimental_view_305 = ttnn_layer_norm_33 = None
    ttnn_from_torch_279 = ttnn_decorators_ttnn_from_torch(arg275_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg275_1 = None
    ttnn_from_torch_280 = ttnn_decorators_ttnn_from_torch(arg276_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg276_1 = None
    ttnn_layer_norm_34 = ttnn_decorators_ttnn_layer_norm(ttnn_add_197, epsilon = 1e-12, weight = ttnn_from_torch_279, bias = ttnn_from_torch_280);  ttnn_add_197 = ttnn_from_torch_279 = ttnn_from_torch_280 = None
    ttnn_experimental_view_306 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_34, [256, 1024])
    ttnn_from_torch_281 = ttnn_decorators_ttnn_from_torch(arg277_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg277_1 = None
    ttnn_transpose_119 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_281, 0, 1);  ttnn_from_torch_281 = None
    ttnn_matmul_136 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_306, ttnn_transpose_119);  ttnn_transpose_119 = None
    ttnn_from_torch_282 = ttnn_decorators_ttnn_from_torch(arg278_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg278_1 = None
    ttnn_add_102 = ttnn_decorators_ttnn_add(ttnn_from_torch_282, ttnn_matmul_136);  ttnn_from_torch_282 = ttnn_matmul_136 = None
    ttnn_experimental_view_307 = ttnn_decorators_ttnn_experimental_view(ttnn_add_102, [1, 256, 1024]);  ttnn_add_102 = None
    ttnn_from_torch_283 = ttnn_decorators_ttnn_from_torch(arg279_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg279_1 = None
    ttnn_transpose_120 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_283, 0, 1);  ttnn_from_torch_283 = None
    ttnn_matmul_137 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_306, ttnn_transpose_120);  ttnn_transpose_120 = None
    ttnn_from_torch_284 = ttnn_decorators_ttnn_from_torch(arg280_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg280_1 = None
    ttnn_add_103 = ttnn_decorators_ttnn_add(ttnn_from_torch_284, ttnn_matmul_137);  ttnn_from_torch_284 = ttnn_matmul_137 = None
    ttnn_experimental_view_309 = ttnn_decorators_ttnn_experimental_view(ttnn_add_103, [1, 256, 1024]);  ttnn_add_103 = None
    ttnn_reshape_70 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_309, [1, 256, 16, 64]);  ttnn_experimental_view_309 = None
    ttnn_permute_68 = ttnn_decorators_ttnn_permute(ttnn_reshape_70, [0, 2, 1, 3]);  ttnn_reshape_70 = None
    ttnn_from_torch_285 = ttnn_decorators_ttnn_from_torch(arg281_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg281_1 = None
    ttnn_transpose_121 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_285, 0, 1);  ttnn_from_torch_285 = None
    ttnn_matmul_138 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_306, ttnn_transpose_121);  ttnn_experimental_view_306 = ttnn_transpose_121 = None
    ttnn_from_torch_286 = ttnn_decorators_ttnn_from_torch(arg282_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg282_1 = None
    ttnn_add_104 = ttnn_decorators_ttnn_add(ttnn_from_torch_286, ttnn_matmul_138);  ttnn_from_torch_286 = ttnn_matmul_138 = None
    ttnn_experimental_view_311 = ttnn_decorators_ttnn_experimental_view(ttnn_add_104, [1, 256, 1024]);  ttnn_add_104 = None
    ttnn_reshape_71 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_311, [1, 256, 16, 64]);  ttnn_experimental_view_311 = None
    ttnn_permute_69 = ttnn_decorators_ttnn_permute(ttnn_reshape_71, [0, 2, 1, 3]);  ttnn_reshape_71 = None
    ttnn_reshape_72 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_307, [1, 256, 16, 64]);  ttnn_experimental_view_307 = None
    ttnn_permute_70 = ttnn_decorators_ttnn_permute(ttnn_reshape_72, [0, 2, 1, 3]);  ttnn_reshape_72 = None
    ttnn_transpose_122 = ttnn_decorators_ttnn_transpose(ttnn_permute_68, 3, 2);  ttnn_permute_68 = None
    ttnn_experimental_view_312 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_70, [16, 256, 64]);  ttnn_permute_70 = None
    ttnn_experimental_view_313 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_122, [16, 64, 256]);  ttnn_transpose_122 = None
    ttnn_matmul_139 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_312, ttnn_experimental_view_313);  ttnn_experimental_view_312 = ttnn_experimental_view_313 = None
    ttnn_experimental_view_314 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_139, [1, 16, 256, 256]);  ttnn_matmul_139 = None
    ttnn_multiply_18 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_314, 0.125);  ttnn_experimental_view_314 = None
    ttnn_add_198 = ttnn_decorators_ttnn_add(ttnn_multiply_18, ttnn_multiply);  ttnn_multiply_18 = None
    ttnn_softmax_17 = ttnn_decorators_ttnn_softmax(ttnn_add_198, -1, numeric_stable = True);  ttnn_add_198 = None
    ttnn_experimental_view_315 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_17, [16, 256, 256]);  ttnn_softmax_17 = None
    ttnn_experimental_view_316 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_69, [16, 256, 64]);  ttnn_permute_69 = None
    ttnn_matmul_140 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_315, ttnn_experimental_view_316);  ttnn_experimental_view_315 = ttnn_experimental_view_316 = None
    ttnn_experimental_view_317 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_140, [1, 16, 256, 64]);  ttnn_matmul_140 = None
    ttnn_permute_71 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_317, [0, 2, 1, 3]);  ttnn_experimental_view_317 = None
    ttnn_reshape_73 = ttnn_decorators_ttnn_reshape(ttnn_permute_71, [1, 256, 1024]);  ttnn_permute_71 = None
    ttnn_experimental_view_318 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_73, [256, 1024]);  ttnn_reshape_73 = None
    ttnn_from_torch_287 = ttnn_decorators_ttnn_from_torch(arg283_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg283_1 = None
    ttnn_transpose_123 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_287, 0, 1);  ttnn_from_torch_287 = None
    ttnn_matmul_141 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_318, ttnn_transpose_123);  ttnn_experimental_view_318 = ttnn_transpose_123 = None
    ttnn_from_torch_288 = ttnn_decorators_ttnn_from_torch(arg284_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg284_1 = None
    ttnn_add_105 = ttnn_decorators_ttnn_add(ttnn_from_torch_288, ttnn_matmul_141);  ttnn_from_torch_288 = ttnn_matmul_141 = None
    ttnn_experimental_view_319 = ttnn_decorators_ttnn_experimental_view(ttnn_add_105, [1, 256, 1024]);  ttnn_add_105 = None
    ttnn_add_199 = ttnn_decorators_ttnn_add(ttnn_experimental_view_319, ttnn_layer_norm_34);  ttnn_experimental_view_319 = ttnn_layer_norm_34 = None
    ttnn_from_torch_289 = ttnn_decorators_ttnn_from_torch(arg285_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg285_1 = None
    ttnn_from_torch_290 = ttnn_decorators_ttnn_from_torch(arg286_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg286_1 = None
    ttnn_layer_norm_35 = ttnn_decorators_ttnn_layer_norm(ttnn_add_199, epsilon = 1e-12, weight = ttnn_from_torch_289, bias = ttnn_from_torch_290);  ttnn_add_199 = ttnn_from_torch_289 = ttnn_from_torch_290 = None
    ttnn_experimental_view_320 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_35, [256, 1024])
    ttnn_from_torch_291 = ttnn_decorators_ttnn_from_torch(arg287_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg287_1 = None
    ttnn_transpose_124 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_291, 0, 1);  ttnn_from_torch_291 = None
    ttnn_matmul_142 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_320, ttnn_transpose_124);  ttnn_experimental_view_320 = ttnn_transpose_124 = None
    ttnn_from_torch_292 = ttnn_decorators_ttnn_from_torch(arg288_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg288_1 = None
    ttnn_add_106 = ttnn_decorators_ttnn_add(ttnn_from_torch_292, ttnn_matmul_142);  ttnn_from_torch_292 = ttnn_matmul_142 = None
    ttnn_experimental_view_321 = ttnn_decorators_ttnn_experimental_view(ttnn_add_106, [1, 256, 4096]);  ttnn_add_106 = None
    ttnn_gelu_17 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_321);  ttnn_experimental_view_321 = None
    ttnn_experimental_view_322 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_17, [256, 4096]);  ttnn_gelu_17 = None
    ttnn_from_torch_293 = ttnn_decorators_ttnn_from_torch(arg289_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg289_1 = None
    ttnn_transpose_125 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_293, 0, 1);  ttnn_from_torch_293 = None
    ttnn_matmul_143 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_322, ttnn_transpose_125);  ttnn_experimental_view_322 = ttnn_transpose_125 = None
    ttnn_from_torch_294 = ttnn_decorators_ttnn_from_torch(arg290_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg290_1 = None
    ttnn_add_107 = ttnn_decorators_ttnn_add(ttnn_from_torch_294, ttnn_matmul_143);  ttnn_from_torch_294 = ttnn_matmul_143 = None
    ttnn_experimental_view_323 = ttnn_decorators_ttnn_experimental_view(ttnn_add_107, [1, 256, 1024]);  ttnn_add_107 = None
    ttnn_add_200 = ttnn_decorators_ttnn_add(ttnn_experimental_view_323, ttnn_layer_norm_35);  ttnn_experimental_view_323 = ttnn_layer_norm_35 = None
    ttnn_from_torch_295 = ttnn_decorators_ttnn_from_torch(arg291_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg291_1 = None
    ttnn_from_torch_296 = ttnn_decorators_ttnn_from_torch(arg292_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg292_1 = None
    ttnn_layer_norm_36 = ttnn_decorators_ttnn_layer_norm(ttnn_add_200, epsilon = 1e-12, weight = ttnn_from_torch_295, bias = ttnn_from_torch_296);  ttnn_add_200 = ttnn_from_torch_295 = ttnn_from_torch_296 = None
    ttnn_experimental_view_324 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_36, [256, 1024])
    ttnn_from_torch_297 = ttnn_decorators_ttnn_from_torch(arg293_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg293_1 = None
    ttnn_transpose_126 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_297, 0, 1);  ttnn_from_torch_297 = None
    ttnn_matmul_144 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_324, ttnn_transpose_126);  ttnn_transpose_126 = None
    ttnn_from_torch_298 = ttnn_decorators_ttnn_from_torch(arg294_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg294_1 = None
    ttnn_add_108 = ttnn_decorators_ttnn_add(ttnn_from_torch_298, ttnn_matmul_144);  ttnn_from_torch_298 = ttnn_matmul_144 = None
    ttnn_experimental_view_325 = ttnn_decorators_ttnn_experimental_view(ttnn_add_108, [1, 256, 1024]);  ttnn_add_108 = None
    ttnn_from_torch_299 = ttnn_decorators_ttnn_from_torch(arg295_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg295_1 = None
    ttnn_transpose_127 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_299, 0, 1);  ttnn_from_torch_299 = None
    ttnn_matmul_145 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_324, ttnn_transpose_127);  ttnn_transpose_127 = None
    ttnn_from_torch_300 = ttnn_decorators_ttnn_from_torch(arg296_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg296_1 = None
    ttnn_add_109 = ttnn_decorators_ttnn_add(ttnn_from_torch_300, ttnn_matmul_145);  ttnn_from_torch_300 = ttnn_matmul_145 = None
    ttnn_experimental_view_327 = ttnn_decorators_ttnn_experimental_view(ttnn_add_109, [1, 256, 1024]);  ttnn_add_109 = None
    ttnn_reshape_74 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_327, [1, 256, 16, 64]);  ttnn_experimental_view_327 = None
    ttnn_permute_72 = ttnn_decorators_ttnn_permute(ttnn_reshape_74, [0, 2, 1, 3]);  ttnn_reshape_74 = None
    ttnn_from_torch_301 = ttnn_decorators_ttnn_from_torch(arg297_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg297_1 = None
    ttnn_transpose_128 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_301, 0, 1);  ttnn_from_torch_301 = None
    ttnn_matmul_146 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_324, ttnn_transpose_128);  ttnn_experimental_view_324 = ttnn_transpose_128 = None
    ttnn_from_torch_302 = ttnn_decorators_ttnn_from_torch(arg298_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg298_1 = None
    ttnn_add_110 = ttnn_decorators_ttnn_add(ttnn_from_torch_302, ttnn_matmul_146);  ttnn_from_torch_302 = ttnn_matmul_146 = None
    ttnn_experimental_view_329 = ttnn_decorators_ttnn_experimental_view(ttnn_add_110, [1, 256, 1024]);  ttnn_add_110 = None
    ttnn_reshape_75 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_329, [1, 256, 16, 64]);  ttnn_experimental_view_329 = None
    ttnn_permute_73 = ttnn_decorators_ttnn_permute(ttnn_reshape_75, [0, 2, 1, 3]);  ttnn_reshape_75 = None
    ttnn_reshape_76 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_325, [1, 256, 16, 64]);  ttnn_experimental_view_325 = None
    ttnn_permute_74 = ttnn_decorators_ttnn_permute(ttnn_reshape_76, [0, 2, 1, 3]);  ttnn_reshape_76 = None
    ttnn_transpose_129 = ttnn_decorators_ttnn_transpose(ttnn_permute_72, 3, 2);  ttnn_permute_72 = None
    ttnn_experimental_view_330 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_74, [16, 256, 64]);  ttnn_permute_74 = None
    ttnn_experimental_view_331 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_129, [16, 64, 256]);  ttnn_transpose_129 = None
    ttnn_matmul_147 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_330, ttnn_experimental_view_331);  ttnn_experimental_view_330 = ttnn_experimental_view_331 = None
    ttnn_experimental_view_332 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_147, [1, 16, 256, 256]);  ttnn_matmul_147 = None
    ttnn_multiply_19 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_332, 0.125);  ttnn_experimental_view_332 = None
    ttnn_add_201 = ttnn_decorators_ttnn_add(ttnn_multiply_19, ttnn_multiply);  ttnn_multiply_19 = None
    ttnn_softmax_18 = ttnn_decorators_ttnn_softmax(ttnn_add_201, -1, numeric_stable = True);  ttnn_add_201 = None
    ttnn_experimental_view_333 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_18, [16, 256, 256]);  ttnn_softmax_18 = None
    ttnn_experimental_view_334 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_73, [16, 256, 64]);  ttnn_permute_73 = None
    ttnn_matmul_148 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_333, ttnn_experimental_view_334);  ttnn_experimental_view_333 = ttnn_experimental_view_334 = None
    ttnn_experimental_view_335 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_148, [1, 16, 256, 64]);  ttnn_matmul_148 = None
    ttnn_permute_75 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_335, [0, 2, 1, 3]);  ttnn_experimental_view_335 = None
    ttnn_reshape_77 = ttnn_decorators_ttnn_reshape(ttnn_permute_75, [1, 256, 1024]);  ttnn_permute_75 = None
    ttnn_experimental_view_336 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_77, [256, 1024]);  ttnn_reshape_77 = None
    ttnn_from_torch_303 = ttnn_decorators_ttnn_from_torch(arg299_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg299_1 = None
    ttnn_transpose_130 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_303, 0, 1);  ttnn_from_torch_303 = None
    ttnn_matmul_149 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_336, ttnn_transpose_130);  ttnn_experimental_view_336 = ttnn_transpose_130 = None
    ttnn_from_torch_304 = ttnn_decorators_ttnn_from_torch(arg300_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg300_1 = None
    ttnn_add_111 = ttnn_decorators_ttnn_add(ttnn_from_torch_304, ttnn_matmul_149);  ttnn_from_torch_304 = ttnn_matmul_149 = None
    ttnn_experimental_view_337 = ttnn_decorators_ttnn_experimental_view(ttnn_add_111, [1, 256, 1024]);  ttnn_add_111 = None
    ttnn_add_202 = ttnn_decorators_ttnn_add(ttnn_experimental_view_337, ttnn_layer_norm_36);  ttnn_experimental_view_337 = ttnn_layer_norm_36 = None
    ttnn_from_torch_305 = ttnn_decorators_ttnn_from_torch(arg301_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg301_1 = None
    ttnn_from_torch_306 = ttnn_decorators_ttnn_from_torch(arg302_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg302_1 = None
    ttnn_layer_norm_37 = ttnn_decorators_ttnn_layer_norm(ttnn_add_202, epsilon = 1e-12, weight = ttnn_from_torch_305, bias = ttnn_from_torch_306);  ttnn_add_202 = ttnn_from_torch_305 = ttnn_from_torch_306 = None
    ttnn_experimental_view_338 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_37, [256, 1024])
    ttnn_from_torch_307 = ttnn_decorators_ttnn_from_torch(arg303_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg303_1 = None
    ttnn_transpose_131 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_307, 0, 1);  ttnn_from_torch_307 = None
    ttnn_matmul_150 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_338, ttnn_transpose_131);  ttnn_experimental_view_338 = ttnn_transpose_131 = None
    ttnn_from_torch_308 = ttnn_decorators_ttnn_from_torch(arg304_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg304_1 = None
    ttnn_add_112 = ttnn_decorators_ttnn_add(ttnn_from_torch_308, ttnn_matmul_150);  ttnn_from_torch_308 = ttnn_matmul_150 = None
    ttnn_experimental_view_339 = ttnn_decorators_ttnn_experimental_view(ttnn_add_112, [1, 256, 4096]);  ttnn_add_112 = None
    ttnn_gelu_18 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_339);  ttnn_experimental_view_339 = None
    ttnn_experimental_view_340 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_18, [256, 4096]);  ttnn_gelu_18 = None
    ttnn_from_torch_309 = ttnn_decorators_ttnn_from_torch(arg305_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg305_1 = None
    ttnn_transpose_132 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_309, 0, 1);  ttnn_from_torch_309 = None
    ttnn_matmul_151 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_340, ttnn_transpose_132);  ttnn_experimental_view_340 = ttnn_transpose_132 = None
    ttnn_from_torch_310 = ttnn_decorators_ttnn_from_torch(arg306_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg306_1 = None
    ttnn_add_113 = ttnn_decorators_ttnn_add(ttnn_from_torch_310, ttnn_matmul_151);  ttnn_from_torch_310 = ttnn_matmul_151 = None
    ttnn_experimental_view_341 = ttnn_decorators_ttnn_experimental_view(ttnn_add_113, [1, 256, 1024]);  ttnn_add_113 = None
    ttnn_add_203 = ttnn_decorators_ttnn_add(ttnn_experimental_view_341, ttnn_layer_norm_37);  ttnn_experimental_view_341 = ttnn_layer_norm_37 = None
    ttnn_from_torch_311 = ttnn_decorators_ttnn_from_torch(arg307_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg307_1 = None
    ttnn_from_torch_312 = ttnn_decorators_ttnn_from_torch(arg308_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg308_1 = None
    ttnn_layer_norm_38 = ttnn_decorators_ttnn_layer_norm(ttnn_add_203, epsilon = 1e-12, weight = ttnn_from_torch_311, bias = ttnn_from_torch_312);  ttnn_add_203 = ttnn_from_torch_311 = ttnn_from_torch_312 = None
    ttnn_experimental_view_342 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_38, [256, 1024])
    ttnn_from_torch_313 = ttnn_decorators_ttnn_from_torch(arg309_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg309_1 = None
    ttnn_transpose_133 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_313, 0, 1);  ttnn_from_torch_313 = None
    ttnn_matmul_152 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_342, ttnn_transpose_133);  ttnn_transpose_133 = None
    ttnn_from_torch_314 = ttnn_decorators_ttnn_from_torch(arg310_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg310_1 = None
    ttnn_add_114 = ttnn_decorators_ttnn_add(ttnn_from_torch_314, ttnn_matmul_152);  ttnn_from_torch_314 = ttnn_matmul_152 = None
    ttnn_experimental_view_343 = ttnn_decorators_ttnn_experimental_view(ttnn_add_114, [1, 256, 1024]);  ttnn_add_114 = None
    ttnn_from_torch_315 = ttnn_decorators_ttnn_from_torch(arg311_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg311_1 = None
    ttnn_transpose_134 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_315, 0, 1);  ttnn_from_torch_315 = None
    ttnn_matmul_153 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_342, ttnn_transpose_134);  ttnn_transpose_134 = None
    ttnn_from_torch_316 = ttnn_decorators_ttnn_from_torch(arg312_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg312_1 = None
    ttnn_add_115 = ttnn_decorators_ttnn_add(ttnn_from_torch_316, ttnn_matmul_153);  ttnn_from_torch_316 = ttnn_matmul_153 = None
    ttnn_experimental_view_345 = ttnn_decorators_ttnn_experimental_view(ttnn_add_115, [1, 256, 1024]);  ttnn_add_115 = None
    ttnn_reshape_78 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_345, [1, 256, 16, 64]);  ttnn_experimental_view_345 = None
    ttnn_permute_76 = ttnn_decorators_ttnn_permute(ttnn_reshape_78, [0, 2, 1, 3]);  ttnn_reshape_78 = None
    ttnn_from_torch_317 = ttnn_decorators_ttnn_from_torch(arg313_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg313_1 = None
    ttnn_transpose_135 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_317, 0, 1);  ttnn_from_torch_317 = None
    ttnn_matmul_154 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_342, ttnn_transpose_135);  ttnn_experimental_view_342 = ttnn_transpose_135 = None
    ttnn_from_torch_318 = ttnn_decorators_ttnn_from_torch(arg314_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg314_1 = None
    ttnn_add_116 = ttnn_decorators_ttnn_add(ttnn_from_torch_318, ttnn_matmul_154);  ttnn_from_torch_318 = ttnn_matmul_154 = None
    ttnn_experimental_view_347 = ttnn_decorators_ttnn_experimental_view(ttnn_add_116, [1, 256, 1024]);  ttnn_add_116 = None
    ttnn_reshape_79 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_347, [1, 256, 16, 64]);  ttnn_experimental_view_347 = None
    ttnn_permute_77 = ttnn_decorators_ttnn_permute(ttnn_reshape_79, [0, 2, 1, 3]);  ttnn_reshape_79 = None
    ttnn_reshape_80 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_343, [1, 256, 16, 64]);  ttnn_experimental_view_343 = None
    ttnn_permute_78 = ttnn_decorators_ttnn_permute(ttnn_reshape_80, [0, 2, 1, 3]);  ttnn_reshape_80 = None
    ttnn_transpose_136 = ttnn_decorators_ttnn_transpose(ttnn_permute_76, 3, 2);  ttnn_permute_76 = None
    ttnn_experimental_view_348 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_78, [16, 256, 64]);  ttnn_permute_78 = None
    ttnn_experimental_view_349 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_136, [16, 64, 256]);  ttnn_transpose_136 = None
    ttnn_matmul_155 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_348, ttnn_experimental_view_349);  ttnn_experimental_view_348 = ttnn_experimental_view_349 = None
    ttnn_experimental_view_350 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_155, [1, 16, 256, 256]);  ttnn_matmul_155 = None
    ttnn_multiply_20 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_350, 0.125);  ttnn_experimental_view_350 = None
    ttnn_add_204 = ttnn_decorators_ttnn_add(ttnn_multiply_20, ttnn_multiply);  ttnn_multiply_20 = None
    ttnn_softmax_19 = ttnn_decorators_ttnn_softmax(ttnn_add_204, -1, numeric_stable = True);  ttnn_add_204 = None
    ttnn_experimental_view_351 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_19, [16, 256, 256]);  ttnn_softmax_19 = None
    ttnn_experimental_view_352 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_77, [16, 256, 64]);  ttnn_permute_77 = None
    ttnn_matmul_156 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_351, ttnn_experimental_view_352);  ttnn_experimental_view_351 = ttnn_experimental_view_352 = None
    ttnn_experimental_view_353 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_156, [1, 16, 256, 64]);  ttnn_matmul_156 = None
    ttnn_permute_79 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_353, [0, 2, 1, 3]);  ttnn_experimental_view_353 = None
    ttnn_reshape_81 = ttnn_decorators_ttnn_reshape(ttnn_permute_79, [1, 256, 1024]);  ttnn_permute_79 = None
    ttnn_experimental_view_354 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_81, [256, 1024]);  ttnn_reshape_81 = None
    ttnn_from_torch_319 = ttnn_decorators_ttnn_from_torch(arg315_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg315_1 = None
    ttnn_transpose_137 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_319, 0, 1);  ttnn_from_torch_319 = None
    ttnn_matmul_157 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_354, ttnn_transpose_137);  ttnn_experimental_view_354 = ttnn_transpose_137 = None
    ttnn_from_torch_320 = ttnn_decorators_ttnn_from_torch(arg316_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg316_1 = None
    ttnn_add_117 = ttnn_decorators_ttnn_add(ttnn_from_torch_320, ttnn_matmul_157);  ttnn_from_torch_320 = ttnn_matmul_157 = None
    ttnn_experimental_view_355 = ttnn_decorators_ttnn_experimental_view(ttnn_add_117, [1, 256, 1024]);  ttnn_add_117 = None
    ttnn_add_205 = ttnn_decorators_ttnn_add(ttnn_experimental_view_355, ttnn_layer_norm_38);  ttnn_experimental_view_355 = ttnn_layer_norm_38 = None
    ttnn_from_torch_321 = ttnn_decorators_ttnn_from_torch(arg317_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg317_1 = None
    ttnn_from_torch_322 = ttnn_decorators_ttnn_from_torch(arg318_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg318_1 = None
    ttnn_layer_norm_39 = ttnn_decorators_ttnn_layer_norm(ttnn_add_205, epsilon = 1e-12, weight = ttnn_from_torch_321, bias = ttnn_from_torch_322);  ttnn_add_205 = ttnn_from_torch_321 = ttnn_from_torch_322 = None
    ttnn_experimental_view_356 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_39, [256, 1024])
    ttnn_from_torch_323 = ttnn_decorators_ttnn_from_torch(arg319_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg319_1 = None
    ttnn_transpose_138 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_323, 0, 1);  ttnn_from_torch_323 = None
    ttnn_matmul_158 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_356, ttnn_transpose_138);  ttnn_experimental_view_356 = ttnn_transpose_138 = None
    ttnn_from_torch_324 = ttnn_decorators_ttnn_from_torch(arg320_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg320_1 = None
    ttnn_add_118 = ttnn_decorators_ttnn_add(ttnn_from_torch_324, ttnn_matmul_158);  ttnn_from_torch_324 = ttnn_matmul_158 = None
    ttnn_experimental_view_357 = ttnn_decorators_ttnn_experimental_view(ttnn_add_118, [1, 256, 4096]);  ttnn_add_118 = None
    ttnn_gelu_19 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_357);  ttnn_experimental_view_357 = None
    ttnn_experimental_view_358 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_19, [256, 4096]);  ttnn_gelu_19 = None
    ttnn_from_torch_325 = ttnn_decorators_ttnn_from_torch(arg321_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg321_1 = None
    ttnn_transpose_139 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_325, 0, 1);  ttnn_from_torch_325 = None
    ttnn_matmul_159 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_358, ttnn_transpose_139);  ttnn_experimental_view_358 = ttnn_transpose_139 = None
    ttnn_from_torch_326 = ttnn_decorators_ttnn_from_torch(arg322_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg322_1 = None
    ttnn_add_119 = ttnn_decorators_ttnn_add(ttnn_from_torch_326, ttnn_matmul_159);  ttnn_from_torch_326 = ttnn_matmul_159 = None
    ttnn_experimental_view_359 = ttnn_decorators_ttnn_experimental_view(ttnn_add_119, [1, 256, 1024]);  ttnn_add_119 = None
    ttnn_add_206 = ttnn_decorators_ttnn_add(ttnn_experimental_view_359, ttnn_layer_norm_39);  ttnn_experimental_view_359 = ttnn_layer_norm_39 = None
    ttnn_from_torch_327 = ttnn_decorators_ttnn_from_torch(arg323_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg323_1 = None
    ttnn_from_torch_328 = ttnn_decorators_ttnn_from_torch(arg324_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg324_1 = None
    ttnn_layer_norm_40 = ttnn_decorators_ttnn_layer_norm(ttnn_add_206, epsilon = 1e-12, weight = ttnn_from_torch_327, bias = ttnn_from_torch_328);  ttnn_add_206 = ttnn_from_torch_327 = ttnn_from_torch_328 = None
    ttnn_experimental_view_360 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_40, [256, 1024])
    ttnn_from_torch_329 = ttnn_decorators_ttnn_from_torch(arg325_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg325_1 = None
    ttnn_transpose_140 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_329, 0, 1);  ttnn_from_torch_329 = None
    ttnn_matmul_160 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_360, ttnn_transpose_140);  ttnn_transpose_140 = None
    ttnn_from_torch_330 = ttnn_decorators_ttnn_from_torch(arg326_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg326_1 = None
    ttnn_add_120 = ttnn_decorators_ttnn_add(ttnn_from_torch_330, ttnn_matmul_160);  ttnn_from_torch_330 = ttnn_matmul_160 = None
    ttnn_experimental_view_361 = ttnn_decorators_ttnn_experimental_view(ttnn_add_120, [1, 256, 1024]);  ttnn_add_120 = None
    ttnn_from_torch_331 = ttnn_decorators_ttnn_from_torch(arg327_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg327_1 = None
    ttnn_transpose_141 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_331, 0, 1);  ttnn_from_torch_331 = None
    ttnn_matmul_161 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_360, ttnn_transpose_141);  ttnn_transpose_141 = None
    ttnn_from_torch_332 = ttnn_decorators_ttnn_from_torch(arg328_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg328_1 = None
    ttnn_add_121 = ttnn_decorators_ttnn_add(ttnn_from_torch_332, ttnn_matmul_161);  ttnn_from_torch_332 = ttnn_matmul_161 = None
    ttnn_experimental_view_363 = ttnn_decorators_ttnn_experimental_view(ttnn_add_121, [1, 256, 1024]);  ttnn_add_121 = None
    ttnn_reshape_82 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_363, [1, 256, 16, 64]);  ttnn_experimental_view_363 = None
    ttnn_permute_80 = ttnn_decorators_ttnn_permute(ttnn_reshape_82, [0, 2, 1, 3]);  ttnn_reshape_82 = None
    ttnn_from_torch_333 = ttnn_decorators_ttnn_from_torch(arg329_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg329_1 = None
    ttnn_transpose_142 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_333, 0, 1);  ttnn_from_torch_333 = None
    ttnn_matmul_162 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_360, ttnn_transpose_142);  ttnn_experimental_view_360 = ttnn_transpose_142 = None
    ttnn_from_torch_334 = ttnn_decorators_ttnn_from_torch(arg330_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg330_1 = None
    ttnn_add_122 = ttnn_decorators_ttnn_add(ttnn_from_torch_334, ttnn_matmul_162);  ttnn_from_torch_334 = ttnn_matmul_162 = None
    ttnn_experimental_view_365 = ttnn_decorators_ttnn_experimental_view(ttnn_add_122, [1, 256, 1024]);  ttnn_add_122 = None
    ttnn_reshape_83 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_365, [1, 256, 16, 64]);  ttnn_experimental_view_365 = None
    ttnn_permute_81 = ttnn_decorators_ttnn_permute(ttnn_reshape_83, [0, 2, 1, 3]);  ttnn_reshape_83 = None
    ttnn_reshape_84 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_361, [1, 256, 16, 64]);  ttnn_experimental_view_361 = None
    ttnn_permute_82 = ttnn_decorators_ttnn_permute(ttnn_reshape_84, [0, 2, 1, 3]);  ttnn_reshape_84 = None
    ttnn_transpose_143 = ttnn_decorators_ttnn_transpose(ttnn_permute_80, 3, 2);  ttnn_permute_80 = None
    ttnn_experimental_view_366 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_82, [16, 256, 64]);  ttnn_permute_82 = None
    ttnn_experimental_view_367 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_143, [16, 64, 256]);  ttnn_transpose_143 = None
    ttnn_matmul_163 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_366, ttnn_experimental_view_367);  ttnn_experimental_view_366 = ttnn_experimental_view_367 = None
    ttnn_experimental_view_368 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_163, [1, 16, 256, 256]);  ttnn_matmul_163 = None
    ttnn_multiply_21 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_368, 0.125);  ttnn_experimental_view_368 = None
    ttnn_add_207 = ttnn_decorators_ttnn_add(ttnn_multiply_21, ttnn_multiply);  ttnn_multiply_21 = None
    ttnn_softmax_20 = ttnn_decorators_ttnn_softmax(ttnn_add_207, -1, numeric_stable = True);  ttnn_add_207 = None
    ttnn_experimental_view_369 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_20, [16, 256, 256]);  ttnn_softmax_20 = None
    ttnn_experimental_view_370 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_81, [16, 256, 64]);  ttnn_permute_81 = None
    ttnn_matmul_164 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_369, ttnn_experimental_view_370);  ttnn_experimental_view_369 = ttnn_experimental_view_370 = None
    ttnn_experimental_view_371 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_164, [1, 16, 256, 64]);  ttnn_matmul_164 = None
    ttnn_permute_83 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_371, [0, 2, 1, 3]);  ttnn_experimental_view_371 = None
    ttnn_reshape_85 = ttnn_decorators_ttnn_reshape(ttnn_permute_83, [1, 256, 1024]);  ttnn_permute_83 = None
    ttnn_experimental_view_372 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_85, [256, 1024]);  ttnn_reshape_85 = None
    ttnn_from_torch_335 = ttnn_decorators_ttnn_from_torch(arg331_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg331_1 = None
    ttnn_transpose_144 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_335, 0, 1);  ttnn_from_torch_335 = None
    ttnn_matmul_165 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_372, ttnn_transpose_144);  ttnn_experimental_view_372 = ttnn_transpose_144 = None
    ttnn_from_torch_336 = ttnn_decorators_ttnn_from_torch(arg332_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg332_1 = None
    ttnn_add_123 = ttnn_decorators_ttnn_add(ttnn_from_torch_336, ttnn_matmul_165);  ttnn_from_torch_336 = ttnn_matmul_165 = None
    ttnn_experimental_view_373 = ttnn_decorators_ttnn_experimental_view(ttnn_add_123, [1, 256, 1024]);  ttnn_add_123 = None
    ttnn_add_208 = ttnn_decorators_ttnn_add(ttnn_experimental_view_373, ttnn_layer_norm_40);  ttnn_experimental_view_373 = ttnn_layer_norm_40 = None
    ttnn_from_torch_337 = ttnn_decorators_ttnn_from_torch(arg333_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg333_1 = None
    ttnn_from_torch_338 = ttnn_decorators_ttnn_from_torch(arg334_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg334_1 = None
    ttnn_layer_norm_41 = ttnn_decorators_ttnn_layer_norm(ttnn_add_208, epsilon = 1e-12, weight = ttnn_from_torch_337, bias = ttnn_from_torch_338);  ttnn_add_208 = ttnn_from_torch_337 = ttnn_from_torch_338 = None
    ttnn_experimental_view_374 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_41, [256, 1024])
    ttnn_from_torch_339 = ttnn_decorators_ttnn_from_torch(arg335_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg335_1 = None
    ttnn_transpose_145 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_339, 0, 1);  ttnn_from_torch_339 = None
    ttnn_matmul_166 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_374, ttnn_transpose_145);  ttnn_experimental_view_374 = ttnn_transpose_145 = None
    ttnn_from_torch_340 = ttnn_decorators_ttnn_from_torch(arg336_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg336_1 = None
    ttnn_add_124 = ttnn_decorators_ttnn_add(ttnn_from_torch_340, ttnn_matmul_166);  ttnn_from_torch_340 = ttnn_matmul_166 = None
    ttnn_experimental_view_375 = ttnn_decorators_ttnn_experimental_view(ttnn_add_124, [1, 256, 4096]);  ttnn_add_124 = None
    ttnn_gelu_20 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_375);  ttnn_experimental_view_375 = None
    ttnn_experimental_view_376 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_20, [256, 4096]);  ttnn_gelu_20 = None
    ttnn_from_torch_341 = ttnn_decorators_ttnn_from_torch(arg337_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg337_1 = None
    ttnn_transpose_146 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_341, 0, 1);  ttnn_from_torch_341 = None
    ttnn_matmul_167 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_376, ttnn_transpose_146);  ttnn_experimental_view_376 = ttnn_transpose_146 = None
    ttnn_from_torch_342 = ttnn_decorators_ttnn_from_torch(arg338_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg338_1 = None
    ttnn_add_125 = ttnn_decorators_ttnn_add(ttnn_from_torch_342, ttnn_matmul_167);  ttnn_from_torch_342 = ttnn_matmul_167 = None
    ttnn_experimental_view_377 = ttnn_decorators_ttnn_experimental_view(ttnn_add_125, [1, 256, 1024]);  ttnn_add_125 = None
    ttnn_add_209 = ttnn_decorators_ttnn_add(ttnn_experimental_view_377, ttnn_layer_norm_41);  ttnn_experimental_view_377 = ttnn_layer_norm_41 = None
    ttnn_from_torch_343 = ttnn_decorators_ttnn_from_torch(arg339_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg339_1 = None
    ttnn_from_torch_344 = ttnn_decorators_ttnn_from_torch(arg340_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg340_1 = None
    ttnn_layer_norm_42 = ttnn_decorators_ttnn_layer_norm(ttnn_add_209, epsilon = 1e-12, weight = ttnn_from_torch_343, bias = ttnn_from_torch_344);  ttnn_add_209 = ttnn_from_torch_343 = ttnn_from_torch_344 = None
    ttnn_experimental_view_378 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_42, [256, 1024])
    ttnn_from_torch_345 = ttnn_decorators_ttnn_from_torch(arg341_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg341_1 = None
    ttnn_transpose_147 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_345, 0, 1);  ttnn_from_torch_345 = None
    ttnn_matmul_168 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_378, ttnn_transpose_147);  ttnn_transpose_147 = None
    ttnn_from_torch_346 = ttnn_decorators_ttnn_from_torch(arg342_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg342_1 = None
    ttnn_add_126 = ttnn_decorators_ttnn_add(ttnn_from_torch_346, ttnn_matmul_168);  ttnn_from_torch_346 = ttnn_matmul_168 = None
    ttnn_experimental_view_379 = ttnn_decorators_ttnn_experimental_view(ttnn_add_126, [1, 256, 1024]);  ttnn_add_126 = None
    ttnn_from_torch_347 = ttnn_decorators_ttnn_from_torch(arg343_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg343_1 = None
    ttnn_transpose_148 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_347, 0, 1);  ttnn_from_torch_347 = None
    ttnn_matmul_169 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_378, ttnn_transpose_148);  ttnn_transpose_148 = None
    ttnn_from_torch_348 = ttnn_decorators_ttnn_from_torch(arg344_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg344_1 = None
    ttnn_add_127 = ttnn_decorators_ttnn_add(ttnn_from_torch_348, ttnn_matmul_169);  ttnn_from_torch_348 = ttnn_matmul_169 = None
    ttnn_experimental_view_381 = ttnn_decorators_ttnn_experimental_view(ttnn_add_127, [1, 256, 1024]);  ttnn_add_127 = None
    ttnn_reshape_86 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_381, [1, 256, 16, 64]);  ttnn_experimental_view_381 = None
    ttnn_permute_84 = ttnn_decorators_ttnn_permute(ttnn_reshape_86, [0, 2, 1, 3]);  ttnn_reshape_86 = None
    ttnn_from_torch_349 = ttnn_decorators_ttnn_from_torch(arg345_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg345_1 = None
    ttnn_transpose_149 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_349, 0, 1);  ttnn_from_torch_349 = None
    ttnn_matmul_170 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_378, ttnn_transpose_149);  ttnn_experimental_view_378 = ttnn_transpose_149 = None
    ttnn_from_torch_350 = ttnn_decorators_ttnn_from_torch(arg346_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg346_1 = None
    ttnn_add_128 = ttnn_decorators_ttnn_add(ttnn_from_torch_350, ttnn_matmul_170);  ttnn_from_torch_350 = ttnn_matmul_170 = None
    ttnn_experimental_view_383 = ttnn_decorators_ttnn_experimental_view(ttnn_add_128, [1, 256, 1024]);  ttnn_add_128 = None
    ttnn_reshape_87 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_383, [1, 256, 16, 64]);  ttnn_experimental_view_383 = None
    ttnn_permute_85 = ttnn_decorators_ttnn_permute(ttnn_reshape_87, [0, 2, 1, 3]);  ttnn_reshape_87 = None
    ttnn_reshape_88 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_379, [1, 256, 16, 64]);  ttnn_experimental_view_379 = None
    ttnn_permute_86 = ttnn_decorators_ttnn_permute(ttnn_reshape_88, [0, 2, 1, 3]);  ttnn_reshape_88 = None
    ttnn_transpose_150 = ttnn_decorators_ttnn_transpose(ttnn_permute_84, 3, 2);  ttnn_permute_84 = None
    ttnn_experimental_view_384 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_86, [16, 256, 64]);  ttnn_permute_86 = None
    ttnn_experimental_view_385 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_150, [16, 64, 256]);  ttnn_transpose_150 = None
    ttnn_matmul_171 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_384, ttnn_experimental_view_385);  ttnn_experimental_view_384 = ttnn_experimental_view_385 = None
    ttnn_experimental_view_386 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_171, [1, 16, 256, 256]);  ttnn_matmul_171 = None
    ttnn_multiply_22 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_386, 0.125);  ttnn_experimental_view_386 = None
    ttnn_add_210 = ttnn_decorators_ttnn_add(ttnn_multiply_22, ttnn_multiply);  ttnn_multiply_22 = None
    ttnn_softmax_21 = ttnn_decorators_ttnn_softmax(ttnn_add_210, -1, numeric_stable = True);  ttnn_add_210 = None
    ttnn_experimental_view_387 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_21, [16, 256, 256]);  ttnn_softmax_21 = None
    ttnn_experimental_view_388 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_85, [16, 256, 64]);  ttnn_permute_85 = None
    ttnn_matmul_172 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_387, ttnn_experimental_view_388);  ttnn_experimental_view_387 = ttnn_experimental_view_388 = None
    ttnn_experimental_view_389 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_172, [1, 16, 256, 64]);  ttnn_matmul_172 = None
    ttnn_permute_87 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_389, [0, 2, 1, 3]);  ttnn_experimental_view_389 = None
    ttnn_reshape_89 = ttnn_decorators_ttnn_reshape(ttnn_permute_87, [1, 256, 1024]);  ttnn_permute_87 = None
    ttnn_experimental_view_390 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_89, [256, 1024]);  ttnn_reshape_89 = None
    ttnn_from_torch_351 = ttnn_decorators_ttnn_from_torch(arg347_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg347_1 = None
    ttnn_transpose_151 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_351, 0, 1);  ttnn_from_torch_351 = None
    ttnn_matmul_173 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_390, ttnn_transpose_151);  ttnn_experimental_view_390 = ttnn_transpose_151 = None
    ttnn_from_torch_352 = ttnn_decorators_ttnn_from_torch(arg348_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg348_1 = None
    ttnn_add_129 = ttnn_decorators_ttnn_add(ttnn_from_torch_352, ttnn_matmul_173);  ttnn_from_torch_352 = ttnn_matmul_173 = None
    ttnn_experimental_view_391 = ttnn_decorators_ttnn_experimental_view(ttnn_add_129, [1, 256, 1024]);  ttnn_add_129 = None
    ttnn_add_211 = ttnn_decorators_ttnn_add(ttnn_experimental_view_391, ttnn_layer_norm_42);  ttnn_experimental_view_391 = ttnn_layer_norm_42 = None
    ttnn_from_torch_353 = ttnn_decorators_ttnn_from_torch(arg349_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg349_1 = None
    ttnn_from_torch_354 = ttnn_decorators_ttnn_from_torch(arg350_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg350_1 = None
    ttnn_layer_norm_43 = ttnn_decorators_ttnn_layer_norm(ttnn_add_211, epsilon = 1e-12, weight = ttnn_from_torch_353, bias = ttnn_from_torch_354);  ttnn_add_211 = ttnn_from_torch_353 = ttnn_from_torch_354 = None
    ttnn_experimental_view_392 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_43, [256, 1024])
    ttnn_from_torch_355 = ttnn_decorators_ttnn_from_torch(arg351_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg351_1 = None
    ttnn_transpose_152 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_355, 0, 1);  ttnn_from_torch_355 = None
    ttnn_matmul_174 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_392, ttnn_transpose_152);  ttnn_experimental_view_392 = ttnn_transpose_152 = None
    ttnn_from_torch_356 = ttnn_decorators_ttnn_from_torch(arg352_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg352_1 = None
    ttnn_add_130 = ttnn_decorators_ttnn_add(ttnn_from_torch_356, ttnn_matmul_174);  ttnn_from_torch_356 = ttnn_matmul_174 = None
    ttnn_experimental_view_393 = ttnn_decorators_ttnn_experimental_view(ttnn_add_130, [1, 256, 4096]);  ttnn_add_130 = None
    ttnn_gelu_21 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_393);  ttnn_experimental_view_393 = None
    ttnn_experimental_view_394 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_21, [256, 4096]);  ttnn_gelu_21 = None
    ttnn_from_torch_357 = ttnn_decorators_ttnn_from_torch(arg353_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg353_1 = None
    ttnn_transpose_153 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_357, 0, 1);  ttnn_from_torch_357 = None
    ttnn_matmul_175 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_394, ttnn_transpose_153);  ttnn_experimental_view_394 = ttnn_transpose_153 = None
    ttnn_from_torch_358 = ttnn_decorators_ttnn_from_torch(arg354_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg354_1 = None
    ttnn_add_131 = ttnn_decorators_ttnn_add(ttnn_from_torch_358, ttnn_matmul_175);  ttnn_from_torch_358 = ttnn_matmul_175 = None
    ttnn_experimental_view_395 = ttnn_decorators_ttnn_experimental_view(ttnn_add_131, [1, 256, 1024]);  ttnn_add_131 = None
    ttnn_add_212 = ttnn_decorators_ttnn_add(ttnn_experimental_view_395, ttnn_layer_norm_43);  ttnn_experimental_view_395 = ttnn_layer_norm_43 = None
    ttnn_from_torch_359 = ttnn_decorators_ttnn_from_torch(arg355_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg355_1 = None
    ttnn_from_torch_360 = ttnn_decorators_ttnn_from_torch(arg356_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg356_1 = None
    ttnn_layer_norm_44 = ttnn_decorators_ttnn_layer_norm(ttnn_add_212, epsilon = 1e-12, weight = ttnn_from_torch_359, bias = ttnn_from_torch_360);  ttnn_add_212 = ttnn_from_torch_359 = ttnn_from_torch_360 = None
    ttnn_experimental_view_396 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_44, [256, 1024])
    ttnn_from_torch_361 = ttnn_decorators_ttnn_from_torch(arg357_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg357_1 = None
    ttnn_transpose_154 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_361, 0, 1);  ttnn_from_torch_361 = None
    ttnn_matmul_176 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_396, ttnn_transpose_154);  ttnn_transpose_154 = None
    ttnn_from_torch_362 = ttnn_decorators_ttnn_from_torch(arg358_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg358_1 = None
    ttnn_add_132 = ttnn_decorators_ttnn_add(ttnn_from_torch_362, ttnn_matmul_176);  ttnn_from_torch_362 = ttnn_matmul_176 = None
    ttnn_experimental_view_397 = ttnn_decorators_ttnn_experimental_view(ttnn_add_132, [1, 256, 1024]);  ttnn_add_132 = None
    ttnn_from_torch_363 = ttnn_decorators_ttnn_from_torch(arg359_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg359_1 = None
    ttnn_transpose_155 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_363, 0, 1);  ttnn_from_torch_363 = None
    ttnn_matmul_177 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_396, ttnn_transpose_155);  ttnn_transpose_155 = None
    ttnn_from_torch_364 = ttnn_decorators_ttnn_from_torch(arg360_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg360_1 = None
    ttnn_add_133 = ttnn_decorators_ttnn_add(ttnn_from_torch_364, ttnn_matmul_177);  ttnn_from_torch_364 = ttnn_matmul_177 = None
    ttnn_experimental_view_399 = ttnn_decorators_ttnn_experimental_view(ttnn_add_133, [1, 256, 1024]);  ttnn_add_133 = None
    ttnn_reshape_90 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_399, [1, 256, 16, 64]);  ttnn_experimental_view_399 = None
    ttnn_permute_88 = ttnn_decorators_ttnn_permute(ttnn_reshape_90, [0, 2, 1, 3]);  ttnn_reshape_90 = None
    ttnn_from_torch_365 = ttnn_decorators_ttnn_from_torch(arg361_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg361_1 = None
    ttnn_transpose_156 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_365, 0, 1);  ttnn_from_torch_365 = None
    ttnn_matmul_178 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_396, ttnn_transpose_156);  ttnn_experimental_view_396 = ttnn_transpose_156 = None
    ttnn_from_torch_366 = ttnn_decorators_ttnn_from_torch(arg362_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg362_1 = None
    ttnn_add_134 = ttnn_decorators_ttnn_add(ttnn_from_torch_366, ttnn_matmul_178);  ttnn_from_torch_366 = ttnn_matmul_178 = None
    ttnn_experimental_view_401 = ttnn_decorators_ttnn_experimental_view(ttnn_add_134, [1, 256, 1024]);  ttnn_add_134 = None
    ttnn_reshape_91 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_401, [1, 256, 16, 64]);  ttnn_experimental_view_401 = None
    ttnn_permute_89 = ttnn_decorators_ttnn_permute(ttnn_reshape_91, [0, 2, 1, 3]);  ttnn_reshape_91 = None
    ttnn_reshape_92 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_397, [1, 256, 16, 64]);  ttnn_experimental_view_397 = None
    ttnn_permute_90 = ttnn_decorators_ttnn_permute(ttnn_reshape_92, [0, 2, 1, 3]);  ttnn_reshape_92 = None
    ttnn_transpose_157 = ttnn_decorators_ttnn_transpose(ttnn_permute_88, 3, 2);  ttnn_permute_88 = None
    ttnn_experimental_view_402 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_90, [16, 256, 64]);  ttnn_permute_90 = None
    ttnn_experimental_view_403 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_157, [16, 64, 256]);  ttnn_transpose_157 = None
    ttnn_matmul_179 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_402, ttnn_experimental_view_403);  ttnn_experimental_view_402 = ttnn_experimental_view_403 = None
    ttnn_experimental_view_404 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_179, [1, 16, 256, 256]);  ttnn_matmul_179 = None
    ttnn_multiply_23 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_404, 0.125);  ttnn_experimental_view_404 = None
    ttnn_add_213 = ttnn_decorators_ttnn_add(ttnn_multiply_23, ttnn_multiply);  ttnn_multiply_23 = None
    ttnn_softmax_22 = ttnn_decorators_ttnn_softmax(ttnn_add_213, -1, numeric_stable = True);  ttnn_add_213 = None
    ttnn_experimental_view_405 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_22, [16, 256, 256]);  ttnn_softmax_22 = None
    ttnn_experimental_view_406 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_89, [16, 256, 64]);  ttnn_permute_89 = None
    ttnn_matmul_180 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_405, ttnn_experimental_view_406);  ttnn_experimental_view_405 = ttnn_experimental_view_406 = None
    ttnn_experimental_view_407 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_180, [1, 16, 256, 64]);  ttnn_matmul_180 = None
    ttnn_permute_91 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_407, [0, 2, 1, 3]);  ttnn_experimental_view_407 = None
    ttnn_reshape_93 = ttnn_decorators_ttnn_reshape(ttnn_permute_91, [1, 256, 1024]);  ttnn_permute_91 = None
    ttnn_experimental_view_408 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_93, [256, 1024]);  ttnn_reshape_93 = None
    ttnn_from_torch_367 = ttnn_decorators_ttnn_from_torch(arg363_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg363_1 = None
    ttnn_transpose_158 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_367, 0, 1);  ttnn_from_torch_367 = None
    ttnn_matmul_181 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_408, ttnn_transpose_158);  ttnn_experimental_view_408 = ttnn_transpose_158 = None
    ttnn_from_torch_368 = ttnn_decorators_ttnn_from_torch(arg364_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg364_1 = None
    ttnn_add_135 = ttnn_decorators_ttnn_add(ttnn_from_torch_368, ttnn_matmul_181);  ttnn_from_torch_368 = ttnn_matmul_181 = None
    ttnn_experimental_view_409 = ttnn_decorators_ttnn_experimental_view(ttnn_add_135, [1, 256, 1024]);  ttnn_add_135 = None
    ttnn_add_214 = ttnn_decorators_ttnn_add(ttnn_experimental_view_409, ttnn_layer_norm_44);  ttnn_experimental_view_409 = ttnn_layer_norm_44 = None
    ttnn_from_torch_369 = ttnn_decorators_ttnn_from_torch(arg365_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg365_1 = None
    ttnn_from_torch_370 = ttnn_decorators_ttnn_from_torch(arg366_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg366_1 = None
    ttnn_layer_norm_45 = ttnn_decorators_ttnn_layer_norm(ttnn_add_214, epsilon = 1e-12, weight = ttnn_from_torch_369, bias = ttnn_from_torch_370);  ttnn_add_214 = ttnn_from_torch_369 = ttnn_from_torch_370 = None
    ttnn_experimental_view_410 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_45, [256, 1024])
    ttnn_from_torch_371 = ttnn_decorators_ttnn_from_torch(arg367_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg367_1 = None
    ttnn_transpose_159 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_371, 0, 1);  ttnn_from_torch_371 = None
    ttnn_matmul_182 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_410, ttnn_transpose_159);  ttnn_experimental_view_410 = ttnn_transpose_159 = None
    ttnn_from_torch_372 = ttnn_decorators_ttnn_from_torch(arg368_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg368_1 = None
    ttnn_add_136 = ttnn_decorators_ttnn_add(ttnn_from_torch_372, ttnn_matmul_182);  ttnn_from_torch_372 = ttnn_matmul_182 = None
    ttnn_experimental_view_411 = ttnn_decorators_ttnn_experimental_view(ttnn_add_136, [1, 256, 4096]);  ttnn_add_136 = None
    ttnn_gelu_22 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_411);  ttnn_experimental_view_411 = None
    ttnn_experimental_view_412 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_22, [256, 4096]);  ttnn_gelu_22 = None
    ttnn_from_torch_373 = ttnn_decorators_ttnn_from_torch(arg369_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg369_1 = None
    ttnn_transpose_160 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_373, 0, 1);  ttnn_from_torch_373 = None
    ttnn_matmul_183 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_412, ttnn_transpose_160);  ttnn_experimental_view_412 = ttnn_transpose_160 = None
    ttnn_from_torch_374 = ttnn_decorators_ttnn_from_torch(arg370_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg370_1 = None
    ttnn_add_137 = ttnn_decorators_ttnn_add(ttnn_from_torch_374, ttnn_matmul_183);  ttnn_from_torch_374 = ttnn_matmul_183 = None
    ttnn_experimental_view_413 = ttnn_decorators_ttnn_experimental_view(ttnn_add_137, [1, 256, 1024]);  ttnn_add_137 = None
    ttnn_add_215 = ttnn_decorators_ttnn_add(ttnn_experimental_view_413, ttnn_layer_norm_45);  ttnn_experimental_view_413 = ttnn_layer_norm_45 = None
    ttnn_from_torch_375 = ttnn_decorators_ttnn_from_torch(arg371_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg371_1 = None
    ttnn_from_torch_376 = ttnn_decorators_ttnn_from_torch(arg372_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg372_1 = None
    ttnn_layer_norm_46 = ttnn_decorators_ttnn_layer_norm(ttnn_add_215, epsilon = 1e-12, weight = ttnn_from_torch_375, bias = ttnn_from_torch_376);  ttnn_add_215 = ttnn_from_torch_375 = ttnn_from_torch_376 = None
    ttnn_experimental_view_414 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_46, [256, 1024])
    ttnn_from_torch_377 = ttnn_decorators_ttnn_from_torch(arg373_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg373_1 = None
    ttnn_transpose_161 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_377, 0, 1);  ttnn_from_torch_377 = None
    ttnn_matmul_184 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_414, ttnn_transpose_161);  ttnn_transpose_161 = None
    ttnn_from_torch_378 = ttnn_decorators_ttnn_from_torch(arg374_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg374_1 = None
    ttnn_add_138 = ttnn_decorators_ttnn_add(ttnn_from_torch_378, ttnn_matmul_184);  ttnn_from_torch_378 = ttnn_matmul_184 = None
    ttnn_experimental_view_415 = ttnn_decorators_ttnn_experimental_view(ttnn_add_138, [1, 256, 1024]);  ttnn_add_138 = None
    ttnn_from_torch_379 = ttnn_decorators_ttnn_from_torch(arg375_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg375_1 = None
    ttnn_transpose_162 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_379, 0, 1);  ttnn_from_torch_379 = None
    ttnn_matmul_185 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_414, ttnn_transpose_162);  ttnn_transpose_162 = None
    ttnn_from_torch_380 = ttnn_decorators_ttnn_from_torch(arg376_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg376_1 = None
    ttnn_add_139 = ttnn_decorators_ttnn_add(ttnn_from_torch_380, ttnn_matmul_185);  ttnn_from_torch_380 = ttnn_matmul_185 = None
    ttnn_experimental_view_417 = ttnn_decorators_ttnn_experimental_view(ttnn_add_139, [1, 256, 1024]);  ttnn_add_139 = None
    ttnn_reshape_94 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_417, [1, 256, 16, 64]);  ttnn_experimental_view_417 = None
    ttnn_permute_92 = ttnn_decorators_ttnn_permute(ttnn_reshape_94, [0, 2, 1, 3]);  ttnn_reshape_94 = None
    ttnn_from_torch_381 = ttnn_decorators_ttnn_from_torch(arg377_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg377_1 = None
    ttnn_transpose_163 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_381, 0, 1);  ttnn_from_torch_381 = None
    ttnn_matmul_186 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_414, ttnn_transpose_163);  ttnn_experimental_view_414 = ttnn_transpose_163 = None
    ttnn_from_torch_382 = ttnn_decorators_ttnn_from_torch(arg378_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg378_1 = None
    ttnn_add_140 = ttnn_decorators_ttnn_add(ttnn_from_torch_382, ttnn_matmul_186);  ttnn_from_torch_382 = ttnn_matmul_186 = None
    ttnn_experimental_view_419 = ttnn_decorators_ttnn_experimental_view(ttnn_add_140, [1, 256, 1024]);  ttnn_add_140 = None
    ttnn_reshape_95 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_419, [1, 256, 16, 64]);  ttnn_experimental_view_419 = None
    ttnn_permute_93 = ttnn_decorators_ttnn_permute(ttnn_reshape_95, [0, 2, 1, 3]);  ttnn_reshape_95 = None
    ttnn_reshape_96 = ttnn_decorators_ttnn_reshape(ttnn_experimental_view_415, [1, 256, 16, 64]);  ttnn_experimental_view_415 = None
    ttnn_permute_94 = ttnn_decorators_ttnn_permute(ttnn_reshape_96, [0, 2, 1, 3]);  ttnn_reshape_96 = None
    ttnn_transpose_164 = ttnn_decorators_ttnn_transpose(ttnn_permute_92, 3, 2);  ttnn_permute_92 = None
    ttnn_experimental_view_420 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_94, [16, 256, 64]);  ttnn_permute_94 = None
    ttnn_experimental_view_421 = ttnn_decorators_ttnn_experimental_view(ttnn_transpose_164, [16, 64, 256]);  ttnn_transpose_164 = None
    ttnn_matmul_187 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_420, ttnn_experimental_view_421);  ttnn_experimental_view_420 = ttnn_experimental_view_421 = None
    ttnn_experimental_view_422 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_187, [1, 16, 256, 256]);  ttnn_matmul_187 = None
    ttnn_multiply_24 = ttnn_decorators_ttnn_multiply(ttnn_experimental_view_422, 0.125);  ttnn_experimental_view_422 = None
    ttnn_add_216 = ttnn_decorators_ttnn_add(ttnn_multiply_24, ttnn_multiply);  ttnn_multiply_24 = ttnn_multiply = None
    ttnn_softmax_23 = ttnn_decorators_ttnn_softmax(ttnn_add_216, -1, numeric_stable = True);  ttnn_add_216 = None
    ttnn_experimental_view_423 = ttnn_decorators_ttnn_experimental_view(ttnn_softmax_23, [16, 256, 256]);  ttnn_softmax_23 = None
    ttnn_experimental_view_424 = ttnn_decorators_ttnn_experimental_view(ttnn_permute_93, [16, 256, 64]);  ttnn_permute_93 = None
    ttnn_matmul_188 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_423, ttnn_experimental_view_424);  ttnn_experimental_view_423 = ttnn_experimental_view_424 = None
    ttnn_experimental_view_425 = ttnn_decorators_ttnn_experimental_view(ttnn_matmul_188, [1, 16, 256, 64]);  ttnn_matmul_188 = None
    ttnn_permute_95 = ttnn_decorators_ttnn_permute(ttnn_experimental_view_425, [0, 2, 1, 3]);  ttnn_experimental_view_425 = None
    ttnn_reshape_97 = ttnn_decorators_ttnn_reshape(ttnn_permute_95, [1, 256, 1024]);  ttnn_permute_95 = None
    ttnn_experimental_view_426 = ttnn_decorators_ttnn_experimental_view(ttnn_reshape_97, [256, 1024]);  ttnn_reshape_97 = None
    ttnn_from_torch_383 = ttnn_decorators_ttnn_from_torch(arg379_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg379_1 = None
    ttnn_transpose_165 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_383, 0, 1);  ttnn_from_torch_383 = None
    ttnn_matmul_189 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_426, ttnn_transpose_165);  ttnn_experimental_view_426 = ttnn_transpose_165 = None
    ttnn_from_torch_384 = ttnn_decorators_ttnn_from_torch(arg380_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg380_1 = None
    ttnn_add_141 = ttnn_decorators_ttnn_add(ttnn_from_torch_384, ttnn_matmul_189);  ttnn_from_torch_384 = ttnn_matmul_189 = None
    ttnn_experimental_view_427 = ttnn_decorators_ttnn_experimental_view(ttnn_add_141, [1, 256, 1024]);  ttnn_add_141 = None
    ttnn_add_217 = ttnn_decorators_ttnn_add(ttnn_experimental_view_427, ttnn_layer_norm_46);  ttnn_experimental_view_427 = ttnn_layer_norm_46 = None
    ttnn_from_torch_385 = ttnn_decorators_ttnn_from_torch(arg381_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg381_1 = None
    ttnn_from_torch_386 = ttnn_decorators_ttnn_from_torch(arg382_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg382_1 = None
    ttnn_layer_norm_47 = ttnn_decorators_ttnn_layer_norm(ttnn_add_217, epsilon = 1e-12, weight = ttnn_from_torch_385, bias = ttnn_from_torch_386);  ttnn_add_217 = ttnn_from_torch_385 = ttnn_from_torch_386 = None
    ttnn_experimental_view_428 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_47, [256, 1024])
    ttnn_from_torch_387 = ttnn_decorators_ttnn_from_torch(arg383_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg383_1 = None
    ttnn_transpose_166 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_387, 0, 1);  ttnn_from_torch_387 = None
    ttnn_matmul_190 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_428, ttnn_transpose_166);  ttnn_experimental_view_428 = ttnn_transpose_166 = None
    ttnn_from_torch_388 = ttnn_decorators_ttnn_from_torch(arg384_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg384_1 = None
    ttnn_add_142 = ttnn_decorators_ttnn_add(ttnn_from_torch_388, ttnn_matmul_190);  ttnn_from_torch_388 = ttnn_matmul_190 = None
    ttnn_experimental_view_429 = ttnn_decorators_ttnn_experimental_view(ttnn_add_142, [1, 256, 4096]);  ttnn_add_142 = None
    ttnn_gelu_23 = ttnn_decorators_ttnn_gelu(ttnn_experimental_view_429);  ttnn_experimental_view_429 = None
    ttnn_experimental_view_430 = ttnn_decorators_ttnn_experimental_view(ttnn_gelu_23, [256, 4096]);  ttnn_gelu_23 = None
    ttnn_from_torch_389 = ttnn_decorators_ttnn_from_torch(arg385_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg385_1 = None
    ttnn_transpose_167 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_389, 0, 1);  ttnn_from_torch_389 = None
    ttnn_matmul_191 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_430, ttnn_transpose_167);  ttnn_experimental_view_430 = ttnn_transpose_167 = None
    ttnn_from_torch_390 = ttnn_decorators_ttnn_from_torch(arg386_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg386_1 = None
    ttnn_add_143 = ttnn_decorators_ttnn_add(ttnn_from_torch_390, ttnn_matmul_191);  ttnn_from_torch_390 = ttnn_matmul_191 = None
    ttnn_experimental_view_431 = ttnn_decorators_ttnn_experimental_view(ttnn_add_143, [1, 256, 1024]);  ttnn_add_143 = None
    ttnn_add_218 = ttnn_decorators_ttnn_add(ttnn_experimental_view_431, ttnn_layer_norm_47);  ttnn_experimental_view_431 = ttnn_layer_norm_47 = None
    ttnn_from_torch_391 = ttnn_decorators_ttnn_from_torch(arg387_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg387_1 = None
    ttnn_from_torch_392 = ttnn_decorators_ttnn_from_torch(arg388_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg388_1 = None
    ttnn_layer_norm_48 = ttnn_decorators_ttnn_layer_norm(ttnn_add_218, epsilon = 1e-12, weight = ttnn_from_torch_391, bias = ttnn_from_torch_392);  ttnn_add_218 = ttnn_from_torch_391 = ttnn_from_torch_392 = None
    ttnn_experimental_view_432 = ttnn_decorators_ttnn_experimental_view(ttnn_layer_norm_48, [256, 1024]);  ttnn_layer_norm_48 = None
    ttnn_from_torch_393 = ttnn_decorators_ttnn_from_torch(arg389_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg389_1 = None
    ttnn_transpose_168 = ttnn_decorators_ttnn_transpose(ttnn_from_torch_393, 0, 1);  ttnn_from_torch_393 = None
    ttnn_matmul_192 = ttnn_decorators_ttnn_matmul(ttnn_experimental_view_432, ttnn_transpose_168);  ttnn_experimental_view_432 = ttnn_transpose_168 = None
    ttnn_from_torch_394 = ttnn_decorators_ttnn_from_torch(arg390_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16);  arg390_1 = None
    ttnn_add_144 = ttnn_decorators_ttnn_add(ttnn_from_torch_394, ttnn_matmul_192);  ttnn_from_torch_394 = ttnn_matmul_192 = None
    ttnn_experimental_view_433 = ttnn_decorators_ttnn_experimental_view(ttnn_add_144, [1, 256, 2]);  ttnn_add_144 = None
    ttnn_to_layout_1 = ttnn_decorators_ttnn_to_layout(ttnn_experimental_view_433, ttnn_ROW_MAJOR_LAYOUT);  ttnn_experimental_view_433 = None
    ttnn_split = ttnn_decorators_ttnn_split(ttnn_to_layout_1, 1, 2);  ttnn_to_layout_1 = None
    getitem_49 = ttnn_split[0]
    getitem_50 = ttnn_split[1];  ttnn_split = None
    ttnn_to_layout_2 = ttnn_decorators_ttnn_to_layout(getitem_49, ttnn_TILE_LAYOUT);  getitem_49 = None
    ttnn_squeeze = ttnn_decorators_ttnn_squeeze(ttnn_to_layout_2, -1);  ttnn_to_layout_2 = None
    ttnn_to_layout_3 = ttnn_decorators_ttnn_to_layout(getitem_50, ttnn_TILE_LAYOUT);  getitem_50 = None
    ttnn_squeeze_1 = ttnn_decorators_ttnn_squeeze(ttnn_to_layout_3, -1);  ttnn_to_layout_3 = None
    ttnn_to_torch = ttnn_decorators_ttnn_to_torch(ttnn_squeeze, dtype = torch.bfloat16);  ttnn_squeeze = None
    ttnn_to_torch_1 = ttnn_decorators_ttnn_to_torch(ttnn_squeeze_1, dtype = torch.bfloat16);  ttnn_squeeze_1 = None
    return (ttnn_to_torch, ttnn_to_torch_1)



def run_bert(ttnn_Specified_Device, iteration):
     # Download model from cloud
    batch_size = 8
    model_name = "phiyodr/bert-large-finetuned-squad2"
    m = AutoModelForQuestionAnswering.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        
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

    for i in params:
        print(i)
    profiler.enable()
    profiler.start(iteration)
    after_attention(ttnn_Specified_Device, *full_args, inputs.data['input_ids'], inputs.data['token_type_ids'], inputs.data['attention_mask'])
    profiler.end(iteration)
    profiler.disable()
      
if __name__ == "__main__":
    device_id = 0
    ttnn_Specified_Device = ttnn.open_device(device_id=device_id)
    ttnn_Specified_Device.disable_and_clear_program_cache()
    disable_persistent_kernel_cache()
    run_bert(ttnn_Specified_Device, 0)
    
    enable_persistent_kernel_cache()
    ttnn_Specified_Device.enable_program_cache()
    
    for i in range(1, 5):
        run_bert(ttnn_Specified_Device, i)

    ttnn.close_device(ttnn_Specified_Device)
    for i in range(5):
        print(f"Runtime Iteration {i}: {profiler.get(i)}")